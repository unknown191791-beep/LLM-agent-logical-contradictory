#!/usr/bin/env python
"""
Main entry point for the Agent Workflow Reasoning Experiment.

Usage:
    # Mock mode (no API calls, for testing)
    python run_experiment.py --mock

    # Generate samples only (no agent runs)
    python run_experiment.py --generate-only

    # Full experiment with real API calls
    python run_experiment.py

    # Specify config files
    python run_experiment.py --config config/experiment.yaml --model config/model.yaml
"""

import argparse
import os
import sys
import logging
from datetime import datetime

# Ensure project root is on path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.core.types import ExperimentConfig
from src.generation.sampler import SampleGenerator
from src.generation.renderer import NaturalLanguageRenderer
from src.agents.llm_client import LLMClient
from src.agents.react import ReActAgent
from src.agents.plan_solve import PlanSolveAgent
from src.experiment.runner import ExperimentRunner
from src.metrics.compute import compute_all, compute_accuracy, compute_conflict_detection
from src.analysis.visualize import generate_all_charts, plot_token_comparison

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


def load_prompts(prompt_dir: str = "config/prompts") -> dict:
    """Load all prompt templates."""
    prompts = {}
    for name in ["base_task", "react_workflow", "plan_solve_workflow"]:
        path = os.path.join(prompt_dir, f"{name}.txt")
        if os.path.exists(path):
            with open(path, "r", encoding="utf-8") as f:
                prompts[name] = f.read()
        else:
            logger.warning(f"Prompt file not found: {path}")
            prompts[name] = ""
    return prompts


def print_metrics_report(metrics: dict) -> None:
    """Print a formatted metrics report."""
    print("\n" + "=" * 70)
    print("EXPERIMENT RESULTS")
    print("=" * 70)

    for wf_name, wf_metrics in metrics.items():
        print(f"\n{'─' * 50}")
        print(f"  {wf_name.upper()}")
        print(f"{'─' * 50}")

        acc = wf_metrics["accuracy"]
        print(f"  Accuracy:           {acc['accuracy']:.3f} "
              f"[95% CI: {acc['ci_lower']:.3f}, {acc['ci_upper']:.3f}]")
        print(f"  Correct / Total:    {acc['correct']} / {acc['total_with_answer']}")
        print(f"  Uncertain:          {acc['uncertain_count']}")
        print(f"  Unparseable:        {acc['unparseable_count']}")

        cd = wf_metrics["conflict_detection"]
        print(f"  Conflict Detection (explicit): {cd['explicit_rate']:.3f}")
        print(f"  Conflict Detection (implicit): {cd['implicit_rate']:.3f}")

        cost = wf_metrics["cost"]
        if cost:
            print(f"  Avg Tokens/Run:     {cost['avg_tokens_per_run']:.0f}")
            print(f"  Total Tokens:       {cost['total_tokens_sum']:,}")

        lat = wf_metrics["latency"]
        if lat:
            print(f"  Avg Latency:        {lat['avg_latency']:.2f}s")
            print(f"  Total Latency:      {lat['total_latency']:.2f}s")

        steps = wf_metrics["reasoning_steps"]
        if steps:
            print(f"  Avg Reasoning Steps:{steps['avg_steps']:.1f}")
            print(f"  Runs with Steps:    {steps['total_runs_with_steps']}")

    print("\n" + "=" * 70)


def main():
    parser = argparse.ArgumentParser(
        description="Agent Workflow Reasoning under Logical Conflict"
    )
    parser.add_argument(
        "--config", default="config/experiment.yaml",
        help="Path to experiment config"
    )
    parser.add_argument(
        "--model-config", default="config/model.yaml",
        help="Path to model config"
    )
    parser.add_argument(
        "--mock", action="store_true",
        help="Run with mock LLM (no API calls)"
    )
    parser.add_argument(
        "--generate-only", action="store_true",
        help="Only generate and save samples, skip agent runs"
    )
    parser.add_argument(
        "--no-charts", action="store_true",
        help="Skip chart generation"
    )
    parser.add_argument(
        "--model-profile", default=None,
        help="Model profile to use (haiku/sonnet)"
    )
    args = parser.parse_args()

    print("=" * 70)
    print("Agent Workflow Reasoning Experiment")
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 70)

    # ── Load Configuration ──
    print("\n[1/5] Loading configuration...")
    config = ExperimentConfig.from_yaml(args.config)
    print(f"  Config: {args.config}")
    print(f"  Factor design: {len(config.chain_lengths)} chain_lengths x "
          f"{len(config.conflict_counts)} conflict_counts x "
          f"{len(config.noise_counts)} noise_counts")
    print(f"  Samples per condition: {config.samples_per_condition}")
    print(f"  Workflows: {config.workflows}")
    print(f"  Repeats: {config.repeats_per_sample}")

    # ── Generate Samples ──
    print("\n[2/5] Generating samples...")
    gen = SampleGenerator(config)
    samples = gen.generate_all()
    gen.print_summary(samples)

    # Save samples
    sample_path = gen.save(samples, os.path.join(config.data_dir, "samples"))

    if args.generate_only:
        print("\nSamples generated. Use --no-generate-only to run experiments.")
        return

    # ── Load Prompts ──
    print("\n[3/5] Loading prompts and initializing agents...")
    prompts = load_prompts()
    for k, v in prompts.items():
        print(f"  {k}: {len(v)} chars")

    # ── Initialize LLM ──
    llm = LLMClient.from_yaml(
        config_path=args.model_config,
        profile=args.model_profile,
        mock_mode=args.mock,
    )
    print(f"  LLM: {llm.model} (mock={args.mock})")

    # ── Create Agents ──
    agents = {}
    if "react" in config.workflows:
        agents["react"] = ReActAgent(
            llm_client=llm,
            task_prompt=prompts.get("base_task", ""),
            workflow_prompt=prompts.get("react_workflow", ""),
        )
        print(f"  Agent: react (ReAct)")

    if "plan_solve" in config.workflows:
        agents["plan_solve"] = PlanSolveAgent(
            llm_client=llm,
            task_prompt=prompts.get("base_task", ""),
            workflow_prompt=prompts.get("plan_solve_workflow", ""),
        )
        print(f"  Agent: plan_solve (Plan-and-Solve)")

    # ── Run Experiment ──
    print("\n[4/5] Running experiment...")
    runner = ExperimentRunner(
        config=config,
        samples=samples,
        agents=agents,
        llm_client=llm,
        checkpoint_dir=config.results_dir,
    )
    results = runner.run(verbose=not args.mock)

    # ── Compute Metrics ──
    print("\n[5/5] Computing metrics and generating charts...")
    metrics = compute_all(results, samples)
    print_metrics_report(metrics)

    # ── Generate Charts ──
    if not args.no_charts and not args.mock:
        print("\nGenerating charts...")
        chart_paths = generate_all_charts(
            results, samples,
            output_dir=config.figures_dir,
        )
        plot_token_comparison(
            results,
            output_path=os.path.join(config.figures_dir, "token_comparison.png"),
        )
        print(f"  Generated {len(chart_paths) + 1} chart(s)")

    # ── Summary ──
    print("\n" + "=" * 70)
    print("Experiment Complete")
    print(f"  Samples: {len(samples)}")
    print(f"  Results: {len(results)}")
    print(f"  Sample file: {sample_path}")
    print(f"  Results file: {runner._checkpoint_path}")
    if llm.total_calls > 0:
        print(f"  Total API calls: {llm.total_calls}")
        print(f"  Total cost: ${llm.total_cost_usd:.4f}")
    print(f"  Charts: {config.figures_dir}/")
    print("=" * 70)


if __name__ == "__main__":
    main()
