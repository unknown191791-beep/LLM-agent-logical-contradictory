"""
End-to-end smoke test using mock LLM (no API calls required).

Validates the full pipeline:
  Sample Generation -> Agent Execution -> Result Collection -> Analysis
"""

import sys
import os
import json
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.core.types import ExperimentConfig
from src.generation.sampler import SampleGenerator
from src.generation.renderer import NaturalLanguageRenderer
from src.agents.llm_client import LLMClient
from src.agents.react import ReActAgent
from src.agents.plan_solve import PlanSolveAgent


def load_prompt_templates():
    """Load prompt templates from config files."""
    base_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "config", "prompts")

    def read(path):
        full = os.path.join(base_dir, path)
        if os.path.exists(full):
            with open(full, "r") as f:
                return f.read()
        return ""

    return {
        "task": read("base_task.txt"),
        "react": read("react_workflow.txt"),
        "plan_solve": read("plan_solve_workflow.txt"),
    }


def run_smoke_test():
    """Full pipeline smoke test with mock LLM."""
    print("=" * 60)
    print("SMOKE TEST: End-to-End Pipeline (Mock Mode)")
    print("=" * 60)

    # -- 1. Load prompts --
    prompts = load_prompt_templates()
    print(f"\nPrompt templates loaded:")
    for k, v in prompts.items():
        print(f"  {k}: {len(v)} chars")

    # -- 2. Generate samples --
    config = ExperimentConfig(
        chain_lengths=[2, 3],
        conflict_counts=[0, 1],
        noise_counts=[0, 2],
        samples_per_condition=3,
        seed=42,
    )
    gen = SampleGenerator(config)
    samples = gen.generate_all()
    gen.print_summary(samples)
    print(f"\nGenerated {len(samples)} samples across {gen.total_conditions} conditions")

    # -- 3. Print sample examples --
    renderer = NaturalLanguageRenderer(seed=42)
    print("\n-- Sample Preview --")
    for sample in samples[:3]:
        text = renderer.render_sample_prompt(sample, shuffle_seed=0)
        print(f"\n[{sample.id}] chain_len={sample.metadata.chain_length}, "
              f"conflicts={sample.metadata.conflict_count}, "
              f"noise={sample.metadata.noise_count}, "
              f"GT={sample.ground_truth}")
        print(text[:300])
        print("...")
        print(f"  GT rationale: {sample.ground_truth_rationale}")

    # -- 4. Create mock LLM client --
    llm = LLMClient(
        model="claude-haiku-3-5",
        mock_mode=True,
    )
    print(f"\nLLM client: mock_mode={llm.mock_mode}")

    # -- 5. Create agents --
    react = ReActAgent(
        llm_client=llm,
        task_prompt=prompts["task"],
        workflow_prompt=prompts["react"],
    )
    plan_solve = PlanSolveAgent(
        llm_client=llm,
        task_prompt=prompts["task"],
        workflow_prompt=prompts["plan_solve"],
    )
    print(f"Agents: {react.name}, {plan_solve.name}")

    # -- 6. Run on 3 samples --
    test_samples = samples[:3]
    results = []

    for sample in test_samples:
        for agent in [react, plan_solve]:
            for run_idx in range(2):  # 2 repeats
                result = agent.run(sample, shuffle_seed=run_idx * 100)
                result.run_index = run_idx
                # Compare with ground truth
                if result.final_answer is not None:
                    result.is_correct = (result.final_answer == sample.ground_truth)
                results.append(result)

                status = "OK" if result.is_correct else ("??" if result.final_answer is None else "XX")
                print(f"  [{sample.id}] {agent.name} run={run_idx}: "
                      f"answer={result.final_answer}, GT={sample.ground_truth} {status}, "
                      f"conflict_explicit={result.conflict_detected_explicit}, "
                      f"conflict_implicit={result.conflict_detected_implicit}, "
                      f"tokens={result.total_tokens}, "
                      f"latency={result.latency_seconds:.2f}s")

    # -- 7. Compute metrics --
    print("\n-- Metrics Summary --")
    for wf_name in ["react", "plan_solve"]:
        wf_results = [r for r in results if r.workflow_name == wf_name]
        correct = sum(1 for r in wf_results if r.is_correct)
        total_with_answer = sum(1 for r in wf_results if r.final_answer is not None)
        accuracy = correct / total_with_answer if total_with_answer > 0 else 0
        avg_tokens = sum(r.total_tokens for r in wf_results) / len(wf_results) if wf_results else 0
        avg_latency = sum(r.latency_seconds for r in wf_results) / len(wf_results) if wf_results else 0

        # Conflict detection rate (on samples that have conflicts)
        conflict_samples_results = [
            r for r in wf_results
            if r.sample_id in [s.id for s in test_samples if s.metadata.conflict_count > 0]
        ]
        cd_rate = (
            sum(1 for r in conflict_samples_results if r.conflict_detected_explicit)
            / len(conflict_samples_results)
            if conflict_samples_results else 0
        )

        print(f"  {wf_name}:")
        print(f"    Accuracy:           {accuracy:.2f} ({correct}/{total_with_answer})")
        print(f"    Avg tokens/call:    {avg_tokens:.0f}")
        print(f"    Avg latency:        {avg_latency:.2f}s")
        print(f"    Conflict detection: {cd_rate:.2f}")

    # -- 8. Verify result serialization --
    print("\n-- Serialization Check --")
    sample_result = {
        "sample_id": results[0].sample_id,
        "workflow": results[0].workflow_name,
        "answer": results[0].final_answer,
        "tokens": results[0].total_tokens,
        "latency": results[0].latency_seconds,
        "model": results[0].model_name,
    }
    json_str = json.dumps(sample_result, indent=2)
    print(f"  Result JSON: {len(json_str)} bytes, valid JSON: True")

    # -- 9. LLM usage summary --
    llm.print_usage_summary()

    print("\n" + "=" * 60)
    print("SMOKE TEST PASSED")
    print("=" * 60)

    return results


if __name__ == "__main__":
    run_smoke_test()
