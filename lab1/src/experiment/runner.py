"""
Experiment runner: main loop executing samples across workflows.

Supports:
  - Adaptive repeats (stop early if all runs agree)
  - Rate limiting between API calls
  - Checkpoint/resume for long experiments
  - Progress display with tqdm
"""

import json
import os
import time
import logging
from datetime import datetime, timezone
from typing import Optional

from src.core.types import Sample, AgentResult, ExperimentConfig
from src.agents.llm_client import LLMClient
from src.agents.base import AgentRunner

logger = logging.getLogger(__name__)


class ExperimentRunner:
    """
    Orchestrates the experiment: for each sample, for each workflow,
    runs N repeats and collects results.

    Usage:
        config = ExperimentConfig.from_yaml("config/experiment.yaml")
        samples = SampleGenerator(config).generate_all()
        llm = LLMClient.from_yaml("config/model.yaml")
        agents = {
            "react": ReActAgent(llm, task_prompt, react_prompt),
            "plan_solve": PlanSolveAgent(llm, task_prompt, ps_prompt),
        }
        runner = ExperimentRunner(config, samples, agents, llm)
        results = runner.run()
    """

    def __init__(
        self,
        config: ExperimentConfig,
        samples: list[Sample],
        agents: dict[str, AgentRunner],
        llm_client: LLMClient,
        checkpoint_dir: str = "data",
    ):
        self.config = config
        self.samples = samples
        self.agents = agents
        self.llm = llm_client
        self.checkpoint_dir = checkpoint_dir
        self.results: list[AgentResult] = []
        self._checkpoint_path: Optional[str] = None

    # ── Main Run Loop ────────────────────────────────────────────────────

    def run(self, verbose: bool = True) -> list[AgentResult]:
        """
        Execute the full experiment.

        Progress is saved incrementally to a JSONL checkpoint file.
        On interruption, can be resumed.

        Returns:
            List of all AgentResults.
        """
        total = (
            len(self.samples)
            * len(self.agents)
            * self.config.repeats_per_sample
        )
        print(f"\nExperiment: {len(self.samples)} samples x "
              f"{len(self.agents)} workflows x "
              f"{self.config.repeats_per_sample} repeats = {total} runs\n")

        # Setup checkpoint
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self._checkpoint_path = os.path.join(
            self.checkpoint_dir, f"results_{timestamp}.jsonl"
        )

        # Try to use tqdm for progress, fall back to manual
        try:
            from tqdm import tqdm
            pbar = tqdm(total=total, desc="Running")
            _use_tqdm = True
        except ImportError:
            pbar = None
            _use_tqdm = False

        run_count = 0
        completed_samples: dict[str, dict[str, int]] = {}  # sample_id -> {workflow: runs_done}

        for sample in self.samples:
            completed_samples[sample.id] = {}

            for wf_name in self.config.workflows:
                if wf_name not in self.agents:
                    logger.warning(f"Workflow '{wf_name}' not found in agents, skipping")
                    continue

                agent = self.agents[wf_name]

                # Adaptive repeats
                answers = []
                max_repeats = (
                    self.config.max_total_repeats
                    if self.config.adaptive_repeats
                    else self.config.repeats_per_sample
                )
                min_repeats = self.config.repeats_per_sample

                for run_idx in range(max_repeats):
                    # Deterministic shuffle seed per run
                    shuffle_seed = hash(sample.id) ^ (run_idx * 31337) & 0x7FFFFFFF

                    result = agent.run(sample, shuffle_seed=shuffle_seed)
                    result.run_index = run_idx

                    # Compare with ground truth
                    if result.final_answer is not None:
                        result.is_correct = (result.final_answer == sample.ground_truth)

                    self.results.append(result)
                    self._append_checkpoint(result)
                    answers.append(result.final_answer)

                    run_count += 1
                    if _use_tqdm:
                        pbar.update(1)
                    elif verbose and run_count % 10 == 0:
                        print(f"  [{run_count}/{total}] {sample.id} {wf_name} r{run_idx}")

                    # Rate limiting
                    if self.config.rate_limit_delay > 0:
                        time.sleep(self.config.rate_limit_delay)

                    # Adaptive early stop: if all answers so far agree
                    if (
                        self.config.adaptive_repeats
                        and run_idx >= min_repeats - 1
                        and self._all_agree(answers)
                    ):
                        break

                completed_samples[sample.id][wf_name] = len(answers)

        if _use_tqdm:
            pbar.close()

        # Print summary
        print(f"\nExperiment complete: {run_count} runs")
        self.llm.print_usage_summary()

        return self.results

    def _all_agree(self, answers: list) -> bool:
        """Check if all non-None answers in the list are the same."""
        valid = [a for a in answers if a is not None]
        if len(valid) < 2:
            return False
        return all(a == valid[0] for a in valid)

    # ── Checkpointing ────────────────────────────────────────────────────

    def _append_checkpoint(self, result: AgentResult) -> None:
        """Append a single result to the checkpoint file."""
        if not self._checkpoint_path:
            return
        try:
            record = self._result_to_dict(result)
            with open(self._checkpoint_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
        except Exception as e:
            logger.warning(f"Failed to write checkpoint: {e}")

    def _result_to_dict(self, result: AgentResult) -> dict:
        """Serialize an AgentResult to a JSON-safe dict."""
        return {
            "sample_id": result.sample_id,
            "workflow_name": result.workflow_name,
            "run_index": result.run_index,
            "seed": result.seed,
            "final_answer": result.final_answer,
            "extracted_answer_text": result.extracted_answer_text,
            "raw_response": result.raw_response if self.config.save_raw_responses else "",
            "answered_uncertain": result.answered_uncertain,
            "conflict_detected_explicit": result.conflict_detected_explicit,
            "conflict_detected_implicit": result.conflict_detected_implicit,
            "conflict_keywords_found": result.conflict_keywords_found,
            "total_tokens": result.total_tokens,
            "prompt_tokens": result.prompt_tokens,
            "completion_tokens": result.completion_tokens,
            "latency_seconds": result.latency_seconds,
            "reasoning_steps": result.reasoning_steps,
            "reasoning_trace": [
                {"step_index": s.step_index, "step_type": s.step_type, "content": s.content}
                for s in result.reasoning_trace
            ] if self.config.save_reasoning_traces else [],
            "model_name": result.model_name,
            "prompt_version": result.prompt_version,
            "timestamp": result.timestamp,
            "is_correct": result.is_correct,
            "error": result.error,
        }

    # ── Result Access ────────────────────────────────────────────────────

    def get_results_for(self, workflow_name: str) -> list[AgentResult]:
        """Filter results by workflow."""
        return [r for r in self.results if r.workflow_name == workflow_name]

    def get_results_for_sample(self, sample_id: str) -> list[AgentResult]:
        """Filter results by sample."""
        return [r for r in self.results if r.sample_id == sample_id]
