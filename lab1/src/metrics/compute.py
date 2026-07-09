"""
Metrics computation for experiment results.

Implements the 5 core metrics:
  1. Accuracy (with bootstrap CI)
  2. Conflict Detection Rate
  3. Token Cost
  4. Latency
  5. Reasoning Steps
"""

import math
from collections import defaultdict
from typing import Optional

from src.core.types import AgentResult, Sample


def compute_accuracy(
    results: list[AgentResult],
    samples: list[Sample],
) -> dict:
    """
    Compute accuracy metrics.

    Returns:
        dict with:
          - accuracy: overall accuracy
          - ci_lower, ci_upper: 95% bootstrap CI
          - correct: number correct
          - total_with_answer: number with parseable answers
          - total_runs: total runs
          - uncertain_count: number of UNCERTAIN responses
          - unparseable_count: number of unparseable responses
    """
    sample_gt = {s.id: s.ground_truth for s in samples}

    correct = 0
    total_with_answer = 0
    uncertain = 0
    unparseable = 0

    for r in results:
        if r.final_answer is None:
            if r.answered_uncertain:
                uncertain += 1
            else:
                unparseable += 1
        else:
            total_with_answer += 1
            if r.final_answer == sample_gt.get(r.sample_id):
                correct += 1

    accuracy = correct / total_with_answer if total_with_answer > 0 else 0.0

    # Bootstrap 95% CI (simple normal approximation)
    if total_with_answer > 0:
        se = math.sqrt(accuracy * (1 - accuracy) / total_with_answer)
        ci_lower = max(0.0, accuracy - 1.96 * se)
        ci_upper = min(1.0, accuracy + 1.96 * se)
    else:
        ci_lower = ci_upper = 0.0

    return {
        "accuracy": accuracy,
        "ci_lower": ci_lower,
        "ci_upper": ci_upper,
        "correct": correct,
        "total_with_answer": total_with_answer,
        "total_runs": len(results),
        "uncertain_count": uncertain,
        "unparseable_count": unparseable,
    }


def compute_conflict_detection(
    results: list[AgentResult],
    samples: list[Sample],
) -> dict:
    """
    Compute conflict detection rate.

    Only evaluates on samples that actually have conflicts.

    Returns:
        dict with:
          - explicit_rate: detection rate via explicit CONFLICT_DETECTED field
          - implicit_rate: detection rate via keyword matching
          - total_conflict_samples: number of samples with conflicts
          - total_conflict_runs: total runs on conflict samples
    """
    conflict_sample_ids = {s.id for s in samples if s.metadata.conflict_count > 0}

    conflict_results = [r for r in results if r.sample_id in conflict_sample_ids]

    if not conflict_results:
        return {
            "explicit_rate": 0.0,
            "implicit_rate": 0.0,
            "total_conflict_samples": len(conflict_sample_ids),
            "total_conflict_runs": 0,
        }

    explicit_count = sum(
        1 for r in conflict_results
        if r.conflict_detected_explicit is True
    )
    implicit_count = sum(
        1 for r in conflict_results
        if r.conflict_detected_implicit
    )

    return {
        "explicit_rate": explicit_count / len(conflict_results),
        "implicit_rate": implicit_count / len(conflict_results),
        "total_conflict_samples": len(conflict_sample_ids),
        "total_conflict_runs": len(conflict_results),
    }


def compute_cost(
    results: list[AgentResult],
) -> dict:
    """
    Compute token cost statistics.

    Returns:
        dict with token and cost statistics per run.
    """
    if not results:
        return {}

    tokens = [r.total_tokens for r in results]
    prompt_tokens = [r.prompt_tokens for r in results]
    completion_tokens = [r.completion_tokens for r in results]

    return {
        "total_tokens_sum": sum(tokens),
        "avg_tokens_per_run": sum(tokens) / len(tokens),
        "median_tokens_per_run": _median(tokens),
        "total_prompt_tokens": sum(prompt_tokens),
        "total_completion_tokens": sum(completion_tokens),
        "min_tokens": min(tokens),
        "max_tokens": max(tokens),
    }


def compute_latency(results: list[AgentResult]) -> dict:
    """Compute latency statistics."""
    if not results:
        return {}

    lats = [r.latency_seconds for r in results]

    return {
        "avg_latency": sum(lats) / len(lats),
        "median_latency": _median(lats),
        "total_latency": sum(lats),
        "min_latency": min(lats),
        "max_latency": max(lats),
    }


def compute_reasoning_steps(results: list[AgentResult]) -> dict:
    """Compute reasoning step statistics."""
    if not results:
        return {}

    steps = [r.reasoning_steps for r in results]

    return {
        "avg_steps": sum(steps) / len(steps),
        "median_steps": _median(steps),
        "min_steps": min(steps),
        "max_steps": max(steps),
        "total_runs_with_steps": sum(1 for s in steps if s > 0),
    }


def compute_all(
    results: list[AgentResult],
    samples: list[Sample],
) -> dict:
    """Compute all metrics, grouped by workflow."""
    metrics = {}
    for wf_name in sorted(set(r.workflow_name for r in results)):
        wf_results = [r for r in results if r.workflow_name == wf_name]
        metrics[wf_name] = {
            "accuracy": compute_accuracy(wf_results, samples),
            "conflict_detection": compute_conflict_detection(wf_results, samples),
            "cost": compute_cost(wf_results),
            "latency": compute_latency(wf_results),
            "reasoning_steps": compute_reasoning_steps(wf_results),
        }
    return metrics


def compute_breakdown(
    results: list[AgentResult],
    samples: list[Sample],
    param: str,
) -> dict:
    """
    Compute accuracy broken down by a parameter value.

    Args:
        results: All results.
        samples: All samples (for metadata lookup).
        param: One of "chain_length", "conflict_count", "noise_count".

    Returns:
        dict: {workflow: {param_value: accuracy}}
    """
    sample_map = {s.id: s for s in samples}
    breakdown = defaultdict(lambda: defaultdict(list))

    for r in results:
        sample = sample_map.get(r.sample_id)
        if sample is None:
            continue
        val = getattr(sample.metadata, param, None)
        if val is None:
            continue
        if r.final_answer is not None:
            correct = r.final_answer == sample.ground_truth
            breakdown[r.workflow_name][val].append(correct)

    result = {}
    for wf_name, vals in breakdown.items():
        result[wf_name] = {}
        for val, corrects in sorted(vals.items()):
            result[wf_name][val] = sum(corrects) / len(corrects) if corrects else 0.0

    return result


def _median(data: list) -> float:
    """Compute median of a list."""
    if not data:
        return 0.0
    sorted_data = sorted(data)
    n = len(sorted_data)
    if n % 2 == 0:
        return (sorted_data[n // 2 - 1] + sorted_data[n // 2]) / 2
    return sorted_data[n // 2]
