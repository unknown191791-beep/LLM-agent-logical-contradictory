"""
Auto-generate experimental charts.

Produces:
  1. Accuracy vs Chain Length
  2. Accuracy vs Conflict Count (as proxy for MIS)
  3. Accuracy vs Noise Count
"""

import os
from collections import defaultdict
from typing import Optional

from src.core.types import AgentResult, Sample
from src.metrics.compute import compute_breakdown


def plot_accuracy_vs_param(
    results: list[AgentResult],
    samples: list[Sample],
    param: str,
    title: str,
    xlabel: str,
    output_path: str,
    figsize: tuple = (8, 5),
) -> None:
    """
    Generate a bar/line chart: Accuracy vs parameter.

    Args:
        results: All experiment results.
        samples: All samples.
        param: Metadata field to group by.
        title: Chart title.
        xlabel: X-axis label.
        output_path: Where to save the figure.
    """
    try:
        import matplotlib.pyplot as plt
        import numpy as np
    except ImportError:
        print("matplotlib not installed. Install with: pip install matplotlib")
        return

    breakdown = compute_breakdown(results, samples, param)

    if not breakdown:
        print(f"No data to plot for {param}")
        return

    fig, ax = plt.subplots(figsize=figsize)

    workflows = sorted(breakdown.keys())
    colors = ["#2196F3", "#FF5722", "#4CAF50", "#9C27B0"]
    markers = ["o", "s", "^", "D"]
    bar_width = 0.35
    x_offset = np.arange(len(next(iter(breakdown.values())))) if breakdown else []

    for i, wf_name in enumerate(workflows):
        data = breakdown[wf_name]
        param_vals = sorted(data.keys())
        accuracies = [data[v] for v in param_vals]

        if len(x_offset) == 0:
            x_offset = np.arange(len(param_vals))

        offset = (i - len(workflows) / 2 + 0.5) * bar_width
        bars = ax.bar(
            x_offset + offset,
            accuracies,
            bar_width / max(len(workflows), 1),
            label=wf_name,
            color=colors[i % len(colors)],
            alpha=0.85,
            edgecolor="white",
            linewidth=0.5,
        )

        # Add value labels on bars
        for bar, acc in zip(bars, accuracies):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.01,
                f"{acc:.2f}",
                ha="center",
                va="bottom",
                fontsize=8,
            )

    ax.set_xlabel(xlabel, fontsize=12)
    ax.set_ylabel("Accuracy", fontsize=12)
    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.set_xticks(x_offset)
    ax.set_xticklabels(param_vals)
    ax.set_ylim(0, 1.1)
    ax.legend(loc="lower right", fontsize=10)
    ax.grid(axis="y", alpha=0.3, linestyle="--")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {output_path}")


def generate_all_charts(
    results: list[AgentResult],
    samples: list[Sample],
    output_dir: str = "outputs/figures",
) -> list[str]:
    """
    Generate all 3 standard charts.

    Returns:
        List of output file paths.
    """
    os.makedirs(output_dir, exist_ok=True)
    paths = []

    # 1. Accuracy vs Chain Length (proxy for Depth)
    path1 = os.path.join(output_dir, "accuracy_vs_chain_length.png")
    plot_accuracy_vs_param(
        results, samples,
        param="chain_length",
        title="Accuracy vs Chain Length (Depth)",
        xlabel="Chain Length (reasoning hops)",
        output_path=path1,
    )
    paths.append(path1)

    # 2. Accuracy vs Conflict Count
    path2 = os.path.join(output_dir, "accuracy_vs_conflict_count.png")
    plot_accuracy_vs_param(
        results, samples,
        param="conflict_count",
        title="Accuracy vs Conflict Count",
        xlabel="Conflict Count",
        output_path=path2,
    )
    paths.append(path2)

    # 3. Accuracy vs Noise Count
    path3 = os.path.join(output_dir, "accuracy_vs_noise_count.png")
    plot_accuracy_vs_param(
        results, samples,
        param="noise_count",
        title="Accuracy vs Noise Count",
        xlabel="Noise Fact Count",
        output_path=path3,
    )
    paths.append(path3)

    return paths


def plot_token_comparison(
    results: list[AgentResult],
    output_path: str = "outputs/figures/token_comparison.png",
) -> None:
    """Bar chart comparing token usage across workflows."""
    try:
        import matplotlib.pyplot as plt
        import numpy as np
    except ImportError:
        return

    workflow_tokens = defaultdict(list)
    for r in results:
        workflow_tokens[r.workflow_name].append(r.total_tokens)

    if not workflow_tokens:
        return

    fig, ax = plt.subplots(figsize=(6, 4))
    workflows = sorted(workflow_tokens.keys())
    means = [np.mean(workflow_tokens[w]) for w in workflows]
    stds = [np.std(workflow_tokens[w]) for w in workflows]
    colors = ["#2196F3", "#FF5722", "#4CAF50", "#9C27B0"]

    bars = ax.bar(workflows, means, color=colors[:len(workflows)], alpha=0.85,
                  edgecolor="white", linewidth=0.5)
    ax.errorbar(workflows, means, yerr=stds, fmt="none", ecolor="black",
                capsize=5, linewidth=1)

    for bar, mean in zip(bars, means):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + max(stds) * 0.1,
                f"{mean:.0f}", ha="center", fontsize=10)

    ax.set_ylabel("Avg Tokens per Run", fontsize=12)
    ax.set_title("Token Usage Comparison", fontsize=14, fontweight="bold")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y", alpha=0.3, linestyle="--")

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {output_path}")
