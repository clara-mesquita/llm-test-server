#!/usr/bin/env python3
"""Compare parameter-tier averages (e.g. 4b vs 8b) from results_average.json
(output of ollama-results-avg.py), for the metrics TTFT, prompt_tps, eval_tps.

TTFT is not measured by the benchmark; load_duration_s (model load time)
is used instead.

results_average.json layout: {tier: {family: {size: {metric: {mean, std}}}}}

Usage: python ollama-compare-avg.py [results_json] [tier_a] [tier_b]
       defaults: last-results-imgs/results_average.json, tiers 4b and 8b
Writes: compare-TTFT.png, compare-prompt_tps.png, compare-eval_tps.png
        (one figure per metric; one subplot per prompt size, grouped bars
        tier_a vs tier_b per family, error bars = std dev)
"""

import json
import os
import sys

METRICS = {
    "TTFT": "load_duration_s",  # model load time, in seconds
    "prompt_tps": "prompt_tps",
    "eval_tps": "eval_tps",
    "total_duration_s": "total_duration_s",
    "prompt_eval_duration_s": "prompt_eval_duration_s",
    "eval_duration_s": "eval_duration_s",
    "prompt_eval_count": "prompt_eval_count",
    "eval_count": "eval_count",
}
SIZES = ["small", "medium", "large"]


def tier_colors(tiers):
    """One color per parameter tier, more vivid for larger tiers."""
    import matplotlib.colors as mcolors
    import matplotlib.pyplot as plt
    import numpy as np
    cmap = plt.get_cmap("YlOrRd")  # pale yellow -> vivid red
    return {t: mcolors.to_hex(cmap(v))
            for t, v in zip(tiers, np.linspace(0.3, 0.9, len(tiers)))}


def load(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def main():
    path = sys.argv[1] if len(sys.argv) > 1 else "last-results-imgs/results_average.json"
    data = load(path)
    tiers = sorted(data)
    tier_a = sys.argv[2] if len(sys.argv) > 2 else tiers[0]
    tier_b = sys.argv[3] if len(sys.argv) > 3 else tiers[-1]
    img_dir = os.path.dirname(path) or "."

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fams = sorted(set(data[tier_a]) | set(data[tier_b]))
    colors = tier_colors(tiers)
    for label, metric in METRICS.items():
        fig, axes = plt.subplots(len(SIZES), 1, figsize=(8, 2.6 * len(SIZES)))
        for i, size in enumerate(SIZES):
            ax = axes[i]
            x = range(len(fams))
            width = 0.35
            bars = []
            for name, tier in ((tier_a, tier_a), (tier_b, tier_b)):
                means = [data[tier][f][size][metric]["mean"] for f in fams]
                stds = [data[tier][f][size][metric]["std"] for f in fams]
                off = width / 2 * (1 if name == tier_b else -1)
                bars.append(ax.bar([k + off for k in x], means, width, yerr=stds,
                                   capsize=4, label=name, color=colors[tier]))
            ax.set_xticks(list(x))
            ax.set_xticklabels(fams)
            ax.set_title(size)
            ax.grid(axis="y", alpha=0.3)
        fig.suptitle(f"{label} ({metric}) — {tier_a} vs {tier_b}, mean ± std")
        fig.legend([b[0] for b in bars], [tier_a, tier_b], loc="upper right")
        fig.tight_layout(rect=(0, 0, 1, 0.92))
        fig.savefig(os.path.join(img_dir, f"compare-{label}.png"), dpi=120)
        plt.close(fig)
        print(f"  wrote {os.path.join(img_dir, 'compare-' + label + '.png')}")


if __name__ == "__main__":
    main()
