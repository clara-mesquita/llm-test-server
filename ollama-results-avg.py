#!/usr/bin/env python3
"""Aggregate ollama benchmark runs into averages + std devs, per tier/family/size.

Reads:  ollama-benchmark-results.json  (flat list; the `run` key identifies the
        benchmark round, mean/std are computed across runs)
Writes: <img_dir>/results_average.json
        <img_dir>/results_avg-<size>.png   (one figure per prompt size)

Each size figure has a subplot per (tier x key metric), with one bar per model
family (llama / gemma / qwen), error bars = std dev.

Usage:  python ollama-results-avg.py [results_json] [img_dir]
        defaults: results_json=ollama-benchmark-results.json,
                  img_dir=last-results-imgs
"""

import json
import math
import os
import statistics
import sys

SIZES = ["small", "medium", "large"]
ALL_METRICS = ["eval_tps", "prompt_tps", "load_duration_s", "total_duration_s",
               "prompt_eval_duration_s", "eval_duration_s", "prompt_eval_count", "eval_count"]
NONMETRIC = {"model", "size", "family", "tier", "params", "run"}

# one hue per family (matches the old results_avg-*.png palette),
# intensity driven by parameter tier: larger params = more vivid
FAMILY_COLORS = {"llama": "#4C72B0", "gemma": "#DD8452", "qwen": "#55A868"}


def bar_color(family, tier, tiers):
    import matplotlib.colors as mcolors
    base = FAMILY_COLORS.get(family, "#4C72B0")
    a = 0.35 + 0.65 * tiers.index(tier) / max(len(tiers) - 1, 1)
    r, g, b = mcolors.to_rgb(base)
    return mcolors.to_hex(tuple(1 - a + a * c for c in (r, g, b)))


def load_runs(path):
    with open(path, encoding="utf-8") as f:
        runs = json.load(f)
    if not runs:
        raise SystemExit(f"no runs in {path}")
    return runs


def metrics(runs):
    return [k for k in runs[0] if k not in NONMETRIC
            and isinstance(runs[0][k], (int, float))]


def mean_std(vals):
    mean = statistics.mean(vals)
    std = statistics.stdev(vals) if len(vals) > 1 else 0.0
    return mean, std


def aggregate(runs, cols):
    avg = {}
    for tier in sorted({r["tier"] for r in runs}):
        avg[tier] = {}
        for f in sorted({r["family"] for r in runs if r["tier"] == tier}):
            avg[tier][f] = {}
            for s in SIZES:
                rows = [r for r in runs if r["tier"] == tier and r["family"] == f and r["size"] == s]
                avg[tier][f][s] = {
                    c: dict(zip(("mean", "std"), mean_std([r[c] for r in rows])))
                    for c in cols
                }
    # durations are reported in nanoseconds by the benchmark; convert to seconds
    for tier in avg:
        for f in avg[tier]:
            for s in SIZES:
                for c in list(avg[tier][f][s]):
                    if c.endswith("_ns"):
                        avg[tier][f][s][c[:-3] + "_s"] = {
                            k: v / 1e9 for k, v in avg[tier][f][s].pop(c).items()
                        }
    return avg


def plot_size(avg, size, fams, params, img_dir):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plot_cols = ALL_METRICS
    tiers = list(avg)
    combos = [(t, f) for f in fams for t in tiers]  # grouped by family

    ncols = 4
    nrows = math.ceil(len(plot_cols) / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(16, 6), dpi=120)
    axes = axes.ravel()
    for j, col in enumerate(plot_cols):
        ax = axes[j]
        means = [avg[t][f][size][col]["mean"] for t, f in combos]
        stds = [avg[t][f][size][col]["std"] for t, f in combos]
        colors = [bar_color(f, t, tiers) for t, f in combos]
        ax.bar(range(len(combos)), means, yerr=stds, capsize=4, color=colors)
        ax.set_xticks(range(len(combos)))
        ax.set_xticklabels([f"{f}\n{params.get((t, f), '')}" for t, f in combos],
                           fontsize=7)
        ax.set_title(col, fontsize=9)
        ax.grid(axis="y", alpha=0.3)
    for j in range(len(plot_cols), nrows * ncols):
        axes[j].set_visible(False)
    fig.suptitle(f"ollama benchmark — {size} prompts — mean ± std", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(os.path.join(img_dir, f"results_avg-{size}.png"))
    plt.close(fig)


def main():
    results_json = sys.argv[1] if len(sys.argv) > 1 else "ollama-benchmark-results.json"
    img_dir = sys.argv[2] if len(sys.argv) > 2 else "last-results-imgs"
    os.makedirs(img_dir, exist_ok=True)

    runs = load_runs(results_json)
    cols = metrics(runs)
    avg = aggregate(runs, cols)

    with open(os.path.join(img_dir, "results_average.json"), "w", encoding="utf-8") as f:
        json.dump(avg, f, indent=2)

    fams = sorted({f for tier in avg for f in avg[tier]})
    params = {(r["tier"], r["family"]): r["params"] for r in runs}
    for size in SIZES:
        plot_size(avg, size, fams, params, img_dir)
        print(f"  {size}: -> {os.path.join(img_dir, 'results_avg-' + size + '.png')}")

    n = len(runs)
    print(f"{results_json}: aggregated {n} runs ({len(cols)} metrics) "
          f"-> {os.path.join(img_dir, 'results_average.json')}")


if __name__ == "__main__":
    main()
