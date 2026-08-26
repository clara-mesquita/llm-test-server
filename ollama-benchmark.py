#!/usr/bin/env python3
"""Benchmark same-size models via the Ollama HTTP API.

Models are grouped in parameter-size tiers (4b / 8b / 16b) so each tier is an
apples-to-apples comparison of models of ~equal size. Exact param counts are
noted inline; no family has an exact match at every size, so each tier uses
the closest real model (e.g. gemma has no 8B -> gemma2:9b).

Run:  python ollama-benchmark.py   (Ollama server must be running)
"""

import json
import urllib.request

BASE = "http://localhost:11434"

# tier -> {family -> (ollama tag, param label)}
MODELS = {
    "4b": {
        "llama": ("llama3.2:3b", "3B"),
        "gemma": ("gemma4:e4b",  "4.5B"),
        "qwen":  ("qwen3:4b",    "4B"),
    },
    "8b": {
        "llama": ("llama3.1:8b", "8B"),
        "gemma": ("gemma2:9b",   "9B"),  # gemma4 has no 8B; gemma2:9b is closest
        "qwen":  ("qwen3:8b",    "8B"),
    },
    "16b": {
        "llama": ("llama4:scout", "17B"),  # MoE: 109B total / 17B active
        "gemma": ("gemma4:12b",   "12B"),
        "qwen":  ("qwen3:14b",    "14B"),
    },
}

PROMPTS = {
    "small": "What is 2+2?",
    "medium": (
        "Explain the key differences between SQL and NoSQL databases, "
        "with one concrete use case for each. Answer in about 150 words."
    ),
    "large": (
        "Write a detailed technical essay (about 800 words) comparing supervised, "
        "unsupervised, and reinforcement learning: definitions, typical algorithms, "
        "data requirements, real-world applications, and common pitfalls. "
        "Structure the essay with headings."
    ),
}


def generate(model, prompt):
    body = json.dumps(
        {
            "model": model,
            "prompt": prompt,
            "stream": False,
            "options": {
                "num_ctx": 8192,
                "num_predict": 256,  # cap output so verbose models don't run for minutes
            },  # consistent ctx so the large prompt isn't truncated
        }
    ).encode()
    req = urllib.request.Request(
        f"{BASE}/api/generate", data=body, headers={"Content-Type": "application/json"}
    )
    with urllib.request.urlopen(req) as resp:
        return json.loads(resp.read())


def tps(count, duration_ns):
    return round(count / (duration_ns / 1e9), 1) if duration_ns > 0 else 0


def main():
    runs = []
    for tier, families in MODELS.items():
        for family, (tag, params) in families.items():
            print(f"=== {tier} / {family} ({tag}, {params}) ===")
            print("  warming up (load model, excluded from results)...")
            generate(tag, "ping")
            for size, prompt in PROMPTS.items():
                print(f"  running {size} prompt...")
                r = generate(tag, prompt)
                runs.append(
                    {
                        "model": tag,
                        "family": family,
                        "tier": tier,
                        "params": params,
                        "size": size,
                        "total_duration_ns": r["total_duration"],
                        "load_duration_ns": r["load_duration"],
                        "prompt_eval_count": r["prompt_eval_count"],
                        "prompt_eval_duration_ns": r["prompt_eval_duration"],
                        "eval_count": r["eval_count"],
                        "eval_duration_ns": r["eval_duration"],
                        "prompt_tps": tps(
                            r["prompt_eval_count"], r["prompt_eval_duration"]
                        ),
                        "eval_tps": tps(r["eval_count"], r["eval_duration"]),
                    }
                )
                print(
                    f"    eval {r['eval_count']} tokens in {r['eval_duration'] / 1e9:.2f}s "
                    f"-> {tps(r['eval_count'], r['eval_duration'])} tok/s"
                )

    out = "ollama-benchmark-results.json"
    with open(out, "w", encoding="utf-8") as f:
        json.dump(runs, f, indent=2)

    print("\n=== Summary ===")
    for row in runs:
        print(
            f"{row['tier']:<4} {row['family']:<6} {row['size']:<7} total={row['total_duration_ns'] / 1e9:8.2f}s "
            f"eval={row['eval_count']:>5} tok {row['eval_duration_ns'] / 1e9:8.2f}s "
            f"-> {row['eval_tps']:>7.1f} tok/s"
        )
    print(f"Results saved to {out}")


if __name__ == "__main__":
    main()
