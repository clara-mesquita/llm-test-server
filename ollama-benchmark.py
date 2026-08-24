#!/usr/bin/env python3
"""Benchmark llama3.2:3b, gemma3:4b, qwen3:8b via the Ollama HTTP API.

Run:  python ollama-benchmark.py   (Ollama server must be running)
"""
import json
import urllib.request

BASE = "http://localhost:11434"
MODELS = ["llama3.2:3b", "gemma3:4b", "qwen3:8b"]

PROMPTS = {
    "small": "What is 2+2?",
    "medium": ("Write a detailed technical essay (about 800 words) comparing supervised, "
               "unsupervised, and reinforcement learning: definitions, typical algorithms, "
               "data requirements, real-world applications, and common pitfalls. "
               "Structure the essay with headings."),
    "large": ("You are a senior data scientist asked to perform a full exploratory data analysis "
              "and modeling plan on the following e-commerce sales dataset (10,000 rows, fields: "
              "order_date, category, units_sold, unit_price, discount_rate, region, customer_segment, "
              "returned_flag).\n"
              "\n"
              "1. Data quality: list the exact steps you would take to clean this dataset (missing values, "
              "outliers, inconsistent categories) and justify each choice statistically.\n"
              "2. Univariate analysis: specify which summary statistics and distribution checks you would "
              "run for the numeric columns and why.\n"
              "3. Hypothesis testing: formulate three concrete business hypotheses (e.g. discount rate vs "
              "return rate, region vs revenue, segment vs order size), pick the appropriate statistical "
              "test for each (with assumptions and their verification), and state how you would interpret "
              "a significant vs non-significant result.\n"
              "4. Predictive model: design a regression model for daily revenue, listing feature engineering "
              "steps, model selection strategy, cross-validation scheme, and the exact metrics you would "
              "report (with formulas).\n"
              "5. Write complete, runnable Python code (pandas, scipy, sklearn) implementing steps 1, 3, and 4 "
              "end-to-end on this dataset, with comments.\n"
              "6. Finally, summarize in under 200 words what the analysis would reveal about discounting "
              "strategy if the results came back the way you expect.\n"
              "\n"
              "Be rigorous, concrete, and specific; no generic filler."),
}


def generate(model, prompt):
    body = json.dumps({
        "model": model,
        "prompt": prompt,
        "stream": False,
        "options": {"num_ctx": 8192},  # consistent ctx so the large prompt isn't truncated
    }).encode()
    req = urllib.request.Request(f"{BASE}/api/generate", data=body,
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req) as resp:
        return json.loads(resp.read())


def tps(count, duration_ns):
    return round(count / (duration_ns / 1e9), 1) if duration_ns > 0 else 0


def main():
    runs = []
    for model in MODELS:
        print(f"=== {model} ===")
        print("  warming up (load model, excluded from results)...")
        generate(model, "ping")
        for size, prompt in PROMPTS.items():
            print(f"  running {size} prompt...")
            r = generate(model, prompt)
            runs.append({
                "model": model,
                "size": size,
                "total_duration_ns": r["total_duration"],
                "load_duration_ns": r["load_duration"],
                "prompt_eval_count": r["prompt_eval_count"],
                "prompt_eval_duration_ns": r["prompt_eval_duration"],
                "eval_count": r["eval_count"],
                "eval_duration_ns": r["eval_duration"],
                "prompt_tps": tps(r["prompt_eval_count"], r["prompt_eval_duration"]),
                "eval_tps": tps(r["eval_count"], r["eval_duration"]),
            })
            print(f"    eval {r['eval_count']} tokens in {r['eval_duration'] / 1e9:.2f}s "
                  f"-> {tps(r['eval_count'], r['eval_duration'])} tok/s")

    out = "ollama-benchmark-results.json"
    with open(out, "w", encoding="utf-8") as f:
        json.dump(runs, f, indent=2)

    print("\n=== Summary ===")
    for row in runs:
        print(f"{row['model']:<12} {row['size']:<7} total={row['total_duration_ns'] / 1e9:8.2f}s "
              f"eval={row['eval_count']:>5} tok {row['eval_duration_ns'] / 1e9:8.2f}s "
              f"-> {row['eval_tps']:>7.1f} tok/s")
    print(f"Results saved to {out}")


if __name__ == "__main__":
    main()
