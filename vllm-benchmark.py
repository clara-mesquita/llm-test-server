#!/usr/bin/env python3
"""Benchmark llama3.2 / gemma3 / qwen3 served by vLLM (WSL2 + Docker).

Run:  python vllm-benchmark.py
Prereqs: vllm-activation.ps1 run once (creates the containers).
The script starts any stopped container itself, and stops containers it
started after each model so models don't fight over VRAM.

Timing note: vLLM's OpenAI-compatible API (pinned image v0.8.5.post1) does
not expose per-request prompt/eval durations, so total_duration_ns is the
client-side wall time and eval/prompt durations are None.
"""
import json
import subprocess
import time
import urllib.request

MODELS = {
    "llama32": {"hf": "meta-llama/Llama-3.2-3B-Instruct", "port": 8000, "container": "vllm-llama32"},
    "gemma3":  {"hf": "google/gemma-3-4b-it",              "port": 8001, "container": "vllm-gemma3"},
    "qwen3":   {"hf": "Qwen/Qwen3-8B-Instruct",            "port": 8002, "container": "vllm-qwen3"},
}

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


def wsl(cmd):
    subprocess.run(["wsl", "-e", "bash", "-lc", cmd], check=True)


def server_up(port):
    try:
        urllib.request.urlopen(f"http://localhost:{port}/v1/models", timeout=3)
        return True
    except OSError:
        return False


def wait_server(port, timeout=900):
    deadline = time.time() + timeout
    while time.time() < deadline:
        if server_up(port):
            return
        time.sleep(3)
    raise TimeoutError(f"server on :{port} not ready")


def chat(model, prompt):
    body = json.dumps({
        "model": model["hf"],
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": 4096,
        "temperature": 0,
        "stream": False,
    }).encode()
    req = urllib.request.Request(f"http://localhost:{model['port']}/v1/chat/completions",
                                 data=body, headers={"Content-Type": "application/json"})
    t0 = time.perf_counter()
    with urllib.request.urlopen(req) as resp:
        r = json.loads(resp.read())
    wall_ns = int((time.perf_counter() - t0) * 1e9)
    return r, wall_ns


def main():
    runs = []
    for name, model in MODELS.items():
        started = False
        if not server_up(model["port"]):
            print(f"=== {name}: starting container {model['container']} ===")
            wsl(f"docker start {model['container']}")
            wait_server(model["port"])
            started = True
        print(f"=== {name} ===")
        print("  warming up (model load excluded from results)...")
        chat(model, "ping")
        for size, prompt in PROMPTS.items():
            print(f"  running {size} prompt...")
            r, wall_ns = chat(model, prompt)
            usage = r.get("usage", {})
            eval_count = usage.get("completion_tokens", 0)
            runs.append({
                "model": name,
                "size": size,
                "total_duration_ns": wall_ns,
                "load_duration_ns": None,          # vLLM loads at server start, not per request
                "prompt_eval_count": usage.get("prompt_tokens", 0),
                "prompt_eval_duration_ns": None,   # not exposed by vLLM API
                "eval_count": eval_count,
                "eval_duration_ns": None,          # not exposed by vLLM API
                "finish_reason": r["choices"][0].get("finish_reason"),
                "eval_tps": round(eval_count / (wall_ns / 1e9), 1) if wall_ns > 0 else 0,
            })
            print(f"    eval {eval_count} tokens in {wall_ns / 1e9:.2f}s "
                  f"-> {runs[-1]['eval_tps']} tok/s (wall)")
        if started:
            wsl(f"docker stop {model['container']}")
            print(f"  stopped {model['container']} (I started it)")

    out = "vllm-benchmark-results.json"
    with open(out, "w", encoding="utf-8") as f:
        json.dump(runs, f, indent=2)

    print("\n=== Summary ===")
    for row in runs:
        print(f"{row['model']:<8} {row['size']:<7} total={row['total_duration_ns'] / 1e9:8.2f}s "
              f"eval={row['eval_count']:>5} tok -> {row['eval_tps']:>7.1f} tok/s "
              f"[{row['finish_reason']}]")
    print(f"Results saved to {out}")


if __name__ == "__main__":
    main()
