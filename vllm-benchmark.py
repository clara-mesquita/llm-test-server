#!/usr/bin/env python3
"""Benchmark same-size models via the vLLM OpenAI-compatible API (WSL2 + Docker).

Mirrors ollama-benchmark.py: same tiers/families/prompts and the same output
schema, so ollama-results-avg.py aggregates the JSON unchanged.

Timing note: vLLM's OpenAI API does not expose per-request prompt/eval
durations, so they are approximated from the stream — prompt_duration ~
time-to-first-token (TTFT), eval_duration ~ total - TTFT.

Run:  python vllm-benchmark.py
Prereqs: Docker Desktop + WSL2 up (vllm-activation.ps1); HF_TOKEN for gated models.
Env:   VLLM_IMAGE       (default vllm/vllm-openai:v0.28.0)
       VLLM_PORT        host port (default 8080; models run one at a time)
       VLLM_EXTRA_ARGS  extra vLLM flags, e.g. "--quantization awq --cpu-offload-gb 8"
"""

import json
import os
import subprocess
import time
import urllib.request

IMAGE = os.environ.get("VLLM_IMAGE", "vllm/vllm-openai:v0.28.0")
PORT = int(os.environ.get("VLLM_PORT", "8080"))
EXTRA_ARGS = os.environ.get("VLLM_EXTRA_ARGS", "")
MAX_TOKENS = 256  # mirror ollama's num_predict cap

# tier -> family -> (HF model id, param label, container name)
# models run one at a time on PORT, so no per-model ports needed
MODELS = {
    "4b": {
        "llama": ("meta-llama/Llama-3.2-3B-Instruct", "3B",   "vllm-llama-3b"),
        "gemma": ("google/gemma-4-e4b-it",            "4.5B", "vllm-gemma-4b"),
        "qwen":  ("Qwen/Qwen3-4B",                    "4B",   "vllm-qwen-4b"),
    },
    "8b": {
        "llama": ("meta-llama/Llama-3.1-8B-Instruct", "8B",   "vllm-llama-8b"),
        "gemma": ("google/gemma-2-9b-it",             "9B",   "vllm-gemma-9b"),
        "qwen":  ("Qwen/Qwen3-8B-Instruct",           "8B",   "vllm-qwen-8b"),
    },
    "16b": {
        "llama": ("meta-llama/Llama-4-Scout-17B-16E-Instruct", "17B", "vllm-llama-17b"),
        "gemma": ("google/gemma-4-12b-it",            "12B",  "vllm-gemma-12b"),
        "qwen":  ("Qwen/Qwen3-14B",                   "14B",  "vllm-qwen-14b"),
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


def wsl(cmd):
    subprocess.run(["wsl", "-e", "bash", "-lc", cmd], check=True)


def server_up():
    try:
        urllib.request.urlopen(f"http://localhost:{PORT}/v1/models", timeout=3)
        return True
    except OSError:
        return False


def wait_server(timeout=1800):
    deadline = time.time() + timeout
    while time.time() < deadline:
        if server_up():
            return
        time.sleep(3)
    raise TimeoutError(f"vLLM on :{PORT} not ready in {timeout}s (docker logs vllm-smoke)")


def start_container(hf, container):
    token = f"-e HF_TOKEN={os.environ['HF_TOKEN']}" if os.environ.get("HF_TOKEN") else ""
    # VLLM_WSL2_ENABLE_PIN_MEMORY=1: vLLM's V2 runner needs pinned memory for UVA,
    # which is off by default on WSL2 -> otherwise "RuntimeError: UVA is not available"
    wsl(
        f"docker rm -f {container} 2>/dev/null; "
        f"docker run -d --name {container} --gpus all -p {PORT}:8000 "
        f"--ipc=host --shm-size=8gb -v vllm-hf-cache:/root/.cache/huggingface "
        f"-e VLLM_WSL2_ENABLE_PIN_MEMORY=1 {token} {IMAGE} "
        f"--model {hf} --max-model-len 8192 --gpu-memory-utilization 0.9 {EXTRA_ARGS}"
    )


def stop_container(container):
    subprocess.run(["wsl", "-e", "bash", "-lc", f"docker rm -f {container} 2>/dev/null"])


def generate(hf, prompt):
    """Stream one chat completion; return timing + token counts (all int)."""
    body = json.dumps({
        "model": hf,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": MAX_TOKENS,
        "temperature": 0,
        "stream": True,
        "stream_options": {"include_usage": True},
    }).encode()
    req = urllib.request.Request(
        f"http://localhost:{PORT}/v1/chat/completions",
        data=body, headers={"Content-Type": "application/json"},
    )
    t0 = time.perf_counter()
    ttft = None
    usage = {}
    with urllib.request.urlopen(req) as resp:
        for line in resp:
            line = line.decode("utf-8").strip()
            if not line.startswith("data:"):
                continue
            payload = line[len("data:"):].strip()
            if payload == "[DONE]":
                break
            chunk = json.loads(payload)
            for choice in chunk.get("choices", []):
                if choice.get("delta", {}).get("content") and ttft is None:
                    ttft = time.perf_counter()
            if chunk.get("usage"):
                usage = chunk["usage"]
    total_ns = int((time.perf_counter() - t0) * 1e9)
    prompt_ns = int(((ttft if ttft is not None else time.perf_counter()) - t0) * 1e9)
    eval_ns = max(total_ns - prompt_ns, 0)
    return {
        "total_duration_ns": total_ns,
        "prompt_duration_ns": prompt_ns,
        "eval_duration_ns": eval_ns,
        "prompt_count": usage.get("prompt_tokens", 0),
        "eval_count": usage.get("completion_tokens", 0),
    }


def tps(count, ns):
    return round(count / (ns / 1e9), 1) if ns > 0 else 0


def main():
    runs = []
    for tier, families in MODELS.items():
        for family, (hf, params, container) in families.items():
            print(f"=== {tier} / {family} ({hf}, {params}) ===")
            print("  starting container...")
            start_container(hf, container)
            wait_server()
            print("  warming up (excluded from results)...")
            generate(hf, "ping")
            for size, prompt in PROMPTS.items():
                print(f"  running {size} prompt...")
                r = generate(hf, prompt)
                runs.append({
                    "model": hf, "family": family, "tier": tier,
                    "params": params, "size": size,
                    "total_duration_ns": r["total_duration_ns"],
                    "load_duration_ns": 0,  # model loaded at server start
                    "prompt_eval_count": r["prompt_count"],
                    "prompt_eval_duration_ns": r["prompt_duration_ns"],
                    "eval_count": r["eval_count"],
                    "eval_duration_ns": r["eval_duration_ns"],
                    "prompt_tps": tps(r["prompt_count"], r["prompt_duration_ns"]),
                    "eval_tps": tps(r["eval_count"], r["eval_duration_ns"]),
                })
                print(f"    {r['eval_count']} tok in {r['eval_duration_ns'] / 1e9:.2f}s "
                      f"-> {runs[-1]['eval_tps']} tok/s")
            stop_container(container)
            print(f"  stopped {container}")

    out = "vllm-benchmark-results.json"
    with open(out, "w", encoding="utf-8") as f:
        json.dump(runs, f, indent=2)

    print("\n=== Summary ===")
    for row in runs:
        print(f"{row['tier']:<4} {row['family']:<6} {row['size']:<7} "
              f"eval={row['eval_count']:>4} tok {row['eval_duration_ns'] / 1e9:7.2f}s "
              f"-> {row['eval_tps']:>7.1f} tok/s")
    print(f"Results saved to {out}")


if __name__ == "__main__":
    main()
