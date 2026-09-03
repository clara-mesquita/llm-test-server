#!/usr/bin/env python3
"""Benchmark same-size models via the llama.cpp OpenAI-compatible API (Docker).

Mirrors vllm-benchmark.py / ollama-benchmark.py: same tiers/families/prompts
and the same output schema, so ollama-results-avg.py aggregates unchanged.

Models are GGUF (Q4_K_M) pulled from Hugging Face via llama.cpp's -hf flag.
Unlike vLLM (bfloat16), Q4_K_M GGUF fits a 4GB GPU for the 4b/8b tiers; bigger
tiers spill layers to CPU automatically (-ngl 999 offloads what fits).

Timing note: like vllm-benchmark.py, prompt_duration ~ time-to-first-token and
eval_duration ~ total - TTFT. llama.cpp also reports real timings
(timings.prompt_ms / predicted_ms) but the TTFT method keeps all three
benchmarks measured the same way.

Run:  python llama-benchmark.py
Prereqs: Docker up (see docker-compose.yml); HF_TOKEN for gated Llama models.
Env:   LLAMA_IMAGE        (default ghcr.io/ggml-org/llama.cpp:server-cuda)
       LLAMA_PORT         host port (default 8081; vLLM uses 8080)
       LLAMA_EXTRA_ARGS   extra llama.cpp flags, e.g. "--parallel 4"
       LLAMA_N_GPU_LAYERS GPU layers to offload (default 999 = all that fit)
"""

import json
import os
import shlex
import subprocess
import time
import urllib.request

IMAGE = os.environ.get("LLAMA_IMAGE", "ghcr.io/ggml-org/llama.cpp:server-cuda")
PORT = int(os.environ.get("LLAMA_PORT", "8081"))
EXTRA_ARGS = shlex.split(os.environ.get("LLAMA_EXTRA_ARGS", ""))
N_GPU_LAYERS = os.environ.get("LLAMA_N_GPU_LAYERS", "999")
MAX_TOKENS = 256  # mirror ollama's num_predict cap

# tier -> family -> (HF GGUF repo:quant, param label, container name)
# Q4_K_M = the standard 4-bit quant. Confirm each repo ships Q4_K_M on first run.
MODELS = {
    "4b": {
        "llama": ("unsloth/Llama-3.2-3B-Instruct-GGUF:Q4_K_M", "3B",  "llamacpp-llama-3b"),
        "gemma": ("ggml-org/gemma-3-4b-it-GGUF:Q4_K_M",         "4B",  "llamacpp-gemma-4b"),
        "qwen":  ("Qwen/Qwen3-4B-GGUF:Q4_K_M",                  "4B",  "llamacpp-qwen-4b"),
    },
    "8b": {
        "llama": ("unsloth/Llama-3.1-8B-Instruct-GGUF:Q4_K_M", "8B",  "llamacpp-llama-8b"),
        "gemma": ("ggml-org/gemma-2-9b-it-GGUF:Q4_K_M",         "9B",  "llamacpp-gemma-9b"),
        "qwen":  ("Qwen/Qwen3-8B-GGUF:Q4_K_M",                  "8B",  "llamacpp-qwen-8b"),
    },
    "16b": {
        "llama": ("unsloth/Llama-4-Scout-17B-16E-Instruct-GGUF:Q4_K_M", "17B", "llamacpp-llama-17b"),
        "gemma": ("ggml-org/gemma-3-12b-it-GGUF:Q4_K_M",                "12B", "llamacpp-gemma-12b"),
        "qwen":  ("Qwen/Qwen3-14B-GGUF:Q4_K_M",                         "14B", "llamacpp-qwen-14b"),
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


def docker(*args, check=True):
    result = subprocess.run(["docker", *args], text=True, capture_output=True)
    if check and result.returncode:
        raise RuntimeError(f"docker {args[0]} failed (exit {result.returncode}):\n{result.stderr.strip()}")
    return result


def server_up(model):
    try:
        with urllib.request.urlopen(f"http://localhost:{PORT}/v1/models", timeout=3) as response:
            return model in {item["id"] for item in json.load(response).get("data", [])}
    except OSError:
        return False


def wait_server(model, container, timeout=1800):
    deadline = time.time() + timeout
    while time.time() < deadline:
        if server_up(model):
            return
        state = docker("inspect", "-f", "{{.State.Status}}", container, check=False).stdout.strip()
        if state in {"exited", "dead"}:
            logs = docker("logs", "--tail", "60", container, check=False).stdout
            raise RuntimeError(f"{container} exited:\n{logs}")
        time.sleep(3)
    raise TimeoutError(f"llama.cpp on :{PORT} did not load {model} within {timeout}s")


def start_container(hf, container):
    docker("rm", "-f", container, check=False)
    # -ngl 999: offload as many layers as fit in VRAM; llama.cpp spills the rest
    # to CPU instead of OOMing, so the 8b/16b tiers still run on a 4GB GPU.
    args = ["run", "-d", "--name", container, "--gpus", "all", "-p", f"{PORT}:8000",
            "-v", "llama-hf-cache:/root/.cache/huggingface"]
    if token := os.environ.get("HF_TOKEN"):
        args += ["-e", f"HF_TOKEN={token}"]
    docker(*args, IMAGE, "-hf", hf, "--host", "0.0.0.0", "--port", "8000",
           "-ngl", N_GPU_LAYERS, "-c", "8192", *EXTRA_ARGS)


def stop_container(container):
    docker("rm", "-f", container, check=False)


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
    if not os.environ.get("HF_TOKEN"):
        print("NOTE: HF_TOKEN not set - gated Llama models will fail (401); Qwen/Gemma are public.")
    runs = []
    for tier, families in MODELS.items():
        for family, (hf, params, container) in families.items():
            print(f"=== {tier} / {family} ({hf}, {params}) ===")
            print("  starting container...")
            start_container(hf, container)
            try:
                wait_server(hf, container)
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
            finally:
                stop_container(container)
                print(f"  stopped {container}")

    out = "llama-benchmark-results.json"
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
