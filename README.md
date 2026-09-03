# vLLM benchmark: fresh Windows device

Requirements: Windows 10/11, an NVIDIA GPU/driver supported by Docker Desktop,
internet access, and enough VRAM for each model. The benchmark has no Python
package dependencies.

Open **PowerShell as Administrator** in this directory and run:

```powershell
Set-ExecutionPolicy -Scope Process Bypass
.\vllm-activation.ps1 -InstallPrerequisites
```

The command installs WSL2 or Docker Desktop if either is missing, then stops
when Windows needs a reboot or Docker needs its first launch. Reboot when told,
start Docker Desktop once and accept its terms, then repeat the command. A
successful smoke test ends with `vLLM works`.
If port 8080 is occupied, the script chooses a free port and sets `VLLM_PORT`
for the benchmark in that same PowerShell session.

The full benchmark includes gated Llama and Gemma models. Accept their model
licenses in Hugging Face, create a read token, and set it in the same shell:

```powershell
$env:HF_TOKEN = 'hf_your_token'
python .\vllm-benchmark.py
```

Use `setx HF_TOKEN hf_your_token` to persist the token for future terminals.
If the GPU check fails, update the NVIDIA driver and make sure Docker Desktop
is using Linux containers; vLLM cannot run with CPU-only Docker.

## Docker setup (docker-compose)

`docker-compose.yml` is the declarative form of the server the benchmark talks
to. It serves ONE fixed model at a time — useful for a smoke test or debugging.
Configure via a `.env` file (copy `.env.example`):

```powershell
Copy-Item .env.example .env   # then edit .env
Set-Content .env 'VLLM_MODEL=Qwen/Qwen2.5-0.5B-Instruct' -Encoding utf8  # public, fits 4GB
docker compose up -d          # wait ~60s, then hit http://localhost:8080/v1/models
docker compose down           # stop it
```

`docker compose up` and `python vllm-benchmark.py` are alternative entry points
and collide on `VLLM_PORT`, so run one at a time. Both share the `vllm-hf-cache`
volume, so models download only once.

## Small-GPU note (4GB VRAM)

`--gpu-memory-utilization` defaults to `0.7` (2.8GB on a 4GB card). The default
`0.9` fails on this GPU: vLLM refuses to start when free memory (3.2GB) is below
the requested 3.6GB. The 3B/8B/17B benchmark models still exceed 4GB VRAM in
bfloat16, so run them with CPU offload (slow but works):

```powershell
$env:VLLM_EXTRA_ARGS = '--cpu-offload-gb 8'
python .\vllm-benchmark.py
```

## llama.cpp benchmark (Docker)

`llama-benchmark.py` mirrors `vllm-benchmark.py` but serves GGUF models via the
llama.cpp Docker image, and writes `llama-benchmark-results.json` in the same
schema. GGUF Q4_K_M fits a 4GB GPU (no CPU offload needed for most tiers).

```powershell
$env:HF_TOKEN = 'hf_your_token'   # only needed for the gated Llama models
python .\llama-benchmark.py
```

Config: `LLAMA_IMAGE` (default `ghcr.io/ggml-org/llama.cpp:server-cuda`),
`LLAMA_PORT` (default 8081), `LLAMA_EXTRA_ARGS`, `LLAMA_N_GPU_LAYERS` (default
999 = offload all layers that fit). Test: `python test_llama_benchmark.py`.
