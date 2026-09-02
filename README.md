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
