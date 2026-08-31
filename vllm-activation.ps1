# vllm-activation.ps1
# Set up vLLM on Windows (WSL2 + Docker) and smoke-test it.
#
# Usage:  .\vllm-activation.ps1 [-Model <hf-id>] [-Image <docker-image>]
#   -Model : HF model for the smoke test (default Qwen/Qwen3-0.6B: non-gated, fits 4GB VRAM)
#   -Image : vLLM image (default vllm/vllm-openai:v0.28.0)
#
# Prereqs: NVIDIA GPU, Windows 10/11 with virtualization enabled.
#          HF_TOKEN env var for gated models (llama / gemma2 / qwen3-8b).
#
# Note: Docker Desktop is auto-started here if its engine is down, and
# VLLM_WSL2_ENABLE_PIN_MEMORY=1 is set so vLLM's V2 runner doesn't crash on WSL2
# with "RuntimeError: UVA is not available".
param(
    [string]$Model = 'Qwen/Qwen3-0.6B',
    [string]$Image = 'vllm/vllm-openai:v0.28.0'
)

$ErrorActionPreference = 'Stop'
$Port = 8000
$Container = 'vllm-smoke'
$TestPrompt = 'What is 2+2?'

function Invoke-Wsl($cmd) {
    wsl -e bash -lc $cmd  # named Invoke-Wsl so it doesn't shadow wsl.exe (self-recursion -> CallDepthOverflow)
    if ($LASTEXITCODE -ne 0) { throw "wsl command failed: $cmd" }
}

# --- 1. WSL2 ---------------------------------------------------------------
wsl -e bash -lc 'echo ok' *> $null  # test by exit code, not output
if ($LASTEXITCODE -ne 0) {
    Write-Host "WSL2 not ready. Run as Administrator:" -ForegroundColor Yellow
    Write-Host "  wsl --install -d Ubuntu" -ForegroundColor Yellow
    Write-Host "then reboot and rerun this script." -ForegroundColor Yellow
    exit 1
}
Write-Host "WSL2 OK" -ForegroundColor Green

# --- 2. Docker Desktop (start it if the engine is down) ----------------------
function Test-Docker {
    wsl -e bash -lc 'docker ps' *> $null
    return $LASTEXITCODE -eq 0
}
if (-not (Test-Docker)) {
    $dockerExe = 'C:\Program Files\Docker\Docker\Docker Desktop.exe'
    if (-not (Test-Path $dockerExe)) {
        Write-Host "Docker Desktop not found. Installing (needs admin + reboot)..." -ForegroundColor Yellow
        winget install -e --id Docker.DockerDesktop
        Write-Host "Reboot, start Docker Desktop once, then rerun this script." -ForegroundColor Yellow
        exit 1
    }
    Write-Host "Starting Docker Desktop..." -ForegroundColor Cyan
    Start-Process $dockerExe
    $deadline = (Get-Date).AddMinutes(3)
    while (-not (Test-Docker) -and (Get-Date) -lt $deadline) { Start-Sleep 5 }
    if (-not (Test-Docker)) {
        Write-Host "Docker engine not reachable from WSL after 3 min." -ForegroundColor Yellow
        Write-Host "Check Docker Desktop -> Settings -> Resources -> WSL Integration -> enable your distro, then rerun." -ForegroundColor Yellow
        exit 1
    }
}
Write-Host "Docker OK" -ForegroundColor Green

# --- 3. GPU check -----------------------------------------------------------
Invoke-Wsl 'nvidia-smi' | Out-Null
Write-Host "GPU OK" -ForegroundColor Green

# --- 4. Image ---------------------------------------------------------------
Write-Host "Pulling $Image (first run, may take a while)..." -ForegroundColor Cyan
Invoke-Wsl "docker pull $Image"

# --- 5. Smoke test ----------------------------------------------------------
if (-not $env:HF_TOKEN) {
    Write-Host "NOTE: HF_TOKEN not set - gated model downloads will fail (401)." -ForegroundColor Yellow
}
Write-Host "Smoke test: $Model on :$Port" -ForegroundColor Cyan
$tokenArg = if ($env:HF_TOKEN) { "-e HF_TOKEN=$($env:HF_TOKEN)" } else { '' }
$cmd = "docker rm -f $Container 2>/dev/null; " +
       "docker run -d --name $Container --gpus all -p ${Port}:8000 " +
       "--ipc=host --shm-size=8gb -v vllm-hf-cache:/root/.cache/huggingface " +
       "-e VLLM_WSL2_ENABLE_PIN_MEMORY=1 $tokenArg " +
       "$Image --model $Model --max-model-len 4096 --gpu-memory-utilization 0.9"
Invoke-Wsl $cmd

$deadline = (Get-Date).AddMinutes(10)
$ready = $false
while ((Get-Date) -lt $deadline) {
    try { Invoke-RestMethod "http://localhost:$Port/v1/models" -TimeoutSec 5 | Out-Null; $ready = $true; break } catch { Start-Sleep 3 }
}
if (-not $ready) {
    Write-Host "server did not become ready. Logs:" -ForegroundColor Red
    Invoke-Wsl "docker logs $Container"
    throw "vLLM server did not start"
}
Write-Host "server ready, testing..." -ForegroundColor Green
$body = @{ model = $Model; messages = @(@{ role = 'user'; content = $TestPrompt }); max_tokens = 32 } | ConvertTo-Json
$resp = Invoke-RestMethod "http://localhost:$Port/v1/chat/completions" -Method Post -Body $body -ContentType 'application/json'
$reply = ($resp.choices[0].message.content -replace '\s+', ' ')
Write-Host "  reply: $($reply.Substring(0, [Math]::Min(120, $reply.Length)))..."

# --- 6. Status --------------------------------------------------------------
Write-Host "`nvLLM works. Stopping smoke-test container..." -ForegroundColor Green
Invoke-Wsl "docker rm -f $Container"
Write-Host "Done. Run:  python vllm-benchmark.py" -ForegroundColor Green
