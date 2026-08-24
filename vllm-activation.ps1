# vllm-activation.ps1
# Set up vLLM on Windows (WSL2 + Docker) and serve llama3.2 / gemma3 / qwen3.
#
# Usage:  .\vllm-activation.ps1 [-Model all|llama32|gemma3|qwen3]
#   default "all": start, test, and stop each model one at a time (VRAM).
#   a single model is left running for the benchmark.
#
# Prereqs:
#   - NVIDIA GPU, Windows 10/11 with virtualization enabled
#   - HF_TOKEN environment variable set (llama3.2/gemma3/qwen3 are gated
#     on Hugging Face: https://huggingface.co/settings/tokens)
param([ValidateSet('all', 'llama32', 'gemma3', 'qwen3')][string]$Model = 'all')

$ErrorActionPreference = 'Stop'
$Image = 'vllm/vllm-openai:v0.8.5.post1'   # pinned: v0.24.0+ images fail on WSL2 (CUDA UVA unsupported)
$TestPrompt = 'Explain the three more important statistical methods for data analysis'

$Models = [ordered]@{
    llama32 = @{ hf = 'meta-llama/Llama-3.2-3B-Instruct'; port = 8000; container = 'vllm-llama32' }
    gemma3  = @{ hf = 'google/gemma-3-4b-it';              port = 8001; container = 'vllm-gemma3' }
    qwen3   = @{ hf = 'Qwen/Qwen3-8B-Instruct';            port = 8002; container = 'vllm-qwen3' }
}

function Wsl($cmd) {
    wsl -e bash -lc $cmd
    if ($LASTEXITCODE -ne 0) { throw "wsl command failed: $cmd" }
}

function Wait-Server($port, $timeoutSec = 900) {
    $deadline = (Get-Date).AddSeconds($timeoutSec)
    while ((Get-Date) -lt $deadline) {
        try { Invoke-RestMethod "http://localhost:$port/v1/models" -TimeoutSec 5 | Out-Null; return } catch { Start-Sleep 3 }
    }
    throw "server on :$port did not become ready in $timeoutSec s (run: wsl -e bash -lc 'docker logs $($container)')"
}

function Start-Model($spec) {
    Write-Host "=== $($spec.container)  ($($spec.hf)) on :$($spec.port) ===" -ForegroundColor Cyan
    $tokenArg = if ($env:HF_TOKEN) { "-e HF_TOKEN=$($env:HF_TOKEN)" } else { '' }
    $cmd = "docker rm -f $($spec.container) 2>/dev/null; " +
           "docker run -d --name $($spec.container) --gpus all -p $($spec.port):8000 " +
           "--ipc=host --shm-size=8gb -v vllm-hf-cache:/root/.cache/huggingface $tokenArg " +
           "$Image --model $($spec.hf) --max-model-len 8192 --gpu-memory-utilization 0.9"
    Wsl $cmd
    Wait-Server $spec.port
    Write-Host "  server ready, testing..."
    $body = @{ model = $spec.hf; messages = @(@{ role = 'user'; content = $TestPrompt }) } | ConvertTo-Json
    $resp = Invoke-RestMethod "http://localhost:$($spec.port)/v1/chat/completions" -Method Post -Body $body -ContentType 'application/json'
    $reply = ($resp.choices[0].message.content -replace '\s+', ' ')
    Write-Host "  reply: $($reply.Substring(0, [Math]::Min(200, $reply.Length)))..."
}

# --- 1. WSL2 ---------------------------------------------------------------
if (-not (wsl -e bash -lc 'echo ok' 2>$null)) {
    Write-Host "WSL2 not ready. Run as Administrator:" -ForegroundColor Yellow
    Write-Host "  wsl --install -d Ubuntu-22.04" -ForegroundColor Yellow
    Write-Host "then reboot and rerun this script." -ForegroundColor Yellow
    exit 1
}
Write-Host "WSL2 OK" -ForegroundColor Green

# --- 2. Docker -------------------------------------------------------------
if (-not (wsl -e bash -lc 'docker --version' 2>$null)) {
    Write-Host "Docker not found inside WSL2. Installing Docker Desktop (needs admin + restart)..." -ForegroundColor Yellow
    winget install -e --id Docker.DockerDesktop
    Write-Host "Start Docker Desktop, enable WSL integration for your distro, then rerun." -ForegroundColor Yellow
    exit 1
}
Write-Host "Docker OK" -ForegroundColor Green

# --- 3. GPU check ----------------------------------------------------------
Wsl 'nvidia-smi' | Out-Null
Write-Host "GPU OK" -ForegroundColor Green

# --- 4. Image --------------------------------------------------------------
Write-Host "Pulling $Image (first run, may take a while)..." -ForegroundColor Cyan
Wsl "docker pull $Image"

# --- 5. Serve + test -------------------------------------------------------
if (-not $env:HF_TOKEN) {
    Write-Host "NOTE: HF_TOKEN not set - gated model downloads may fail (401)." -ForegroundColor Yellow
}
$targets = if ($Model -eq 'all') { $Models.Keys } else { @($Model) }
foreach ($name in $targets) {
    Start-Model $Models[$name]
    if ($Model -eq 'all' -and $targets.Count -gt 1) {
        Wsl "docker stop $($Models[$name].container)"
        Write-Host "  stopped $($Models[$name].container) to free VRAM for the next model" -ForegroundColor DarkGray
    }
}

# --- 6. Status -------------------------------------------------------------
Write-Host "`n=== docker ps ===" -ForegroundColor Cyan
wsl -e bash -lc 'docker ps --format "table {{.Names}}\t{{.Status}}\t{{.Ports}}"'
Write-Host "=== nvidia-smi ===" -ForegroundColor Cyan
wsl -e bash -lc 'nvidia-smi'

if ($Model -eq 'all') { Write-Host "`nAll models tested and stopped. Start one for benchmarking:" -ForegroundColor Green }
else { Write-Host "`n$Model is running on :$($Models[$Model].port). Benchmark it with vllm-benchmark." -ForegroundColor Green }
