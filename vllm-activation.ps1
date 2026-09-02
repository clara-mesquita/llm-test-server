# vllm-activation.ps1
# Set up vLLM with Docker Desktop and smoke-test it.
#
# Usage:  .\vllm-activation.ps1 [-InstallPrerequisites] [-Model <hf-id>] [-Image <docker-image>] [-Port <port>] [-Reset] [-DockerCheck]
#   -InstallPrerequisites : install WSL2 and Docker Desktop when absent (run PowerShell as Administrator)
#   -Model : HF model for the smoke test (default Qwen/Qwen2.5-0.5B-Instruct: non-gated,
#            0.5B -> fits a 4GB VRAM GPU)
#   -Image : vLLM image (default vllm/vllm-openai:latest)
#   -Port  : host port (default 8080; avoid 8000 which this machine uses for other apps)
#   -Reset : remove vLLM containers, images, and the vLLM Hugging Face cache before the smoke test
#   -DockerCheck : start Docker if needed and verify it can run hello-world; does not require a GPU
#
# Prereqs: NVIDIA GPU, Docker Desktop using Linux containers.
#          HF_TOKEN env var for gated models (llama / gemma2 / qwen3-8b).
#
# Notes: Docker Desktop is auto-started if its engine is down. Flags are tuned for a
# small GPU: --enforce-eager (skip CUDA graphs, which can fail on WSL2), low
# --max-model-len and --gpu-memory-utilization so it fits 4GB VRAM.
# VLLM_WSL2_ENABLE_PIN_MEMORY=1 avoids "RuntimeError: UVA is not available" on WSL2.
param(
    [string]$Model = 'Qwen/Qwen2.5-0.5B-Instruct',
    [string]$Image = 'vllm/vllm-openai:latest',
    [int]$Port = 8080,
    [switch]$Reset,
    [switch]$DockerCheck,
    [switch]$InstallPrerequisites
)

$ErrorActionPreference = 'Stop'
# PowerShell 7 otherwise turns Docker stderr into an exception before this
# script can show Docker's useful error message.
if ($PSVersionTable.PSVersion.Major -ge 7) { $PSNativeCommandUseErrorActionPreference = $false }
$Container = 'vllm-smoke'
$TestPrompt = 'What is 2+2?'

function Invoke-Docker([string[]]$DockerArgs) {
    # Docker reports normal pull progress on stderr.  Keep Stop for this
    # script, but not while collecting a native command's output.
    $previousErrorAction = $ErrorActionPreference
    try {
        $ErrorActionPreference = 'Continue'
        $output = & docker @DockerArgs 2>&1
        $exitCode = $LASTEXITCODE
    } finally {
        $ErrorActionPreference = $previousErrorAction
    }
    if ($output) { $output | Write-Host }
    if ($exitCode -ne 0) { throw "docker $($DockerArgs[0]) failed (exit $exitCode)" }
}

function Test-Docker {
    try {
        & docker version --format '{{.Server.Version}}' *> $null
        return $LASTEXITCODE -eq 0
    } catch {
        return $false
    }
}

function Test-FreePort([int]$Candidate) {
    $listener = [Net.Sockets.TcpListener]::new([Net.IPAddress]::Any, $Candidate)
    try {
        $listener.Start()
        return $true
    } catch {
        return $false
    } finally {
        $listener.Stop()
    }
}

function Install-Prerequisites {
    if (-not (Get-Command wsl -ErrorAction SilentlyContinue)) {
        $principal = [Security.Principal.WindowsPrincipal] [Security.Principal.WindowsIdentity]::GetCurrent()
        if (-not $principal.IsInRole([Security.Principal.WindowsBuiltInRole]::Administrator)) {
            throw 'WSL2 is missing. Open PowerShell as Administrator and rerun with -InstallPrerequisites.'
        }
        Write-Host 'Installing WSL2 (a reboot may be required)...' -ForegroundColor Cyan
        & wsl --install --no-distribution
        if ($LASTEXITCODE -ne 0) { throw 'WSL2 installation failed.' }
        throw 'WSL2 was installed. Reboot Windows, then rerun this script.'
    }

    $dockerExe = 'C:\Program Files\Docker\Docker\Docker Desktop.exe'
    if (-not (Test-Path $dockerExe)) {
        if (-not (Get-Command winget -ErrorAction SilentlyContinue)) {
            throw 'Docker Desktop is missing and winget is unavailable. Install Docker Desktop manually, then rerun.'
        }
        Write-Host 'Installing Docker Desktop (a reboot may be required)...' -ForegroundColor Cyan
        & winget install --exact --id Docker.DockerDesktop --accept-package-agreements --accept-source-agreements
        if ($LASTEXITCODE -ne 0) { throw 'Docker Desktop installation failed.' }
        throw 'Docker Desktop was installed. Reboot if prompted, start Docker Desktop once, then rerun this script.'
    }
}

if ($InstallPrerequisites) { Install-Prerequisites }

if (-not (Test-FreePort $Port)) {
    if ($PSBoundParameters.ContainsKey('Port')) {
        throw "Port $Port is already in use. Choose a free port, for example: -Port 8081"
    }
    $Port = 8081..8090 | Where-Object { Test-FreePort $_ } | Select-Object -First 1
    if (-not $Port) { throw 'No free vLLM port found in 8080-8090. Use -Port <free-port>.' }
    Write-Host "Port 8080 is in use; using :$Port instead." -ForegroundColor Yellow
}
$env:VLLM_PORT = "$Port"

if (-not (Test-Docker)) {
    $dockerExe = 'C:\Program Files\Docker\Docker\Docker Desktop.exe'
    if (-not (Test-Path $dockerExe)) {
        throw 'Docker Desktop is not installed. Run this script as Administrator with -InstallPrerequisites.'
    }
    Write-Host "Starting Docker Desktop..." -ForegroundColor Cyan
    Start-Process $dockerExe
    $deadline = (Get-Date).AddMinutes(3)
    while (-not (Test-Docker) -and (Get-Date) -lt $deadline) { Start-Sleep 5 }
    if (-not (Test-Docker)) {
        throw 'Docker Desktop did not become ready within 3 minutes.'
    }
}
Write-Host "Docker OK" -ForegroundColor Green

if ($Reset) {
    Write-Host "Removing vLLM Docker resources..." -ForegroundColor Cyan
    $containers = @(& docker ps -aq --filter 'name=^vllm-')
    if ($containers) { Invoke-Docker (@('rm', '-f') + $containers) }
    $images = @(& docker images -q 'vllm/*')
    if ($images) { Invoke-Docker (@('rmi', '-f') + $images) }
    try { & docker volume rm vllm-hf-cache *> $null } catch {}
    Write-Host "vLLM Docker resources removed" -ForegroundColor Green
}

if ($DockerCheck) {
    Write-Host "Testing Docker with hello-world..." -ForegroundColor Cyan
    Invoke-Docker @('run', '--rm', 'hello-world')
    Write-Host "Docker can initialize and run containers" -ForegroundColor Green
    exit 0
}

# Docker exit 125 means its daemon rejected the container before vLLM ran.
# Check GPU passthrough with the tiny image first so the error is actionable.
Write-Host 'Testing Docker GPU access...' -ForegroundColor Cyan
Invoke-Docker @('run', '--rm', '--gpus', 'all', 'hello-world')
Write-Host 'Docker GPU access OK' -ForegroundColor Green

# --- Image -----------------------------------------------------------------
Write-Host "Pulling $Image (first run, may take a while)..." -ForegroundColor Cyan
Invoke-Docker @('pull', $Image)

# --- 5. Smoke test ----------------------------------------------------------
if (-not $env:HF_TOKEN) {
    Write-Host "NOTE: HF_TOKEN not set - gated model downloads will fail (401)." -ForegroundColor Yellow
    Write-Host "      $Model is public, so this smoke test does not need it." -ForegroundColor Yellow
}
Write-Host "Smoke test: $Model on :$Port" -ForegroundColor Cyan
try { & docker rm -f $Container *> $null } catch {}
$dockerArgs = @('run', '-d', '--name', $Container, '--gpus', 'all', '-p', "${Port}:8000",
                '--ipc=host', '--shm-size=8gb', '-v', 'vllm-hf-cache:/root/.cache/huggingface',
                '-e', 'VLLM_WSL2_ENABLE_PIN_MEMORY=1')
if ($env:HF_TOKEN) { $dockerArgs += @('-e', "HF_TOKEN=$env:HF_TOKEN") }
$dockerArgs += @($Image, '--model', $Model, '--max-model-len', '1024', '--gpu-memory-utilization', '0.6', '--enforce-eager')
Invoke-Docker $dockerArgs

$deadline = (Get-Date).AddMinutes(12)
$ready = $false
while ((Get-Date) -lt $deadline) {
    # fail fast if the container crashed (OOM / CUDA error) instead of waiting 12 min
    $state = (& docker inspect -f '{{.State.Status}}' $Container 2>$null).Trim()
    if ($state -eq 'exited' -or $state -eq 'dead') {
        Write-Host "container exited early ($state). Logs:" -ForegroundColor Red
        & docker logs --tail 60 $Container
        throw "vLLM container exited: $state"
    }
    try { Invoke-RestMethod "http://localhost:$Port/v1/models" -TimeoutSec 5 | Out-Null; $ready = $true; break } catch { Start-Sleep 3 }
}
if (-not $ready) {
    Write-Host "server did not become ready. Logs:" -ForegroundColor Red
    & docker logs --tail 60 $Container
    throw "vLLM server did not start"
}
Write-Host "server ready, testing..." -ForegroundColor Green
$body = @{ model = $Model; messages = @(@{ role = 'user'; content = $TestPrompt }); max_tokens = 32 } | ConvertTo-Json
$resp = Invoke-RestMethod "http://localhost:$Port/v1/chat/completions" -Method Post -Body $body -ContentType 'application/json'
$reply = ($resp.choices[0].message.content -replace '\s+', ' ')
Write-Host "  reply: $($reply.Substring(0, [Math]::Min(120, $reply.Length)))..."

# --- 6. Status --------------------------------------------------------------
Write-Host "`nvLLM works. Stopping smoke-test container..." -ForegroundColor Green
Invoke-Docker @('rm', '-f', $Container)
Write-Host "Done. Run:  python vllm-benchmark.py" -ForegroundColor Green
