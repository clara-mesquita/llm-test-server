# run-all-tests.ps1
# Runs in order: ollama-activation -> ollama-benchmark -> vllm-activation -> vllm-benchmark,
# waiting for each to finish before starting the next, for N rounds (default 10).
# Each round's JSON results are moved into .\results\.
#
# Usage:  powershell -ExecutionPolicy Bypass -File .\run-all-tests.ps1 [-Rounds 10]
param([int]$Rounds = 10)

$ErrorActionPreference = 'Stop'
Set-Location $PSScriptRoot
$ResultsDir = Join-Path $PSScriptRoot 'results_personal_pc'
New-Item -ItemType Directory -Force -Path $ResultsDir | Out-Null

function Invoke-Ps1($path) {
    Write-Host "`n>>> $path" -ForegroundColor Cyan
    & $path
    if (-not $?) { throw "script failed: $path" }
}

function Invoke-Py($file) {
    Write-Host "`n>>> python $file" -ForegroundColor Cyan
    python $file
    if ($LASTEXITCODE -ne 0) { throw "script failed: python $file (exit $LASTEXITCODE)" }
}

for ($r = 1; $r -le $Rounds; $r++) {
    $tag = 'round-{0:d2}' -f $r
    Write-Host "`n########## $tag / $Rounds ##########" -ForegroundColor Magenta

    Invoke-Ps1 '.\ollama-activation.ps1'
    Invoke-Py   'ollama-benchmark.py'
    Move-Item -Force '.\ollama-benchmark-results.json' (Join-Path $ResultsDir "ollama-benchmark-$tag.json")

    # Invoke-Ps1 '.\vllm-activation.ps1'   # default -Model all: tests each model, stops containers after
    # Invoke-Py   'vllm-benchmark.py'
    # Move-Item -Force '.\vllm-benchmark-results.json' (Join-Path $ResultsDir "vllm-benchmark-$tag.json")
}

Write-Host "`nAll $Rounds rounds done. Results in $ResultsDir" -ForegroundColor Green
Get-ChildItem $ResultsDir | Select-Object Name, Length | Format-Table -AutoSize
