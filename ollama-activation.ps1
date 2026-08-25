$ErrorActionPreference = 'Stop'

Write-Host "=== Installing Ollama ===" -ForegroundColor Cyan
irm https://ollama.com/install.ps1 | iex

# Refresh PATH so the newly installed ollama binary is visible in this session
$env:Path = [System.Environment]::GetEnvironmentVariable('Path', 'Machine') + ';' +
            [System.Environment]::GetEnvironmentVariable('Path', 'User')

# Only start the server if none is already running (the tray app may already own port 11434)
ollama list *> $null
if ($LASTEXITCODE -ne 0) {
    Write-Host "=== Starting Ollama server ===" -ForegroundColor Cyan
    Start-Process "$env:LOCALAPPDATA\Programs\Ollama\ollama.exe" -ArgumentList 'serve' -WindowStyle Hidden
    $deadline = (Get-Date).AddSeconds(90)
    do {
        Start-Sleep 2
        ollama list *> $null
    } until ($LASTEXITCODE -eq 0 -or (Get-Date) -gt $deadline)
    if ($LASTEXITCODE -ne 0) { throw 'Ollama server did not start within 90s' }
}
Write-Host '  server ready' -ForegroundColor Green

Write-Host "=== Pulling models ===" -ForegroundColor Cyan
foreach ($m in @('llama3.2:3b', 'gemma3:4b', 'qwen3:8b')) {
    Write-Host "  pulling $m..."
    ollama pull $m
    if ($LASTEXITCODE -ne 0) { throw "ollama pull $m failed (exit $LASTEXITCODE)" }
}

Write-Host "=== Testing models ===" -ForegroundColor Cyan
$prompt = 'Explain the three more important statistical methods for data analysis'
foreach ($model in @('llama3.2:3b', 'gemma3:4b', 'qwen3:8b')) {
    Write-Host "--- $model ---" -ForegroundColor Yellow
    ollama run $model $prompt
    if ($LASTEXITCODE -ne 0) { throw "ollama run $model failed (exit $LASTEXITCODE)" }
}

Write-Host "=== ollama ps ===" -ForegroundColor Cyan
ollama ps

Write-Host "=== nvidia-smi ===" -ForegroundColor Cyan
nvidia-smi

Write-Host "Done." -ForegroundColor Green
