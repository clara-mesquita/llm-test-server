$ErrorActionPreference = 'Stop'

Write-Host "=== Installing Ollama ===" -ForegroundColor Cyan
irm https://ollama.com/install.ps1 | iex

# Refresh PATH so the newly installed ollama binary is visible in this session
$env:Path = [System.Environment]::GetEnvironmentVariable('Path', 'Machine') + ';' +
            [System.Environment]::GetEnvironmentVariable('Path', 'User')

Write-Host "=== Pulling models ===" -ForegroundColor Cyan
ollama pull llama3.2:3b
ollama pull gemma3:4b
ollama pull qwen3:8b

Write-Host "=== Testing models ===" -ForegroundColor Cyan
$prompt = 'Explain the three more important statistical methods for data analysis'
foreach ($model in @('llama3.2:3b', 'gemma3:4b', 'qwen3:8b')) {
    Write-Host "--- $model ---" -ForegroundColor Yellow
    ollama run $model $prompt
}

Write-Host "=== ollama ps ===" -ForegroundColor Cyan
ollama ps

Write-Host "=== nvidia-smi ===" -ForegroundColor Cyan
nvidia-smi

Write-Host "Done." -ForegroundColor Green
