param(
    [string]$Summary = "Manual checkpoint"
)

$ErrorActionPreference = 'Stop'
$repoRoot = Split-Path -Parent $PSScriptRoot

Write-Host "Creating restart snapshot..." -ForegroundColor Cyan
python (Join-Path $repoRoot 'tools\restart_archive.py') snapshot --summary "$Summary"

Write-Host ""
Write-Host "Snapshot complete. Current status:" -ForegroundColor Green
& (Join-Path $repoRoot 'tools\status_now.ps1')
