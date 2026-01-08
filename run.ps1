# Audio Transcript - Quick Start Script
# Usage: .\run.ps1
# Requires: Run setup-and-run.ps1 first for initial setup

$ErrorActionPreference = "Stop"

# Change to script directory
Set-Location $PSScriptRoot

# Check if virtual environment exists
if (-not (Test-Path ".\.venv\Scripts\Activate.ps1")) {
    Write-Host "[ERROR] Virtual environment not found!" -ForegroundColor Red
    Write-Host "Run setup-and-run.ps1 first to create the environment." -ForegroundColor Yellow
    exit 1
}

# Activate virtual environment and run
& .\.venv\Scripts\Activate.ps1
python -m app.main
