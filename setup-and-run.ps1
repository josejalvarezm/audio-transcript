# Audio Transcript Setup & Run Script
# This script installs all dependencies and launches the application
# Usage: .\setup-and-run.ps1

$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot

Write-Host "========================================" -ForegroundColor Cyan
Write-Host "  Audio Transcript - Setup & Run" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""

# Check if virtual environment exists
$venvPath = ".\.venv"
if (-not (Test-Path $venvPath)) {
    Write-Host "[1/5] Creating virtual environment..." -ForegroundColor Yellow
    python -m venv .venv
    if (-not $?) {
        Write-Host "  [ERROR] Failed to create virtual environment" -ForegroundColor Red
        exit 1
    }
} else {
    Write-Host "[1/5] Virtual environment exists" -ForegroundColor Green
}

# Activate virtual environment
Write-Host "[2/5] Activating virtual environment..." -ForegroundColor Yellow
& .\.venv\Scripts\Activate.ps1

# Check for CUDA availability and install appropriate PyTorch
Write-Host "[3/5] Checking PyTorch installation..." -ForegroundColor Yellow
$pytorchInstalled = pip show torch 2>$null
if (-not $pytorchInstalled) {
    # Check if NVIDIA GPU is available
    $nvidiaSmi = Get-Command nvidia-smi -ErrorAction SilentlyContinue
    if ($nvidiaSmi) {
        Write-Host "  NVIDIA GPU detected, installing PyTorch with CUDA 12.6..." -ForegroundColor Yellow
        pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126
    } else {
        Write-Host "  No NVIDIA GPU detected, installing CPU-only PyTorch..." -ForegroundColor Yellow
        pip install torch torchvision torchaudio
    }
} else {
    Write-Host "  PyTorch already installed" -ForegroundColor Green
}

# Install core dependencies
Write-Host "[4/5] Installing dependencies..." -ForegroundColor Yellow
pip install -r requirements.txt

# Install optional enhancements
Write-Host "  Installing punctuation restoration (recommended)..." -ForegroundColor Yellow
pip install deepmultilingualpunctuation 2>$null
if ($LASTEXITCODE -eq 0) {
    Write-Host "  [OK] Punctuation restoration installed" -ForegroundColor Green
} else {
    Write-Host "  [WARNING] Punctuation restoration failed (will use basic fallback)" -ForegroundColor Yellow
}

# Check if pyannote is installed
$pyannoteInstalled = pip show pyannote.audio 2>$null
if (-not $pyannoteInstalled) {
    Write-Host ""
    Write-Host "  Note: For speaker diarization, install pyannote.audio:" -ForegroundColor Cyan
    Write-Host "    pip install pyannote.audio" -ForegroundColor Gray
    Write-Host "    Then set HF_TOKEN in .env file" -ForegroundColor Gray
}

# Run the application
Write-Host ""
Write-Host "[5/5] Launching Audio Transcript..." -ForegroundColor Green
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""

python -m app.main
