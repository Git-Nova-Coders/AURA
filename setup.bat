@echo off
:: ═══════════════════════════════════════════════════════════════
::  AURA — One-Click Setup Script for Windows
::  Run this once after cloning the repo.
:: ═══════════════════════════════════════════════════════════════

echo.
echo  ╔══════════════════════════════════════════════════════╗
echo  ║    AURA — Adaptive Understanding ^& Reasoning          ║
echo  ║    Environment Setup                                  ║
echo  ╚══════════════════════════════════════════════════════╝
echo.

:: ── Check Python ──────────────────────────────────────────────
echo [1/7] Checking Python...
python --version >nul 2>&1
if errorlevel 1 (
    echo  ERROR: Python is not installed or not in PATH.
    echo  Download from: https://www.python.org/downloads/
    pause
    exit /b 1
)
python --version
echo  OK.
echo.

:: ── Create virtual environment ────────────────────────────────
echo [2/7] Creating virtual environment (.venv)...
if exist .venv (
    echo  .venv already exists — skipping creation.
) else (
    python -m venv .venv
    echo  .venv created.
)
echo.

:: ── Activate venv ────────────────────────────────────────────
echo [3/7] Activating virtual environment...
call .venv\Scripts\activate.bat
echo  Activated.
echo.

:: ── Upgrade pip ──────────────────────────────────────────────
echo [4/7] Upgrading pip...
python -m pip install --upgrade pip --quiet
echo  pip up to date.
echo.

:: ── Install core requirements ────────────────────────────────
echo [5/7] Installing Python dependencies (requirements.txt)...
echo  This may take several minutes on first run.
echo.
pip install -r requirements.txt
if errorlevel 1 (
    echo.
    echo  ERROR: Some packages failed to install.
    echo  Try running:  pip install -r requirements.txt  manually.
    pause
    exit /b 1
)
echo.
echo  Core dependencies installed.
echo.

:: ── Install PyTorch ──────────────────────────────────────────
echo [6/7] Installing PyTorch...
echo.
echo  Detecting GPU...
python -c "import subprocess; r=subprocess.run(['nvidia-smi'],capture_output=True); exit(0 if r.returncode==0 else 1)" >nul 2>&1
if errorlevel 1 (
    echo  No NVIDIA GPU detected — installing CPU-only PyTorch.
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
) else (
    echo  NVIDIA GPU detected — installing CUDA 12.6 PyTorch.
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126
)
echo.

:: ── Download YOLO26 Model Weights ────────────────────────────
echo [7/7] Downloading YOLO26 model weights (yoloe-26s-seg.pt)...
echo.
python scripts\download_models.py
if errorlevel 1 (
    echo.
    echo  WARNING: Model download had issues. The model will be
    echo  auto-downloaded by Ultralytics on first run.
)
echo.

:: ── Build frontend ────────────────────────────────────────────
echo  Building web frontend...
cd web
where node >nul 2>&1
if errorlevel 1 (
    echo  WARNING: Node.js not found. Install from https://nodejs.org/
    echo  Then run:  cd web ^&^& npm install ^&^& npm run build
) else (
    call npm install --silent
    call npm run build
    echo  Frontend built.
)
cd ..
echo.

:: ── Done ─────────────────────────────────────────────────────
echo.
echo  ╔══════════════════════════════════════════════════════╗
echo  ║  Setup Complete!                                      ║
echo  ║                                                       ║
echo  ║  To start AURA:                                       ║
echo  ║    .venv\Scripts\activate                             ║
echo  ║    python run_dashboard.py                            ║
echo  ║                                                       ║
echo  ║  For mobile camera (HTTPS mode):                      ║
echo  ║    python run_dashboard.py --ssl                      ║
echo  ╚══════════════════════════════════════════════════════╝
echo.
pause
