@echo off
REM ========================================================================
REM VeriGraph Windows Launcher
REM Enterprise Agentic GraphRAG & Verifiable Retrieval Engine
REM ========================================================================

title VeriGraph Launcher

echo ========================================================================
echo   VeriGraph: Enterprise Agentic GraphRAG Launcher (Windows)
echo ========================================================================

where python >nul 2>nul
if %errorlevel% neq 0 (
    echo [ERROR] Python is not installed or not in PATH!
    echo Please install Python 3.10+ from https://www.python.org/downloads/
    echo Make sure to check "Add python.exe to PATH" during installation.
    pause
    exit /b 1
)

REM Setup virtual environment if not present
if not exist ".venv" (
    echo [*] Setting up virtual environment in .venv ...
    python -m venv .venv
    call .venv\Scripts\activate.bat
    echo [*] Installing dependencies from requirements.txt ...
    python -m pip install --upgrade pip
    pip install -r requirements.txt
) else (
    call .venv\Scripts\activate.bat
)

REM Run universal launcher
python run.py %*

if %errorlevel% neq 0 (
    echo [!] Server exited with an error code.
    pause
)
