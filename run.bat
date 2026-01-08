@echo off
REM Audio Transcript - Quick Start Script
REM Just double-click this file or run: .\run.bat

cd /d "%~dp0"
call .venv\Scripts\activate.bat
python app/main.py
pause
  