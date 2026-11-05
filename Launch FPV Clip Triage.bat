@echo off
cd /d "%~dp0"
if not exist ".venv\Scripts\pythonw.exe" (
  echo Setting up the app for the first time...
  python -m venv .venv
  call .venv\Scripts\activate.bat
  pip install -r requirements.txt
)
start "" "%~dp0.venv\Scripts\pythonw.exe" -m fpv_triage
