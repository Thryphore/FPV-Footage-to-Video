@echo off
cd /d "%~dp0"

if not exist "app_runtime\FPVFootageToVideo.exe" (
  call "%~dp0scripts\stamp_host_icon.bat"
)

if exist "%~dp0FPV Footage to Video.exe" (
  start "" "%~dp0FPV Footage to Video.exe"
  exit /b 0
)

REM Direct fallback if the branded shortcut exe is missing.
if not exist "app_runtime\FPVFootageToVideo.exe" (
  echo Failed to create app_runtime host.
  pause
  exit /b 1
)

set "PYTHONHOME="
if exist "app_runtime\python_home.txt" for /f "usebackq tokens=1,* delims==" %%A in ("app_runtime\python_home.txt") do set "PYTHONHOME=%%B"
if not defined PYTHONHOME set "PYTHONHOME=C:\Program Files\Python314"

set "VENV=%~dp0.venv"
if not exist "%VENV%\Lib\site-packages" set "VENV=%USERPROFILE%\fpv-clip-triage\.venv"

set "PYTHONPATH=%~dp0;%VENV%\Lib\site-packages"
set "PYTHONNOUSERSITE=1"
set "PATH=%PYTHONHOME%;%PYTHONHOME%\Scripts;%VENV%\Scripts;%PATH%"

start "" "%~dp0app_runtime\FPVFootageToVideo.exe" -m fpv_triage
