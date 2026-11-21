@echo off
setlocal
cd /d "%~dp0.."

set "ICO=%CD%\fpv_triage\assets\app_icon.ico"
set "RCEDIT=%CD%\tools\rcedit-x64.exe"
set "RUNTIME=%CD%\app_runtime"
set "HOST=%RUNTIME%\FPVFootageToVideo.exe"

REM Prefer the real install pythonw (not the tiny venv stub that re-execs as "Python").
set "PYDIR="
for %%P in (
  "%LocalAppData%\Programs\Python\Python314"
  "%LocalAppData%\Programs\Python\Python313"
  "%LocalAppData%\Programs\Python\Python312"
  "C:\Program Files\Python314"
  "C:\Program Files\Python313"
  "C:\Program Files\Python312"
) do (
  if exist "%%~P\pythonw.exe" if exist "%%~P\python314.dll" set "PYDIR=%%~P"
  if exist "%%~P\pythonw.exe" if exist "%%~P\python313.dll" set "PYDIR=%%~P"
  if exist "%%~P\pythonw.exe" if exist "%%~P\python312.dll" set "PYDIR=%%~P"
)

REM Fall back to home= from the active venv config.
if not defined PYDIR (
  set "VENVCFG=%CD%\.venv\pyvenv.cfg"
  if not exist "%VENVCFG%" set "VENVCFG=%USERPROFILE%\fpv-clip-triage\.venv\pyvenv.cfg"
  if exist "%VENVCFG%" for /f "tokens=1,* delims==" %%A in ('findstr /b /c:"home =" "%VENVCFG%"') do set "PYDIR=%%B"
)
if defined PYDIR set "PYDIR=%PYDIR: =%"

if not defined PYDIR (
  echo Could not locate a base Python install ^(pythonw.exe + python3xx.dll^).
  exit /b 1
)
if not exist "%PYDIR%\pythonw.exe" (
  echo Missing "%PYDIR%\pythonw.exe"
  exit /b 1
)
if not exist "%ICO%" (
  echo Missing icon: %ICO%
  exit /b 1
)

if not exist "%CD%\tools" mkdir "%CD%\tools"
if not exist "%RCEDIT%" (
  echo Downloading rcedit...
  powershell -NoProfile -Command "Invoke-WebRequest -Uri 'https://github.com/electron/rcedit/releases/download/v2.0.0/rcedit-x64.exe' -OutFile '%RCEDIT%' -UseBasicParsing"
)

if not exist "%RUNTIME%" mkdir "%RUNTIME%"
copy /Y "%PYDIR%\pythonw.exe" "%HOST%" >nul
for %%D in (python314.dll python313.dll python312.dll python3.dll) do (
  if exist "%PYDIR%\%%D" copy /Y "%PYDIR%\%%D" "%RUNTIME%\%%D" >nul
)

"%RCEDIT%" "%HOST%" --set-icon "%ICO%"
"%RCEDIT%" "%HOST%" --set-version-string "FileDescription" "FPV Footage to Video"
"%RCEDIT%" "%HOST%" --set-version-string "ProductName" "FPV Footage to Video"
"%RCEDIT%" "%HOST%" --set-version-string "InternalName" "FPVFootageToVideo"
"%RCEDIT%" "%HOST%" --set-version-string "OriginalFilename" "FPVFootageToVideo.exe"
"%RCEDIT%" "%HOST%" --set-version-string "CompanyName" "FPV Footage to Video"
"%RCEDIT%" "%HOST%" --set-version-string "LegalCopyright" "FPV Footage to Video"
echo home=%PYDIR%> "%RUNTIME%\python_home.txt"
echo Stamped %HOST% ^(base %PYDIR%^)

if exist "%CD%\fpv_triage_host.exe" del /q "%CD%\fpv_triage_host.exe"
