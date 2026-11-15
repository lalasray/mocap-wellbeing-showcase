@echo off
setlocal
cd /d "%~dp0"
if not exist ".venv\Scripts\python.exe" (
  call setup_demo.bat
  if errorlevel 1 exit /b 1
)
.venv\Scripts\python.exe demo.py %*
if errorlevel 1 (
  echo Launch failed. Run setup_demo.bat to install or repair dependencies.
  pause
  exit /b 1
)
