@echo off
setlocal
cd /d "%~dp0"
if not exist ".venv\Scripts\python.exe" (
  py -3.13 -m venv .venv
  if errorlevel 1 goto failure
)
.venv\Scripts\python.exe -m pip install -r requirements.txt
if errorlevel 1 goto failure
.venv\Scripts\python.exe -m pip install -r requirements-model.txt
if errorlevel 1 goto failure
.venv\Scripts\python.exe setup_model.py
if errorlevel 1 goto failure
.venv\Scripts\python.exe setup_protective_model.py
if errorlevel 1 goto failure
echo Setup complete. Run launch_demo.bat.
exit /b 0
:failure
echo Setup failed. Check Python 3.13, the Python launcher, and your internet connection.
pause
exit /b 1
