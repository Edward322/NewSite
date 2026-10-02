@echo off
setlocal
cd /d "%~dp0.."
set "MODE=%~1"
if "%MODE%"=="" set "MODE=demo"
set PYTHONUTF8=1
".venv\Scripts\python.exe" -m bot.engine status --mode %MODE%
schtasks /Query /TN "HypeBot-%MODE%" /FO LIST 2>nul
pause
