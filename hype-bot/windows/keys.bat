@echo off
rem Re-enter API keys (checked on the exchange right away), then a quick self-check.
setlocal
cd /d "%~dp0.."
set "MODE=%~1"
if "%MODE%"=="" set "MODE=demo"
set PYTHONUTF8=1
".venv\Scripts\python.exe" -m bot.setup_env keys %MODE% || goto :end
".venv\Scripts\python.exe" -m bot.selfcheck --mode %MODE% --quick
:end
pause
