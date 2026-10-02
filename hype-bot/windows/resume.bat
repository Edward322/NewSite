@echo off
rem Removes the stop (after an emergency stop or the 40%% drawdown halt) and starts the bot again.
setlocal
cd /d "%~dp0.."
set "MODE=%~1"
if "%MODE%"=="" set "MODE=demo"
set PYTHONUTF8=1
".venv\Scripts\python.exe" -m bot.engine resume --mode %MODE% || goto :end
schtasks /Change /TN "HypeBot-%MODE%" /ENABLE >nul 2>nul
schtasks /Run /TN "HypeBot-%MODE%" >nul 2>nul
echo Bot started.
:end
pause
