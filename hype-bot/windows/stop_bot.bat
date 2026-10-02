@echo off
rem EMERGENCY STOP: close all bot positions, cancel orders, switch the bot off.
setlocal
cd /d "%~dp0.."
set "MODE=%~1"
if "%MODE%"=="" set "MODE=demo"
set PYTHONUTF8=1
echo EMERGENCY STOP (%MODE%): closing positions, cancelling orders, stopping the bot...
echo emergency stop> STOP
".venv\Scripts\python.exe" -m bot.engine kill --mode %MODE%
schtasks /Change /TN "HypeBot-%MODE%" /DISABLE >nul 2>nul
echo.
echo Done. The bot will not start again until you run windows\resume.bat
pause
