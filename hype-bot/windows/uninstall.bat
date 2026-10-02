@echo off
rem Removes the bot from Task Scheduler. Does NOT close positions: run stop_bot.bat first if needed.
setlocal
set "MODE=%~1"
if "%MODE%"=="" set "MODE=demo"
schtasks /Delete /TN "HypeBot-%MODE%" /F
pause
