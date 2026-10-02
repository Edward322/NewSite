@echo off
rem Keeps the bot running: restarts it 30 s after a crash. Started by the scheduled task.
rem Exit code 0 = emergency stop (stays off), 3 = another copy already runs.
setlocal
cd /d "%~dp0.."
set "MODE=%~1"
if "%MODE%"=="" set "MODE=demo"
set PYTHONUTF8=1
set "VPY=.venv\Scripts\python.exe"
:loop
"%VPY%" -m bot.engine run --mode %MODE%
set "RC=%ERRORLEVEL%"
if "%RC%"=="0" exit /b 0
if "%RC%"=="3" exit /b 3
if "%RC%"=="2" (
  ping -n 301 127.0.0.1 >nul
  goto loop
)
ping -n 31 127.0.0.1 >nul
goto loop
