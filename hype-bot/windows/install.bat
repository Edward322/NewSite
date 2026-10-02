@echo off
rem One-time installation of the trading bot on Windows (no Docker).
rem   - Python environment and libraries
rem   - config and API keys (asked in the window), Telegram (optional)
rem   - quick self-check against Bybit
rem   - autostart via Windows Task Scheduler (at boot and at logon, restart on failure)
rem   - disables sleep/hibernate on AC power
rem Usage: double-click (demo). Real money only after explicit decision: install.bat live
setlocal
set "MODE=%~1"
if "%MODE%"=="" set "MODE=demo"

net session >nul 2>&1
if errorlevel 1 (
  echo Requesting administrator rights ^(needed for the scheduled task^)...
  powershell -NoProfile -Command "Start-Process -FilePath '%~f0' -ArgumentList '%MODE%' -Verb RunAs"
  exit /b 0
)
cd /d "%~dp0.."
set PYTHONUTF8=1
echo Bot folder: %CD%
echo Mode: %MODE%
echo.

set "PY="
for %%V in (3.13 3.12 3.14 3.11) do (
  if not defined PY (
    py -%%V -c "import sys" >nul 2>nul && set "PY=py -%%V"
  )
)
if not defined PY (
  python -c "import sys; sys.exit(0 if (3,11)<=sys.version_info[:2]<=(3,14) else 1)" >nul 2>nul && set "PY=python"
)
if not defined PY (
  echo Python not found - trying to install Python 3.12 with winget...
  winget install -e --id Python.Python.3.12 --scope machine --silent --accept-package-agreements --accept-source-agreements
  py -3.12 -c "import sys" >nul 2>nul && set "PY=py -3.12"
)
if not defined PY (
  if exist "%ProgramFiles%\Python312\python.exe" set PY="%ProgramFiles%\Python312\python.exe"
)
if not defined PY goto :nopython

set "VPY=.venv\Scripts\python.exe"
if not exist "%VPY%" (
  echo [1/6] Creating Python environment...
  %PY% -m venv .venv || goto :fail
)
echo [2/6] Installing libraries ^(first run takes a few minutes^)...
"%VPY%" -m pip install --disable-pip-version-check -q --upgrade pip >nul 2>nul
"%VPY%" -m pip install --disable-pip-version-check -q -r requirements.txt || goto :fail

echo [3/6] Settings: config, API keys, Telegram...
"%VPY%" -m bot.setup_env %MODE% || goto :fail

echo [4/6] Quick self-check against Bybit ^(no orders^)...
"%VPY%" -m bot.selfcheck --mode %MODE% --quick || goto :checkfail

echo [5/6] Autostart: Windows Task Scheduler...
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0register_task.ps1" -Mode %MODE% -BotDir "%CD%" || goto :fail
powercfg /change standby-timeout-ac 0 >nul 2>nul
powercfg /change hibernate-timeout-ac 0 >nul 2>nul

echo [6/6] Starting the bot...
schtasks /Run /TN "HypeBot-%MODE%" >nul || goto :fail
echo.
echo DONE. The bot runs in the background and starts automatically after reboot.
echo   status:          windows\status.bat
echo   EMERGENCY STOP:  windows\stop_bot.bat   ^(closes positions, cancels orders, stops the bot^)
echo   log:             logs\bot_%MODE%.log
pause
exit /b 0

:checkfail
echo.
echo Self-check FAILED - the bot was NOT started. Read the messages above,
echo fix the problem ^(usually API keys or internet^) and run install.bat again.
pause
exit /b 1

:nopython
echo.
echo Python 3.11-3.14 not found and could not be installed automatically.
echo Install Python 3.12 from https://www.python.org/downloads/
echo ^(tick "Add python.exe to PATH" in the installer^), then run install.bat again.
pause
exit /b 1

:fail
echo.
echo ERROR. Make a screenshot of this window and send it.
pause
exit /b 1
