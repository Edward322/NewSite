@echo off
rem Downloads Bybit history for HYPEUSDT and packs it into upload\hype-data.zip
rem Safe to run again: download resumes from where it stopped.
setlocal
cd /d "%~dp0.."

set "PY="
for %%V in (3.13 3.12 3.14 3.11) do (
  if not defined PY (
    py -%%V -c "import sys" >nul 2>nul && set "PY=py -%%V"
  )
)
if not defined PY (
  python -c "import sys; sys.exit(0 if (3,11)<=sys.version_info[:2]<=(3,14) else 1)" >nul 2>nul && set "PY=python"
)
if not defined PY goto :nopython

set "VPY=.venv\Scripts\python.exe"
if not exist "%VPY%" (
  echo [1/4] Creating Python environment...
  %PY% -m venv .venv || goto :fail
)
echo [2/4] Installing libraries (first run takes a few minutes)...
"%VPY%" -m pip install --disable-pip-version-check -q -r requirements.txt || goto :fail

echo [3/4] Downloading history from Bybit (30-60 minutes)...
"%VPY%" -m bot.cli data download || goto :fail

echo [4/4] Packing data...
"%VPY%" -m bot.cli data pack || goto :fail

start "" explorer "%CD%\upload"
echo.
echo DONE. Upload the file upload\hype-data.zip as described in the instructions.
pause
exit /b 0

:nopython
echo.
echo Python 3.11-3.14 not found.
echo Install Python 3.13 from https://www.python.org/downloads/
echo and tick "Add python.exe to PATH" in the installer, then run this file again.
pause
exit /b 1

:fail
echo.
echo ERROR. Make a screenshot of this window and send it.
echo Full log: %CD%\logs\bot.log
pause
exit /b 1
