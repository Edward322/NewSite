@echo off
rem Downloads the latest bot code from GitHub and restarts the bot.
rem Keeps your .env (keys), config\bot.yaml, data\ and logs\. Open positions stay protected by exchange stops.
setlocal
cd /d "%~dp0.."
set "MODE=%~1"
if "%MODE%"=="" set "MODE=demo"
set PYTHONUTF8=1
set "BRANCH=claude/nice-allen-xl48t9"
set "URL=https://github.com/Edward322/NewSite/archive/refs/heads/%BRANCH%.zip"
set "TMPD=%TEMP%\hypebot_update"
if exist "%TMPD%" rmdir /s /q "%TMPD%"
mkdir "%TMPD%"
echo Downloading %URL% ...
powershell -NoProfile -Command "Invoke-WebRequest -UseBasicParsing -Uri '%URL%' -OutFile '%TMPD%\code.zip'; Expand-Archive -Force '%TMPD%\code.zip' '%TMPD%\code'" || goto :fail
set "SRC="
for /d %%D in ("%TMPD%\code\*") do set "SRC=%%D\hype-bot"
if not exist "%SRC%\bot" goto :fail
echo Stopping the bot (positions stay under exchange stops)...
schtasks /End /TN "HypeBot-%MODE%" >nul 2>nul
powershell -NoProfile -Command "Get-CimInstance Win32_Process -Filter \"Name='python.exe'\" | Where-Object { $_.CommandLine -like '*bot.engine run*' } | ForEach-Object { Stop-Process -Id $_.ProcessId -Force }"
robocopy "%SRC%" "%CD%" /E /XF .env bot.yaml STOP >nul
if errorlevel 8 goto :fail
".venv\Scripts\python.exe" -m pip install --disable-pip-version-check -q -r requirements.txt || goto :fail
schtasks /Run /TN "HypeBot-%MODE%" >nul 2>nul
echo Updated and restarted.
pause
exit /b 0
:fail
echo ERROR during update. Make a screenshot and send it.
pause
exit /b 1
