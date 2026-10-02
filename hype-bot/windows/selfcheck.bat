@echo off
rem Full self-check against the real Bybit API. On DEMO it places one minimal test order
rem (about 5 USDT of virtual money) with an attached stop and closes it immediately.
rem Send the resulting file reports\selfcheck\selfcheck_*.txt to the developer.
setlocal
cd /d "%~dp0.."
set "MODE=%~1"
if "%MODE%"=="" set "MODE=demo"
set PYTHONUTF8=1
".venv\Scripts\python.exe" -m bot.selfcheck --mode %MODE%
start "" explorer "%CD%\reports\selfcheck"
pause
