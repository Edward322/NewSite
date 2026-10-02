@echo off
rem Builds the demo test report (reports\DEMO_REPORT.md) and opens it.
setlocal
cd /d "%~dp0.."
set "MODE=%~1"
if "%MODE%"=="" set "MODE=demo"
set PYTHONUTF8=1
".venv\Scripts\python.exe" -m bot.report.demo --mode %MODE% > "reports\demo_report_console.txt" 2>&1
if "%MODE%"=="demo" (start "" notepad "reports\DEMO_REPORT.md") else (start "" notepad "reports\LIVE_REPORT.md")
echo Report written. Send the file reports\DEMO_REPORT.md to the developer.
pause
