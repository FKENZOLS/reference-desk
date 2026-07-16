@echo off
setlocal
cd /d "%~dp0"
title Reference Desk

if not exist "%~dp0.venv\Scripts\python.exe" goto :not_ready

powershell.exe -NoLogo -NoProfile -ExecutionPolicy Bypass -File "%~dp0start.ps1"
if errorlevel 1 goto :start_failed
exit /b 0

:not_ready
echo.
echo Reference Desk has not been prepared yet.
echo Double-click SETUP.bat first. You only need to do that once.
echo.
pause
exit /b 1

:start_failed
echo.
echo Reference Desk stopped because of the error shown above.
echo.
pause
exit /b 1
