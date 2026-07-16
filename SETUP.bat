@echo off
setlocal
cd /d "%~dp0"
title Reference Desk - First-time setup

echo.
echo ============================================================
echo   Reference Desk - first-time setup
echo ============================================================
echo.
echo This prepares the app for this computer. The first setup can
echo take a while because the search and document models are large.
echo.

where python >nul 2>&1
if errorlevel 1 goto :python_missing

powershell.exe -NoLogo -NoProfile -ExecutionPolicy Bypass -File "%~dp0scripts\setup.ps1" -Backend auto
if errorlevel 1 goto :setup_failed

echo.
echo Setup finished successfully.
echo You can now double-click START.bat whenever you want to use the app.
echo.
pause
exit /b 0

:python_missing
echo Python 3.12 (64-bit) was not found.
echo Install it from https://www.python.org/downloads/windows/
echo During installation, enable "Add Python to PATH", then run SETUP.bat again.
echo.
pause
exit /b 1

:setup_failed
echo.
echo Setup could not finish. Read the message above for the cause.
echo The INSTALL_WINDOWS.txt file contains the most common solutions.
echo.
pause
exit /b 1
