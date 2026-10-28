@echo off
setlocal enabledelayedexpansion
title NeuroCast Web Server (Concept ^& Design by K.S.Chandra 2026)

:: ==============================================================================
:: Cross-Environment Python 3 Web Launcher for Windows
:: Concept & Design by K.S.Chandra 2026
:: ==============================================================================

set "PY_CMD="

:: 1. Check py launcher
where py >nul 2>&1
if not errorlevel 1 (
    py -3 -c "import sys; exit(0)" >nul 2>&1
    if not errorlevel 1 (
        set "PY_CMD=py -3"
        goto :found
    )
)

:: 2. Check python in PATH
where python >nul 2>&1
if not errorlevel 1 (
    python -c "import sys; exit(0)" >nul 2>&1
    if not errorlevel 1 (
        set "PY_CMD=python"
        goto :found
    )
)

:: 3. Check python3 in PATH
where python3 >nul 2>&1
if not errorlevel 1 (
    python3 -c "import sys; exit(0)" >nul 2>&1
    if not errorlevel 1 (
        set "PY_CMD=python3"
        goto :found
    )
)

:: 4. Check standard AppData Local Python install paths
if exist "%LOCALAPPDATA%\Python\bin\python.exe" (
    "%LOCALAPPDATA%\Python\bin\python.exe" -c "import sys; exit(0)" >nul 2>&1
    if not errorlevel 1 (
        set "PY_CMD="%LOCALAPPDATA%\Python\bin\python.exe""
        goto :found
    )
)

for /d %%D in ("%LOCALAPPDATA%\Programs\Python\Python3*") do (
    if exist "%%D\python.exe" (
        "%%D\python.exe" -c "import sys; exit(0)" >nul 2>&1
        if not errorlevel 1 (
            set "PY_CMD="%%D\python.exe""
            goto :found
        )
    )
)

for /d %%D in ("C:\Python3*") do (
    if exist "%%D\python.exe" (
        "%%D\python.exe" -c "import sys; exit(0)" >nul 2>&1
        if not errorlevel 1 (
            set "PY_CMD="%%D\python.exe""
            goto :found
        )
    )
)

for /d %%D in ("%ProgramFiles%\Python3*") do (
    if exist "%%D\python.exe" (
        "%%D\python.exe" -c "import sys; exit(0)" >nul 2>&1
        if not errorlevel 1 (
            set "PY_CMD="%%D\python.exe""
            goto :found
        )
    )
)

:notfound
echo =====================================================================
echo  [ERROR] Python 3 could not be detected automatically.
echo  Please install Python 3 from: https://www.python.org/downloads/
echo =====================================================================
pause
exit /b 1

:found
echo Starting NeuroCast Web Server...
!PY_CMD! "%~dp0server.py" %*
if errorlevel 1 (
    echo.
    echo Application exited with return code %errorlevel%.
    pause
)
