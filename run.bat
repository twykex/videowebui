@echo off
setlocal

:: --- CONFIGURATION ---
set "ENGINE_DIR=..\kandinsky-5"
set "VENV_PYTHON=%ENGINE_DIR%\venv\Scripts\python.exe"
set "UI_SCRIPT=web_ui.py"

:: --- CHECKS ---
if not exist "%ENGINE_DIR%" (
    echo [ERROR] Could not find Engine folder at: %ENGINE_DIR%
    pause
    exit /b
)

if not exist "%VENV_PYTHON%" (
    if exist "%ENGINE_DIR%\weights\venv\Scripts\python.exe" (
        set "VENV_PYTHON=%ENGINE_DIR%\weights\venv\Scripts\python.exe"
    ) else (
        echo [ERROR] Virtual Environment not found in %ENGINE_DIR%
        pause
        exit /b
    )
)

:: --- SYNC & REPAIR ---
echo.
echo [1/4] Syncing UI code to Engine...
copy /Y "%UI_SCRIPT%" "%ENGINE_DIR%\%UI_SCRIPT%" >nul

echo [2/4] Checking Engine dependencies...
"%VENV_PYTHON%" -m pip install fastapi uvicorn --quiet

echo [3/4] Switching to Engine Context...
:: === CRITICAL FIX: GO INSIDE THE FOLDER ===
cd /d "%ENGINE_DIR%"

echo [4/4] Launching Studio...
echo.
echo Open http://localhost:8000 in your browser when ready.
echo.

:: Run the script from INSIDE the engine folder
"%VENV_PYTHON%" "%UI_SCRIPT%"

pause