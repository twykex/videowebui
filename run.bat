@echo off
setlocal

:: --- CONFIGURATION ---
:: This assumes the heavy engine folder is right next to this one.
set "ENGINE_DIR=..\kandinsky-5"
set "VENV_PYTHON=%ENGINE_DIR%\venv\Scripts\python.exe"
set "UI_SCRIPT=web_ui.py"

:: --- CHECKS ---
if not exist "%ENGINE_DIR%" (
    echo [ERROR] Could not find the Kandinsky-5 engine folder at: %ENGINE_DIR%
    echo Please ensure this folder is next to your Kandinsky-Studio folder.
    pause
    exit /b
)

if not exist "%VENV_PYTHON%" (
    echo [ERROR] Virtual Environment not found.
    echo Please ensure you have set up the engine correctly in %ENGINE_DIR%
    pause
    exit /b
)

:: --- SYNC & LAUNCH ---
echo.
echo [1/3] Syncing latest UI code to Engine Room...
copy /Y "%UI_SCRIPT%" "%ENGINE_DIR%\%UI_SCRIPT%" >nul

echo [2/3] Activating Neural Engine...
echo [3/3] Launching Studio...
echo.

:: Run the script using the HEAVY environment's Python
"%VENV_PYTHON%" "%ENGINE_DIR%\%UI_SCRIPT%"

pause