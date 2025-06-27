@echo off
echo *** NOISEGEN ***
echo ComfyUI-NoiseGen Installation Script (Windows)
echo ================================================

REM Check if Python is available
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo ERROR: Python is not installed or not in PATH
    echo Please install Python 3.8+ from https://python.org
    pause
    exit /b 1
)

echo Python version:
python --version

REM Check if we're in the right directory
if not exist "noise_nodes.py" (
    echo ERROR: Please run this script from the noisegen directory
    pause
    exit /b 1
)

REM Install dependencies
echo.
echo Installing dependencies...
python -m pip install -r requirements.txt
if %errorlevel% neq 0 (
    echo ERROR: Failed to install dependencies
    echo You may need to install manually: pip install torch numpy scipy
    pause
    exit /b 1
)

REM Verify installation
echo.
echo Verifying installation...
python -c "from noise_nodes import NODE_CLASS_MAPPINGS; print(f'SUCCESS Found {len(NODE_CLASS_MAPPINGS)} node classes')"
if %errorlevel% neq 0 (
    echo ERROR: Installation verification failed
    pause
    exit /b 1
)

echo.
echo.
echo Installation completed successfully!
echo.
echo Next steps:
echo 1. Restart ComfyUI
echo 2. Look for NoiseGen categories in the Add Node menu
echo 3. Start with the Noise Generator node
echo.
echo Repository: https://github.com/eg0pr0xy/noisegen
echo Generate chaos. Destroy silence. Create music.

pause 