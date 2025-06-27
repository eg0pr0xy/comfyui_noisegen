@echo off
echo 🎵 ComfyUI-NoiseGen Installation Script (Windows)
echo ================================================

echo.
echo 🔍 Checking Python installation...
python --version
if errorlevel 1 (
    echo ❌ Error: Python is not installed or not in PATH
    echo Please install Python 3.8 or higher from https://python.org
    pause
    exit /b 1
)

echo.
echo 📦 Installing dependencies...
python -m pip install --upgrade pip
python -m pip install -r requirements.txt

if errorlevel 1 (
    echo ❌ Error: Failed to install dependencies
    pause
    exit /b 1
)

echo.
echo 🧪 Verifying installation...
python -c "from noise_nodes import NODE_CLASS_MAPPINGS; print(f'✅ Found {len(NODE_CLASS_MAPPINGS)} node classes')"

if errorlevel 1 (
    echo ❌ Error: Installation verification failed
    pause
    exit /b 1
)

echo.
echo 🎉 Installation completed successfully!
echo.
echo 📋 Next steps:
echo 1. Restart ComfyUI
echo 2. Look for NoiseGen nodes in the 'audio' category
echo 3. Check the examples/ directory for workflow templates
echo 4. Visit the web interface for documentation
echo.
pause 