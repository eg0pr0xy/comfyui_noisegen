#!/usr/bin/env python3
"""
Installation script for ComfyUI-NoiseGen
Automatically installs dependencies and verifies the installation.
"""

import subprocess
import sys
import os
from pathlib import Path

def run_command(command, description):
    """Run a command and handle errors gracefully."""
    print(f"🔄 {description}...")
    try:
        result = subprocess.run(command, shell=True, check=True, 
                              capture_output=True, text=True)
        print(f"✅ {description} completed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ {description} failed:")
        print(f"Error: {e.stderr}")
        return False

def check_dependencies():
    """Check if required dependencies are available."""
    print("🔍 Checking dependencies...")
    
    required_packages = ['numpy', 'scipy']
    missing_packages = []
    
    for package in required_packages:
        try:
            __import__(package)
            print(f"✅ {package} is available")
        except ImportError:
            missing_packages.append(package)
            print(f"❌ {package} is missing")
    
    return missing_packages

def main():
    """Main installation function."""
    print("🎵 ComfyUI-NoiseGen Installation Script")
    print("=" * 50)
    
    # Check if we're in the right directory
    if not Path("__init__.py").exists() or not Path("noise_nodes.py").exists():
        print("❌ Error: Please run this script from the noisegen directory")
        sys.exit(1)
    
    # Check current Python version
    print(f"🐍 Python version: {sys.version}")
    if sys.version_info < (3, 8):
        print("❌ Error: Python 3.8 or higher is required")
        sys.exit(1)
    
    # Check for missing dependencies
    missing = check_dependencies()
    
    if missing:
        print(f"\n📦 Installing missing packages: {', '.join(missing)}")
        success = run_command(
            f"{sys.executable} -m pip install {' '.join(missing)}", 
            "Installing dependencies"
        )
        if not success:
            print("❌ Failed to install dependencies")
            sys.exit(1)
    else:
        print("✅ All dependencies are already installed")
    
    # Verify installation by importing the module
    print("\n🧪 Verifying installation...")
    try:
        from noise_nodes import NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS
        print(f"✅ Found {len(NODE_CLASS_MAPPINGS)} node classes")
        print(f"✅ Node types: {list(NODE_CLASS_MAPPINGS.keys())}")
    except Exception as e:
        print(f"❌ Installation verification failed: {e}")
        sys.exit(1)
    
    print("\n🎉 Installation completed successfully!")
    print("\n📋 Next steps:")
    print("1. Restart ComfyUI")
    print("2. Look for NoiseGen nodes in the 'audio' category")
    print("3. Check the examples/ directory for workflow templates")
    print("4. Visit the web interface for documentation")

if __name__ == "__main__":
    main() 