#!/usr/bin/env python3
"""
ComfyUI-NoiseGen Installation Script
Installs dependencies and verifies the node pack installation.
"""

import sys
import os
import subprocess
import importlib.util

def run_command(command, description="Command"):
    """Run a shell command and handle errors gracefully."""
    try:
        result = subprocess.run(command, shell=True, check=True, capture_output=True, text=True)
        print(f"SUCCESS {description} completed successfully")
        if result.stdout.strip():
            print(f"Output: {result.stdout.strip()}")
        return True
    except subprocess.CalledProcessError as e:
        print(f"ERROR {description} failed:")
        print(f"Command: {command}")
        print(f"Error: {e.stderr}")
        return False

def check_python_version():
    """Check if Python version is 3.8 or higher."""
    if sys.version_info < (3, 8):
        return False
    return True

def check_package_availability(package):
    """Check if a Python package is available."""
    try:
        spec = importlib.util.find_spec(package)
        if spec is not None:
            print(f"OK {package} is available")
            return True
        else:
            print(f"MISSING {package} is missing")
            return False
    except ImportError:
        print(f"ERROR {package} is missing")
        return False

def main():
    print("NOISEGEN")
    print("ComfyUI-NoiseGen Installation Script")
    print("-" * 50)
    
    # Check if we're in the right directory
    if not os.path.exists("noise_nodes.py"):
        print("ERROR: Please run this script from the noisegen directory")
        print("Expected files: noise_nodes.py, requirements.txt")
        sys.exit(1)
    
    # Check Python version
    if not check_python_version():
        print("ERROR: Python 3.8 or higher is required")
        print(f"Current version: {sys.version}")
        sys.exit(1)
    
    print(f"Python version: {sys.version.split()[0]}")
    
    # Install dependencies
    print("\nInstalling Dependencies")
    print("-" * 25)
    
    if os.path.exists("requirements.txt"):
        print("Installing from requirements.txt...")
        success = run_command(
            f"{sys.executable} -m pip install -r requirements.txt",
            "pip install"
        )
        if not success:
            print("ERROR Failed to install dependencies")
            print("You may need to install manually:")
            print("pip install torch numpy scipy")
            sys.exit(1)
    else:
        print("OK All dependencies are already installed")
    
    # Verify installation
    print("\nVerifying installation...")
    print("-" * 25)
    
    try:
        from noise_nodes import NODE_CLASS_MAPPINGS
        print(f"SUCCESS Found {len(NODE_CLASS_MAPPINGS)} node classes")
        print(f"Node types: {list(NODE_CLASS_MAPPINGS.keys())}")
    except Exception as e:
        print(f"ERROR Installation verification failed: {e}")
        sys.exit(1)
    
    # Success message
    print("\nInstallation completed successfully!")
    print("\nNext steps:")
    print("1. Restart ComfyUI")
    print("2. Look for 'NoiseGen' categories in the Add Node menu")
    print("3. Start with the 'Noise Generator' node")
    print("\nRepository: https://github.com/eg0pr0xy/noisegen")
    print("Generate chaos. Destroy silence. Create music.")

if __name__ == "__main__":
    main() 