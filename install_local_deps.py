#!/usr/bin/env python
"""
Installation script for local transcription dependencies.

This script checks for and installs the dependencies required for local transcription
using Whisper without requiring an API key.
"""
import subprocess
import sys
import importlib.util
from typing import List, Tuple


def check_dependency(package_name: str) -> bool:
    """Check if a package is installed."""
    return importlib.util.find_spec(package_name) is not None


def check_numpy_version() -> Tuple[bool, str]:
    """Check if numpy is installed and has a compatible version."""
    if not check_dependency("numpy"):
        return False, "not installed"
    
    import numpy
    version = numpy.__version__
    if version.startswith("2."):
        return False, f"incompatible version {version} (need <2.0.0)"
    
    return True, version


def get_missing_dependencies() -> List[str]:
    """Get a list of missing dependencies."""
    missing = []
    
    # Check whisper
    if not check_dependency("whisper"):
        missing.append("openai-whisper")
    
    # Check torch
    if not check_dependency("torch"):
        missing.append("torch")
    
    # Check numpy
    numpy_ok, numpy_status = check_numpy_version()
    if not numpy_ok:
        if numpy_status == "not installed":
            missing.append("numpy<2.0.0")
        else:
            # Need to uninstall and reinstall numpy
            missing.append("numpy<2.0.0")
            print(f"⚠️ Numpy {numpy_status}")
            print("   Will need to downgrade numpy")
    
    # Check numba
    if not check_dependency("numba"):
        missing.append("numba")
    
    return missing


def install_dependencies(packages: List[str]) -> bool:
    """Install the specified packages using pip."""
    if not packages:
        return True
    
    print(f"Installing {len(packages)} packages: {', '.join(packages)}")
    
    # Check if numpy needs to be downgraded
    if "numpy<2.0.0" in packages and check_dependency("numpy"):
        print("Uninstalling numpy to downgrade...")
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "uninstall", "-y", "numpy"])
        except subprocess.CalledProcessError:
            print("❌ Failed to uninstall numpy")
            return False
    
    # Install packages
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install"] + packages)
        return True
    except subprocess.CalledProcessError:
        print("❌ Failed to install packages")
        return False


def main():
    """Main function."""
    print("Checking for local transcription dependencies...")
    
    missing = get_missing_dependencies()
    
    if not missing:
        print("✅ All dependencies are already installed!")
        return
    
    print(f"Missing {len(missing)} dependencies: {', '.join(missing)}")
    
    install = input("Do you want to install these dependencies? (y/n): ").lower()
    if install != "y":
        print("Installation cancelled.")
        return
    
    if install_dependencies(missing):
        print("✅ Dependencies installed successfully!")
    else:
        print("❌ Failed to install some dependencies.")
        print("Please try installing them manually:")
        print(f"pip install {' '.join(missing)}")


if __name__ == "__main__":
    main()
