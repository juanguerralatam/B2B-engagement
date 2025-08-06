#!/usr/bin/env python3
"""
Simplified Environment Checker for B2B Engagement Project
Checks and installs required dependencies with minimal output
"""

import sys
import os
import subprocess
import importlib
import shutil
import json
from typing import List, Tuple, Dict

# Load configuration
def load_config():
    """Load configuration from config.json"""
    config_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'config.json')
    with open(config_path, 'r') as f:
        return json.load(f)

# Add analysis directory to path to import utils
analysis_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'analysis')
sys.path.insert(0, analysis_dir)

try:
    from utils import (
        check_package_installed, install_package_safe, check_gpu_availability,
        check_system_dependencies, print_status
    )
except ImportError:
    # Fallback if utils can't be imported
    def check_package_installed(package_name: str, import_name: str = None) -> bool:
        if import_name is None:
            import_name = package_name.replace('-', '_').split('[')[0]
        try:
            importlib.import_module(import_name)
            return True
        except ImportError:
            return False
    
    def install_package_safe(package_name: str, import_name: str = None, quiet: bool = True) -> bool:
        if check_package_installed(package_name, import_name):
            return True
        if not quiet:
            print(f"Installing {package_name}...")
        try:
            subprocess.check_call([
                sys.executable, "-m", "pip", "install", package_name
            ], stdout=subprocess.DEVNULL if quiet else None, 
               stderr=subprocess.DEVNULL if quiet else None)
            return True
        except subprocess.CalledProcessError:
            return False
    
    def check_gpu_availability() -> bool:
        try:
            import torch
            return torch.cuda.is_available()
        except ImportError:
            return False
    
    def check_system_dependencies() -> Dict[str, bool]:
        return {
            "ffmpeg": shutil.which("ffmpeg") is not None,
            "python": sys.version_info >= (3, 8),
        }
    
    def print_status(message: str, level: str = "INFO") -> None:
        levels = {"INFO": "", "WARNING": "⚠️ ", "ERROR": "❌ ", "SUCCESS": "✅ "}
        prefix = levels.get(level.upper(), "")
        print(f"{prefix}{message}")

class EnvChecker:
    """Simplified environment checker and package installer."""
    
    def __init__(self, quiet: bool = False):
        self.quiet = quiet
        self.config = load_config()
    
    def check_package(self, package_name: str, import_name: str = None) -> bool:
        """Check if a package is installed."""
        return check_package_installed(package_name, import_name)
    
    def install_package(self, package_name: str, import_name: str = None) -> bool:
        """Install a package if not available."""
        return install_package_safe(package_name, import_name, quiet=self.quiet)
    
    def check_gpu(self) -> bool:
        """Check if GPU is available for processing."""
        return check_gpu_availability()
    
    def check_system_deps(self) -> Dict[str, bool]:
        """Check system dependencies."""
        return check_system_dependencies()
    
    def get_required_packages(self) -> List[Tuple[str, str]]:
        """Get required packages from config or defaults."""
        try:
            # Try to get packages from config if available
            if 'packages' in self.config:
                return [(pkg, imp) for pkg, imp in self.config['packages'].items()]
        except (KeyError, TypeError):
            pass
        
        # Fallback to hardcoded packages
        return [
            ("moviepy", None),
            ("openai-whisper", "whisper"),
            ("torch", None),
            ("requests", None),
            ("scenedetect[opencv]", "scenedetect"),
            ("yt-dlp", "yt_dlp"),
            ("browser-cookie3", "browser_cookie3"),
            ("opencv-python", "cv2"),
            ("numpy", "numpy"),
            ("deepface", "deepface"),
            ("psutil", None),
        ]
    
    def install_required_packages(self) -> bool:
        """Install all required packages."""
        packages = self.get_required_packages()
        
        failed = []
        for package, import_name in packages:
            if not self.install_package(package, import_name):
                failed.append(package)
        
        if failed and not self.quiet:
            print(f"Failed to install: {', '.join(failed)}")
        
        return len(failed) == 0
    
    def run_check(self, install: bool = True) -> Dict[str, bool]:
        """Run complete environment check."""
        results = {
            "system_deps": all(self.check_system_deps().values()),
            "packages": True,
            "gpu": self.check_gpu()
        }
        
        if install:
            results["packages"] = self.install_required_packages()
        else:
            # Just check without installing - use core packages from config or defaults
            core_packages = self.get_required_packages()[:5]  # First 5 are core
            results["packages"] = all(
                self.check_package(pkg, imp) for pkg, imp in core_packages
            )
        
        return results

def main():
    """Main function."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Environment checker")
    parser.add_argument("--check-only", action="store_true", 
                       help="Only check, don't install")
    parser.add_argument("--quiet", action="store_true",
                       help="Minimal output")
    
    args = parser.parse_args()
    
    checker = EnvChecker(quiet=args.quiet)
    results = checker.run_check(install=not args.check_only)
    
    if not args.quiet:
        print_status(f"System dependencies: {'✅' if results['system_deps'] else '❌'}", "INFO")
        print_status(f"Python packages: {'✅' if results['packages'] else '❌'}", "INFO")
        print_status(f"GPU available: {'✅' if results['gpu'] else '❌'}", "INFO")
    
    # Exit with error if critical components missing
    if not (results['system_deps'] and results['packages']):
        sys.exit(1)

if __name__ == "__main__":
    main()
