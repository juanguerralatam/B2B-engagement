#!/usr/bin/env python3
"""
Simplified Environment Checker for B2B Engagement Project
Checks and installs required dependencies with minimal output
"""

import sys
import subprocess
import importlib
import shutil
from typing import List, Tuple, Dict

class EnvChecker:
    """Simplified environment checker and package installer."""
    
    def __init__(self, quiet: bool = False):
        self.quiet = quiet
    
    def check_package(self, package_name: str, import_name: str = None) -> bool:
        """Check if a package is installed."""
        if import_name is None:
            import_name = package_name.replace('-', '_').split('[')[0]
        
        try:
            importlib.import_module(import_name)
            return True
        except ImportError:
            return False
    
    def install_package(self, package_name: str, import_name: str = None) -> bool:
        """Install a package if not available."""
        if self.check_package(package_name, import_name):
            return True
        
        if not self.quiet:
            print(f"Installing {package_name}...")
        
        try:
            subprocess.check_call([
                sys.executable, "-m", "pip", "install", package_name
            ], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            return True
        except subprocess.CalledProcessError:
            return False
    
    def check_gpu(self) -> bool:
        """Check if GPU is available for processing."""
        try:
            import torch
            return torch.cuda.is_available()
        except ImportError:
            return False
    
    def check_system_deps(self) -> Dict[str, bool]:
        """Check system dependencies."""
        return {
            "ffmpeg": shutil.which("ffmpeg") is not None,
            "python": sys.version_info >= (3, 8),
        }
    
    def install_required_packages(self) -> bool:
        """Install all required packages."""
        packages = [
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
            # Just check without installing
            core_packages = [
                ("moviepy", None),
                ("openai-whisper", "whisper"),
                ("torch", None),
                ("scenedetect[opencv]", "scenedetect"),
                ("yt-dlp", "yt_dlp")
            ]
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
        print(f"System dependencies: {'✅' if results['system_deps'] else '❌'}")
        print(f"Python packages: {'✅' if results['packages'] else '❌'}")
        print(f"GPU available: {'✅' if results['gpu'] else '❌'}")
    
    # Exit with error if critical components missing
    if not (results['system_deps'] and results['packages']):
        sys.exit(1)

if __name__ == "__main__":
    main()
