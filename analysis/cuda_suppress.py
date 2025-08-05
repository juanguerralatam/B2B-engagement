#!/usr/bin/env python3
"""
CUDA Warning Suppressor for Whisper
Sets environment variables to disable CUDA and suppress warnings
"""

import os
import sys
import warnings

def suppress_cuda_warnings():
    """Suppress CUDA-related warnings and force CPU usage."""
    
    # Environment variables to disable CUDA
    os.environ['CUDA_VISIBLE_DEVICES'] = ''
    os.environ['TORCH_USE_CUDA_DSA'] = '0'
    os.environ['CUDA_LAUNCH_BLOCKING'] = '0'
    
    # Suppress specific warnings
    warnings.filterwarnings("ignore", message=".*CUDA initialization.*")
    warnings.filterwarnings("ignore", message=".*FP16 is not supported on CPU.*")
    warnings.filterwarnings("ignore", category=UserWarning, module="torch.cuda")
    warnings.filterwarnings("ignore", category=UserWarning, module="whisper.transcribe")
    
    print("🔇 CUDA warnings suppressed, using CPU for processing")

# Apply suppression when module is imported
suppress_cuda_warnings()
