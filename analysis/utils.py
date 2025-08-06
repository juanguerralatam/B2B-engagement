#!/usr/bin/env python3
"""
Utility functions for the B2B Engagement project
Common utilities extracted from various modules for better code organization
"""

import os
import csv
import json
import subprocess
import sys
import shutil
import importlib
from typing import Dict, List, Optional, Any, Tuple

# Optional imports with fallbacks
try:
    import chardet
except ImportError:
    chardet = None


# =============================================================================
# File and Path Utilities
# =============================================================================

def get_project_root():
    """Get the absolute path to the project root directory."""
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def ensure_directory_exists(directory_path: str) -> bool:
    """Create directory if it doesn't exist."""
    try:
        os.makedirs(directory_path, exist_ok=True)
        return True
    except (OSError, PermissionError) as e:
        print(f"Error creating directory {directory_path}: {e}")
        return False


def get_output_directory(subdir: str = None) -> str:
    """Get output directory path with optional subdirectory."""
    project_root = get_project_root()
    if subdir:
        return os.path.join(project_root, "output", subdir)
    return os.path.join(project_root, "output")


def find_file_in_locations(filename: str, locations: List[str]) -> Optional[str]:
    """Find a file in multiple possible locations."""
    for location in locations:
        filepath = os.path.join(location, filename)
        if os.path.exists(filepath):
            return filepath
    return None


def get_file_extension_variants(video_id: str, extensions: List[str]) -> List[str]:
    """Generate file paths with different extensions."""
    return [f"{video_id}{ext}" for ext in extensions]


# =============================================================================
# File Encoding and Reading Utilities
# =============================================================================

def detect_file_encoding(file_path: str) -> str:
    """Detect file encoding using chardet."""
    if chardet is None:
        return 'utf-8'
    
    try:
        with open(file_path, 'rb') as f:
            raw_data = f.read()
            result = chardet.detect(raw_data)
            return result.get('encoding', 'utf-8')
    except Exception:
        return 'utf-8'


def load_text_file_safe(file_path: str) -> Optional[str]:
    """Load text file with automatic encoding detection."""
    try:
        # Try UTF-8 first
        with open(file_path, 'r', encoding='utf-8') as f:
            return f.read()
    except UnicodeDecodeError:
        try:
            # Detect encoding and retry
            encoding = detect_file_encoding(file_path)
            with open(file_path, 'r', encoding=encoding) as f:
                return f.read()
        except Exception as e:
            print(f"Error reading file {file_path}: {e}")
            return None
    except Exception as e:
        print(f"Error reading file {file_path}: {e}")
        return None


def save_json_file(data: Any, file_path: str, ensure_dir: bool = True) -> bool:
    """Save data to JSON file with optional directory creation."""
    try:
        if ensure_dir:
            directory = os.path.dirname(file_path)
            if directory and not ensure_directory_exists(directory):
                return False
        
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        return True
    except Exception as e:
        print(f"Error saving JSON file {file_path}: {e}")
        return False


def load_json_file(file_path: str) -> Optional[Dict]:
    """Load JSON file safely."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        print(f"Error loading JSON file {file_path}: {e}")
        return None


# =============================================================================
# CSV Utilities
# =============================================================================

def detect_csv_format(csv_file: str) -> Dict[str, Any]:
    """Detect CSV format and delimiter."""
    try:
        with open(csv_file, 'r', encoding='utf-8') as f:
            sample = f.read(1024)
            
        # Try to detect delimiter
        sniffer = csv.Sniffer()
        delimiter = sniffer.sniff(sample).delimiter
        
        # Check if first line looks like a header
        lines = sample.split('\n')
        has_header = len(lines) > 1 and sniffer.has_header(sample)
        
        return {
            'delimiter': delimiter,
            'has_header': has_header,
            'encoding': 'utf-8'
        }
    except Exception:
        return {
            'delimiter': ',',
            'has_header': True,
            'encoding': 'utf-8'
        }


def read_csv_safe(csv_file: str, expected_columns: List[str] = None) -> List[Dict]:
    """Read CSV file with format detection and validation."""
    try:
        csv_format = detect_csv_format(csv_file)
        
        with open(csv_file, 'r', encoding=csv_format['encoding']) as f:
            reader = csv.DictReader(f, delimiter=csv_format['delimiter'])
            
            data = list(reader)
            
            # Validate expected columns if provided
            if expected_columns and data:
                available_columns = set(data[0].keys())
                missing_columns = set(expected_columns) - available_columns
                if missing_columns:
                    print(f"Warning: Missing columns in CSV: {missing_columns}")
            
            return data
            
    except Exception as e:
        print(f"Error reading CSV file {csv_file}: {e}")
        return []


def write_csv_safe(data: List[Dict], csv_file: str, fieldnames: List[str] = None) -> bool:
    """Write data to CSV file safely."""
    try:
        if not data:
            print("No data to write")
            return False
        
        if fieldnames is None:
            fieldnames = list(data[0].keys())
        
        # Ensure directory exists
        directory = os.path.dirname(csv_file)
        if directory and not ensure_directory_exists(directory):
            return False
        
        with open(csv_file, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(data)
        
        return True
        
    except Exception as e:
        print(f"Error writing CSV file {csv_file}: {e}")
        return False


def read_channel_ids_from_csv(csv_file: str) -> List[str]:
    """Read channel IDs from CSV file with flexible format support."""
    try:
        data = read_csv_safe(csv_file)
        if not data:
            return []
        
        # Try different possible column names for channel IDs
        possible_columns = ['channel_id', 'channelId', 'Channel ID', 'id', 'ID']
        
        for row in data:
            for col in possible_columns:
                if col in row and row[col]:
                    # Found a valid column, extract all channel IDs
                    return [row[col] for row in data if row.get(col)]
        
        # If no column found, try to extract from first non-empty value
        for row in data:
            for value in row.values():
                if value and len(value) == 24:  # YouTube channel ID length
                    return [row_data[list(row.keys())[0]] for row_data in data 
                           if row_data.get(list(row.keys())[0])]
        
        print(f"No valid channel ID column found in {csv_file}")
        return []
        
    except Exception as e:
        print(f"Error reading channel IDs from {csv_file}: {e}")
        return []


# =============================================================================
# Hardware Detection Utilities
# =============================================================================

def check_gpu_availability() -> bool:
    """Check if GPU is available for processing."""
    try:
        import torch
        return torch.cuda.is_available()
    except ImportError:
        return False


def get_system_memory_gb() -> float:
    """Get system memory in GB."""
    try:
        import psutil
        return psutil.virtual_memory().total / (1024**3)
    except ImportError:
        return 8.0  # Default assumption


def detect_optimal_device_settings() -> Dict[str, Any]:
    """Detect optimal device settings for ML operations."""
    settings = {
        'device': 'cpu',
        'gpu_available': False,
        'memory_gb': get_system_memory_gb(),
        'fp16_enabled': False
    }
    
    if check_gpu_availability():
        import torch
        settings.update({
            'device': 'cuda',
            'gpu_available': True,
            'gpu_memory_gb': torch.cuda.get_device_properties(0).total_memory / (1024**3),
            'fp16_enabled': True
        })
    
    return settings


# =============================================================================
# Package and Dependency Utilities
# =============================================================================

def check_package_installed(package_name: str, import_name: str = None) -> bool:
    """Check if a package is installed."""
    if import_name is None:
        import_name = package_name.replace('-', '_').split('[')[0]
    
    try:
        importlib.import_module(import_name)
        return True
    except ImportError:
        return False


def install_package_safe(package_name: str, import_name: str = None, quiet: bool = True) -> bool:
    """Install a package safely."""
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


def check_system_dependencies() -> Dict[str, bool]:
    """Check system dependencies."""
    return {
        "ffmpeg": shutil.which("ffmpeg") is not None,
        "python": sys.version_info >= (3, 8),
    }


# =============================================================================
# Progress and Status Utilities
# =============================================================================

def print_status(message: str, level: str = "INFO") -> None:
    """Print status message with level indicator."""
    levels = {
        "INFO": "",
        "WARNING": "⚠️ ",
        "ERROR": "❌ ",
        "SUCCESS": "✅ "
    }
    prefix = levels.get(level.upper(), "")
    print(f"{prefix}{message}")


def print_progress(current: int, total: int, prefix: str = "Progress") -> None:
    """Print progress indicator."""
    if total <= 0:
        return
    
    percentage = (current / total) * 100
    print(f"{prefix}: {current}/{total} ({percentage:.1f}%)")


def get_timestamp() -> str:
    """Get current timestamp as string."""
    try:
        return subprocess.check_output(['date', '+%Y-%m-%d %H:%M:%S']).decode().strip()
    except subprocess.CalledProcessError:
        from datetime import datetime
        return datetime.now().strftime('%Y-%m-%d %H:%M:%S')


# =============================================================================
# Data Validation Utilities
# =============================================================================

def validate_video_id(video_id: str) -> bool:
    """Validate YouTube video ID format."""
    if not video_id or len(video_id) != 11:
        return False
    
    # YouTube video IDs contain alphanumeric characters, hyphens, and underscores
    import re
    return bool(re.match(r'^[a-zA-Z0-9_-]{11}$', video_id))


def validate_file_path(file_path: str, must_exist: bool = True) -> bool:
    """Validate file path."""
    if not file_path:
        return False
    
    if must_exist:
        return os.path.exists(file_path)
    
    # Check if parent directory exists or can be created
    parent_dir = os.path.dirname(file_path)
    return not parent_dir or os.path.exists(parent_dir) or ensure_directory_exists(parent_dir)


def sanitize_filename(filename: str) -> str:
    """Sanitize filename for cross-platform compatibility."""
    import re
    # Remove or replace invalid characters
    sanitized = re.sub(r'[<>:"/\\|?*]', '_', filename)
    # Remove leading/trailing spaces and dots
    sanitized = sanitized.strip(' .')
    # Limit length
    return sanitized[:255]


# =============================================================================
# Configuration Utilities
# =============================================================================

def load_config_file(config_path: str = None) -> Dict[str, Any]:
    """Load configuration from file."""
    if config_path is None:
        config_path = os.path.join(get_project_root(), 'config.json')
    
    if os.path.exists(config_path):
        return load_json_file(config_path) or {}
    
    # Return default configuration
    return {
        "output_directories": {
            "video": "output/video",
            "audio": "output/audio", 
            "text": "output/text",
            "json": "output/json"
        },
        "whisper": {
            "model": "base",
            "language": "en"
        },
        "video": {
            "download_format": "mp4",
            "max_resolution": "720p"
        }
    }


def save_config_file(config: Dict[str, Any], config_path: str = None) -> bool:
    """Save configuration to file."""
    if config_path is None:
        config_path = os.path.join(get_project_root(), 'config.json')
    
    return save_json_file(config, config_path)
