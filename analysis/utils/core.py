#!/usr/bin/env python3
"""
Core Utilities for B2B Engagement Analysis
Consolidated utilities for file operations, data processing, hardware detection, and more.
"""

import os
import csv
import json
import subprocess
import sys
import shutil
import importlib
import re
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime

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
    return os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


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
    config = load_config_file()
    
    if subdir:
        # Check if we have a specific config path for this subdirectory
        config_key = f"{subdir}_dir"
        if config.get("paths", {}).get(config_key):
            path = config["paths"][config_key]
            # Handle relative paths and home directory expansion
            if path.startswith("~/"):
                return os.path.expanduser(path)
            elif not os.path.isabs(path):
                return os.path.join(project_root, path)
            else:
                return path
        else:
            # Default fallback
            return os.path.join(project_root, "output", subdir)
    
    # For base output directory
    output_dir = config.get("paths", {}).get("output_dir", "output")
    if not os.path.isabs(output_dir):
        return os.path.join(project_root, output_dir)
    return output_dir


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
# Data Processing Utilities
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


def update_csv_record(csv_file: str, record_data: Dict, id_column: str = 'videoId') -> bool:
    """
    Update or append a record to a CSV file. If a record with the same ID exists, 
    it will be replaced. Otherwise, the record will be appended.
    
    Args:
        csv_file: Path to the CSV file
        record_data: Dictionary containing the record data
        id_column: Name of the column used as identifier (default: 'VideoId')
    
    Returns:
        bool: True if successful, False otherwise
    """
    try:
        import pandas as pd
        
        record_id = record_data.get(id_column)
        if not record_id:
            print(f"Error: Record data missing {id_column} field")
            return False
        
        # Ensure directory exists
        directory = os.path.dirname(csv_file) if os.path.dirname(csv_file) else '.'
        if not ensure_directory_exists(directory):
            return False
        
        if os.path.exists(csv_file):
            # Read existing data
            try:
                existing_df = pd.read_csv(csv_file)
                # Remove existing entry for this ID if it exists
                existing_df = existing_df[existing_df[id_column] != record_id]
                # Add new record
                new_row_df = pd.DataFrame([record_data])
                if len(existing_df) > 0:
                    new_df = pd.concat([existing_df, new_row_df], ignore_index=True)
                else:
                    new_df = new_row_df
            except pd.errors.EmptyDataError:
                # Handle empty CSV file
                new_df = pd.DataFrame([record_data])
        else:
            # Create new file
            new_df = pd.DataFrame([record_data])
        
        # Save updated data
        new_df.to_csv(csv_file, index=False)
        return True
        
    except Exception as e:
        print(f"Error updating CSV file {csv_file}: {e}")
        return False


def check_record_exists(csv_file: str, record_id: str, id_column: str = 'VideoId') -> bool:
    """
    Check if a record with the given ID already exists in the CSV file.
    
    Args:
        csv_file: Path to the CSV file
        record_id: ID to check for
        id_column: Name of the column used as identifier (default: 'VideoId')
    
    Returns:
        bool: True if record exists, False otherwise
    """
    try:
        import pandas as pd
        
        if not os.path.exists(csv_file):
            return False
            
        df = pd.read_csv(csv_file)
        return record_id in df[id_column].values
        
    except Exception:
        return False


def format_analysis_data(data: Dict, precision: int = 3) -> Dict:
    """
    Format analysis data with proper rounding and None handling.
    
    Args:
        data: Dictionary with analysis results
        precision: Number of decimal places for rounding
    
    Returns:
        Dict: Formatted data with proper types
    """
    formatted = {}
    
    for key, value in data.items():
        if value is None:
            formatted[key] = None
        elif isinstance(value, (int, float)):
            try:
                formatted[key] = round(float(value), precision)
            except (ValueError, TypeError):
                formatted[key] = None
        else:
            formatted[key] = value
    
    return formatted


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
# Configuration Management
# =============================================================================

# Configuration singleton cache
_cached_config = None

def get_shared_config() -> Dict[str, Any]:
    """Get cached configuration - loads only once per session."""
    global _cached_config
    if _cached_config is None:
        _cached_config = load_config_file()
    return _cached_config

def clear_config_cache() -> None:
    """Clear config cache - useful for testing."""
    global _cached_config
    _cached_config = None

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


# =============================================================================
# Hardware Detection and System Utilities
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

def detect_optimal_whisper_settings() -> Dict[str, Any]:
    """Detect optimal Whisper settings based on available hardware."""
    try:
        # Check for whisper import
        try:
            import whisper
        except ImportError:
            pass
        
        # Use utils function for device detection
        device_settings = detect_optimal_device_settings()
        
        # Determine recommended model based on memory
        memory_gb = device_settings['memory_gb']
        if device_settings['gpu_available'] and memory_gb >= 16:
            recommended_model = "large"
        elif memory_gb >= 8:
            recommended_model = "medium"
        else:
            recommended_model = "base"
        
        return {
            "device": device_settings['device'],
            "recommended_model": recommended_model,
            "fp16": device_settings['fp16_enabled']
        }
        
    except ImportError as e:
        return {"device": "cpu", "recommended_model": "base", "fp16": False}
    except Exception as e:
        return {"device": "cpu", "recommended_model": "base", "fp16": False}


# =============================================================================
# Progress, Status, and Validation Utilities
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
        return datetime.now().strftime('%Y-%m-%d %H:%M:%S')


def validate_video_id(video_id: str) -> bool:
    """Validate YouTube video ID format."""
    if not video_id or len(video_id) != 11:
        return False
    
    # YouTube video IDs contain alphanumeric characters, hyphens, and underscores
    import re
    return bool(re.match(r'^[a-zA-Z0-9_-]{11}$', video_id))


# =============================================================================
# Video Analysis Utilities
# =============================================================================

def get_video_dimensions(video_path: str) -> Optional[Dict[str, Any]]:
    """
    Extract video dimensions and format information.
    
    Args:
        video_path: Path to the video file
        
    Returns:
        Dict with keys: width, height, format, orientation
        format: "WidthxHeight" (e.g., "1920x1080")
        orientation: "horizontal", "vertical", or "square"
    """
    if not os.path.exists(video_path):
        return None
    
    try:
        import cv2
        
        # Open video capture
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return None
        
        # Get video dimensions
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        # Release the capture
        cap.release()
        
        if width == 0 or height == 0:
            return None
        
        # Determine orientation
        if width > height:
            orientation = "horizontal"
        elif height > width:
            orientation = "vertical"
        else:
            orientation = "square"
        
        # Create format string
        format_str = f"{width}x{height}"
        
        return {
            "width": width,
            "height": height,
            "format": format_str,
            "orientation": orientation
        }
        
    except ImportError:
        print_status("OpenCV not available for video dimension extraction", "WARNING")
        return None
    except Exception as e:
        print_status(f"Error extracting video dimensions from {video_path}: {e}", "WARNING")
        return None
