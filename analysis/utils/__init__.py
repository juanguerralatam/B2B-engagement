"""
Utils module for B2B Engagement Analysis.
Consolidated utilities in core.py and yolo.py for better organization.
"""

# Import all core utilities
from .core import (
    # File operations
    get_project_root,
    ensure_directory_exists,
    get_output_directory,
    find_file_in_locations,
    get_file_extension_variants,
    validate_file_path,
    sanitize_filename,
    
    # Data processing
    detect_file_encoding,
    load_text_file_safe,
    save_json_file,
    load_json_file,
    detect_csv_format,
    read_csv_safe,
    write_csv_safe,
    read_channel_ids_from_csv,
    
    # Hardware detection
    check_gpu_availability,
    get_system_memory_gb,
    detect_optimal_device_settings,
    check_package_installed,
    install_package_safe,
    check_system_dependencies,
    
    # Configuration management
    load_config_file,
    save_config_file,
    get_shared_config,
    clear_config_cache,
    detect_optimal_whisper_settings,
    
    # Status and progress
    print_status,
    print_progress,
    get_timestamp,
    validate_video_id,
    
    # Video utilities
    get_video_dimensions
)

# YOLO functions - with fallback for import errors
try:
    from .yolo import (
        ProductionFaceAnalyzer,
        YOLOFaceDetector,
        AdvancedFaceTracker,
        ProductionGenderClassifier,
        detect_faces_yolo_production,
        classify_gender_production,
        detect_smile_production,
        analyze_motion_production,
        analyze_color_production,
        analyze_video_comprehensive_production,
        detect_gender_in_video
    )
    YOLO_AVAILABLE = True
except ImportError as e:
    print(f"YOLO module not available: {e}")
    YOLO_AVAILABLE = False

# Export all for backward compatibility
__all__ = [
    # File operations
    'get_project_root',
    'ensure_directory_exists', 
    'get_output_directory',
    'find_file_in_locations',
    'get_file_extension_variants',
    'validate_file_path',
    'sanitize_filename',
    
    # Data processing
    'detect_file_encoding',
    'load_text_file_safe',
    'save_json_file',
    'load_json_file',
    'detect_csv_format',
    'read_csv_safe',
    'write_csv_safe',
    'read_channel_ids_from_csv',
    
    # Hardware detection
    'check_gpu_availability',
    'get_system_memory_gb',
    'detect_optimal_device_settings',
    'check_package_installed',
    'install_package_safe',
    'check_system_dependencies',
    
    # Configuration
    'load_config_file',
    'save_config_file',
    'get_shared_config',
    'clear_config_cache',
    'detect_optimal_whisper_settings',
    
    # Status and progress
    'print_status',
    'print_progress',
    'get_timestamp',
    'validate_video_id',
    
    # Video utilities
    'get_video_dimensions',
    
    # YOLO availability flag
    'YOLO_AVAILABLE',
]

# Conditionally add YOLO exports
if YOLO_AVAILABLE:
    __all__.extend([
        'ProductionFaceAnalyzer',
        'YOLOFaceDetector',
        'AdvancedFaceTracker',
        'ProductionGenderClassifier',
        'detect_faces_yolo_production',
        'classify_gender_production',
        'detect_smile_production',
        'analyze_motion_production',
        'analyze_color_production',
        'analyze_video_comprehensive_production',
        'detect_gender_in_video'
    ])
