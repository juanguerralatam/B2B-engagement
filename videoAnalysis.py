#!/usr/bin/env python3
"""
Simplified Video to Text Converter
"""

import os
import argparse
import sys
import glob
import pandas as pd
from datetime import datetime

# Add analysis directory to Python path
analysis_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'analysis')
sys.path.insert(0, analysis_dir)

from core_functions import (
    read_video_csv, download_video, process_single_video,
    detect_optimal_whisper_settings, save_scene_analysis, read_scene_analysis
)
from image_functions import detect_scenes, analyze_video_comprehensive
from basic_functions import check_already_processed, save_features_to_csv
from utils import (
    load_config_file, validate_video_id, print_status, print_progress
)

def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Video to Text Converter")
    
    parser.add_argument('--force', action='store_true',
                       help='Force regeneration')
    
    # Processing modes
    parser.add_argument('--download-only', action='store_true')
    parser.add_argument('--transcribe-only', action='store_true')
    parser.add_argument('--extract-audio-only', action='store_true')
    parser.add_argument('--narration-only', action='store_true',
                       help='Evaluate narration quality only')
    parser.add_argument('--detect-scenes', action='store_true')
    parser.add_argument('--analyze-image-features', action='store_true',
                       help='Analyze comprehensive image features (faces, motion, color, etc.)')
    parser.add_argument('--analyze-audio-features', action='store_true',
                       help='Analyze comprehensive audio features (arousal, valence, pitch)')
    parser.add_argument('--analyze-text-features', action='store_true',
                       help='Analyze comprehensive text features (sentiment, technicality, content)')
    parser.add_argument('--analyze-basic-features', action='store_true',
                       help='Extract basic features (followers, age, length, scenes)')
    
    return parser.parse_args()

def check_dependencies(args):
    """Check required dependencies."""
    try:
        import moviepy
        import whisper
        if args.narration_only:
            import requests
        if args.detect_scenes:
            import scenedetect
        return True
    except ImportError as e:
        print_status(f"Missing package: {e.name}", "ERROR")
        print_status("Run: python test/env_check.py", "INFO")
        return False

def find_csv_file(csv_file):
    """Find CSV file in various locations."""
    if os.path.exists(csv_file):
        return csv_file
    
    config = load_config_file()
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Get potential locations from config
    input_dir = config.get("paths", {}).get("input_dir", "input/")
    
    locations = [
        os.path.join(script_dir, input_dir, os.path.basename(csv_file)),
        os.path.join(script_dir, 'analysis', csv_file),
        os.path.join(script_dir, csv_file)
    ]
    
    for location in locations:
        if os.path.exists(location):
            return location
    return csv_file

def find_video_file(video_id):
    """Find existing video file with different naming patterns."""
    if not validate_video_id(video_id):
        return None
    
    # Load config to get video directory
    from utils import load_config_file
    config = load_config_file()
    video_dir = os.path.expanduser(config.get('paths', {}).get('video_dir', '~/Downloads/youtube/'))
    
    # Try different file naming patterns
    extensions = ['.mp4', '.mkv', '.webm', '.avi', '.mov']
    
    # First try simple pattern: videoId.ext
    for ext in extensions:
        simple_path = os.path.join(video_dir, f"{video_id}{ext}")
        if os.path.exists(simple_path):
            return simple_path
    
    # Then try pattern with title: videoId - title.ext
    import glob
    for ext in extensions:
        pattern = os.path.join(video_dir, f"{video_id} - *{ext}")
        matching_files = glob.glob(pattern)
        if matching_files:
            return matching_files[0]  # Return first match
    
    # Finally, try any file that starts with videoId
    search_pattern = os.path.join(video_dir, f"{video_id}*")
    matching_files = glob.glob(search_pattern)
    video_files = [f for f in matching_files if any(f.endswith(ext) for ext in extensions)]
    if video_files:
        return video_files[0]
    
    return None

def process_scene_detection(video_id, video_path, existing_scenes, args):
    """Process scene detection for a video."""
    if video_id in existing_scenes and not args.force:
        return True
    
    if not video_path or not os.path.exists(video_path):
        return False
    
    scene_results = detect_scenes(video_path, video_id, 30.0)
    if scene_results:
        return save_scene_analysis(video_id, scene_results)
    return False

def process_features_unified(video_id, video_path, feature_type, args):
    """Unified feature processing function to eliminate code duplication."""
    config = load_config_file()
    
    # Map feature types to their configurations
    feature_configs = {
        'image': {
            'output_file': config.get("output_files", {}).get("image_features", "output/image_features_analysis.csv"),
            'processor': 'process_image_features_core',
            'name': 'image features'
        },
        'audio': {
            'output_file': config.get("output_files", {}).get("audio_features", "output/audio_features_analysis.csv"),
            'processor': 'process_audio_features_core',
            'name': 'audio features'
        },
        'text': {
            'output_file': config.get("output_files", {}).get("text_features", "output/text_features_analysis.csv"),
            'processor': 'process_text_features_core',
            'name': 'text features'
        }
    }
    
    if feature_type not in feature_configs:
        return False
    
    feat_config = feature_configs[feature_type]
    
    # Check if already processed
    if check_already_processed(video_id, feat_config['output_file'], args.force):
        print(f"{feat_config['name'].title()} already analyzed for {video_id}, use --force to regenerate")
        return True
    
    if feature_type != 'text' and (not video_path or not os.path.exists(video_path)):
        print(f"Video file not found: {video_path}")
        return False
    
    print(f"Analyzing {feat_config['name']} for {video_id}...")
    
    # Process features using the specific processor
    processor_func = globals()[feat_config['processor']]
    features_data = processor_func(video_id, video_path, config)
    
    if features_data:
        return save_features_to_csv(video_id, features_data, feat_config['output_file'])
    else:
        print(f"Failed to analyze {feat_config['name']} for {video_id}")
        return False

def main():
    """Main function."""
    args = parse_arguments()
    
    # Load config file to get settings
    from utils import load_config_file
    config = load_config_file()
    
    # Get configuration values
    model = config.get("defaults", {}).get("model", "base")
    csv_file = config.get("input_files", {}).get("primary_csv", "output/videos_statistics.csv")
    video_path = config.get("paths", {}).get("video_path", None)
    video_id = config.get("defaults", {}).get("video_id", None)
    force = config.get("defaults", {}).get("force", False) or args.force
    max_videos = config.get("defaults", {}).get("max_videos", None)
    
    # Create a config-enhanced args object
    class ConfigArgs:
        def __init__(self, args, config_values):
            # Copy all original args
            for attr in dir(args):
                if not attr.startswith('_'):
                    setattr(self, attr, getattr(args, attr))
            # Override with config values
            self.model = config_values['model']
            self.csv_file = config_values['csv_file']
            self.video_path = config_values['video_path']
            self.video_id = config_values['video_id']
            self.force = config_values['force']
            self.max_videos = config_values['max_videos']
            # Ensure narration attribute exists
            if not hasattr(self, 'narration'):
                self.narration = False
    
    args = ConfigArgs(args, {
        'model': model,
        'csv_file': csv_file,
        'video_path': video_path,
        'video_id': video_id,
        'force': force,
        'max_videos': max_videos
    })
    
    # Check dependencies - skip for basic features
    if not args.analyze_basic_features and not check_dependencies(args):
        return
    
    # Setup
    whisper_settings = detect_optimal_whisper_settings()
    if args.model == 'base' and 'recommended_model' in whisper_settings:
        args.model = whisper_settings['recommended_model']
    
    # Read videos
    csv_file = find_csv_file(args.csv_file)
    videos = read_video_csv(csv_file)
    if not videos:
        print_status("No videos found", "ERROR")
        return
    
    # Filter specific video
    if args.video_id:
        videos = [v for v in videos if v['videoId'] == args.video_id]
        if not videos:
            print_status(f"Video ID '{args.video_id}' not found", "ERROR")
            return
    
    # Handle basic features analysis separately (processes all videos at once)
    if args.analyze_basic_features:
        from basic_functions import extract_basic_features_from_data, load_config
        config = load_config()
        
        # Add max_videos to config if specified
        if hasattr(args, 'max_videos') and args.max_videos:
            config['max_videos'] = args.max_videos
        
        if extract_basic_features_from_data(config):
            print_status("Basic features extraction completed", "SUCCESS")
            return
        else:
            print_status("Basic features extraction failed", "ERROR")
            return
    
    # Load existing analysis
    existing_scenes = read_scene_analysis() if args.detect_scenes else {}
    
    # Apply max_videos limit if specified
    if hasattr(args, 'max_videos') and args.max_videos and args.max_videos < len(videos):
        print_status(f"Limiting processing to first {args.max_videos} videos (out of {len(videos)})", "INFO")
        videos = videos[:args.max_videos]
    
    # Process videos
    successful = 0
    failed = 0
    total_videos = len(videos)
    
    for i, video in enumerate(videos, 1):
        video_id = video['videoId']
        video_url = video['video_url']
        
        print_progress(i, total_videos, f"Processing {video_id}")
        
        # Find or download video
        video_path = None
        if not args.transcribe_only:
            video_path = find_video_file(video_id)
            
            if not video_path:
                if args.video_path:
                    video_path = args.video_path
                else:
                    video_path = download_video(video_url, video_id)
                    if not video_path:
                        failed += 1
                        continue
        
        # Handle different modes
        if args.download_only:
            if video_path:
                successful += 1
            else:
                failed += 1
            continue
        
        if args.detect_scenes:
            if process_scene_detection(video_id, video_path, existing_scenes, args):
                successful += 1
            else:
                failed += 1
            continue
        
        if args.analyze_image_features:
            if process_features_unified(video_id, video_path, 'image', args):
                successful += 1
            else:
                failed += 1
            continue
        
        if args.analyze_audio_features:
            if process_features_unified(video_id, video_path, 'audio', args):
                successful += 1
            else:
                failed += 1
            continue
        
        if args.analyze_text_features:
            if process_features_unified(video_id, video_path, 'text', args):
                successful += 1
            else:
                failed += 1
            continue
        
        if args.narration_only:
            # Process narration quality evaluation
            try:
                if process_single_video(video_id, video_path, args):
                    successful += 1
                else:
                    failed += 1
            except Exception as e:
                print_status(f"Error processing narration for {video_id}: {e}", "ERROR")
                failed += 1
            continue
        
        # Normal processing
        try:
            if process_single_video(video_id, video_path, args):
                successful += 1
            else:
                failed += 1
        except Exception as e:
            print_status(f"Error processing {video_id}: {e}", "ERROR")
            failed += 1
    
    print_status(f"Completed: {successful} successful, {failed} failed", "SUCCESS")

if __name__ == "__main__":
    main()
