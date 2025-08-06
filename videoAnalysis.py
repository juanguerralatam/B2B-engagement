#!/usr/bin/env python3
"""
Simplified Video to Text Converter
"""

import os
import argparse
import sys

# Add analysis directory to Python path
analysis_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'analysis')
sys.path.insert(0, analysis_dir)

from core_functions import (
    read_video_csv, download_video, process_single_video,
    detect_optimal_whisper_settings, save_scene_analysis, read_scene_analysis,
    save_gender_analysis, read_gender_analysis
)
from image_functions import detect_scenes, detect_gender_in_video

def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Video to Text Converter")
    
    parser.add_argument('-n', '--narration', action='store_true',
                       help='Evaluate narration quality')
    parser.add_argument('--api-key', type=str, help='API key')
    parser.add_argument('--video-path', type=str, help='Path to video file')
    parser.add_argument('--model', type=str, choices=['tiny', 'base', 'small', 'medium', 'large'],
                       default='base', help='Whisper model size')
    parser.add_argument('--force-cpu', action='store_true', help='Force CPU usage')
    parser.add_argument('--force', action='store_true', help='Force regeneration')
    parser.add_argument('--csv-file', type=str, default='input/downloadVideo.csv',
                       help='CSV file with video IDs')
    parser.add_argument('--video-id', type=str, help='Process specific video ID')
    
    # Processing modes
    parser.add_argument('--download-only', action='store_true')
    parser.add_argument('--transcribe-only', action='store_true')
    parser.add_argument('--extract-audio-only', action='store_true')
    parser.add_argument('--skip-audio', action='store_true')
    parser.add_argument('--detect-scenes', action='store_true')
    parser.add_argument('--detect-gender', action='store_true')
    parser.add_argument('--scene-threshold', type=float, default=30.0)
    parser.add_argument('--gender-frames', type=int, default=30)
    
    return parser.parse_args()

def check_dependencies(args):
    """Check required dependencies."""
    try:
        import moviepy
        import whisper
        if args.narration:
            import requests
        if args.detect_scenes:
            import scenedetect
        if args.detect_gender:
            try:
                from deepface import DeepFace
            except ImportError:
                import cv2
        return True
    except ImportError as e:
        print(f"Missing package: {e.name}")
        print("Run: python test/env_check.py")
        return False

def find_csv_file(csv_file):
    """Find CSV file in various locations."""
    if os.path.exists(csv_file):
        return csv_file
    
    script_dir = os.path.dirname(os.path.abspath(__file__))
    locations = [
        os.path.join(script_dir, 'input', os.path.basename(csv_file)),
        os.path.join('analysis', csv_file),
        os.path.join(script_dir, csv_file)
    ]
    
    for path in locations:
        if os.path.exists(path):
            return path
    
    return csv_file

def find_video_file(video_id):
    """Find existing video file."""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    video_dir = os.path.join(script_dir, "output", "video")
    
    for ext in ['.mp4', '.mkv', '.webm', '.avi', '.mov']:
        path = os.path.join(video_dir, f"{video_id}{ext}")
        if os.path.exists(path):
            return path
    return None

def process_scene_detection(video_id, video_path, existing_scenes, args):
    """Process scene detection for a video."""
    if video_id in existing_scenes and not args.force:
        return True
    
    if not video_path or not os.path.exists(video_path):
        return False
    
    scene_results = detect_scenes(video_path, video_id, args.scene_threshold)
    if scene_results:
        return save_scene_analysis(video_id, scene_results)
    return False

def process_gender_detection(video_id, video_path, existing_gender, args):
    """Process gender detection for a video."""
    if video_id in existing_gender and not args.force:
        return True
    
    if not video_path or not os.path.exists(video_path):
        return False
    
    gender_results = detect_gender_in_video(video_path, video_id, args.gender_frames)
    if gender_results:
        return save_gender_analysis(video_id, gender_results)
    return False

def main():
    """Main function."""
    args = parse_arguments()
    
    # Check dependencies
    if not check_dependencies(args):
        return
    
    # Setup
    whisper_settings = detect_optimal_whisper_settings(args.force_cpu)
    if args.model == 'base' and 'recommended_model' in whisper_settings:
        args.model = whisper_settings['recommended_model']
    
    # Read videos
    csv_file = find_csv_file(args.csv_file)
    videos = read_video_csv(csv_file)
    if not videos:
        print("No videos found")
        return
    
    # Filter specific video
    if args.video_id:
        videos = [v for v in videos if v['videoId'] == args.video_id]
        if not videos:
            print(f"Video ID '{args.video_id}' not found")
            return
    
    # Load existing analysis
    existing_scenes = read_scene_analysis() if args.detect_scenes else {}
    existing_gender = read_gender_analysis() if args.detect_gender else {}
    
    # Process videos
    successful = 0
    failed = 0
    
    for video in videos:
        video_id = video['videoId']
        video_url = video['video_url']
        
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
        
        if args.detect_gender:
            if process_gender_detection(video_id, video_path, existing_gender, args):
                successful += 1
            else:
                failed += 1
            continue
        
        # Normal processing
        try:
            if process_single_video(video_id, video_path, args):
                successful += 1
            else:
                failed += 1
        except Exception as e:
            print(f"Error processing {video_id}: {e}")
            failed += 1
    
    print(f"Completed: {successful} successful, {failed} failed")

if __name__ == "__main__":
    main()
