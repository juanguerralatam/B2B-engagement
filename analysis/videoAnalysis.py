#!/usr/bin/env python3
"""
Simplified Video to Text Converter
Main processing script with reduced complexity
"""

import os
import argparse

# Suppress CUDA warnings before importing other modules
try:
    import cuda_suppress  # This will suppress CUDA warnings
except ImportError:
    pass

from functions import (
    install_package, read_video_csv, download_video, process_single_video,
    detect_optimal_whisper_settings
)

def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Video to Text Converter",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python videoAnalysis.py                    # Basic transcription
  python videoAnalysis.py -n                # Include narration evaluation
  python videoAnalysis.py --download-only   # Only download videos
  python videoAnalysis.py --transcribe-only # Only transcribe audio
        """
    )
    
    parser.add_argument('-n', '--narration', action='store_true',
                       help='Evaluate narration quality using DeepSeek API')
    parser.add_argument('--api-key', type=str,
                       help='DeepSeek API key')
    parser.add_argument('--video-path', type=str,
                       help='Path to video file')
    parser.add_argument('--model', type=str, choices=['tiny', 'base', 'small', 'medium', 'large'],
                       default='base', help='Whisper model size')
    parser.add_argument('--force', action='store_true',
                       help='Force regeneration of files')
    parser.add_argument('--csv-file', type=str, default='downloadVideo.csv',
                       help='CSV file with video IDs and URLs')
    parser.add_argument('--video-id', type=str,
                       help='Process specific video ID')
    
    # Stage control
    parser.add_argument('--download-only', action='store_true',
                       help='Only download videos')
    parser.add_argument('--extract-audio-only', action='store_true',
                       help='Only extract audio')
    parser.add_argument('--transcribe-only', action='store_true',
                       help='Only transcribe audio')
    parser.add_argument('--skip-download', action='store_true',
                       help='Skip download phase')
    parser.add_argument('--skip-audio', action='store_true',
                       help='Skip audio extraction')
    
    return parser.parse_args()

def main():
    """Main function."""
    args = parse_arguments()
    
    print("🎥 Video to Text Converter")
    if args.narration:
        print("📊 Narration evaluation enabled")
    
    # Detect optimal settings
    whisper_settings = detect_optimal_whisper_settings()
    
    # Override model if user didn't specify and we have a recommendation
    if args.model == 'base' and 'recommended_model' in whisper_settings:
        args.model = whisper_settings['recommended_model']
        print(f"🎯 Using optimized model: {args.model}")
    
    # Install packages
    install_package("moviepy")
    install_package("openai-whisper")
    if args.narration:
        install_package("requests")
    
    # Read videos
    # Handle CSV file path - check multiple locations
    csv_file = args.csv_file
    if not os.path.exists(csv_file):
        # If not found, try in analysis/ subdirectory
        alt_path = os.path.join('analysis', csv_file)
        if os.path.exists(alt_path):
            csv_file = alt_path
        else:
            # Try in current script directory
            script_dir = os.path.dirname(os.path.abspath(__file__))
            script_path = os.path.join(script_dir, csv_file)
            if os.path.exists(script_path):
                csv_file = script_path
    
    videos = read_video_csv(csv_file)
    if not videos:
        print("❌ No videos found in CSV")
        return
    
    # Filter specific video
    if args.video_id:
        videos = [v for v in videos if v['videoId'] == args.video_id]
        if not videos:
            print(f"❌ Video ID '{args.video_id}' not found")
            return
    
    print(f"📊 Processing {len(videos)} videos")
    
    # Process videos
    successful = 0
    failed = 0
    
    for i, video in enumerate(videos, 1):
        video_id = video['videoId']
        video_url = video['video_url']
        
        print(f"\n[{i}/{len(videos)}] {video_id}")
        
        # Find or download video
        video_path = None
        if not args.transcribe_only:
            # Check for existing video
            for ext in ['.mp4', '.mkv', '.webm', '.avi', '.mov']:
                test_path = f"../video/{video_id}{ext}"
                if os.path.exists(test_path):
                    video_path = test_path
                    break
            
            # Download if needed
            if not video_path and not args.skip_download:
                if args.video_path:
                    video_path = args.video_path
                else:
                    print(f"📥 Downloading...")
                    video_path = download_video(video_url, video_id)
                    if not video_path:
                        print(f"❌ Download failed")
                        failed += 1
                        continue
        else:
            video_path = f"../video/{video_id}"  # Dummy for transcribe-only
        
        # Handle download-only
        if args.download_only:
            if video_path and video_path.startswith("../video/"):
                print(f"✅ Downloaded")
                successful += 1
            else:
                print(f"❌ Download failed")
                failed += 1
            continue
        
        # Process video
        try:
            if process_single_video(video_id, video_path, args):
                successful += 1
                print(f"✅ Success")
            else:
                failed += 1
                print(f"❌ Failed")
        except Exception as e:
            print(f"❌ Error: {e}")
            failed += 1
    
    # Summary
    print(f"\n📊 Summary: {successful} success, {failed} failed")

if __name__ == "__main__":
    main()
