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

from analysis.core_functions import (
    read_video_csv, download_video, process_single_video,
    detect_optimal_whisper_settings, save_scene_analysis, read_scene_analysis
)
from analysis.image_functions import detect_scenes, detect_gender_in_video, analyze_video_comprehensive
from analysis.utils.core import (
    find_file_in_locations, get_file_extension_variants, validate_video_id,
    print_status, print_progress, get_output_directory, load_config_file
)

def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Video to Text Converter")
    
    parser.add_argument('--force', action='store_true',
                       help='Force regeneration')
    
    # Video selection
    parser.add_argument('--video-id', type=str,
                       help='Process specific video by ID')
    parser.add_argument('--download-only', action='store_true')
    parser.add_argument('--transcribe-only', action='store_true')
    parser.add_argument('--extract-audio-only', action='store_true')
    parser.add_argument('--narration-only', action='store_true',
                       help='Evaluate narration quality only')
    parser.add_argument('--detect-scenes', action='store_true')
    # Processing modes
    parser.add_argument('--analyze-basic-features', action='store_true',
                       help='Extract basic features (followers, age, length, scenes)')    
    parser.add_argument('--analyze-image-features', action='store_true',
                       help='Analyze comprehensive image features (faces, motion, color, etc.)')
    parser.add_argument('--analyze-audio-features', action='store_true',
                       help='Analyze comprehensive audio features (arousal, valence, pitch)')
    parser.add_argument('--analyze-text-features', action='store_true',
                       help='Analyze comprehensive text features (sentiment, technicality, content)')
    
    # Parallel processing options
    parser.add_argument('--parallel', action='store_true',
                       help='Enable parallel processing of videos')
    parser.add_argument('--workers', type=int, default=None,
                       help='Number of parallel workers (default: auto-detect)')
    parser.add_argument('--chunk-size', type=int, default=10,
                       help='Number of videos per chunk (default: 10)')
    parser.add_argument('--max-videos', type=int, default=None,
                       help='Maximum number of videos to process (default: all)')

    
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
    analysis_dir = os.path.join(script_dir, 'analysis')
    
    locations = [
        os.path.join(script_dir, input_dir, os.path.basename(csv_file)),
        os.path.join(analysis_dir, csv_file),
        os.path.join(script_dir, csv_file)
    ]
    
    return find_file_in_locations(os.path.basename(csv_file), locations) or csv_file

def find_video_file(video_id):
    """Find existing video file with different naming patterns."""
    if not validate_video_id(video_id):
        return None
    
    # Load config to get video directory
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

def process_image_features(video_id, video_path, args):
    """Process comprehensive image feature analysis for a video."""
    from datetime import datetime
    from analysis.utils.core import check_record_exists, update_csv_record, format_analysis_data
    
    # Load config to get output file path
    config = load_config_file()
    output_file = config.get("output_files", {}).get("image_features", "output/image_features_analysis.csv")
    
    # Check if already processed
    if not args.force and check_record_exists(output_file, video_id):
        print(f"Image features already analyzed for {video_id}, use --force to regenerate")
        return True
    
    if not video_path or not os.path.exists(video_path):
        print(f"Video file not found: {video_path}")
        return False
    
    print(f"Analyzing image features for {video_id}...")
    
    # Run comprehensive analysis
    results = analyze_video_comprehensive(video_path, video_id, sample_frames=25)
    
    if not results:
        print(f"Failed to analyze image features for {video_id}")
        return False
    
    # Prepare data for CSV using utility functions
    row_data = {
        'videoId': video_id,
        'frames_analyzed': results.get('frames_analyzed'),
        'analysis_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    }
    
    # Add formatted analysis results (excluding non-numeric fields)
    numeric_results = {k: v for k, v in results.items() 
                      if k not in ['frames_analyzed', 'videoId']}
    formatted_results = format_analysis_data(numeric_results, precision=3)
    row_data.update(formatted_results)
    
    # Save using utility function
    if update_csv_record(output_file, row_data):
        print(f"✓ Image features saved for {video_id}")
        print(f"  Human: {row_data.get('humanPresence')}, Faces: {row_data.get('faceSum')}, "
              f"Gender: {row_data.get('Gender')}, Brightness: {row_data.get('Brightness')}")
        return True
    else:
        print(f"Error saving image features for {video_id}")
        return False

def process_audio_features(video_id, video_path, args):
    """Process comprehensive audio feature analysis for a video."""
    from datetime import datetime
    from analysis.audio_functions import extract_audio, analyze_audio_comprehensive
    from analysis.utils.core import check_record_exists, update_csv_record, format_analysis_data
    
    # Load config to get output file path
    config = load_config_file()
    output_file = config.get("output_files", {}).get("audio_features", "output/audio_features_analysis.csv")
    
    # Check if already processed
    if not args.force and check_record_exists(output_file, video_id):
        print(f"Audio features already analyzed for {video_id}, use --force to regenerate")
        return True
    
    if not video_path or not os.path.exists(video_path):
        print(f"Video file not found: {video_path}")
        return False
    
    print(f"Analyzing audio features for {video_id}...")
    
    # Extract audio first
    audio_dir = config.get("paths", {}).get("audio_dir", "output/audio/")
    audio_path = os.path.join(audio_dir, f"{video_id}.wav")
    if not os.path.exists(audio_path):
        print(f"Extracting audio from {video_id}...")
        if not extract_audio(video_path, audio_path):
            print(f"Failed to extract audio from {video_path}")
            return False
    
    # Run comprehensive audio analysis
    results = analyze_audio_comprehensive(audio_path, video_id)
    
    if not results:
        print(f"Failed to analyze audio features for {video_id}")
        return False
    
    # Prepare data for CSV using utility functions
    row_data = {
        'videoId': video_id,
        'analysis_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    }
    
    # Add formatted analysis results
    formatted_results = format_analysis_data(results, precision=3)
    row_data.update(formatted_results)
    
    # Save using utility function
    if update_csv_record(output_file, row_data):
        print(f"✓ Audio features saved for {video_id}")
        print(f"  Arousal: {row_data.get('Arousal')}, Valence: {row_data.get('Valence')}, "
              f"Pitch: {row_data.get('Pitch')}")
        return True
    else:
        print(f"Error saving audio features for {video_id}")
        return False

def process_text_features(video_id, video_path, args):
    """Process comprehensive text feature analysis for a video."""
    from datetime import datetime
    from analysis.text_functions import analyze_text_comprehensive
    from analysis.audio_functions import extract_audio, transcribe_audio
    from analysis.core_functions import detect_optimal_whisper_settings
    from analysis.utils.core import check_record_exists, update_csv_record, format_analysis_data
    
    # Load config to get output file paths
    config = load_config_file()
    output_file = config.get("output_files", {}).get("text_features", "output/text_features_analysis.csv")
    
    # Check if already processed
    if not args.force and check_record_exists(output_file, video_id):
        print(f"Text features already analyzed for {video_id}, use --force to regenerate")
        return True
    
    print(f"Analyzing text features for {video_id}...")
    
    # Load video metadata
    title, description = load_video_metadata(video_id, config)
    if not title and not description:
        print(f"Could not extract text data for video {video_id}")
        return False
    
    print(f"Loaded text data - Title: {len(title)} chars, Description: {len(description)} chars")
    
    # Get or generate transcript
    transcript_path = get_or_generate_transcript(video_id, video_path, args, config)
    
    # Run comprehensive text analysis
    results = analyze_text_comprehensive(video_id, title, description, transcript_path)
    
    if not results:
        print(f"Failed to analyze text features for {video_id}")
        return False
    
    # Prepare data for CSV using utility functions
    row_data = {
        'videoId': video_id,
        'analysis_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    }
    
    # Add formatted analysis results
    formatted_results = format_analysis_data(results, precision=3)
    row_data.update(formatted_results)
    
    # Save using utility function
    if update_csv_record(output_file, row_data):
        print(f"✓ Text features saved for {video_id}")
        print(f"  Title: Sentiment={row_data.get('titleSentiment')}, Tech={row_data.get('titleTechnicality')}")
        print(f"  Description: Sentiment={row_data.get('descriptionSentiment')}, Tech={row_data.get('descriptionTechnicality')}")
        print(f"  Content: Hashtags={row_data.get('hashtagsDescription')}, URLs={row_data.get('URLDescription')}")
        if row_data.get('scriptSentiment') is not None:
            print(f"  Script: Sentiment={row_data.get('scriptSentiment')}, Tech={row_data.get('scriptTechnicality')}")
        return True
    else:
        print(f"Error saving text features for {video_id}")
        return False


def load_video_metadata(video_id, config):
    """Load video title and description from metadata file."""
    import pandas as pd
    
    description_file = config.get("output_files", {}).get("video_statistics", "output/videos_statistics.csv")
    if not os.path.exists(description_file):
        print(f"Video metadata file not found: {description_file}")
        return "", ""
    
    try:
        # Read CSV with proper multiline handling
        description_df = pd.read_csv(description_file, encoding='utf-8', quotechar='"', escapechar='\\')
        
        if description_df.empty or 'videoId' not in description_df.columns:
            print(f"Invalid or empty description file")
            return "", ""
            
        video_row = description_df[description_df['videoId'] == video_id]
        
        if video_row.empty:
            print(f"Video {video_id} not found in description file")
            return "", ""
            
        # Safely extract title and description
        title = str(video_row.iloc[0]['title']) if 'title' in video_row.columns else ""
        description = str(video_row.iloc[0]['description']) if 'description' in video_row.columns else ""
        
        # Handle NaN values
        if title == 'nan':
            title = ""
        if description == 'nan':
            description = ""
            
        return title, description
        
    except Exception as e:
        print(f"Error reading description file: {e}")
        # Try alternative reading method for problematic CSV files
        return load_video_metadata_fallback(video_id, description_file)


def load_video_metadata_fallback(video_id, description_file):
    """Fallback method for loading video metadata from problematic CSV files."""
    try:
        with open(description_file, 'r', encoding='utf-8') as f:
            lines = f.readlines()
            
        if len(lines) < 2:
            return "", ""
            
        # Parse header
        header = lines[0].strip().split(',')
        
        # Find video data
        for line in lines[1:]:
            if video_id in line:
                parts = line.split(',', len(header) - 1)
                if len(parts) >= len(header):
                    try:
                        video_id_col = header.index('videoId')
                        title_col = header.index('title')
                        desc_col = header.index('description')
                        
                        if parts[video_id_col].strip() == video_id:
                            title = parts[title_col].strip().strip('"')
                            description = parts[desc_col].strip().strip('"')
                            return title, description
                    except (ValueError, IndexError):
                        continue
                        
        return "", ""
        
    except Exception as e:
        print(f"Alternative reading method also failed: {e}")
        return "", ""


def get_or_generate_transcript(video_id, video_path, args, config):
    """Get existing transcript or generate new one from video."""
    from analysis.audio_functions import extract_audio, transcribe_audio
    from analysis.core_functions import detect_optimal_whisper_settings
    
    text_dir = config.get("paths", {}).get("text_dir", "output/text/")
    transcript_path = os.path.join(text_dir, f"{video_id}.txt")
    
    # Generate transcript if it doesn't exist
    if not os.path.exists(transcript_path) or args.force:
        print(f"Transcript not found, generating from video...")
        
        if not video_path or not os.path.exists(video_path):
            print(f"Video file required for transcription but not found: {video_path}")
            return transcript_path  # Return path anyway, analysis can work without transcript
        
        try:
            # Extract audio first
            audio_dir = config.get("paths", {}).get("audio_dir", "output/audio/")
            audio_path = os.path.join(audio_dir, f"{video_id}.wav")
            
            if not os.path.exists(audio_path):
                print(f"Extracting audio from {video_id}...")
                if not extract_audio(video_path, audio_path):
                    print(f"Failed to extract audio from {video_path}")
                    return transcript_path
            
            if os.path.exists(audio_path):
                # Transcribe audio
                print(f"Transcribing audio for {video_id}...")
                settings = detect_optimal_whisper_settings()
                model = config.get("defaults", {}).get("model", "base")
                transcript = transcribe_audio(audio_path, model, settings["device"], settings["fp16"])
                
                if transcript:
                    # Save transcript
                    os.makedirs(text_dir, exist_ok=True)
                    with open(transcript_path, 'w', encoding='utf-8') as f:
                        f.write(transcript)
                    print(f"✓ Transcript saved: {len(transcript)} characters")
                else:
                    print("Failed to generate transcript")
                    
        except Exception as e:
            print(f"Error generating transcript: {e}")
    
    return transcript_path

def main():
    """Main function."""
    args = parse_arguments()
    
    # Load config file to get settings
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
            # Override with config values only if command line args are not set
            self.model = config_values['model']
            self.csv_file = config_values['csv_file']
            self.video_path = config_values['video_path']
            # Don't override video_id if set via command line
            if not hasattr(args, 'video_id') or not args.video_id:
                self.video_id = config_values['video_id']
            self.force = config_values['force'] or getattr(args, 'force', False)
            # Don't override max_videos if we have a specific video_id
            if not (hasattr(args, 'video_id') and args.video_id):
                self.max_videos = config_values['max_videos']
            else:
                self.max_videos = None  # No limit when processing specific video
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
    
    # Filter specific video first
    if hasattr(args, 'video_id') and args.video_id:
        videos = [v for v in videos if v['videoId'] == args.video_id]
        if not videos:
            print_status(f"Video ID '{args.video_id}' not found", "ERROR")
            return
    
    # Handle basic features analysis separately (processes all videos at once)
    if args.analyze_basic_features:
        from analysis.basic_functions import extract_basic_features_from_data, load_config
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
    
    # Apply max_videos limit if specified and no specific video selected
    if hasattr(args, 'max_videos') and args.max_videos and not (hasattr(args, 'video_id') and args.video_id):
        if args.max_videos < len(videos):
            print_status(f"Limiting processing to first {args.max_videos} videos (out of {len(videos)})", "INFO")
            videos = videos[:args.max_videos]
    
    # Use parallel processing if requested
    if args.parallel and not args.analyze_basic_features:
        return process_videos_parallel(videos, args)
    
    # Process videos sequentially (original behavior)
    return process_videos_sequential(videos, args)


def process_videos_parallel(videos, args):
    """Process videos using simple parallel processing."""
    from analysis.utils.parallel_processor import process_videos_parallel_simple
    
    # Determine number of workers
    if args.workers:
        num_workers = args.workers
    else:
        import multiprocessing
        # Auto-detect optimal number of workers based on task type
        cpu_count = multiprocessing.cpu_count()
        
        if args.analyze_image_features:
            # GPU-bound task, use fewer workers
            num_workers = min(8, cpu_count // 4)
        elif args.analyze_audio_features or args.analyze_text_features:
            # CPU-bound tasks, use more workers
            num_workers = min(16, cpu_count // 2)
        else:
            # Default
            num_workers = min(8, cpu_count // 4)
    
    print_status(f"Starting parallel processing with {num_workers} workers", "INFO")
    
    # Determine processing mode
    if args.analyze_image_features:
        task_type = 'image'
    elif args.analyze_audio_features:
        task_type = 'audio'
    elif args.analyze_text_features:
        task_type = 'text'
    elif args.detect_scenes:
        task_type = 'scenes'
    elif args.narration_only:
        task_type = 'narration'
    else:
        task_type = 'normal'
    
    # Process videos
    successful, failed = process_videos_parallel_simple(
        videos, task_type, max_workers=num_workers, force=args.force
    )
    
    print_status(f"Parallel processing completed: {successful} successful, {failed} failed", "SUCCESS")
    return successful > 0


def process_videos_sequential(videos, args):
    """Process videos sequentially (original behavior)."""
    # Load existing analysis
    existing_scenes = read_scene_analysis() if args.detect_scenes else {}
    
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
            if process_image_features(video_id, video_path, args):
                successful += 1
            else:
                failed += 1
            continue
        
        if args.analyze_audio_features:
            if process_audio_features(video_id, video_path, args):
                successful += 1
            else:
                failed += 1
            continue
        
        if args.analyze_text_features:
            if process_text_features(video_id, video_path, args):
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
    return successful > 0
    

if __name__ == "__main__":
    main()
