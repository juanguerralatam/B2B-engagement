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
    detect_optimal_whisper_settings, save_scene_analysis, read_scene_analysis
)
from image_functions import detect_scenes, detect_gender_in_video, analyze_video_comprehensive
from utils import (
    find_file_in_locations, get_file_extension_variants, validate_video_id,
    print_status, print_progress, get_output_directory
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
    
    from utils import load_config_file
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

def process_image_features(video_id, video_path, args):
    """Process comprehensive image feature analysis for a video."""
    import pandas as pd
    from datetime import datetime
    from utils import load_config_file
    
    # Load config to get output file path
    config = load_config_file()
    output_file = config.get("output_files", {}).get("image_features", "output/image_features_analysis.csv")
    
    # Check if already processed
    if os.path.exists(output_file) and not args.force:
        try:
            existing_df = pd.read_csv(output_file)
            if video_id in existing_df['VideoId'].values:
                print(f"Image features already analyzed for {video_id}, use --force to regenerate")
                return True
        except:
            pass
    
    if not video_path or not os.path.exists(video_path):
        print(f"Video file not found: {video_path}")
        return False
    
    print(f"Analyzing image features for {video_id}...")
    
    # Run comprehensive analysis
    results = analyze_video_comprehensive(video_path, video_id, sample_frames=25)
    
    if results:
        # Prepare data for CSV
        row_data = {
            'VideoId': video_id,
            'humanPresence': round(results['humanPresence'], 3),
            'faceSum': round(results['faceSum'], 3) if results['faceSum'] is not None else None,
            'Gender': round(results['Gender'], 3) if results['Gender'] is not None else None,
            'Smile': round(results['Smile'], 3) if results['Smile'] is not None else None,
            'motionMagnitude': round(results['motionMagnitude'], 3) if results['motionMagnitude'] is not None else None,
            'motionDirection': round(results['motionDirection'], 3) if results['motionDirection'] is not None else None,
            'Saturation': round(results['Saturation'], 3),
            'Brightness': round(results['Brightness'], 3),
            'frames_analyzed': results['frames_analyzed'],
            'analysis_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }
        
        # Save to CSV
        try:
            if os.path.exists(output_file):
                # Append to existing file
                existing_df = pd.read_csv(output_file)
                # Remove existing entry for this video if it exists
                existing_df = existing_df[existing_df['VideoId'] != video_id]
                # Add new entry - use safer concatenation
                new_row_df = pd.DataFrame([row_data])
                if len(existing_df) > 0:
                    new_df = pd.concat([existing_df, new_row_df], ignore_index=True)
                else:
                    new_df = new_row_df
            else:
                # Create new file
                output_dir = config.get("paths", {}).get("output_dir", "output/")
                os.makedirs(output_dir, exist_ok=True)
                new_df = pd.DataFrame([row_data])
            
            new_df.to_csv(output_file, index=False)
            
            print(f"✓ Image features saved for {video_id}")
            print(f"  Human: {row_data['humanPresence']}, Faces: {row_data['faceSum']}, "
                  f"Gender: {row_data['Gender']}, Brightness: {row_data['Brightness']:.3f}")
            
            return True
            
        except Exception as e:
            print(f"Error saving image features for {video_id}: {e}")
            return False
    else:
        print(f"Failed to analyze image features for {video_id}")
        return False

def process_audio_features(video_id, video_path, args):
    """Process comprehensive audio feature analysis for a video."""
    import pandas as pd
    from datetime import datetime
    from audio_functions import extract_audio, analyze_audio_comprehensive
    from utils import load_config_file
    
    # Load config to get output file path
    config = load_config_file()
    output_file = config.get("output_files", {}).get("audio_features", "output/audio_features_analysis.csv")
    
    # Check if already processed
    if os.path.exists(output_file) and not args.force:
        try:
            existing_df = pd.read_csv(output_file)
            if video_id in existing_df['VideoId'].values:
                print(f"Audio features already analyzed for {video_id}, use --force to regenerate")
                return True
        except:
            pass
    
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
    
    if results:
        # Prepare data for CSV
        row_data = {
            'VideoId': video_id,
            'Arousal': round(float(results['Arousal']), 3) if results['Arousal'] is not None else None,
            'Valence': round(float(results['Valence']), 3) if results['Valence'] is not None else None,
            'Pitch': round(float(results['Pitch']), 3) if results['Pitch'] is not None else None,
            'analysis_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }
        
        # Save to CSV
        try:
            if os.path.exists(output_file):
                # Append to existing file
                existing_df = pd.read_csv(output_file)
                # Remove existing entry for this video if it exists
                existing_df = existing_df[existing_df['VideoId'] != video_id]
                # Add new entry - use safer concatenation
                new_row_df = pd.DataFrame([row_data])
                if len(existing_df) > 0:
                    new_df = pd.concat([existing_df, new_row_df], ignore_index=True)
                else:
                    new_df = new_row_df
            else:
                # Create new file
                output_dir = config.get("paths", {}).get("output_dir", "output/")
                os.makedirs(output_dir, exist_ok=True)
                new_df = pd.DataFrame([row_data])
            
            new_df.to_csv(output_file, index=False)
            
            print(f"✓ Audio features saved for {video_id}")
            print(f"  Arousal: {row_data['Arousal']}, Valence: {row_data['Valence']}, "
                  f"Pitch: {row_data['Pitch']}")
            
            return True
            
        except Exception as e:
            print(f"Error saving audio features for {video_id}: {e}")
            return False
    else:
        print(f"Failed to analyze audio features for {video_id}")
        return False

def process_text_features(video_id, video_path, args):
    """Process comprehensive text feature analysis for a video."""
    import pandas as pd
    from datetime import datetime
    from text_functions import analyze_text_comprehensive
    from utils import load_config_file
    
    # Load config to get output file paths
    config = load_config_file()
    output_file = config.get("output_files", {}).get("text_features", "output/text_features_analysis.csv")
    
    # Check if already processed
    if os.path.exists(output_file) and not args.force:
        try:
            existing_df = pd.read_csv(output_file)
            if video_id in existing_df['VideoId'].values:
                print(f"Text features already analyzed for {video_id}, use --force to regenerate")
                return True
        except:
            pass
    
    print(f"Analyzing text features for {video_id}...")
    
    # Load video metadata from the configured CSV file (same as basic features)
    description_file = config.get("paths", {}).get("csv_file", "output/videos_statistics_250_trial.csv")
    if not os.path.exists(description_file):
        print(f"Video metadata file not found: {description_file}")
        return False
    
    try:
        # Read CSV with proper multiline handling
        description_df = pd.read_csv(description_file, encoding='utf-8', quotechar='"', escapechar='\\')
        
        if description_df.empty:
            print(f"Description file is empty")
            return False
            
        if 'videoId' not in description_df.columns:
            print(f"videoId column not found in description file. Available columns: {list(description_df.columns)}")
            return False
            
        video_row = description_df[description_df['videoId'] == video_id]
        
        if video_row.empty:
            print(f"Video {video_id} not found in description file")
            return False
            
        # Safely extract title and description
        title = str(video_row.iloc[0]['title']) if 'title' in video_row.columns else ""
        description = str(video_row.iloc[0]['description']) if 'description' in video_row.columns else ""
        
        # Handle NaN values
        if title == 'nan':
            title = ""
        if description == 'nan':
            description = ""
            
        print(f"Loaded text data - Title: {len(title)} chars, Description: {len(description)} chars")
        
    except Exception as e:
        print(f"Error reading description file: {e}")
        print(f"Trying alternative CSV reading method...")
        
        # Alternative reading method for problematic CSV files
        try:
            with open(description_file, 'r', encoding='utf-8') as f:
                lines = f.readlines()
                
            if len(lines) < 2:
                print("Description file has insufficient data")
                return False
                
            # Parse header
            header = lines[0].strip().split(',')
            
            # Find video data
            title = ""
            description = ""
            
            for line in lines[1:]:
                if video_id in line:
                    # Simple parsing for the specific video
                    parts = line.split(',', len(header) - 1)
                    if len(parts) >= len(header):
                        try:
                            video_id_col = header.index('videoId')
                            title_col = header.index('title')
                            desc_col = header.index('description')
                            
                            if parts[video_id_col].strip() == video_id:
                                title = parts[title_col].strip().strip('"')
                                description = parts[desc_col].strip().strip('"')
                                break
                        except (ValueError, IndexError):
                            continue
                            
            if not title and not description:
                print(f"Could not extract text data for video {video_id}")
                return False
                
        except Exception as e2:
            print(f"Alternative reading method also failed: {e2}")
            return False
    
    # Set transcript path
    text_dir = config.get("paths", {}).get("text_dir", "output/text/")
    transcript_path = os.path.join(text_dir, f"{video_id}.txt")
    
    # Run comprehensive text analysis
    results = analyze_text_comprehensive(video_id, title, description, transcript_path)
    
    if results:
        # Prepare data for CSV
        row_data = {
            'VideoId': video_id,
            'titleSentiment': round(float(results['titleSentiment']), 3) if results['titleSentiment'] is not None else None,
            'titleTechnicality': round(float(results['titleTechnicality']), 3) if results['titleTechnicality'] is not None else None,
            'descriptionSentiment': round(float(results['descriptionSentiment']), 3) if results['descriptionSentiment'] is not None else None,
            'descriptionTechnicality': round(float(results['descriptionTechnicality']), 3) if results['descriptionTechnicality'] is not None else None,
            'hashtagsDescription': round(float(results['hashtagsDescription']), 3) if results['hashtagsDescription'] is not None else None,
            'URLDescription': round(float(results['URLDescription']), 3) if results['URLDescription'] is not None else None,
            'scriptSentiment': round(float(results['scriptSentiment']), 3) if results['scriptSentiment'] is not None else None,
            'scriptTechnicality': round(float(results['scriptTechnicality']), 3) if results['scriptTechnicality'] is not None else None,
            'analysis_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }
        
        # Save to CSV
        try:
            if os.path.exists(output_file):
                # Append to existing file
                existing_df = pd.read_csv(output_file)
                # Remove existing entry for this video if it exists
                existing_df = existing_df[existing_df['VideoId'] != video_id]
                # Add new entry - use safer concatenation
                new_row_df = pd.DataFrame([row_data])
                if len(existing_df) > 0:
                    new_df = pd.concat([existing_df, new_row_df], ignore_index=True)
                else:
                    new_df = new_row_df
            else:
                # Create new file
                output_dir = config.get("paths", {}).get("output_dir", "output/")
                os.makedirs(output_dir, exist_ok=True)
                new_df = pd.DataFrame([row_data])
            
            new_df.to_csv(output_file, index=False)
            
            print(f"✓ Text features saved for {video_id}")
            print(f"  Title: Sentiment={row_data['titleSentiment']}, Tech={row_data['titleTechnicality']}")
            print(f"  Description: Sentiment={row_data['descriptionSentiment']}, Tech={row_data['descriptionTechnicality']}")
            print(f"  Content: Hashtags={row_data['hashtagsDescription']}, URLs={row_data['URLDescription']}")
            if row_data['scriptSentiment'] is not None:
                print(f"  Script: Sentiment={row_data['scriptSentiment']}, Tech={row_data['scriptTechnicality']}")
            
            return True
            
        except Exception as e:
            print(f"Error saving text features for {video_id}: {e}")
            return False
    else:
        print(f"Failed to analyze text features for {video_id}")
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

if __name__ == "__main__":
    main()
