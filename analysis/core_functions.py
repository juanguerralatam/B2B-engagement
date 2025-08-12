#!/usr/bin/env python3
"""
Core functions for video analysis
Main processing logic and data operations
"""

import os
import csv
import subprocess
from .utils.core import (
    detect_optimal_device_settings, read_csv_safe, write_csv_safe,
    get_output_directory, ensure_directory_exists, print_status,
    get_timestamp, validate_video_id, detect_optimal_whisper_settings
)

def download_video(video_url, video_id, output_dir=None):
    """Download video using yt-dlp with videoId as filename."""
    try:
        from .utils.core import get_shared_config
        config = get_shared_config()
        
        if output_dir is None:
            output_dir = get_output_directory("video")
        
        ensure_directory_exists(output_dir)
        output_path = os.path.join(output_dir, f"{video_id}.%(ext)s")
        
        # Get cookies file locations from config
        cookies_locations = config.get("cookies", {}).get("search_locations", [])
        
        # Add dynamic locations relative to project
        project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        dynamic_locations = [
            os.path.join(project_root, loc) for loc in cookies_locations
        ]
        dynamic_locations.extend([
            os.path.join(os.path.dirname(os.path.abspath(__file__)), config.get("paths", {}).get("cookies_file", "youtube_cookies.txt")),
            os.path.join(project_root, config.get("paths", {}).get("cookies_file", "youtube_cookies.txt"))
        ])
        
        cookies_path = None
        for cookies_file in dynamic_locations:
            if os.path.exists(cookies_file):
                cookies_path = cookies_file
                break
        
        download_methods = []
        
        if cookies_path:
            download_methods.append([
                'yt-dlp', '--cookies', cookies_path,
                '-f', 'best[height<=720]', '-o', output_path, video_url
            ])
        
        download_methods.extend([
            ['yt-dlp', '--cookies-from-browser', 'chrome', '-f', 'best[height<=720]', '-o', output_path, video_url],
            ['yt-dlp', '-f', 'best[height<=720]', '-o', output_path, video_url]
        ])
        
        for i, cmd in enumerate(download_methods, 1):
            try:
                result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
                if result.returncode == 0:
                    # Get video format from config
                    video_formats = config.get("output_formats", {}).get("video_format", "mp4")
                    if isinstance(video_formats, str):
                        video_formats = [video_formats]
                    else:
                        video_formats = ['.mp4', '.mkv', '.webm', '.avi', '.mov']
                    
                    for ext in [f'.{fmt}' if not fmt.startswith('.') else fmt for fmt in video_formats]:
                        test_path = os.path.join(output_dir, f"{video_id}{ext}")
                        if os.path.exists(test_path):
                            return test_path
            except:
                continue
        
        return None
        
    except Exception as e:
        return None

def read_video_csv(csv_file):
    """Read video IDs and URLs from CSV file."""
    return read_csv_safe(csv_file, expected_columns=['videoId', 'video_url'])

def process_single_video(video_id, video_path, args):
    """Process a single video for transcription and evaluation."""
    # Import here to avoid circular imports
    try:
        from .audio_functions import extract_audio, transcribe_audio
        from .text_functions import evaluate_narration_quality, save_evaluation_report
    except ImportError as e:
        print_status(f"Error importing required modules: {e}", "ERROR")
        return False
    
    audio_path = os.path.join(get_output_directory("audio"), f"{video_id}.wav")
    transcript_path = os.path.join(get_output_directory("text"), f"{video_id}.txt")
    
    if not args.transcribe_only and not os.path.exists(video_path):
        print_status(f"Video file not found: {video_path}", "ERROR")
        return False
    
    if args.download_only:
        return True
    elif args.extract_audio_only:
        if not args.transcribe_only:
            if not extract_audio(video_path, audio_path):
                print_status(f"Failed to extract audio from {video_path}", "ERROR")
                return False
        
        if args.extract_audio_only:
            return True
    
    if not os.path.exists(audio_path):
        if not args.transcribe_only:
            if not extract_audio(video_path, audio_path):
                print_status(f"Failed to extract audio from {video_path}", "ERROR")
                return False
    
    transcript = None
    if os.path.exists(transcript_path) and not args.force:
        try:
            with open(transcript_path, 'r', encoding='utf-8') as f:
                transcript = f.read().strip()
        except Exception as e:
            print(f"Error reading existing transcript: {e}")
            transcript = None
    
    if not transcript:
        try:
            settings = detect_optimal_whisper_settings()
            transcript = transcribe_audio(audio_path, args.model, settings["device"], settings["fp16"])
        except Exception as e:
            print(f"Error transcribing audio: {e}")
            return False
    
    if transcript:
        if not os.path.exists(transcript_path) or args.force:
            os.makedirs(os.path.dirname(transcript_path), exist_ok=True)
            with open(transcript_path, 'w', encoding='utf-8') as f:
                f.write(transcript)
        
        if args.narration_only:
            try:
                evaluation = evaluate_narration_quality(transcript, None)
                if evaluation:
                    save_evaluation_report(evaluation, transcript_path)
            except Exception as e:
                print(f"Error evaluating narration: {e}")
        
        return True
    else:
        print("Failed to generate transcript")
        return False

# Data operations for scene analysis - integrate into basic features
def save_scene_analysis(video_id, scene_results, csv_file=None):
    """Save scene analysis results integrated into basic features CSV file."""
    try:
        from .utils.core import get_shared_config
        config = get_shared_config()
        
        # Use basic_features file instead of separate scene analysis file
        if csv_file is None:
            csv_file = config.get("output_files", {}).get("basic_features", "output/basic_features.csv")
        
        if isinstance(scene_results, (int, float)):
            scene_data = {'scene_count': scene_results, 'average_scene_length': 0, 'total_duration': 0}
        elif isinstance(scene_results, dict):
            scene_data = scene_results
        else:
            return False
        
        csv_path = csv_file if os.path.isabs(csv_file) else os.path.join(get_output_directory(), os.path.basename(csv_file))
        
        # Read existing data
        existing_data = {}
        if os.path.exists(csv_path):
            data_list = read_csv_safe(csv_path)
            for row in data_list:
                existing_data[row.get('videoId', '')] = row
        
        # Update existing row with scene data or create new entry
        if video_id in existing_data:
            existing_data[video_id].update({
                'sceneNumber': scene_data['scene_count'],
                'averageSceneLength': scene_data.get('average_scene_length', 0) if scene_data['scene_count'] > 0 else '',
                'analysis_date': get_timestamp()
            })
        else:
            # Create minimal entry if video not in basic features yet
            existing_data[video_id] = {
                'videoId': video_id,
                'sceneNumber': scene_data['scene_count'],
                'averageSceneLength': scene_data.get('average_scene_length', 0) if scene_data['scene_count'] > 0 else '',
                'analysis_date': get_timestamp()
            }
        
        # Write updated data with proper fieldnames
        all_possible_fields = ['videoId', 'channelId', 'Followers', 'videoAge', 'videoLength', 
                              'sceneNumber', 'averageSceneLength', 'analysis_date']
        return write_csv_safe(list(existing_data.values()), csv_path, all_possible_fields)
        
    except Exception as e:
        print_status(f"Error saving scene analysis: {e}", "ERROR")
        return False

def read_scene_analysis(csv_file=None):
    """Read existing scene analysis results from basic features CSV."""
    try:
        from .utils.core import get_shared_config
        config = get_shared_config()
        
        # Read from basic_features file instead of separate scene analysis file
        if csv_file is None:
            csv_file = config.get("output_files", {}).get("basic_features", "output/basic_features.csv")
        
        csv_path = csv_file if os.path.isabs(csv_file) else os.path.join(get_output_directory(), os.path.basename(csv_file))
        
        if not os.path.exists(csv_path):
            return {}
        
        data_list = read_csv_safe(csv_path)
        scene_data = {}
        
        for row in data_list:
            try:
                video_id = row.get('videoId')
                if video_id and row.get('sceneNumber'):
                    scene_data[video_id] = {
                        'scene_count': int(row.get('sceneNumber', 0)),
                        'average_scene_length': float(row.get('averageSceneLength', 0)) if row.get('averageSceneLength') else 0,
                        'analysis_date': row.get('analysis_date', '')
                    }
            except (ValueError, KeyError):
                continue
        
        return scene_data
        
    except Exception as e:
        return {}

# Data operations for gender analysis - integrate into image features
def save_gender_analysis(video_id, gender_results, csv_file=None):
    """Save gender analysis results integrated into image features CSV file."""
    try:
        from .utils.core import get_shared_config
        config = get_shared_config()
        
        if not gender_results:
            return False
        
        # Use image_features file instead of separate gender analysis file
        if csv_file is None:
            csv_file = config.get("output_files", {}).get("image_features", "output/image_features_analysis.csv")
        
        csv_path = csv_file if os.path.isabs(csv_file) else os.path.join(get_output_directory(), os.path.basename(csv_file))
        
        # Read existing data
        existing_data = {}
        if os.path.exists(csv_path):
            data_list = read_csv_safe(csv_path)
            for row in data_list:
                existing_data[row.get('VideoId', '')] = row
        
        # Update existing row with gender data or create new entry
        if video_id in existing_data:
            existing_data[video_id].update({
                'Gender': gender_results.get('score', ''),
                'analysis_date': get_timestamp()
            })
        else:
            # Create minimal entry if video not in image features yet
            existing_data[video_id] = {
                'VideoId': video_id,
                'Gender': gender_results.get('score', ''),
                'analysis_date': get_timestamp()
            }
        
        # Write updated data with proper fieldnames for image features
        all_possible_fields = ['VideoId', 'humanPresence', 'faceSum', 'Gender', 'Smile', 
                              'motionMagnitude', 'motionDirection', 'Saturation', 'Brightness', 
                              'frames_analyzed', 'analysis_date']
        return write_csv_safe(list(existing_data.values()), csv_path, all_possible_fields)
        
    except Exception as e:
        print_status(f"Error saving gender analysis: {e}", "ERROR")
        return False
        
        with open(csv_path, 'w', newline='', encoding='utf-8') as f:
            if existing_data:
                fieldnames = ['videoId', 'gender', 'score', 'confidence', 'male_detections', 
                             'female_detections', 'faces_detected', 'frames_analyzed', 'analysis_date']
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                for data in existing_data.values():
                    writer.writerow(data)
        
        return True
        
    except Exception as e:
        return False

def read_gender_analysis(csv_file=None):
    """Read existing gender analysis results from image features CSV."""
    try:
        from .utils.core import get_shared_config
        config = get_shared_config()
        
        # Read from image_features file instead of separate gender analysis file
        if csv_file is None:
            csv_file = config.get("output_files", {}).get("image_features", "output/image_features_analysis.csv")
        
        csv_path = csv_file if os.path.isabs(csv_file) else os.path.join(get_output_directory(), os.path.basename(csv_file))
        
        if not os.path.exists(csv_path):
            return {}
        
        data_list = read_csv_safe(csv_path)
        gender_data = {}
        
        for row in data_list:
            try:
                video_id = row.get('VideoId')
                if video_id and row.get('Gender') is not None:
                    gender_data[video_id] = {
                        'gender': 'male' if float(row.get('Gender', 0.5)) < 0.5 else 'female',
                        'score': row.get('Gender', ''),
                        'analysis_date': row.get('analysis_date', '')
                    }
            except (ValueError, KeyError):
                continue
        
        return gender_data
        
    except Exception as e:
        return {}

# =============================================================================
# Video Analysis Orchestration Functions
# =============================================================================

def analyze_video_complete(video_path, video_id, config=None):
    """Complete video analysis orchestrator - all features."""
    if config is None:
        from .utils.core import get_shared_config
        config = get_shared_config()
    
    results = {
        'video_id': video_id,
        'video_path': video_path,
        'timestamp': get_timestamp()
    }
    
    print_status(f"Starting complete analysis for {video_id}", "INFO")
    
    # Basic features analysis
    try:
        print_status("Extracting basic features...", "INFO")
        results['basic'] = analyze_basic_only(video_path, video_id)
        print_status("Basic features completed", "SUCCESS")
    except Exception as e:
        print_status(f"Basic features failed: {e}", "ERROR")
        results['basic'] = None
    
    # Image analysis
    try:
        from .image_functions import analyze_video_comprehensive
        print_status("Analyzing video content...", "INFO")
        results['image'] = analyze_video_comprehensive(video_path, video_id)
        print_status("Image analysis completed", "SUCCESS")
    except Exception as e:
        print_status(f"Image analysis failed: {e}", "ERROR")
        results['image'] = None
    
    # Audio analysis (if audio file exists)
    try:
        from .audio_functions import analyze_audio_comprehensive
        audio_dir = get_output_directory("audio")
        audio_path = os.path.join(audio_dir, f"{video_id}.wav")
        
        if os.path.exists(audio_path):
            print_status("Analyzing audio features...", "INFO")
            results['audio'] = analyze_audio_comprehensive(audio_path, video_id)
            print_status("Audio analysis completed", "SUCCESS")
        else:
            print_status(f"Audio file not found: {audio_path}", "WARNING")
            results['audio'] = None
    except Exception as e:
        print_status(f"Audio analysis failed: {e}", "ERROR")
        results['audio'] = None
    
    # Text analysis (if transcript exists)
    try:
        from .text_functions import analyze_text_comprehensive
        transcript_dir = get_output_directory("text")
        transcript_path = os.path.join(transcript_dir, f"{video_id}.txt")
        
        if os.path.exists(transcript_path):
            print_status("Analyzing text features...", "INFO")
            # Get title and description from video metadata if available
            title = ""  # Could be extracted from metadata
            description = ""  # Could be extracted from metadata
            results['text'] = analyze_text_comprehensive(video_id, title, description, transcript_path)
            print_status("Text analysis completed", "SUCCESS")
        else:
            print_status(f"Transcript file not found: {transcript_path}", "WARNING")
            results['text'] = None
    except Exception as e:
        print_status(f"Text analysis failed: {e}", "ERROR")
        results['text'] = None
    
    print_status(f"Complete analysis finished for {video_id}", "SUCCESS")
    return results

def analyze_basic_only(video_path, video_id):
    """Quick basic analysis workflow for a single video."""
    from .utils.core import get_shared_config
    config = get_shared_config()
    
    try:
        from .basic_functions import simple_scene_analysis, calculate_video_age, parse_duration_to_seconds
        from .utils.core import get_video_dimensions
        import pandas as pd
        import os
        
        print_status(f"Analyzing basic features for {video_id}", "INFO")
        
        # Load video metadata to get video info
        csv_file = config.get("input_files", {}).get("primary_csv", "output/videos_statistics.csv")
        if not os.path.exists(csv_file):
            print_status(f"Video metadata file not found: {csv_file}", "ERROR")
            return None
        
        # Read metadata for this specific video
        df = pd.read_csv(csv_file)
        video_row = df[df['videoId'] == video_id]
        
        if video_row.empty:
            print_status(f"Video {video_id} not found in metadata", "WARNING")
            return {
                'videoId': video_id,
                'channelId': None,
                'Followers': None,
                'videoAge': None,
                'videoLength': None,
                'sceneNumber': None,
                'averageSceneLength': None,
                'format': None,
                'orientation': None
            }
        
        row = video_row.iloc[0]
        
        # Load followers data
        followers_map = {}
        channels_file = config.get('paths', {}).get('channels_file', 'input/channels.csv')
        if os.path.exists(channels_file):
            try:
                channels_df = pd.read_csv(channels_file)
                followers_map = dict(zip(channels_df['id'], channels_df['Followers']))
            except Exception as e:
                print_status(f"Error loading followers data: {e}", "WARNING")
        
        # Extract basic info
        channel_id = row.get('channelId', '')
        followers = followers_map.get(channel_id, None)
        
        # Calculate video age
        video_age = None
        upload_date = row.get('publishedAt', row.get('upload_date', row.get('published_at', '')))
        if upload_date:
            video_age = calculate_video_age(str(upload_date))
        
        # Get video length (duration)
        video_length = None
        duration_field = row.get('duration', row.get('videoLength', row.get('video_length', None)))
        if duration_field:
            video_length = parse_duration_to_seconds(str(duration_field))
        
        # Analyze scenes if video file exists
        scene_analysis = {"sceneNumber": None, "averageSceneLength": None}
        if video_path and os.path.exists(video_path):
            scene_analysis = simple_scene_analysis(video_path)
        
        # Extract video dimensions and format
        video_format = None
        video_orientation = None
        if video_path and os.path.exists(video_path):
            try:
                dims = get_video_dimensions(video_path)
                if dims:
                    video_format = dims['format']
                    video_orientation = dims['orientation']
            except Exception as e:
                print_status(f"Could not extract video dimensions: {e}", "WARNING")
        
        result = {
            'videoId': video_id,
            'channelId': channel_id,
            'Followers': followers,
            'videoAge': video_age,
            'videoLength': video_length,
            'sceneNumber': scene_analysis['sceneNumber'],
            'averageSceneLength': scene_analysis['averageSceneLength'],
            'format': video_format,
            'orientation': video_orientation
        }
        
        print_status(f"Basic analysis completed for {video_id}", "SUCCESS")
        return result
        
    except Exception as e:
        print_status(f"Basic analysis failed: {e}", "ERROR")
        return None

def analyze_image_only(video_path, video_id):
    """Quick image analysis workflow."""
    try:
        from .image_functions import analyze_video_comprehensive
        return analyze_video_comprehensive(video_path, video_id)
    except Exception as e:
        print_status(f"Image analysis failed: {e}", "ERROR")
        return None

def analyze_lightweight(video_path, video_id):
    """Fast analysis - basic + image only."""
    print_status(f"Starting lightweight analysis for {video_id}", "INFO")
    
    results = {
        'video_id': video_id,
        'video_path': video_path,
        'timestamp': get_timestamp()
    }
    
    # Basic features
    basic_result = analyze_basic_only(video_path, video_id)
    results['basic'] = basic_result
    
    # Image analysis
    image_result = analyze_image_only(video_path, video_id)
    results['image'] = image_result
    
    print_status(f"Lightweight analysis completed for {video_id}", "SUCCESS")
    return results
