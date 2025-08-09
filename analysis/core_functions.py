#!/usr/bin/env python3
"""
Core functions for video analysis
Main processing logic and data operations
"""

import os
import csv
import subprocess
from utils import (
    detect_optimal_device_settings, read_csv_safe, write_csv_safe,
    get_output_directory, ensure_directory_exists, print_status,
    get_timestamp, validate_video_id
)

def detect_optimal_whisper_settings():
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

def download_video(video_url, video_id, output_dir=None):
    """Download video using yt-dlp with videoId as filename."""
    try:
        from utils import load_config_file
        config = load_config_file()
        
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
        from audio_functions import extract_audio, transcribe_audio
        from text_functions import evaluate_narration_quality, save_evaluation_report
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
        from utils import load_config_file
        config = load_config_file()
        
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
        from utils import load_config_file
        config = load_config_file()
        
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
        from utils import load_config_file
        config = load_config_file()
        
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
        from utils import load_config_file
        config = load_config_file()
        
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
