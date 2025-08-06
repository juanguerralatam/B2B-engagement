#!/usr/bin/env python3
"""
Core functions for video analysis
Main processing logic and data operations
"""

import os
import csv
import subprocess

def detect_optimal_whisper_settings():
    """Detect optimal Whisper settings based on available hardware."""
    try:
        import psutil
        import torch
        
        # Check for whisper import inside try block 
        try:
            import whisper
        except ImportError:
            pass
        
        memory_gb = psutil.virtual_memory().total / (1024**3)
        
        if torch.cuda.is_available():
            device = "cuda"
            fp16 = True
            if memory_gb >= 16:
                recommended_model = "large"
            elif memory_gb >= 8:
                recommended_model = "medium"
            else:
                recommended_model = "base"
        else:
            device = "cpu"
            recommended_model = "base"
            fp16 = False
        
        return {
            "device": device,
            "recommended_model": recommended_model,
            "fp16": fp16
        }
        
    except ImportError as e:
        return {"device": "cpu", "recommended_model": "base", "fp16": False}
    except Exception as e:
        return {"device": "cpu", "recommended_model": "base", "fp16": False}

def download_video(video_url, video_id, output_dir=None):
    """Download video using yt-dlp with videoId as filename."""
    try:
        if output_dir is None:
            script_dir = os.path.dirname(os.path.abspath(__file__))
            project_root = os.path.dirname(script_dir)
            output_dir = os.path.join(project_root, "output", "video")
        
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, f"{video_id}.%(ext)s")
        
        # Look for cookies file in multiple locations
        cookies_files = [
            "youtube_cookies.txt",
            os.path.join(os.path.dirname(os.path.abspath(__file__)), "youtube_cookies.txt"),
            os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "youtube_cookies.txt"),
            os.path.join("test", "youtube_cookies.txt")
        ]
        
        cookies_path = None
        for cookies_file in cookies_files:
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
                    for ext in ['.mp4', '.mkv', '.webm', '.avi', '.mov']:
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
    videos = []
    try:
        with open(csv_file, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                videos.append(row)
        return videos
    except Exception as e:
        return []

def process_single_video(video_id, video_path, args):
    """Process a single video for transcription and evaluation."""
    # Import here to avoid circular imports
    try:
        from audio_functions import extract_audio, transcribe_audio
        from text_functions import evaluate_narration_quality, save_evaluation_report
    except ImportError as e:
        print(f"Error importing required modules: {e}")
        return False
    
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    
    audio_path = os.path.join(project_root, "output", "audio", f"{video_id}.wav")
    transcript_path = os.path.join(project_root, "output", "text", f"{video_id}.txt")
    
    if not args.transcribe_only and not os.path.exists(video_path):
        print(f"Video file not found: {video_path}")
        return False
    
    if args.download_only:
        return True
    elif args.extract_audio_only:
        if not args.transcribe_only:
            if not extract_audio(video_path, audio_path):
                print(f"Failed to extract audio from {video_path}")
                return False
        
        if args.extract_audio_only:
            return True
    
    if not os.path.exists(audio_path):
        if not args.transcribe_only:
            if not extract_audio(video_path, audio_path):
                print(f"Failed to extract audio from {video_path}")
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
        
        if args.narration:
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

# Data operations for scene analysis
def save_scene_analysis(video_id, scene_results, csv_file="analysisYoutube.csv"):
    """Save scene analysis results to CSV file."""
    try:
        if isinstance(scene_results, (int, float)):
            scene_data = {'scene_count': scene_results, 'average_scene_length': 0, 'total_duration': 0}
        elif isinstance(scene_results, dict):
            scene_data = scene_results
        else:
            return False
        
        script_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.dirname(script_dir)
        csv_path = os.path.join(project_root, "output", csv_file)
        
        file_exists = os.path.exists(csv_path)
        existing_data = {}
        
        if file_exists:
            try:
                with open(csv_path, 'r', encoding='utf-8') as f:
                    reader = csv.DictReader(f)
                    for row in reader:
                        existing_data[row['videoId']] = row
            except:
                pass
        
        existing_data[video_id] = {
            'videoId': video_id,
            'scene_count': scene_data['scene_count'],
            'average_scene_length': scene_data.get('average_scene_length', 0) if scene_data['scene_count'] > 0 else '',
            'total_duration': scene_data.get('total_duration', 0) if scene_data['scene_count'] > 0 else '',
            'analysis_date': subprocess.check_output(['date', '+%Y-%m-%d %H:%M:%S']).decode().strip()
        }
        
        with open(csv_path, 'w', newline='', encoding='utf-8') as f:
            if existing_data:
                fieldnames = ['videoId', 'scene_count', 'average_scene_length', 'total_duration', 'analysis_date']
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                for data in existing_data.values():
                    writer.writerow(data)
        
        return True
        
    except Exception as e:
        return False

def read_scene_analysis(csv_file="analysisYoutube.csv"):
    """Read existing scene analysis results from CSV."""
    try:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.dirname(script_dir)
        csv_path = os.path.join(project_root, "output", csv_file)
        
        if not os.path.exists(csv_path):
            return {}
        
        scene_data = {}
        with open(csv_path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                try:
                    scene_data[row['videoId']] = {
                        'scene_count': int(row['scene_count']),
                        'average_scene_length': float(row['average_scene_length']) if row['average_scene_length'] else 0,
                        'total_duration': float(row['total_duration']) if row['total_duration'] else 0,
                        'analysis_date': row.get('analysis_date', '')
                    }
                except (ValueError, KeyError):
                    continue
        
        return scene_data
        
    except Exception as e:
        return {}

# Data operations for gender analysis  
def save_gender_analysis(video_id, gender_results, csv_file="genderAnalysis.csv"):
    """Save gender analysis results to CSV file."""
    try:
        if not gender_results:
            return False
        
        script_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.dirname(script_dir)
        csv_path = os.path.join(project_root, "output", csv_file)
        
        file_exists = os.path.exists(csv_path)
        existing_data = {}
        
        if file_exists:
            try:
                with open(csv_path, 'r', encoding='utf-8') as f:
                    reader = csv.DictReader(f)
                    for row in reader:
                        existing_data[row['videoId']] = row
            except:
                pass
        
        existing_data[video_id] = {
            'videoId': video_id,
            'gender': gender_results['gender'],
            'score': gender_results.get('score', ''),
            'confidence': gender_results['confidence'],
            'male_detections': gender_results.get('male_detections', 0),
            'female_detections': gender_results.get('female_detections', 0),
            'faces_detected': gender_results['faces_detected'],
            'frames_analyzed': gender_results['frames_analyzed'],
            'analysis_date': subprocess.check_output(['date', '+%Y-%m-%d %H:%M:%S']).decode().strip()
        }
        
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

def read_gender_analysis(csv_file="genderAnalysis.csv"):
    """Read existing gender analysis results from CSV."""
    try:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.dirname(script_dir)
        csv_path = os.path.join(project_root, "output", csv_file)
        
        if not os.path.exists(csv_path):
            return {}
        
        gender_data = {}
        with open(csv_path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                try:
                    # Handle both old and new CSV formats
                    gender = row.get('gender', row.get('predicted_gender', 'unknown'))
                    confidence = float(row.get('confidence', 0))
                    
                    gender_data[row['videoId']] = {
                        'gender': gender,
                        'score': row.get('score', ''),
                        'confidence': confidence,
                        'male_detections': int(row.get('male_detections', 0)),
                        'female_detections': int(row.get('female_detections', 0)),
                        'faces_detected': int(row['faces_detected']),
                        'frames_analyzed': int(row['frames_analyzed']),
                        'analysis_date': row.get('analysis_date', '')
                    }
                except (ValueError, KeyError):
                    continue
        
        return gender_data
        
    except Exception as e:
        return {}
