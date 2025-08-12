#!/usr/bin/env python3
"""
Basic Functions for Feature Extraction
Contains core functions for extracting basic video features
"""

import os
import sys
import json
import pandas as pd
from datetime import datetime
from typing import Dict, Optional, List

# Import utilities
try:
    from .utils.core import ensure_directory_exists, print_status, get_shared_config, get_video_dimensions
except ImportError:
    # Fallback for direct execution
    try:
        from utils.core import ensure_directory_exists, print_status, get_shared_config, get_video_dimensions
    except ImportError:
        def ensure_directory_exists(path):
            os.makedirs(path, exist_ok=True)
        
        def print_status(message, level="INFO"):
            levels = {"INFO": "", "WARNING": "⚠️ ", "ERROR": "❌ ", "SUCCESS": "✅ "}
            prefix = levels.get(level.upper(), "")
            print(f"{prefix}{message}")
        
        def get_shared_config():
            return {}
        
        def get_video_dimensions(video_path):
            return None

def load_config() -> Dict:
    """Load configuration from config.json"""
    from .utils.core import get_shared_config
    return get_shared_config()

def load_followers_data(config: Dict) -> Dict[str, int]:
    """Load follower data from channels file"""
    channels_file = config['paths']['channels_file']
    
    if not os.path.exists(channels_file):
        print_status(f"Channels file not found: {channels_file}", "ERROR")
        return {}
    
    try:
        df = pd.read_csv(channels_file)
        # Create mapping from channel id to followers
        followers_map = dict(zip(df['id'], df['Followers']))
        print_status(f"Loaded follower data for {len(followers_map)} channels", "SUCCESS")
        return followers_map
    except Exception as e:
        print_status(f"Error loading followers data: {e}", "ERROR")
        return {}

def load_video_metadata(config: Dict) -> Optional[pd.DataFrame]:
    """Load video metadata from videos_statistics.csv file"""
    
    # Get the primary CSV file from config
    csv_file = config.get("input_files", {}).get("primary_csv", "output/videos_statistics.csv")
    
    if not os.path.exists(csv_file):
        print_status(f"Video metadata file not found: {csv_file}", "ERROR")
        return None
    
    try:
        df = pd.read_csv(csv_file)
        print_status(f"Loaded video metadata from {csv_file}", "SUCCESS")
        print_status(f"Available columns: {list(df.columns)}", "INFO")
        return df
    except Exception as e:
        print_status(f"Error reading {csv_file}: {e}", "ERROR")
        return None

def calculate_video_age(upload_date_str: str) -> Optional[int]:
    """Calculate video age in days from upload date string"""
    if not upload_date_str or pd.isna(upload_date_str):
        return None
    
    try:
        # Try different date formats
        date_formats = [
            '%Y-%m-%d %H:%M:%S UTC',  # 2025-07-17 00:07:21 UTC
            '%Y-%m-%d', 
            '%Y-%m-%dT%H:%M:%SZ', 
            '%Y-%m-%d %H:%M:%S',
            '%Y-%m-%dT%H:%M:%S.%fZ',
            '%m/%d/%Y',
            '%d/%m/%Y'
        ]
        
        for fmt in date_formats:
            try:
                upload_date = datetime.strptime(str(upload_date_str), fmt)
                age_days = (datetime.now() - upload_date).days
                return age_days
            except ValueError:
                continue
        
        print_status(f"Could not parse date: {upload_date_str}", "WARNING")
        return None
    except Exception as e:
        return None

def parse_duration_to_seconds(duration_str: str) -> Optional[int]:
    """Parse duration string to seconds"""
    if not duration_str or pd.isna(duration_str):
        return None
    
    try:
        duration_str = str(duration_str).strip()
        
        # If it's already a number, return it
        try:
            return int(float(duration_str))
        except ValueError:
            pass
        
        # Parse time format (HH:MM:SS or MM:SS)
        if ':' in duration_str:
            parts = duration_str.split(':')
            if len(parts) == 2:  # MM:SS
                return int(parts[0]) * 60 + int(parts[1])
            elif len(parts) == 3:  # HH:MM:SS
                return int(parts[0]) * 3600 + int(parts[1]) * 60 + int(parts[2])
        
        # Parse ISO 8601 duration (PT#M#S format)
        if duration_str.startswith('PT'):
            import re
            pattern = r'PT(?:(\d+)H)?(?:(\d+)M)?(?:(\d+)S)?'
            match = re.match(pattern, duration_str)
            if match:
                hours = int(match.group(1) or 0)
                minutes = int(match.group(2) or 0)
                seconds = int(match.group(3) or 0)
                return hours * 3600 + minutes * 60 + seconds
        
        return None
    except Exception as e:
        return None

def simple_scene_analysis(video_path: str) -> Dict[str, Optional[float]]:
    """Simple scene analysis - try to use existing function or provide estimates"""
    if not os.path.exists(video_path):
        return {"sceneNumber": None, "averageSceneLength": None}
    
    try:
        # Try to import and use scene detection
        from .image_functions import detect_scenes
        
        # Extract video_id from path
        video_id = os.path.basename(video_path).replace('.mp4', '')
        
        # Try scene detection with default threshold first
        scene_data = detect_scenes(video_path, video_id, threshold=30.0)
        
        # If no scenes detected, try with a lower threshold
        if scene_data and scene_data.get('scene_count', 0) == 0:
            scene_data = detect_scenes(video_path, video_id, threshold=15.0)
        
        # If still no scenes, try with an even lower threshold
        if scene_data and scene_data.get('scene_count', 0) == 0:
            scene_data = detect_scenes(video_path, video_id, threshold=5.0)
        
        if scene_data and isinstance(scene_data, dict):
            return {
                "sceneNumber": scene_data.get('scene_count'),
                "averageSceneLength": scene_data.get('average_scene_length')
            }
    except ImportError:
        # Fallback: estimate scenes based on video length
        try:
            # Get video duration using moviepy or similar
            from moviepy.editor import VideoFileClip
            with VideoFileClip(video_path) as clip:
                duration = clip.duration
                # Estimate ~1 scene per 10-30 seconds (rough estimate)
                estimated_scenes = max(1, int(duration / 20))
                avg_scene_length = duration / estimated_scenes
                
                return {
                    "sceneNumber": estimated_scenes,
                    "averageSceneLength": avg_scene_length
                }
        except Exception:
            pass
    except Exception as e:
        print_status(f"Error analyzing scenes for {video_path}: {e}", "WARNING")
    
    return {"sceneNumber": None, "averageSceneLength": None}

def extract_basic_features_from_data(config: Dict) -> bool:
    """Main function to extract basic features"""
    print_status("Starting basic feature extraction...", "INFO")
    
    # Load followers data
    followers_map = load_followers_data(config)
    if not followers_map:
        print_status("No follower data available", "ERROR")
        return False
    
    # Load video metadata
    video_metadata = load_video_metadata(config)
    if video_metadata is None:
        print_status("No video metadata available", "ERROR")
        return False
    
    print_status(f"Found metadata for {len(video_metadata)} videos", "INFO")
    print_status(f"Available columns: {list(video_metadata.columns)}", "INFO")
    
    # Prepare output data
    output_data = []
    video_dir = os.path.expanduser(config['paths']['video_dir'])  # Expand ~ to home directory
    
    # Check for max_videos limit
    max_videos = config.get('defaults', {}).get('max_videos', None)
    total_videos = len(video_metadata)
    
    if max_videos and max_videos < total_videos:
        print_status(f"Limiting processing to first {max_videos} videos (out of {total_videos})", "INFO")
        video_metadata = video_metadata.head(max_videos)
    
    print_status(f"Processing {len(video_metadata)} videos...", "INFO")
    
    for idx, row in video_metadata.iterrows():
        try:
            # Get basic info - try different column names
            video_id = row.get('videoId', row.get('id', row.get('video_id', '')))
            channel_id = row.get('channelId', row.get('channel_id', row.get('channelID', '')))
            
            if not video_id or not channel_id:
                print_status(f"Skipping row {idx}: missing videoId or channelId", "WARNING")
                continue
            
            # Get followers
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
            
            # Analyze scenes - handle different video file naming patterns
            video_path = None
            
            # Try different file naming patterns
            possible_patterns = [
                f"{video_id}.mp4",  # Simple: videoId.mp4
                f"{video_id} - *.mp4",  # Pattern: videoId - title.mp4
            ]
            
            for pattern in possible_patterns:
                if '*' in pattern:
                    # Use glob to find files matching the pattern
                    import glob
                    matching_files = glob.glob(os.path.join(video_dir, pattern))
                    if matching_files:
                        video_path = matching_files[0]  # Take the first match
                        break
                else:
                    # Direct file check
                    test_path = os.path.join(video_dir, pattern)
                    if os.path.exists(test_path):
                        video_path = test_path
                        break
            
            # If no video found, try a more thorough search
            if not video_path:
                import glob
                # Search for any file that starts with the video ID
                search_pattern = os.path.join(video_dir, f"{video_id}*")
                matching_files = glob.glob(search_pattern)
                video_files = [f for f in matching_files if f.endswith(('.mp4', '.mkv', '.webm', '.avi', '.mov'))]
                if video_files:
                    video_path = video_files[0]
            
            scene_analysis = simple_scene_analysis(video_path) if video_path else {"sceneNumber": None, "averageSceneLength": None}
            
            # Extract video dimensions and format
            video_dimensions = None
            video_format = None
            video_orientation = None
            
            if video_path:
                try:
                    dims = get_video_dimensions(video_path)
                    if dims:
                        video_format = dims['format']
                        video_orientation = dims['orientation']
                        video_dimensions = dims
                except Exception as e:
                    print_status(f"Could not extract video dimensions for {video_id}: {e}", "WARNING")
            
            # Create output record
            record = {
                'channelId': channel_id,
                'videoId': video_id,
                'Followers': followers,
                'videoAge': video_age,
                'videoLength': video_length,
                'sceneNumber': scene_analysis['sceneNumber'],
                'averageSceneLength': scene_analysis['averageSceneLength'],
                'format': video_format,
                'orientation': video_orientation
            }
            
            output_data.append(record)
            
            if (idx + 1) % 10 == 0:
                print_status(f"Processed {idx + 1}/{len(video_metadata)} videos", "INFO")
        
        except Exception as e:
            print_status(f"Error processing video {idx}: {e}", "WARNING")
            continue
    
    # Save results
    if output_data:
        # Use output_files config if available, otherwise fallback to old method
        output_file = config.get('output_files', {}).get('basic_features')
        if not output_file:
            output_file = os.path.join(config['paths']['output_dir'], 'basic_features.csv')
        ensure_directory_exists(os.path.dirname(output_file))
        
        output_df = pd.DataFrame(output_data)
        output_df.to_csv(output_file, index=False)
        
        print_status(f"Saved {len(output_data)} records to {output_file}", "SUCCESS")
        print_status(f"Features extracted: channelId, videoId, Followers, videoAge, videoLength, sceneNumber, averageSceneLength, format, orientation", "INFO")
        return True
    else:
        print_status("No data to save", "ERROR")
        return False
