#!/usr/bin/env python3
"""
Parallel Video Processing Module

This module provides efficient parallel processing capabilities for video analysis tasks.
It uses multiprocessing with the 'spawn' method to ensure CUDA compatibility for GPU-accelerated
image analysis while maintaining optimal performance for CPU-intensive tasks.

Key Features:
- CUDA-compatible multiprocessing using 'spawn' method
- Automatic worker optimization based on system resources
- Support for image, audio, and text analysis tasks
- Real-time progress monitoring and error handling

Performance Guidelines:
- Image Analysis (GPU): 8-16 workers recommended, up to 2.5+ videos/second
- Text Analysis (CPU): 16-32 workers recommended, transcription dependent
- Audio Analysis (CPU): 8-12 workers recommended

Author: Enhanced parallel processing system
Date: August 2025
"""

import os
import sys
import time
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import List, Dict, Any
import argparse

# Add analysis directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def process_single_video_simple(video_data):
    """Simple worker function for processing a single video."""
    try:
        video_id, video_path, task_type, force = video_data
        
        # Set up paths
        current_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        sys.path.insert(0, current_dir)
        
        # Create minimal args object
        class SimpleArgs:
            def __init__(self):
                self.force = force
                self.analyze_image_features = task_type == 'image'
                self.analyze_audio_features = task_type == 'audio' 
                self.analyze_text_features = task_type == 'text'
                self.detect_scenes = task_type == 'scenes'
                self.narration_only = task_type == 'narration'
        
        args = SimpleArgs()
        
        # Import and call appropriate function
        if task_type == 'image':
            from videoAnalysis import process_image_features
            result = process_image_features(video_id, video_path, args)
        elif task_type == 'audio':
            from videoAnalysis import process_audio_features  
            result = process_audio_features(video_id, video_path, args)
        elif task_type == 'text':
            from videoAnalysis import process_text_features
            result = process_text_features(video_id, video_path, args)
        else:
            result = False
            
        return (video_id, result, None)
        
    except Exception as e:
        return (video_id, False, str(e))

def find_video_file(video_id):
    """Find video file for a given video ID using the main videoAnalysis function."""
    import sys
    import os
    
    # Add current directory to path to import videoAnalysis
    current_dir = os.path.dirname(os.path.abspath(__file__))
    sys.path.insert(0, current_dir)
    
    from videoAnalysis import find_video_file
    return find_video_file(video_id)

def process_videos_parallel_simple(videos, task_type, max_workers=None, force=False):
    """Process videos in parallel using simple multiprocessing."""
    
    # Set multiprocessing start method to 'spawn' for CUDA compatibility
    import multiprocessing as mp
    try:
        mp.set_start_method('spawn', force=True)
    except RuntimeError:
        # Already set, ignore
        pass
    
    if max_workers is None:
        max_workers = min(8, mp.cpu_count() // 2)  # Conservative default
    
    print(f"🚀 Processing {len(videos)} videos with {max_workers} workers")
    print(f"📊 Task type: {task_type}")
    
    # Prepare work items
    work_items = []
    for video in videos:
        video_id = video['videoId']
        video_path = find_video_file(video_id)
        
        if video_path or task_type in ['text']:  # Text can work without video file
            work_items.append((video_id, video_path, task_type, force))
    
    print(f"📁 Found {len(work_items)} videos to process")
    
    if not work_items:
        print("❌ No videos found to process")
        return 0, 0
    
    # Process in parallel
    successful = 0
    failed = 0
    start_time = time.time()
    
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        # Submit all jobs
        future_to_video = {
            executor.submit(process_single_video_simple, item): item[0] 
            for item in work_items
        }
        
        # Process results as they complete
        for i, future in enumerate(as_completed(future_to_video), 1):
            video_id = future_to_video[future]
            
            try:
                video_id_result, success, error = future.result()
                
                if success:
                    successful += 1
                    print(f"✅ {i:3d}/{len(work_items)} - {video_id}")
                else:
                    failed += 1
                    error_msg = f" ({error})" if error else ""
                    print(f"❌ {i:3d}/{len(work_items)} - {video_id}{error_msg}")
                    
            except Exception as e:
                failed += 1
                print(f"💥 {i:3d}/{len(work_items)} - {video_id} - Exception: {e}")
            
            # Show progress every 10 videos
            if i % 10 == 0:
                elapsed = time.time() - start_time
                rate = i / elapsed if elapsed > 0 else 0
                eta = (len(work_items) - i) / rate if rate > 0 else 0
                print(f"⚡ Progress: {i}/{len(work_items)} ({i/len(work_items)*100:.1f}%) - Rate: {rate:.1f}/s - ETA: {eta/60:.1f}m")
    
    total_time = time.time() - start_time
    rate = len(work_items) / total_time if total_time > 0 else 0
    
    print(f"\n🎉 Parallel processing completed!")
    print(f"✅ Successful: {successful}")
    print(f"❌ Failed: {failed}")
    print(f"⏱️ Total time: {total_time:.1f}s")
    print(f"⚡ Average rate: {rate:.2f} videos/second")
    
    return successful, failed

if __name__ == "__main__":
    # Simple test
    from analysis.core_functions import read_video_csv
    from analysis.utils.core import find_csv_file, load_config_file
    
    config = load_config_file()
    csv_file = config.get("input_files", {}).get("primary_csv", "output/videos_statistics.csv")
    csv_file = find_csv_file(csv_file)
    
    videos = read_video_csv(csv_file)
    if videos:
        # Test with first 5 videos
        test_videos = videos[:5]
        process_videos_parallel_simple(test_videos, 'image', max_workers=4, force=True)
