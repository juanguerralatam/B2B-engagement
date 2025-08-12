#!/usr/bin/env python3
"""
Main image processing module for B2B engagement analysis.
Uses state-of-the-art YOLO implementation from utils.yolo for production-ready accuracy.
"""

import os
import cv2
import numpy as np
from typing import Dict, Any, Tuple, Optional
import time

# Import YOLO production functions
try:
    from .utils.yolo import (
        analyze_video_comprehensive_production,
        detect_gender_in_video as yolo_detect_gender,
        detect_faces_yolo_production,
        classify_gender_production,
        detect_smile_production,
        analyze_motion_production,
        analyze_color_production,
        ProductionFaceAnalyzer,
        YOLOFaceDetector
    )
    YOLO_AVAILABLE = True
    print("🚀 YOLO production module loaded successfully")
except ImportError as e:
    YOLO_AVAILABLE = False
    print(f"⚠️  YOLO module not available: {e}")

# Configuration
DEFAULT_SAMPLE_FRAMES = 30

def detect_scenes(video_path, video_id, threshold=30.0):
    """Fast scene detection using PySceneDetect."""
    try:
        from scenedetect import detect, ContentDetector
        scene_list = detect(video_path, ContentDetector(threshold=threshold))
        
        scene_count = len(scene_list)
        scene_lengths = []
        total_duration = 0
        
        if scene_count > 0:
            for scene in scene_list:
                start_time = scene[0].get_seconds()
                end_time = scene[1].get_seconds()
                scene_length = end_time - start_time
                scene_lengths.append(scene_length)
                total_duration += scene_length
            
            average_scene_length = total_duration / scene_count
        else:
            average_scene_length = 0
            total_duration = 0
        
        return {
            'scene_count': scene_count,
            'average_scene_length': round(average_scene_length, 2),
            'total_duration': round(total_duration, 2),
            'scene_lengths': [round(length, 2) for length in scene_lengths]
        }
        
    except ImportError:
        print(f"PySceneDetect not available for {video_id}")
        return None
    except Exception as e:
        print(f"Scene detection error for {video_id}: {e}")
        return None

def analyze_video_comprehensive(video_path: str, video_id: str, sample_frames: int = DEFAULT_SAMPLE_FRAMES) -> Dict[str, Any]:
    """
    Main video analysis function using state-of-the-art YOLO implementation.
    
    This function provides comprehensive analysis including:
    - Face detection and counting
    - Gender classification
    - Smile detection
    - Motion analysis
    - Color analysis
    """
    if not YOLO_AVAILABLE:
        print(f"❌ YOLO not available, cannot analyze {video_id}")
        return None
    
    if not os.path.exists(video_path):
        print(f"❌ Video file not found: {video_path}")
        return None
    
    print(f"🎬 Analyzing video: {video_id} with YOLO production pipeline")
    
    # Use the production YOLO implementation
    result = analyze_video_comprehensive_production(video_path, video_id, sample_frames)
    
    if result:
        print(f"✅ Analysis complete for {video_id}")
        print(f"   📊 Results: {result['humanPresence']:.1f} human, {result.get('faceSum', 0)} faces")
        print(f"   ⚡ Performance: {result['performance']['analysis_fps']:.1f} FPS")
    
    return result

def detect_gender_in_video(video_path: str, video_id: str, sample_frames: int = DEFAULT_SAMPLE_FRAMES) -> Dict[str, Any]:
    """
    Gender detection wrapper using YOLO production pipeline.
    """
    if not YOLO_AVAILABLE:
        return {
            'gender': 'unknown',
            'score': None,
            'confidence': 0.0,
            'method': 'yolo_unavailable'
        }
    
    return yolo_detect_gender(video_path, video_id, sample_frames)

def count_faces_simple(video_path: str, video_id: str, sample_frames: int = DEFAULT_SAMPLE_FRAMES) -> int:
    """
    Simple face counting using YOLO production pipeline.
    """
    if not YOLO_AVAILABLE:
        print(f"⚠️  YOLO not available for face counting: {video_id}")
        return 0
    
    try:
        face_analysis = detect_faces_yolo_production(video_path, video_id, sample_frames)
        
        if 'error' in face_analysis:
            print(f"❌ Face counting failed: {face_analysis['error']}")
            return 0
        
        unique_faces = face_analysis.get('unique_faces', 0)
        print(f"🎯 Face count for {video_id}: {unique_faces}")
        
        return unique_faces
        
    except Exception as e:
        print(f"Face counting error for {video_id}: {e}")
        return 0

def detect_smile_in_video(video_path: str, video_id: str, sample_frames: int = DEFAULT_SAMPLE_FRAMES) -> bool:
    """
    Smile detection wrapper using YOLO production pipeline.
    """
    if not YOLO_AVAILABLE:
        print(f"⚠️  YOLO not available for smile detection: {video_id}")
        return False
    
    return detect_smile_production(video_path, video_id, sample_frames)

def analyze_motion_in_video(video_path: str, video_id: str, sample_frames: int = DEFAULT_SAMPLE_FRAMES) -> Tuple[float, float]:
    """
    Motion analysis wrapper using YOLO production pipeline.
    """
    if not YOLO_AVAILABLE:
        print(f"⚠️  YOLO not available for motion analysis: {video_id}")
        return 0.0, 0.0
    
    return analyze_motion_production(video_path, video_id, sample_frames)

def analyze_color_in_video(video_path: str, video_id: str, sample_frames: int = DEFAULT_SAMPLE_FRAMES) -> Tuple[float, float]:
    """
    Color analysis wrapper using YOLO production pipeline.
    """
    if not YOLO_AVAILABLE:
        print(f"⚠️  YOLO not available for color analysis: {video_id}")
        return 0.0, 0.0
    
    return analyze_color_production(video_path, video_id, sample_frames)

def get_video_info(video_path: str) -> Dict[str, Any]:
    """
    Get basic video information.
    """
    try:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return {'error': 'Cannot open video'}
        
        info = {
            'frame_count': int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
            'fps': cap.get(cv2.CAP_PROP_FPS),
            'width': int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            'height': int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
            'duration': int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) / cap.get(cv2.CAP_PROP_FPS) if cap.get(cv2.CAP_PROP_FPS) > 0 else 0
        }
        
        cap.release()
        return info
        
    except Exception as e:
        return {'error': str(e)}

# Fallback functions for when YOLO is not available
def analyze_video_comprehensive_fallback(video_path: str, video_id: str, sample_frames: int = DEFAULT_SAMPLE_FRAMES) -> Dict[str, Any]:
    """
    Fallback analysis using basic OpenCV when YOLO is not available.
    """
    print(f"⚠️  Using fallback analysis for {video_id}")
    
    try:
        # Basic video info
        video_info = get_video_info(video_path)
        if 'error' in video_info:
            return None
        
        # Basic motion and color analysis using OpenCV
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return None
        
        total_frames = video_info['frame_count']
        frame_interval = max(1, total_frames // sample_frames)
        
        motion_values = []
        saturation_values = []
        brightness_values = []
        prev_gray = None
        
        for frame_idx in range(0, total_frames, frame_interval):
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = cap.read()
            
            if not ret:
                continue
            
            # Color analysis
            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            saturation_values.append(hsv[:, :, 1].mean() / 255.0)
            brightness_values.append(hsv[:, :, 2].mean() / 255.0)
            
            # Motion analysis
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            if prev_gray is not None:
                diff = cv2.absdiff(prev_gray, gray)
                motion_values.append(diff.mean() / 255.0)
            prev_gray = gray
        
        cap.release()
        
        # Return basic results
        return {
            'humanPresence': 0.0,  # Cannot detect faces without YOLO
            'faceSum': None,
            'Gender': None,
            'Smile': 0.0,
            'motionMagnitude': np.mean(motion_values) if motion_values else 0.0,
            'motionDirection': 0.5,
            'Saturation': np.mean(saturation_values) if saturation_values else 0.0,
            'Brightness': np.mean(brightness_values) if brightness_values else 0.0,
            'frames_analyzed': sample_frames,
            'video_id': video_id,
            'method': 'opencv_fallback',
            'performance': {
                'total_time': 0,
                'analysis_fps': 0,
                'detection_fps': 0,
                'gpu_used': False
            }
        }
        
    except Exception as e:
        print(f"Fallback analysis error for {video_id}: {e}")
        return None

# Use fallback if YOLO is not available
if not YOLO_AVAILABLE:
    analyze_video_comprehensive = analyze_video_comprehensive_fallback

# Export main functions
__all__ = [
    'detect_scenes',
    'analyze_video_comprehensive',
    'detect_gender_in_video',
    'count_faces_simple',
    'detect_smile_in_video',
    'analyze_motion_in_video',
    'analyze_color_in_video',
    'get_video_info'
]
