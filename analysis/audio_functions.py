#!/usr/bin/env python3
"""
Audio processing functions
Audio extraction and transcription using Whisper
"""

import os
import subprocess

def extract_audio(video_path, audio_path):
    """Extract audio from video file using FFmpeg or MoviePy."""
    os.makedirs(os.path.dirname(audio_path), exist_ok=True)
    
    # Try FFmpeg first
    try:
        cmd = [
            'ffmpeg', '-i', video_path, '-vn', '-acodec', 'pcm_s16le',
            '-ar', '16000', '-ac', '1', '-y', audio_path
        ]
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode == 0:
            return True
    except:
        pass
    
    # Fallback to MoviePy
    try:
        from moviepy.editor import VideoFileClip
        video = VideoFileClip(video_path)
        if video.audio is None:
            video.close()
            return False
        video.audio.write_audiofile(audio_path, verbose=False, logger=None)
        video.close()
        return True
    except Exception as e:
        return False

def transcribe_audio(audio_path, model_size="base", device=None, fp16=None):
    """Transcribe audio to text using OpenAI Whisper on GPU."""
    try:
        import whisper
        import torch
        
        # Force GPU settings if not provided
        if device is None or fp16 is None:
            try:
                from core_functions import detect_optimal_whisper_settings
                settings = detect_optimal_whisper_settings()
                device = settings["device"]
                fp16 = settings["fp16"]
            except Exception:
                device = "cpu"
                fp16 = False
        
        model = whisper.load_model(model_size, device=device)
        result = model.transcribe(audio_path, fp16=fp16, verbose=False)
        
        transcript = result["text"].strip()
        language = result.get("language", "unknown")
        
        if transcript:
            return transcript
        else:
            return None
            
    except ImportError as e:
        raise ImportError("Required packages not installed.")
    except Exception as e:
        return None
