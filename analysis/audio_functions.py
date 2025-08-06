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

def analyze_audio_comprehensive(audio_path, video_id):
    """
    Comprehensive audio analysis extracting Arousal, Valence, and Pitch.
    All values normalized to 0-1 range.
    Returns dict with features or None values if extraction fails.
    """
    try:
        if not os.path.exists(audio_path):
            return {
                'Arousal': None,
                'Valence': None, 
                'Pitch': None
            }
        
        # Extract features using primary methods with fallbacks
        arousal = extract_arousal(audio_path)
        valence = extract_valence(audio_path)
        pitch = extract_pitch_gpu(audio_path)
        
        return {
            'Arousal': arousal,
            'Valence': valence,
            'Pitch': pitch
        }
        
    except Exception as e:
        return {
            'Arousal': None,
            'Valence': None,
            'Pitch': None
        }

def extract_arousal(audio_path):
    """Extract arousal (energy/activation) normalized to 0-1."""
    try:
        # Try OpenSMILE first
        try:
            import opensmile
            smile = opensmile.Smile(
                feature_set=opensmile.FeatureSet.ComParE_2016,
                feature_level=opensmile.FeatureLevel.Functionals,
            )
            features = smile.process_file(audio_path)
            
            # Extract arousal-related features
            arousal_features = []
            for col in features.columns:
                if any(keyword in col.lower() for keyword in ['energy', 'loudness', 'intensity', 'rms']):
                    arousal_features.append(features[col].iloc[0])
            
            if arousal_features:
                arousal_raw = sum(arousal_features) / len(arousal_features)
                # Normalize to 0-1 (assuming typical range)
                return min(max(arousal_raw / 100.0, 0.0), 1.0)
        except ImportError:
            pass
        
        # Fallback to librosa
        import librosa
        import numpy as np
        
        y, sr = librosa.load(audio_path, sr=None)
        
        # Combine multiple energy indicators
        rms_energy = librosa.feature.rms(y=y)[0]
        spectral_rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)[0]
        tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
        
        # Normalize components
        energy_score = np.mean(rms_energy) * 10  # Scale RMS
        rolloff_score = np.mean(spectral_rolloff) / sr * 2  # Normalize by sample rate
        tempo_score = tempo / 200.0  # Normalize tempo (typical range 60-200 BPM)
        
        # Combine scores
        arousal = (energy_score + rolloff_score + tempo_score) / 3
        return min(max(arousal, 0.0), 1.0)
        
    except Exception as e:
        return None

def extract_valence(audio_path):
    """Extract valence (emotional positivity) normalized to 0-1."""
    try:
        # Try OpenSMILE first
        try:
            import opensmile
            smile = opensmile.Smile(
                feature_set=opensmile.FeatureSet.ComParE_2016,
                feature_level=opensmile.FeatureLevel.Functionals,
            )
            features = smile.process_file(audio_path)
            
            # Extract valence-related features
            valence_features = []
            for col in features.columns:
                if any(keyword in col.lower() for keyword in ['mfcc', 'chroma', 'spectral_contrast']):
                    valence_features.append(features[col].iloc[0])
            
            if valence_features:
                valence_raw = sum(valence_features) / len(valence_features)
                # Normalize to 0-1
                return min(max((valence_raw + 50) / 100.0, 0.0), 1.0)
        except ImportError:
            pass
        
        # Fallback to librosa
        import librosa
        import numpy as np
        
        y, sr = librosa.load(audio_path, sr=None)
        
        # Extract valence indicators
        chroma = librosa.feature.chroma_stft(y=y, sr=sr)
        spectral_contrast = librosa.feature.spectral_contrast(y=y, sr=sr)
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
        
        # Calculate positivity indicators
        chroma_mean = np.mean(chroma)
        contrast_mean = np.mean(spectral_contrast)
        mfcc_brightness = np.mean(mfcc[1:4])  # Higher MFCC coefficients indicate brightness
        
        # Combine indicators (higher values = more positive)
        valence = (chroma_mean * 2 + contrast_mean * 0.5 + abs(mfcc_brightness) * 0.1) / 3
        return min(max(valence, 0.0), 1.0)
        
    except Exception as e:
        return None

def extract_pitch_gpu(audio_path):
    """Extract pitch (fundamental frequency) with GPU acceleration, normalized to 0-1."""
    try:
        # Try GPU-accelerated approach first
        try:
            import torch
            import torchaudio
            import numpy as np
            
            # Use existing device detection
            try:
                from core_functions import detect_optimal_whisper_settings
                settings = detect_optimal_whisper_settings()
                device = torch.device(settings["device"])
            except:
                device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            
            # Load audio with torchaudio
            waveform, sample_rate = torchaudio.load(audio_path)
            waveform = waveform.to(device)
            
            # Extract pitch using torchaudio transforms
            pitch_transform = torchaudio.transforms.PitchShift(sample_rate, n_steps=0).to(device)
            
            # Use a simpler approach - spectral analysis for pitch
            stft = torch.stft(waveform.squeeze(), n_fft=2048, hop_length=512, return_complex=True)
            magnitude = torch.abs(stft)
            
            # Find dominant frequency (pitch proxy)
            freq_bins = torch.linspace(0, sample_rate/2, magnitude.shape[0], device=device)
            weighted_freq = torch.sum(magnitude * freq_bins.unsqueeze(1), dim=0) / torch.sum(magnitude, dim=0)
            
            # Filter out very low and high frequencies (focus on speech range)
            speech_mask = (weighted_freq > 80) & (weighted_freq < 400)
            if torch.sum(speech_mask) > 0:
                mean_pitch = torch.mean(weighted_freq[speech_mask]).cpu().item()
                # Normalize to 0-1 using typical speech range (80-400 Hz)
                normalized_pitch = (mean_pitch - 80) / (400 - 80)
                return min(max(normalized_pitch, 0.0), 1.0)
            
        except (ImportError, AttributeError):
            pass
        
        # Fallback to librosa
        import librosa
        import numpy as np
        
        y, sr = librosa.load(audio_path, sr=None)
        
        # Extract fundamental frequency
        pitches, magnitudes = librosa.piptrack(y=y, sr=sr, threshold=0.1)
        
        # Get pitch values where magnitude is highest
        pitch_values = []
        for t in range(pitches.shape[1]):
            index = magnitudes[:, t].argmax()
            pitch = pitches[index, t]
            if pitch > 0:
                pitch_values.append(pitch)
        
        if pitch_values:
            mean_pitch = np.mean(pitch_values)
            # Normalize to 0-1 using typical speech range (80-400 Hz)
            normalized_pitch = (mean_pitch - 80) / (400 - 80)
            return min(max(normalized_pitch, 0.0), 1.0)
        
        return None
        
    except Exception as e:
        return None
