#!/usr/bin/env python3
"""
Simple Video to Text Converter
Extracts audio from video and converts speech to text
"""

import os
import sys
import subprocess

def install_package(package):
    """Install a package if it's not already installed."""
    try:
        if package == "moviepy":
            from moviepy.editor import VideoFileClip
        elif package == "SpeechRecognition":
            import speech_recognition as sr
        else:
            __import__(package)
        print(f"✓ {package} already installed")
    except ImportError:
        print(f"📦 Installing {package}...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", package])
        print(f"✓ {package} installed successfully")

def extract_audio_ffmpeg(video_path, audio_path):
    """Extract audio using FFmpeg directly (most reliable)."""
    try:
        import subprocess
        
        print(f"🎵 Extracting audio using FFmpeg...")
        
        cmd = [
            'ffmpeg', 
            '-i', video_path,
            '-vn',  # No video
            '-acodec', 'pcm_s16le',  # PCM format
            '-ar', '16000',  # 16kHz sample rate
            '-ac', '1',  # Mono audio
            '-y',  # Overwrite output file
            audio_path
        ]
        
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        if result.returncode == 0:
            print(f"✓ Audio extracted to: {audio_path}")
            return True
        else:
            print(f"❌ FFmpeg error: {result.stderr}")
            return False
            
    except FileNotFoundError:
        print("❌ FFmpeg not found. Trying MoviePy...")
        return False
    except Exception as e:
        print(f"❌ FFmpeg error: {e}")
        return False

def extract_audio(video_path, audio_path):
    """Extract audio from video file using multiple methods."""
    # Try FFmpeg first (most reliable)
    if extract_audio_ffmpeg(video_path, audio_path):
        return True
    
    # Fallback to MoviePy
    print(f"🎵 Extracting audio using MoviePy...")
    try:
        # Force reload MoviePy in case it was just installed
        import importlib
        import sys
        if 'moviepy.editor' in sys.modules:
            importlib.reload(sys.modules['moviepy.editor'])
        
        from moviepy.editor import VideoFileClip
        
        video = VideoFileClip(video_path)
        if video.audio is None:
            print("❌ Video file has no audio track")
            video.close()
            return False
        
        video.audio.write_audiofile(audio_path, verbose=False, logger=None)
        video.close()
        
        print(f"✓ Audio extracted to: {audio_path}")
        return True
        
    except ImportError as e:
        print(f"❌ MoviePy import error: {e}")
        print("💡 Try installing MoviePy manually:")
        print("   pip install moviepy --upgrade --force-reinstall")
        return False
    except Exception as e:
        print(f"❌ Error extracting audio with MoviePy: {e}")
        print("💡 Trying alternative method...")
        return extract_audio_simple(video_path, audio_path)

def extract_audio_simple(video_path, audio_path):
    """Simple audio extraction using system commands."""
    try:
        import subprocess
        
        print("🎵 Trying simple extraction method...")
        
        # Try with basic ffmpeg command
        cmd = ['ffmpeg', '-i', video_path, '-q:a', '0', '-map', 'a', audio_path, '-y']
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        if result.returncode == 0:
            print(f"✓ Audio extracted to: {audio_path}")
            return True
        else:
            print("❌ All audio extraction methods failed")
            print("💡 Please install FFmpeg or fix MoviePy installation")
            return False
            
    except Exception as e:
        print(f"❌ Simple extraction failed: {e}")
        return False

def transcribe_audio(audio_path):
    """Transcribe audio to text using Google Speech Recognition."""
    try:
        import speech_recognition as sr
        
        print(f"🎤 Transcribing audio: {audio_path}")
        
        recognizer = sr.Recognizer()
        with sr.AudioFile(audio_path) as source:
            # Adjust for ambient noise
            recognizer.adjust_for_ambient_noise(source)
            audio_data = recognizer.record(source)
        
        # Use Google's free speech recognition
        transcript = recognizer.recognize_google(audio_data, language='en-US')
        return transcript
        
    except sr.UnknownValueError:
        print("❌ Could not understand the audio")
        return None
    except sr.RequestError as e:
        print(f"❌ Speech recognition service error: {e}")
        return None
    except Exception as e:
        print(f"❌ Error during transcription: {e}")
        return None

def main():
    """Main function."""
    print("=" * 60)
    print("🎥 SIMPLE VIDEO TO TEXT CONVERTER")
    print("=" * 60)
    
    # Install required packages
    print("Checking required packages...")
    install_package("moviepy")
    install_package("SpeechRecognition")
    
    # Check if FFmpeg is available
    try:
        subprocess.run(['ffmpeg', '-version'], capture_output=True, check=True)
        print("✓ FFmpeg is available")
    except (subprocess.CalledProcessError, FileNotFoundError):
        print("⚠️ FFmpeg not found - will use MoviePy only")
        print("💡 For better performance, install FFmpeg:")
        print("   Ubuntu/Debian: sudo apt install ffmpeg")
        print("   macOS: brew install ffmpeg")
    
    # Video file configuration
    video_path = "AI Agents for Real-Time Lead Generation： Tools, Frameworks, & LLMs.mp4"
    audio_path = "extracted_audio.wav"
    transcript_path = "transcript.txt"
    
    # Check if video file exists
    if not os.path.exists(video_path):
        print(f"\n❌ Video file not found: {video_path}")
        print("\n💡 Solutions:")
        print("1. Make sure the video file is in the current directory")
        print("2. Update the video_path in the script")
        print("3. Check the exact filename (including special characters)")
        
        # List available video files
        video_extensions = ['.mp4', '.avi', '.mkv', '.mov', '.m4v', '.wmv']
        video_files = [f for f in os.listdir('.') if any(f.lower().endswith(ext) for ext in video_extensions)]
        
        if video_files:
            print(f"\n📁 Found these video files in current directory:")
            for i, file in enumerate(video_files, 1):
                print(f"   {i}. {file}")
        else:
            print("\n📁 No video files found in current directory")
        
        return
    
    print(f"\n✓ Video file found: {video_path}")
    file_size = os.path.getsize(video_path) / (1024 * 1024)
    print(f"📁 File size: {file_size:.1f} MB")
    
    # Extract audio
    if not extract_audio(video_path, audio_path):
        return
    
    # Transcribe audio
    transcript = transcribe_audio(audio_path)
    
    if transcript:
        # Save transcript
        with open(transcript_path, 'w', encoding='utf-8') as f:
            f.write(transcript)
        
        # Display results
        print("\n" + "=" * 60)
        print("📝 TRANSCRIPT:")
        print("=" * 60)
        print(transcript)
        
        print(f"\n✓ Transcript saved to: {transcript_path}")
        
        # Statistics
        words = len(transcript.split())
        print(f"\n📊 Statistics:")
        print(f"   • Characters: {len(transcript)}")
        print(f"   • Words: {words}")
        print(f"   • Estimated reading time: {words // 200} minutes")
        
    else:
        print("\n❌ Transcription failed")
        print("\n💡 Troubleshooting:")
        print("1. Check your internet connection (required for Google Speech Recognition)")
        print("2. Ensure the video has clear, audible speech")
        print("3. Try with a shorter video clip first")
        print("4. Make sure the audio is in English")
        print("5. If MoviePy installation failed, try: pip install moviepy --upgrade")

if __name__ == "__main__":
    # First try to ensure MoviePy is properly installed
    try:
        from moviepy.editor import VideoFileClip
        print("✓ MoviePy is working")
    except ImportError:
        print("📦 MoviePy not found, installing...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "moviepy", "--upgrade"])
        try:
            from moviepy.editor import VideoFileClip
            print("✓ MoviePy installed and working")
        except ImportError as e:
            print(f"❌ MoviePy installation failed: {e}")
            print("💡 You can try installing FFmpeg instead:")
            print("   Ubuntu/Debian: sudo apt install ffmpeg")
    
    main()