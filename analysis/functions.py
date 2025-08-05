#!/usr/bin/env python3
"""
Additional functions for video analysis
Contains LLM evaluation and other utility functions
"""

import os
import sys
import subprocess
import json
import requests
import csv

def install_package(package):
    """Install a package if it's not already installed."""
    try:
        __import__(package)
        print(f"✓ {package} already installed")
    except ImportError:
        print(f"📦 Installing {package}...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", package])
        print(f"✓ {package} installed successfully")

def evaluate_narration_quality(transcript, api_key=None):
    """
    Evaluate narration quality using DeepSeek API.
    Returns a score from 0 to 1 based on narrative quality.
    
    Args:
        transcript (str): The text transcript to evaluate
        api_key (str): DeepSeek API key (if None, tries to get from environment)
    
    Returns:
        dict: Contains score (0-1), explanation, and details
    """
    try:
        # Get API key from parameter or environment
        if not api_key:
            api_key = os.getenv('DEEPSEEK_API_KEY')
            
        if not api_key:
            print("❌ DeepSeek API key not found")
            print("💡 Set DEEPSEEK_API_KEY environment variable or pass api_key parameter")
            return None
        
        print("🤖 Evaluating narration quality with DeepSeek...")
        
        # Prepare the prompt for narration evaluation
        prompt = f"""
Analyze the following transcript and evaluate its narration quality. Consider these factors:
1. Clarity and coherence of speech
2. Logical flow and structure
3. Engagement and storytelling quality
4. Professional presentation
5. Educational or informational value

Please provide a score from 0 to 1 (where 1 is excellent narration) and explain your reasoning.

IMPORTANT: Return ONLY a valid JSON object with exactly this structure:
{{
  "score": 0.85,
  "explanation": "Your detailed explanation here",
  "strengths": ["Strength 1", "Strength 2", "Strength 3"],
  "improvements": ["Improvement 1", "Improvement 2", "Improvement 3"]
}}

Do not include any markdown formatting, code blocks, or additional text outside the JSON.

Transcript to analyze:
{transcript[:3000]}{"..." if len(transcript) > 3000 else ""}
"""

        # DeepSeek API endpoint
        url = "https://api.deepseek.com/v1/chat/completions"
        
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }
        
        data = {
            "model": "deepseek-chat",
            "messages": [
                {
                    "role": "system",
                    "content": "You are an expert in communication, storytelling, and presentation analysis. Provide detailed, constructive feedback on narration quality."
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            "temperature": 0,
            "max_tokens": 1000
        }
        
        print("📡 Sending request to DeepSeek API...")
        response = requests.post(url, headers=headers, json=data, timeout=30)
        
        if response.status_code == 200:
            result = response.json()
            content = result['choices'][0]['message']['content']
            
            # Try to parse JSON response
            try:
                # First try direct JSON parsing
                evaluation = json.loads(content)
                print(f"✅ Narration evaluation complete!")
                return evaluation
            except json.JSONDecodeError:
                # If direct parsing fails, try to extract JSON from markdown code blocks
                print("🔧 Extracting JSON from markdown response...")
                
                # Look for JSON content within ```json blocks
                import re
                json_match = re.search(r'```json\s*(.*?)\s*```', content, re.DOTALL)
                if json_match:
                    try:
                        json_content = json_match.group(1)
                        evaluation = json.loads(json_content)
                        print(f"✅ Narration evaluation complete!")
                        return evaluation
                    except json.JSONDecodeError:
                        print("❌ Could not parse JSON from markdown block")
                
                # If still no valid JSON, try to extract any JSON-like content
                json_match = re.search(r'\{.*\}', content, re.DOTALL)
                if json_match:
                    try:
                        json_content = json_match.group(0)
                        evaluation = json.loads(json_content)
                        print(f"✅ Narration evaluation complete!")
                        return evaluation
                    except json.JSONDecodeError:
                        print("❌ Could not parse extracted JSON content")
                
                # If all parsing fails, create a structured response from the raw content
                print("⚠️ Using fallback parsing method...")
                
                # Try to extract score using regex
                score_match = re.search(r'"score":\s*([0-9.]+)', content)
                score = float(score_match.group(1)) if score_match else 0.5
                
                # Try to extract explanation
                explanation_match = re.search(r'"explanation":\s*"([^"]*)"', content)
                explanation = explanation_match.group(1) if explanation_match else content[:500] + "..."
                
                # Try to extract strengths and improvements
                strengths = []
                improvements = []
                
                strengths_match = re.search(r'"strengths":\s*\[(.*?)\]', content, re.DOTALL)
                if strengths_match:
                    strengths_content = strengths_match.group(1)
                    strengths = [s.strip(' "') for s in strengths_content.split(',') if s.strip()]
                
                improvements_match = re.search(r'"improvements":\s*\[(.*?)\]', content, re.DOTALL)
                if improvements_match:
                    improvements_content = improvements_match.group(1)
                    improvements = [i.strip(' "') for i in improvements_content.split(',') if i.strip()]
                
                return {
                    "score": score,
                    "explanation": explanation,
                    "strengths": strengths if strengths else ["AI provided detailed feedback"],
                    "improvements": improvements if improvements else ["See explanation for detailed suggestions"]
                }
        else:
            print(f"❌ API request failed: {response.status_code}")
            print(f"Error: {response.text}")
            return None
            
    except requests.RequestException as e:
        print(f"❌ Network error: {e}")
        return None
    except Exception as e:
        print(f"❌ Error evaluating narration: {e}")
        return None

def save_evaluation_report(evaluation, transcript_path, output_dir="text"):
    """
    Save the narration evaluation report to a file.
    
    Args:
        evaluation (dict): Evaluation results from evaluate_narration_quality
        transcript_path (str): Path to the original transcript file
        output_dir (str): Directory to save the report
    """
    try:
        if not evaluation:
            print("❌ No evaluation data to save")
            return None
            
        # Ensure output directory exists
        os.makedirs(output_dir, exist_ok=True)
        
        # Create report filename
        base_name = os.path.splitext(os.path.basename(transcript_path))[0]
        report_path = os.path.join(output_dir, f"{base_name}_evaluation.json")
        
        # Add metadata to evaluation
        report_data = {
            "timestamp": subprocess.check_output(['date', '+%Y-%m-%d %H:%M:%S']).decode().strip(),
            "transcript_file": transcript_path,
            "evaluation": evaluation
        }
        
        # Save as JSON
        with open(report_path, 'w', encoding='utf-8') as f:
            json.dump(report_data, f, indent=2, ensure_ascii=False)
        
        print(f"📄 Evaluation report saved to: {report_path}")
        return report_path
        
    except Exception as e:
        print(f"❌ Error saving evaluation report: {e}")
        return None

def display_evaluation_summary(evaluation):
    """
    Display a formatted summary of the narration evaluation.
    
    Args:
        evaluation (dict): Evaluation results from evaluate_narration_quality
    """
    if not evaluation:
        print("❌ No evaluation data to display")
        return
    
    print("\n" + "=" * 60)
    print("📊 NARRATION QUALITY EVALUATION")
    print("=" * 60)
    
    # Score with visual indicator
    score = evaluation.get('score', 0)
    print(f"🎯 Overall Score: {score:.2f}/1.00")
    
    # Visual score bar
    bar_length = 20
    filled_length = int(bar_length * score)
    bar = "█" * filled_length + "░" * (bar_length - filled_length)
    print(f"📈 Quality: [{bar}] {score*100:.1f}%")
    
    # Score interpretation
    if score >= 0.8:
        interpretation = "🌟 Excellent narration quality"
    elif score >= 0.6:
        interpretation = "👍 Good narration quality"
    elif score >= 0.4:
        interpretation = "⚠️ Average narration quality"
    elif score >= 0.2:
        interpretation = "👎 Below average narration quality"
    else:
        interpretation = "❌ Poor narration quality"
    
    print(f"💬 {interpretation}")
    
    # Explanation
    if 'explanation' in evaluation:
        print(f"\n📝 Analysis:")
        print(f"   {evaluation['explanation']}")
    
    # Strengths
    if 'strengths' in evaluation and evaluation['strengths']:
        print(f"\n✅ Strengths:")
        for strength in evaluation['strengths']:
            print(f"   • {strength}")
    
    # Areas for improvement
    if 'improvements' in evaluation and evaluation['improvements']:
        print(f"\n🔧 Areas for Improvement:")
        for improvement in evaluation['improvements']:
            print(f"   • {improvement}")
    
    print("=" * 60)

def fix_existing_evaluation(evaluation_path):
    """
    Fix an existing evaluation file that has parsing issues.
    
    Args:
        evaluation_path (str): Path to the evaluation JSON file
    
    Returns:
        dict: Fixed evaluation data or None if can't be fixed
    """
    try:
        with open(evaluation_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        evaluation = data.get('evaluation', {})
        explanation = evaluation.get('explanation', '')
        
        # If explanation contains the raw API response with JSON, extract it
        if '```json' in explanation:
            import re
            json_match = re.search(r'```json\s*(.*?)\s*```', explanation, re.DOTALL)
            if json_match:
                try:
                    json_content = json_match.group(1)
                    fixed_evaluation = json.loads(json_content)
                    
                    # Update the evaluation data
                    data['evaluation'] = fixed_evaluation
                    data['timestamp'] = subprocess.check_output(['date', '+%Y-%m-%d %H:%M:%S']).decode().strip()
                    
                    # Save the fixed version
                    with open(evaluation_path, 'w', encoding='utf-8') as f:
                        json.dump(data, f, indent=2, ensure_ascii=False)
                    
                    print("✅ Fixed existing evaluation file!")
                    return fixed_evaluation
                    
                except json.JSONDecodeError:
                    print("❌ Could not parse JSON from existing evaluation")
                    return None
        
        return evaluation
        
    except Exception as e:
        print(f"❌ Error fixing evaluation file: {e}")
        return None

def get_deepseek_api_key():
    """
    Get DeepSeek API key from environment or prompt user.
    
    Returns:
        str: API key or None if not available
    """
    # Try environment variable first
    api_key = os.getenv('DEEPSEEK_API_KEY')
    
    if api_key:
        print("✓ Found DeepSeek API key in environment")
        return api_key
    
    # If not found, provide instructions
    print("❌ DeepSeek API key not found")
    print("\n💡 To use narration evaluation, you need a DeepSeek API key:")
    print("1. Visit https://platform.deepseek.com/")
    print("2. Create an account and get your API key")
    print("3. Set it as environment variable: export DEEPSEEK_API_KEY='your-key-here'")
    print("4. Or pass it directly when running the script")
    
    return None

def detect_optimal_whisper_settings():
    """Detect optimal Whisper settings based on available hardware."""
    try:
        import torch
        import psutil
        
        # Check available memory
        memory_gb = psutil.virtual_memory().total / (1024**3)
        
        # Force CPU usage due to CUDA compatibility issues
        device = "cpu"
        
        # Recommend model size based on available RAM
        if memory_gb >= 16:
            recommended_model = "large"
        elif memory_gb >= 8:
            recommended_model = "medium"
        elif memory_gb >= 4:
            recommended_model = "small"
        else:
            recommended_model = "tiny"
        
        print(f"💻 System: {memory_gb:.1f}GB RAM, using CPU")
        print(f"🎯 Recommended model: {recommended_model}")
        
        return {
            "device": device,
            "recommended_model": recommended_model,
            "fp16": False  # Always False for CPU
        }
        
    except ImportError:
        return {
            "device": "cpu",
            "recommended_model": "base",
            "fp16": False
        }

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
        print(f"❌ Audio extraction failed: {e}")
        return False

def transcribe_audio(audio_path, model_size="base"):
    """Transcribe audio to text using OpenAI Whisper."""
    try:
        import whisper
        import torch
        
        # Force CPU usage to avoid CUDA compatibility issues
        device = "cpu"
        
        print(f"🤖 Loading Whisper model: {model_size} (CPU)")
        model = whisper.load_model(model_size, device=device)
        
        print("🎤 Transcribing audio...")
        # Use CPU and disable FP16 warnings
        result = model.transcribe(audio_path, fp16=False, verbose=False)
        
        transcript = result["text"].strip()
        language = result.get("language", "unknown")
        
        if transcript:
            print(f"✅ Transcription complete ({language})")
            return transcript
        else:
            print("❌ No text transcribed")
            return None
            
    except ImportError:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "openai-whisper"])
        return transcribe_audio(audio_path, model_size)
    except Exception as e:
        print(f"❌ Transcription failed: {e}")
        return None

def download_video(video_url, video_id, output_dir="../video"):
    """Download video using yt-dlp with videoId as filename."""
    try:
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, f"{video_id}.%(ext)s")
        
        # Check for extracted cookies file
        cookies_file = "youtube_cookies.txt"
        script_dir = os.path.dirname(os.path.abspath(__file__))
        cookies_path = os.path.join(script_dir, cookies_file)
        
        # Try multiple methods to download
        download_methods = []
        
        # Method 1: Use extracted cookies if available
        if os.path.exists(cookies_path):
            download_methods.append([
                'yt-dlp', 
                '--cookies', cookies_path,
                '-f', 'best[height<=720]', 
                '-o', output_path, 
                video_url
            ])
        
        # Method 2: Use cookies from Chrome browser
        download_methods.extend([
            [
                'yt-dlp', 
                '--cookies-from-browser', 'chrome',
                '-f', 'best[height<=720]', 
                '-o', output_path, 
                video_url
            ],
            # Method 3: Basic download
            [
                'yt-dlp', 
                '-f', 'best[height<=720]', 
                '-o', output_path, 
                video_url
            ]
        ])
        
        for i, cmd in enumerate(download_methods, 1):
            print(f"📥 Trying download method {i}...")
            result = subprocess.run(cmd, capture_output=True, text=True)
            
            if result.returncode == 0:
                # Find the actual downloaded file
                video_files = [f for f in os.listdir(output_dir) if f.startswith(video_id)]
                if video_files:
                    file_path = os.path.join(output_dir, video_files[0])
                    print(f"✅ Download successful")
                    return file_path
            else:
                error_msg = result.stderr.split('ERROR:')[-1].strip() if 'ERROR:' in result.stderr else 'Unknown error'
                print(f"❌ Method {i} failed: {error_msg[:100]}...")
        
        print(f"❌ All download methods failed for {video_id}")
        print("💡 Try running: python fix_youtube_auth.py")
        return None
        
    except Exception as e:
        print(f"❌ Download failed for {video_id}: {e}")
        return None

def read_video_csv(csv_file):
    """Read video IDs and URLs from CSV file."""
    videos = []
    try:
        with open(csv_file, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                videos.append({
                    'videoId': row['videoId'],
                    'video_url': row['video_url']
                })
        return videos
    except Exception as e:
        print(f"❌ Error reading CSV: {e}")
        return []

def process_single_video(video_id, video_path, args):
    """Process a single video for transcription and evaluation."""
    audio_path = f"../audio/{video_id}.wav"
    transcript_path = f"../text/{video_id}.txt"
    
    # Stage 1: Check video file
    if not args.transcribe_only and not os.path.exists(video_path):
        print(f"❌ Video not found: {video_path}")
        return False
    
    # Stage 2: Audio extraction
    if args.download_only:
        return True
    elif args.extract_audio_only or not args.skip_audio:
        if not args.transcribe_only:
            if os.path.exists(audio_path) and not args.force:
                print(f"✓ Audio exists: {video_id}")
            else:
                print(f"🎵 Extracting audio: {video_id}")
                if not extract_audio(video_path, audio_path):
                    return False
        
        if args.extract_audio_only:
            return True
    
    # Stage 3: Transcription
    if not args.skip_audio and not os.path.exists(audio_path):
        print(f"❌ Audio not found: {audio_path}")
        return False
    
    # Load or create transcript
    transcript = None
    if os.path.exists(transcript_path) and not args.force:
        try:
            with open(transcript_path, 'r', encoding='utf-8') as f:
                transcript = f.read().strip()
            print(f"✓ Transcript loaded: {video_id}")
        except:
            transcript = None
    
    if not transcript:
        print(f"🎤 Transcribing: {video_id}")
        transcript = transcribe_audio(audio_path, args.model)
    
    if transcript:
        # Save transcript
        if not os.path.exists(transcript_path) or args.force:
            os.makedirs(os.path.dirname(transcript_path), exist_ok=True)
            with open(transcript_path, 'w', encoding='utf-8') as f:
                f.write(transcript)
        
        # Narration evaluation
        if args.narration:
            evaluation_path = transcript_path.replace('.txt', '_evaluation.json')
            evaluation = None
            
            if os.path.exists(evaluation_path) and not args.force:
                try:
                    import json
                    with open(evaluation_path, 'r', encoding='utf-8') as f:
                        report_data = json.load(f)
                        evaluation = report_data.get('evaluation')
                    
                    if evaluation and evaluation.get('strengths') == ["Response received from AI"]:
                        evaluation = fix_existing_evaluation(evaluation_path)
                except:
                    evaluation = None
            
            if not evaluation:
                print(f"🤖 Evaluating: {video_id}")
                api_key = args.api_key or get_deepseek_api_key()
                if api_key:
                    evaluation = evaluate_narration_quality(transcript, api_key)
                    if evaluation:
                        save_evaluation_report(evaluation, transcript_path)
        
        return True
    else:
        return False
