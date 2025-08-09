#!/usr/bin/env python3
"""
Text analysis functions
Narration evaluation and NLP processing with GPU acceleration
"""

import os
import json
import subprocess
import re
import torch
import numpy as np
from typing import Dict, List, Optional, Tuple
from utils import (
    load_text_file_safe, save_json_file, get_output_directory,
    ensure_directory_exists, print_status, get_timestamp, get_project_root
)

# Import technical vocabularies from external file
import sys
import json
# Add the project root to the path to import from input folder
project_root = get_project_root()
sys.path.insert(0, os.path.join(project_root, 'input'))

try:
    # Load config to get vocabulary file path
    from utils import load_config_file
    config = load_config_file()
    vocab_file = config.get("paths", {}).get("technical_vocabularies", "input/technical_vocabularies.json")
    
    # Make absolute path if relative
    if not os.path.isabs(vocab_file):
        vocab_file = os.path.join(project_root, vocab_file)
    
    if os.path.exists(vocab_file):
        with open(vocab_file, 'r') as f:
            TECHNICAL_VOCABULARIES = json.load(f)
        print_status("Loaded technical vocabularies from JSON file", "SUCCESS")
    else:
        # Try to import from Python module as fallback
        from technical_vocabularies import TECHNICAL_VOCABULARIES
except (ImportError, FileNotFoundError, json.JSONDecodeError) as e:
    # Fallback if import fails
    print_status(f"Warning: Could not load technical vocabularies ({e}), using default", "WARNING")
    TECHNICAL_VOCABULARIES = {
        'cloud_infrastructure': {
            'aws': ['ec2', 'lambda', 's3', 'vpc', 'cloudformation'],
            'generic': ['kubernetes', 'docker', 'terraform']
        }
    }

# Normalization ranges for different metrics
NORMALIZATION_RANGES = {
    'hashtags': (0, 10),
    'urls': (0, 15),
    'technicality_base': (0.0, 1.0)
}

def evaluate_narration_quality(transcript, api_key=None):
    """Evaluate narration quality using DeepSeek API."""
    try:
        import requests
        
        if not api_key:
            api_key = os.getenv('DEEPSEEK_API_KEY')
            
        if not api_key:
            print("No API key provided for narration evaluation")
            return None
        
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

Transcript to analyze:
{transcript[:3000]}{"..." if len(transcript) > 3000 else ""}
"""

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
        
        response = requests.post(url, headers=headers, json=data, timeout=30)
        
        if response.status_code == 200:
            result = response.json()
            content = result['choices'][0]['message']['content'].strip()
            
            # Clean up content if it has markdown formatting
            if content.startswith('```json'):
                content = content.replace('```json', '').replace('```', '').strip()
            
            try:
                evaluation = json.loads(content)
                return evaluation
            except json.JSONDecodeError as e:
                print(f"Error parsing API response: {e}")
                return None
        else:
            print(f"API request failed with status {response.status_code}")
            return None
            
    except requests.RequestException as e:
        print(f"Request error: {e}")
        return None
    except Exception as e:
        print(f"Evaluation error: {e}")
        return None

def analyze_text_comprehensive(video_id: str, title: str, description: str, transcript_path: str) -> Dict:
    """
    Comprehensive text analysis extracting all features with GPU acceleration.
    Returns dict with 8 text features normalized to 0-1 range.
    """
    try:
        # Load transcript
        transcript_text = load_transcript_safe(transcript_path)
        
        # Extract all features
        title_sentiment = extract_sentiment_bert(title)
        title_technicality = extract_technicality_advanced(title)
        
        description_sentiment = extract_sentiment_bert(description)
        description_technicality = extract_technicality_advanced(description)
        
        hashtag_count, url_count = count_content_elements_regex(description)
        
        script_sentiment = extract_sentiment_bert(transcript_text) if transcript_text else None
        script_technicality = extract_technicality_advanced(transcript_text) if transcript_text else None
        
        return {
            'titleSentiment': title_sentiment,
            'titleTechnicality': title_technicality,
            'descriptionSentiment': description_sentiment,
            'descriptionTechnicality': description_technicality,
            'hashtagsDescription': hashtag_count,
            'URLDescription': url_count,
            'scriptSentiment': script_sentiment,
            'scriptTechnicality': script_technicality
        }
        
    except Exception as e:
        return {
            'titleSentiment': None,
            'titleTechnicality': None,
            'descriptionSentiment': None,
            'descriptionTechnicality': None,
            'hashtagsDescription': None,
            'URLDescription': None,
            'scriptSentiment': None,
            'scriptTechnicality': None
        }

def extract_sentiment_bert(text: str) -> Optional[float]:
    """Extract sentiment using GPU-accelerated BERT model."""
    try:
        if not text or not text.strip():
            return None
            
        from transformers import pipeline
        import torch
        
        # Use existing device detection
        try:
            from core_functions import detect_optimal_whisper_settings
            settings = detect_optimal_whisper_settings()
            device = 0 if settings["device"] == "cuda" else -1
        except:
            device = 0 if torch.cuda.is_available() else -1
        
        # Initialize BERT sentiment analyzer with GPU
        sentiment_analyzer = pipeline(
            "sentiment-analysis",
            model="nlptown/bert-base-multilingual-uncased-sentiment",
            device=device
        )
        
        # Analyze sentiment
        result = sentiment_analyzer(text[:512])  # Limit text length
        
        # Extract confidence score and normalize to 0-1
        if result[0]['label'] in ['POSITIVE', '4 stars', '5 stars']:
            sentiment_score = result[0]['score']
        elif result[0]['label'] in ['NEGATIVE', '1 star', '2 stars']:
            sentiment_score = 1 - result[0]['score']
        else:  # NEUTRAL or '3 stars'
            sentiment_score = 0.5
            
        return min(max(sentiment_score, 0.0), 1.0)
        
    except Exception as e:
        print(f"BERT sentiment analysis failed: {e}")
        return None

def extract_technicality_advanced(text: str, domain: str = 'cloud') -> Optional[float]:
    """Extract technicality score using multi-factor analysis."""
    try:
        if not text or not text.strip():
            return None
            
        text_lower = text.lower()
        words = re.findall(r'\b\w+\b', text_lower)
        
        if not words:
            return 0.0
            
        # Factor 1: Domain-specific technical terms
        technical_score = calculate_domain_technical_score(text_lower)
        
        # Factor 2: Acronym density
        acronym_score = calculate_acronym_density(text)
        
        # Factor 3: Linguistic complexity
        complexity_score = calculate_linguistic_complexity(text, words)
        
        # Factor 4: Technical jargon patterns
        jargon_score = calculate_jargon_patterns(text_lower)
        
        # Weighted combination
        weights = [0.4, 0.2, 0.2, 0.2]  # Domain terms weighted highest
        scores = [technical_score, acronym_score, complexity_score, jargon_score]
        
        final_score = sum(w * s for w, s in zip(weights, scores))
        return min(max(final_score, 0.0), 1.0)
        
    except Exception as e:
        return None

def calculate_domain_technical_score(text: str) -> float:
    """Calculate technical score based on domain vocabularies."""
    total_words = len(re.findall(r'\b\w+\b', text))
    if total_words == 0:
        return 0.0
        
    technical_matches = 0
    for domain_group in TECHNICAL_VOCABULARIES.values():
        for term_category in domain_group.values():
            for term in term_category:
                # Count exact matches and partial matches
                pattern = r'\b' + re.escape(term.replace('-', r'[\s\-]?')) + r'\b'
                matches = len(re.findall(pattern, text))
                technical_matches += matches
    
    # Normalize by text length
    density = technical_matches / total_words
    return min(density * 5, 1.0)  # Scale up to reach 1.0 for highly technical content

def calculate_acronym_density(text: str) -> float:
    """Calculate density of acronyms (uppercase sequences)."""
    words = re.findall(r'\b\w+\b', text)
    if not words:
        return 0.0
        
    # Find acronyms (2+ uppercase letters)
    acronyms = re.findall(r'\b[A-Z]{2,}\b', text)
    density = len(acronyms) / len(words)
    return min(density * 10, 1.0)  # Scale to 0-1 range

def calculate_linguistic_complexity(text: str, words: List[str]) -> float:
    """Calculate linguistic complexity metrics."""
    if not words:
        return 0.0
        
    # Average word length
    avg_word_length = sum(len(word) for word in words) / len(words)
    length_score = min((avg_word_length - 3) / 7, 1.0)  # Normalize assuming 3-10 char range
    
    # Sentence complexity (average sentence length)
    sentences = re.split(r'[.!?]+', text)
    sentences = [s.strip() for s in sentences if s.strip()]
    if sentences:
        avg_sentence_length = sum(len(s.split()) for s in sentences) / len(sentences)
        sentence_score = min((avg_sentence_length - 5) / 25, 1.0)  # Normalize assuming 5-30 word range
    else:
        sentence_score = 0.0
    
    return (length_score + sentence_score) / 2

def calculate_jargon_patterns(text: str) -> float:
    """Detect technical jargon patterns."""
    # Technical patterns
    patterns = [
        r'\b\w+[-]?\w*\s+(service|platform|solution|framework|architecture)\b',
        r'\b(cloud|enterprise|scalable|distributed|microservice)\s+\w+\b',
        r'\b\w+\s+(optimization|integration|implementation|deployment)\b',
        r'\b(api|sdk|cli|ide|gui)\b',
        r'\b\w+\s+(as\s+a\s+service|driven|based|oriented)\b'
    ]
    
    total_matches = 0
    for pattern in patterns:
        matches = len(re.findall(pattern, text, re.IGNORECASE))
        total_matches += matches
    
    # Normalize by text length
    words_count = len(re.findall(r'\b\w+\b', text))
    if words_count == 0:
        return 0.0
        
    jargon_density = total_matches / words_count
    return min(jargon_density * 20, 1.0)  # Scale to 0-1 range

def count_content_elements_regex(description: str) -> Tuple[float, float]:
    """Count hashtags and URLs, normalized to 0-1 range."""
    try:
        if not description:
            return 0.0, 0.0
            
        # Count hashtags
        hashtags = re.findall(r'#\w+', description)
        hashtag_count = len(hashtags)
        
        # Count URLs
        urls = re.findall(r'https?://[^\s]+', description)
        url_count = len(urls)
        
        # Normalize to 0-1 range
        hashtag_normalized = min(hashtag_count / NORMALIZATION_RANGES['hashtags'][1], 1.0)
        url_normalized = min(url_count / NORMALIZATION_RANGES['urls'][1], 1.0)
        
        return hashtag_normalized, url_normalized
        
    except Exception as e:
        return 0.0, 0.0

def load_transcript_safe(transcript_path: str) -> Optional[str]:
    """Safely load transcript with encoding detection."""
    return load_text_file_safe(transcript_path)

def save_evaluation_report(evaluation, transcript_path, output_dir=None):
    """Save the narration evaluation report to a file."""
    try:
        if not evaluation:
            print_status("No evaluation data to save", "WARNING")
            return None
        
        if output_dir is None:
            output_dir = get_output_directory("json")
            
        ensure_directory_exists(output_dir)
        
        base_name = os.path.splitext(os.path.basename(transcript_path))[0]
        report_path = os.path.join(output_dir, f"{base_name}_evaluation.json")
        
        report_data = {
            "timestamp": get_timestamp(),
            "transcript_file": transcript_path,
            "evaluation": evaluation
        }
        
        if save_json_file(report_data, report_path, ensure_dir=False):
            print_status(f"Evaluation report saved to: {report_path}", "SUCCESS")
            return report_path
        else:
            return None
        
    except Exception as e:
        print_status(f"Error saving evaluation report: {e}", "ERROR")
        return None
