#!/usr/bin/env python3
"""
Text analysis functions
Narration evaluation and NLP processing
"""

import os
import json
import subprocess

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

def save_evaluation_report(evaluation, transcript_path, output_dir=None):
    """Save the narration evaluation report to a file."""
    try:
        if not evaluation:
            print("No evaluation data to save")
            return None
        
        if output_dir is None:
            # Get the project root and use output/json directory
            script_dir = os.path.dirname(os.path.abspath(__file__))
            project_root = os.path.dirname(script_dir)
            output_dir = os.path.join(project_root, "output", "json")
            
        os.makedirs(output_dir, exist_ok=True)
        
        base_name = os.path.splitext(os.path.basename(transcript_path))[0]
        report_path = os.path.join(output_dir, f"{base_name}_evaluation.json")
        
        report_data = {
            "timestamp": subprocess.check_output(['date', '+%Y-%m-%d %H:%M:%S']).decode().strip(),
            "transcript_file": transcript_path,
            "evaluation": evaluation
        }
        
        with open(report_path, 'w', encoding='utf-8') as f:
            json.dump(report_data, f, indent=2, ensure_ascii=False)
        
        print(f"Evaluation report saved to: {report_path}")
        return report_path
        
    except Exception as e:
        print(f"Error saving evaluation report: {e}")
        return None
