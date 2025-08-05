#!/usr/bin/env python3
"""
Quick YouTube Download Tester
Tests different download methods to identify the working solution
"""

import subprocess
import sys
import os

def test_download_methods(video_url):
    """Test different download methods."""
    print("🧪 Testing YouTube download methods...")
    print(f"📹 Video URL: {video_url}")
    print("=" * 60)
    
    methods = [
        {
            "name": "Chrome Cookies",
            "cmd": ["yt-dlp", "--cookies-from-browser", "chrome", "--simulate", video_url]
        },
        {
            "name": "Firefox Cookies", 
            "cmd": ["yt-dlp", "--cookies-from-browser", "firefox", "--simulate", video_url]
        },
        {
            "name": "Basic Download",
            "cmd": ["yt-dlp", "--simulate", video_url]
        },
        {
            "name": "Custom User Agent",
            "cmd": ["yt-dlp", "--user-agent", "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36", "--simulate", video_url]
        }
    ]
    
    working_methods = []
    
    for method in methods:
        print(f"\n🔍 Testing: {method['name']}")
        try:
            result = subprocess.run(method['cmd'], capture_output=True, text=True, timeout=30)
            
            if result.returncode == 0:
                print(f"✅ {method['name']} - SUCCESS")
                working_methods.append(method['name'])
            else:
                error_msg = result.stderr.split('ERROR:')[-1].strip() if 'ERROR:' in result.stderr else 'Unknown error'
                print(f"❌ {method['name']} - FAILED: {error_msg}")
                
        except subprocess.TimeoutExpired:
            print(f"⏰ {method['name']} - TIMEOUT")
        except Exception as e:
            print(f"💥 {method['name']} - ERROR: {e}")
    
    print("\n" + "=" * 60)
    print("📊 RESULTS:")
    if working_methods:
        print(f"✅ Working methods: {', '.join(working_methods)}")
        print("\n💡 Recommendation: Use the first working method in your script")
    else:
        print("❌ No methods worked. Try these solutions:")
        print("1. Update yt-dlp: pip install --upgrade yt-dlp")
        print("2. Clear browser cookies and sign in again")
        print("3. Try with a different video URL")
        print("4. Check if the video is restricted in your region")

if __name__ == "__main__":
    # Test with one of your video URLs
    test_urls = [
        "https://www.youtube.com/watch?v=0nJG5axbhk4",
        "https://www.youtube.com/watch?v=9NfWmOrSsmI"
    ]
    
    print("🎯 YouTube Download Troubleshooter")
    print("=" * 60)
    
    # Update yt-dlp first
    print("📦 Updating yt-dlp...")
    try:
        subprocess.run([sys.executable, "-m", "pip", "install", "--upgrade", "yt-dlp"], 
                      capture_output=True, check=True)
        print("✅ yt-dlp updated successfully")
    except:
        print("⚠️ Could not update yt-dlp")
    
    # Test each URL
    for url in test_urls:
        test_download_methods(url)
        print("\n" + "=" * 80)
