#!/usr/bin/env python3
"""
YouTube Cookie Extractor and Downloader
Helps solve authentication issues with YouTube downloads
"""

import subprocess
import sys
import os
import json

def install_browser_cookie3():
    """Install browser-cookie3 for cookie extraction."""
    try:
        import browser_cookie3
        print("✅ browser-cookie3 already installed")
        return True
    except ImportError:
        print("📦 Installing browser-cookie3...")
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install", "browser-cookie3"])
            print("✅ browser-cookie3 installed successfully")
            return True
        except:
            print("❌ Failed to install browser-cookie3")
            return False

def extract_youtube_cookies():
    """Extract YouTube cookies from Chrome."""
    try:
        import browser_cookie3
        
        print("🍪 Extracting YouTube cookies from Chrome...")
        
        # Get cookies for YouTube
        cj = browser_cookie3.chrome(domain_name='youtube.com')
        
        # Save cookies to file
        cookies_file = 'youtube_cookies.txt'
        
        # Convert to Netscape format
        with open(cookies_file, 'w') as f:
            f.write("# Netscape HTTP Cookie File\n")
            f.write("# This is a generated file!  Do not edit.\n\n")
            
            for cookie in cj:
                if 'youtube.com' in cookie.domain:
                    # Netscape format: domain, domain_specified, path, secure, expires, name, value
                    domain = cookie.domain
                    domain_specified = 'TRUE' if domain.startswith('.') else 'FALSE'
                    path = cookie.path
                    secure = 'TRUE' if cookie.secure else 'FALSE'
                    expires = str(int(cookie.expires)) if cookie.expires else '0'
                    name = cookie.name
                    value = cookie.value
                    
                    f.write(f"{domain}\t{domain_specified}\t{path}\t{secure}\t{expires}\t{name}\t{value}\n")
        
        print(f"✅ Cookies saved to {cookies_file}")
        return cookies_file
        
    except Exception as e:
        print(f"❌ Failed to extract cookies: {e}")
        return None

def test_download_with_cookies(video_url, cookies_file):
    """Test download with extracted cookies."""
    try:
        print(f"🧪 Testing download with cookies...")
        
        cmd = ['yt-dlp', '--cookies', cookies_file, '--simulate', video_url]
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        if result.returncode == 0:
            print("✅ Cookie authentication successful!")
            return True
        else:
            print(f"❌ Still failed: {result.stderr}")
            return False
            
    except Exception as e:
        print(f"❌ Test failed: {e}")
        return False

def main():
    print("🔧 YouTube Authentication Fixer")
    print("=" * 50)
    
    # Step 1: Install required package
    if not install_browser_cookie3():
        print("\n💡 Manual solution:")
        print("1. Install Chrome extension 'Get cookies.txt LOCALLY'")
        print("2. Go to youtube.com")
        print("3. Export cookies to 'youtube_cookies.txt'")
        print("4. Use: yt-dlp --cookies youtube_cookies.txt [URL]")
        return
    
    # Step 2: Extract cookies
    cookies_file = extract_youtube_cookies()
    if not cookies_file:
        print("❌ Cookie extraction failed")
        return
    
    # Step 3: Test with a video
    test_urls = [
        "https://www.youtube.com/watch?v=0nJG5axbhk4",
        "https://www.youtube.com/watch?v=9NfWmOrSsmI"
    ]
    
    for url in test_urls:
        print(f"\n🎯 Testing: {url}")
        if test_download_with_cookies(url, cookies_file):
            print("✅ Authentication working!")
            break
    else:
        print("\n❌ Authentication still not working")
        print("💡 Try these manual steps:")
        print("1. Close all Chrome windows")
        print("2. Open Chrome and sign into YouTube")
        print("3. Run this script again")
        print("4. Or try Firefox: python -c \"import browser_cookie3; browser_cookie3.firefox()\"")

if __name__ == "__main__":
    main()
