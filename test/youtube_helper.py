#!/usr/bin/env python3
"""
Simplified YouTube Download Helper
Tests and fixes YouTube download authentication issues
"""

import subprocess
import sys

def install_package(package):
    """Install a package if not available."""
    try:
        __import__(package.replace('-', '_'))
        return True
    except ImportError:
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install", package], 
                                 stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            return True
        except:
            return False

def extract_cookies():
    """Extract YouTube cookies from browser."""
    if not install_package("browser-cookie3"):
        return None
    
    try:
        import browser_cookie3
        cj = browser_cookie3.chrome(domain_name='youtube.com')
        
        cookies_file = 'youtube_cookies.txt'
        with open(cookies_file, 'w') as f:
            f.write("# Netscape HTTP Cookie File\n")
            for cookie in cj:
                if 'youtube.com' in cookie.domain:
                    domain = cookie.domain
                    domain_specified = 'TRUE' if domain.startswith('.') else 'FALSE'
                    path = cookie.path
                    secure = 'TRUE' if cookie.secure else 'FALSE'
                    expires = str(int(cookie.expires)) if cookie.expires else '0'
                    name = cookie.name
                    value = cookie.value
                    f.write(f"{domain}\t{domain_specified}\t{path}\t{secure}\t{expires}\t{name}\t{value}\n")
        
        return cookies_file
    except:
        return None

def test_download_methods(video_url):
    """Test different download methods and return the working one."""
    methods = [
        ["yt-dlp", "--cookies-from-browser", "chrome", "--simulate", video_url],
        ["yt-dlp", "--cookies-from-browser", "firefox", "--simulate", video_url],
        ["yt-dlp", "--simulate", video_url]
    ]
    
    # Try with cookies file if available
    cookies_file = extract_cookies()
    if cookies_file:
        methods.insert(0, ["yt-dlp", "--cookies", cookies_file, "--simulate", video_url])
    
    for method in methods:
        try:
            result = subprocess.run(method, capture_output=True, text=True, timeout=30)
            if result.returncode == 0:
                return method[1:3] if len(method) > 3 else []  # Return the auth method
        except:
            continue
    
    return None

def main():
    """Main function."""
    # Update yt-dlp first
    subprocess.run([sys.executable, "-m", "pip", "install", "--upgrade", "yt-dlp"], 
                  capture_output=True)
    
    # Test URLs
    test_urls = [
        "https://www.youtube.com/watch?v=0nJG5axbhk4",
        "https://www.youtube.com/watch?v=9NfWmOrSsmI"
    ]
    
    for url in test_urls:
        working_method = test_download_methods(url)
        if working_method:
            print(f"Working method: {' '.join(working_method)}")
            return working_method
    
    print("No working method found")
    return None

if __name__ == "__main__":
    main()
