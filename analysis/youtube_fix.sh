#!/bin/bash
# YouTube Cookie Export Helper
# This script helps you manually export YouTube cookies if needed

echo "🔧 YouTube Download Troubleshooting Helper"
echo "=========================================="
echo ""

# Check if yt-dlp is installed
if ! command -v yt-dlp &> /dev/null; then
    echo "❌ yt-dlp not found. Installing..."
    pip install yt-dlp
else
    echo "✅ yt-dlp is installed"
fi

echo ""
echo "🍪 Cookie Solutions:"
echo "1. Automatic cookie extraction (recommended):"
echo "   yt-dlp --cookies-from-browser chrome [URL]"
echo ""
echo "2. Manual cookie export:"
echo "   - Install 'Get cookies.txt LOCALLY' Chrome extension"
echo "   - Visit youtube.com and export cookies to cookies.txt"
echo "   - Use: yt-dlp --cookies cookies.txt [URL]"
echo ""
echo "3. Alternative download methods:"
echo "   - Try different quality: yt-dlp -f 'best[height<=480]' [URL]"
echo "   - Use mobile format: yt-dlp -f 'best[ext=mp4]' [URL]"
echo ""

# Test with a simple video
echo "🧪 Testing download capability..."
echo "Enter a YouTube URL to test (or press Enter to skip):"
read -r test_url

if [ -n "$test_url" ]; then
    echo "Testing download with cookies from Chrome..."
    yt-dlp --cookies-from-browser chrome --simulate "$test_url"
fi

echo ""
echo "💡 Additional Tips:"
echo "- Make sure Chrome is closed when extracting cookies"
echo "- Try signing out and back into YouTube"
echo "- Clear browser cache and cookies, then sign in again"
echo "- Use incognito mode to test if it's a cookie issue"
