#!/usr/bin/env python3
"""
Optimized Video Download Utility for B2B Engagement Analysis

Fast and simple video downloader using yt-dlp with cookie-based authentication.
Includes bot detection avoidance and optimized performance settings.
"""

import os
import time
import random
import logging
from pathlib import Path
from typing import Optional, List, Dict, Any

import yt_dlp

# Import core utilities
from .core import (
    print_status, print_progress, ensure_directory_exists, 
    get_output_directory, get_project_root
)

# Setup logging
logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
log = logging.getLogger(__name__)


class VideoDownloader:
    """
    Optimized video downloader with bot detection avoidance.
    """
    
    def __init__(self, output_dir: Optional[str] = None):
        """Initialize the video downloader."""
        self.output_dir = Path(output_dir or get_output_directory("video"))
        ensure_directory_exists(str(self.output_dir))
        
        # Optimized settings for bot avoidance
        self.cpu_count = os.cpu_count() or 4
        self.concurrent_fragments = min(8, self.cpu_count)  # Keep low to avoid detection
        
        # Stats
        self.stats = {
            'successful': 0,
            'failed': 0,
            'already_exists': 0,
            'start_time': None
        }
        
        # Setup cookies and yt-dlp options
        self.cookies_file = self._get_cookies_file()
        self.ydl_opts = self._build_ydl_options()
    
    def _get_cookies_file(self) -> Optional[str]:
        """Get cookies file from main folder."""
        cookies_path = os.path.join(get_project_root(), "cookies.txt")
        
        if os.path.exists(cookies_path):
            log.info(f"Using cookies from: {cookies_path}")
            return cookies_path
        else:
            log.warning("No cookies.txt found - place cookies.txt in project root")
            print_status("No cookies.txt found - export with: yt-dlp --cookies-from-browser chrome --cookies cookies.txt", "WARNING")
            return None
    
    def _build_ydl_options(self) -> Dict[str, Any]:
        """Build yt-dlp options with bot detection avoidance."""
        opts = {
            "outtmpl": str(self.output_dir / "%(id)s.%(ext)s"),
            "format_sort": ["+size", "+br", "+res", "+fps"],  # Prefer smallest filesize
            
            # Bot detection avoidance
            "concurrent_fragments": self.concurrent_fragments,  # Keep low
            "sleep_interval": 2,  # Fixed 2 second delay between requests
            "max_sleep_interval": 4,  # Maximum random delay
            
            # Player client fallback order
            "extractor_args": {
                "youtube": {
                    "player_client": ["android", "web"],  # Android first, then web
                    "player_skip": ["webpage"],
                }
            },
            
            # Retry settings
            "retries": 3,
            "fragment_retries": 5,
            "quiet": True,
            "no_warnings": True,
        }
        
        # Cookie handling
        if self.cookies_file:
            opts["cookiefile"] = self.cookies_file
            log.info("Configured with cookies for bot detection avoidance")
        else:
            log.warning("No cookies - may encounter bot detection")
        
        return opts
    
    def _check_existing_video(self, video_id: str) -> Optional[str]:
        """Check if video exists."""
        extensions = ['.mp4', '.mkv', '.webm', '.avi', '.mov']
        
        for ext in extensions:
            video_path = self.output_dir / f"{video_id}{ext}"
            if video_path.exists() and video_path.stat().st_size > 1000:
                return str(video_path)
        
        return None
    
    def download_video(self, video_url: str, video_id: str, 
                      force_redownload: bool = False) -> Optional[str]:
        """Download a single video with bot detection avoidance."""
        if not self.stats['start_time']:
            self.stats['start_time'] = time.time()
        
        # Check existing
        if not force_redownload:
            existing_path = self._check_existing_video(video_id)
            if existing_path:
                self.stats['already_exists'] += 1
                log.info(f"Video {video_id} already exists")
                return existing_path
        
        log.info(f"Downloading {video_id}...")
        print_status(f"Downloading {video_id}...", "INFO")
        
        try:
            opts = self.ydl_opts.copy()
            opts["outtmpl"] = str(self.output_dir / f"{video_id}.%(ext)s")
            
            with yt_dlp.YoutubeDL(opts) as ydl:
                ydl.download([video_url])
                
                # Check if downloaded
                downloaded_file = self._check_existing_video(video_id)
                if downloaded_file:
                    self.stats['successful'] += 1
                    log.info(f"Downloaded: {video_id}")
                    print_status(f"Downloaded: {video_id}", "SUCCESS")
                    
                    # Small delay between downloads to avoid detection
                    time.sleep(random.uniform(0.5, 1.5))
                    return downloaded_file
                else:
                    self.stats['failed'] += 1
                    log.error(f"Failed: {video_id} - file not found after download")
                    print_status(f"Failed: {video_id}", "ERROR")
                    return None
                    
        except Exception as e:
            self.stats['failed'] += 1
            error_msg = str(e)[:100]
            log.error(f"Error downloading {video_id}: {error_msg}")
            print_status(f"Error downloading {video_id}: {error_msg}", "ERROR")
            return None
    
    def download_batch(self, video_list: List[Dict[str, str]], 
                      force_redownload: bool = False,
                      batch_size: int = 50) -> Dict[str, Any]:
        """Download multiple videos with batch processing and bot avoidance."""
        total_videos = len(video_list)
        log.info(f"Starting batch download of {total_videos} videos")
        print_status(f"Downloading {total_videos} videos in batches of {batch_size}", "INFO")
        
        results = {'successful': [], 'failed': [], 'already_existed': []}
        
        # Process in batches
        for batch_start in range(0, total_videos, batch_size):
            batch_end = min(batch_start + batch_size, total_videos)
            batch = video_list[batch_start:batch_end]
            batch_num = (batch_start // batch_size) + 1
            total_batches = ((total_videos - 1) // batch_size) + 1
            
            log.info(f"=== Processing batch {batch_num}/{total_batches} ({len(batch)} videos) ===")
            print_status(f"Batch {batch_num}/{total_batches} ({len(batch)} videos)", "INFO")
            
            batch_successful = 0
            batch_failed = 0
            
            for i, video_info in enumerate(batch, 1):
                video_id = video_info.get('video_id') or video_info.get('videoId')
                video_url = video_info.get('video_url') or video_info.get('url')
                
                if not video_id or not video_url:
                    log.error(f"Invalid video info: {video_info}")
                    results['failed'].append(video_info)
                    batch_failed += 1
                    continue
                
                global_index = batch_start + i
                print_progress(global_index, total_videos, f"Processing {video_id}")
                
                # Check existing first
                if not force_redownload:
                    existing_path = self._check_existing_video(video_id)
                    if existing_path:
                        results['already_existed'].append({
                            'video_id': video_id,
                            'path': existing_path
                        })
                        continue
                
                # Download
                downloaded_path = self.download_video(video_url, video_id, force_redownload)
                
                if downloaded_path:
                    results['successful'].append({
                        'video_id': video_id,
                        'path': downloaded_path
                    })
                    batch_successful += 1
                else:
                    results['failed'].append({
                        'video_id': video_id,
                        'url': video_url
                    })
                    batch_failed += 1
            
            # Batch summary with timing
            elapsed = time.time() - self.stats['start_time'] if self.stats['start_time'] else 0
            total_processed = self.stats['successful'] + self.stats['failed']
            avg_time = elapsed / max(total_processed, 1)
            
            log.info(f"Batch {batch_num} complete: {batch_successful} success, {batch_failed} failed (avg {avg_time:.1f}s/video)")
            print_status(f"Batch {batch_num}: {batch_successful} success, {batch_failed} failed", "INFO")
        
        self._print_summary(results)
        return results
    
    def _print_summary(self, results: Dict[str, Any]) -> None:
        """Enhanced summary with detailed statistics."""
        total = len(results['successful']) + len(results['failed']) + len(results['already_existed'])
        
        # Calculate timing statistics
        elapsed = time.time() - self.stats['start_time'] if self.stats['start_time'] else 0
        total_processed = self.stats['successful'] + self.stats['failed']
        avg_time = elapsed / max(total_processed, 1)
        
        # Log summary
        log.info("=== DOWNLOAD COMPLETE ===")
        log.info(f"Total: {self.stats['successful']} success, {self.stats['failed']} failed")
        log.info(f"Time: {elapsed:.1f}s (avg {avg_time:.1f}s/video)")
        
        # Print summary
        print_status("=" * 50, "INFO")
        print_status("DOWNLOAD SUMMARY", "INFO")
        print_status("=" * 50, "INFO")
        print_status(f"Successfully downloaded: {len(results['successful'])}", "SUCCESS")
        print_status(f"Already existed: {len(results['already_existed'])}", "INFO")
        print_status(f"Failed: {len(results['failed'])}", "ERROR")
        print_status(f"Total time: {elapsed/60:.1f} minutes", "INFO")
        print_status(f"Average per video: {avg_time:.1f} seconds", "INFO")
        print_status(f"Concurrent fragments: {self.concurrent_fragments}", "INFO")
        print_status("=" * 50, "INFO")
        
        # Show failed downloads
        if results['failed']:
            print_status("Failed downloads:", "ERROR")
            for failed in results['failed'][:5]:
                print_status(f"  - {failed.get('video_id', 'Unknown ID')}", "ERROR")
            if len(results['failed']) > 5:
                print_status(f"  ... and {len(results['failed']) - 5} more", "ERROR")


# Simple convenience function
def download_video(video_url: str, video_id: str, output_dir: Optional[str] = None) -> Optional[str]:
    """Download a single video."""
    downloader = VideoDownloader(output_dir=output_dir)
    return downloader.download_video(video_url, video_id)


def download_videos_from_csv(csv_file: str, output_dir: Optional[str] = None,
                           force_redownload: bool = False) -> Dict[str, Any]:
    """Download videos from CSV."""
    import csv as csv_module
    
    video_list = []
    
    try:
        with open(csv_file, 'r', encoding='utf-8') as f:
            reader = csv_module.DictReader(f)
            for row in reader:
                if 'videoId' in row and ('video_url' in row or 'url' in row):
                    video_list.append({
                        'video_id': row['videoId'],
                        'video_url': row.get('video_url') or row.get('url')
                    })
    except Exception as e:
        print_status(f"Error reading CSV: {e}", "ERROR")
        return {'successful': [], 'failed': [], 'already_existed': []}
    
    if not video_list:
        print_status("No videos found in CSV", "ERROR")
        return {'successful': [], 'failed': [], 'already_existed': []}
    
    downloader = VideoDownloader(output_dir=output_dir)
    return downloader.download_batch(video_list, force_redownload=force_redownload)



