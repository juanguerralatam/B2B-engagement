import os
import csv
from datetime import datetime
import googleapiclient.discovery
import isodate
from utils import (
    read_channel_ids_from_csv, write_csv_safe, print_status, 
    print_progress, validate_video_id, load_config_file
)

# Get API key from environment variable or default
API_KEY = os.getenv('YOUTUBE_API_KEY', "AIzaSyAElw-2HxeBlTQAdbn647_dIP0rAF5u-d8")

# Load config for file paths
config = load_config_file()
MAX_RESULTS = None  # Set to None to retrieve all videos
# Use main video_statistics file instead of separate all_channels_videos
CSV_FILENAME = config.get("output_files", {}).get("video_statistics", "output/videos_statistics.csv")
CHANNEL_CSV = config.get("input_files", {}).get("channel_csv", "input/channels.csv")

def get_youtube_service():
    try:
        return googleapiclient.discovery.build("youtube", "v3", developerKey=API_KEY)
    except Exception as e:
        print_status(f"Error initializing YouTube service: {e}", "ERROR")
        return None

def get_uploads_playlist_id(youtube, channel_id):
    try:
        response = youtube.channels().list(part="contentDetails", id=channel_id).execute()
        return response["items"][0]["contentDetails"]["relatedPlaylists"]["uploads"]
    except Exception as e:
        print_status(f"Error fetching uploads playlist ID: {e}", "ERROR")
        return None

def get_video_metadata(youtube, playlist_id, max_results=None):
    videos = []
    next_page_token = None

    try:
        # Collect video IDs from playlist - get ALL videos if max_results is None
        while True:
            # If max_results is None, we want all videos, so don't limit the collection
            if max_results is not None and len(videos) >= max_results:
                break
                
            # Calculate how many more videos we need for this request
            videos_needed = 50  # YouTube API max per request
            if max_results is not None:
                videos_needed = min(50, max_results - len(videos))
            
            response = youtube.playlistItems().list(
                part="snippet",
                playlistId=playlist_id,
                maxResults=videos_needed,
                pageToken=next_page_token
            ).execute()

            items = response.get("items", [])
            if not items:
                break

            for item in items:
                videos.append({
                    "id": item["snippet"]["resourceId"]["videoId"],
                    "snippet": item["snippet"]
                })

            next_page_token = response.get("nextPageToken")
            if not next_page_token:
                break

        print(f"Collected {len(videos)} video IDs from playlist")

        # Batch video ID lookups (max 50 IDs per request)
        video_data = []
        for i in range(0, len(videos), 50):
            batch = videos[i:i+50]
            video_ids = ",".join(video["id"] for video in batch)
            details = youtube.videos().list(
                part="snippet,contentDetails,statistics",
                id=video_ids
            ).execute()

            details_map = {v["id"]: v for v in details.get("items", [])}

            for video in batch:
                vid = video["id"]
                snippet = video["snippet"]
                detail = details_map.get(vid, {})

                published = snippet.get("publishedAt")
                if published:
                    try:
                        published = datetime.fromisoformat(published.replace("Z", "+00:00")).strftime('%Y-%m-%d %H:%M:%S UTC')
                    except:
                        pass

                duration = isodate.parse_duration(detail.get("contentDetails", {}).get("duration", "PT0S"))
                duration_str = str(duration)

                video_data.append({
                    "videoId": vid,
                    "title": snippet.get("title"),
                    "description": snippet.get("description"),
                    "publishedAt": published,
                    "channelId": snippet.get("channelId"),
                    "channelTitle": snippet.get("channelTitle"),
                    "thumbnails": snippet.get("thumbnails", {}).get("high", {}).get("url"),
                    "duration": duration_str,
                    "definition": detail.get("contentDetails", {}).get("definition"),
                    "caption": detail.get("contentDetails", {}).get("caption"),
                    "licensedContent": detail.get("contentDetails", {}).get("licensedContent"),
                    "dimension": detail.get("contentDetails", {}).get("dimension"),
                    "viewCount": detail.get("statistics", {}).get("viewCount", "0"),
                    "likeCount": detail.get("statistics", {}).get("likeCount", "0"),
                    "dislikeCount": detail.get("statistics", {}).get("dislikeCount", "0"),
                    "favoriteCount": detail.get("statistics", {}).get("favoriteCount", "0")
                })

        return video_data

    except Exception as e:
        print(f"Error retrieving video metadata: {e}")
        return []


def write_to_csv(data, filename):
    if not data:
        print_status("No data to write", "WARNING")
        return

    if write_csv_safe(data, filename):
        print_status(f"CSV written: {filename} with {len(data)} videos", "SUCCESS")
    else:
        print_status(f"CSV write error for: {filename}", "ERROR")

def process_all_channels(channel_ids, youtube_service, max_results_per_channel=None):
    """Process all channels and collect video data."""
    all_videos = []
    
    for i, channel_id in enumerate(channel_ids, 1):
        print_progress(i, len(channel_ids), f"Processing channel {channel_id}")
        
        playlist_id = get_uploads_playlist_id(youtube_service, channel_id)
        if playlist_id:
            videos = get_video_metadata(youtube_service, playlist_id, max_results_per_channel)
            if videos:
                print_status(f"Retrieved {len(videos)} videos from channel {channel_id}", "SUCCESS")
                all_videos.extend(videos)
            else:
                print_status(f"No videos found for channel {channel_id}", "WARNING")
        else:
            print_status(f"Could not get playlist ID for channel {channel_id}", "ERROR")
    
    return all_videos

if __name__ == "__main__":
    if API_KEY == "YOUR_API_KEY":
        print_status("Please set your API key", "ERROR")
    else:
        youtube = get_youtube_service()
        if youtube:
            # Read channel IDs from CSV
            channel_ids = read_channel_ids_from_csv(CHANNEL_CSV)
            
            if not channel_ids:
                print_status(f"No channel IDs found in {CHANNEL_CSV}. Please add channel IDs to the CSV file", "ERROR")
                exit(1)
            
            print_status(f"Processing {len(channel_ids)} channels...", "INFO")
            
            # Process all channels
            all_videos = process_all_channels(channel_ids, youtube, MAX_RESULTS)
            
            if all_videos:
                write_to_csv(all_videos, CSV_FILENAME)
                print_status(f"Total videos collected: {len(all_videos)}", "SUCCESS")
            else:
                print_status("No video metadata retrieved from any channel", "ERROR")
        else:
            print_status("Failed to initialize YouTube service", "ERROR")
