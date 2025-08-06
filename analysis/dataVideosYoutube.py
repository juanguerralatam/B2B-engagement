import os
import csv
from datetime import datetime
import googleapiclient.discovery
import isodate

# Get API key from environment variable or default
API_KEY = os.getenv('YOUTUBE_API_KEY', "AIzaSyAElw-2HxeBlTQAdbn647_dIP0rAF5u-d8")
MAX_RESULTS = None  # Set to None to retrieve all videos
CSV_FILENAME = "all_channels_videos.csv"
CHANNEL_CSV = "channel.csv"

def get_youtube_service():
    try:
        return googleapiclient.discovery.build("youtube", "v3", developerKey=API_KEY)
    except Exception as e:
        print(f"Error initializing YouTube service: {e}")
        return None

def get_uploads_playlist_id(youtube, channel_id):
    try:
        response = youtube.channels().list(part="contentDetails", id=channel_id).execute()
        return response["items"][0]["contentDetails"]["relatedPlaylists"]["uploads"]
    except Exception as e:
        print(f"Error fetching uploads playlist ID: {e}")
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
        print("No data to write.")
        return

    try:
        with open(filename, 'w', newline='', encoding='utf-8') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=data[0].keys())
            writer.writeheader()
            writer.writerows(data)
        print(f"CSV written: {filename} with {len(data)} videos")
    except IOError as e:
        print(f"CSV write error: {e}")

def process_all_channels(channel_ids, youtube_service, max_results_per_channel=None):
    """Process all channels and collect video data."""
    all_videos = []
    
    for i, channel_id in enumerate(channel_ids, 1):
        print(f"\nProcessing channel {i}/{len(channel_ids)}: {channel_id}")
        
        playlist_id = get_uploads_playlist_id(youtube_service, channel_id)
        if playlist_id:
            videos = get_video_metadata(youtube_service, playlist_id, max_results_per_channel)
            if videos:
                print(f"Retrieved {len(videos)} videos from channel {channel_id}")
                all_videos.extend(videos)
            else:
                print(f"No videos found for channel {channel_id}")
        else:
            print(f"Could not get playlist ID for channel {channel_id}")
    
    return all_videos

def read_channel_ids_from_csv(csv_filename):
    """Read channel IDs from a CSV file. Supports different CSV formats."""
    channel_ids = []
    
    if not os.path.exists(csv_filename):
        print(f"CSV file {csv_filename} not found.")
        return channel_ids
    
    try:
        with open(csv_filename, 'r', encoding='utf-8') as csvfile:
            # Check if file is empty
            content = csvfile.read().strip()
            if not content:
                print(f"CSV file {csv_filename} is empty.")
                return channel_ids
            
            csvfile.seek(0)  # Reset file pointer
            
            # Try to detect CSV format
            reader = csv.reader(csvfile)
            rows = list(reader)
            
            if not rows:
                print(f"No data found in {csv_filename}")
                return channel_ids
            
            # Check if first row might be a header
            first_row = rows[0]
            
            # Look for common header names for channel IDs
            header_indicators = ['channel_id', 'channelid', 'id', 'channel', 'youtube_id']
            has_header = any(cell.lower().strip() in header_indicators for cell in first_row)
            
            if has_header:
                # Find the column with channel IDs
                header_row = [cell.lower().strip() for cell in first_row]
                channel_id_col = None
                
                # Look for 'id' column specifically (your file has 'name,id' structure)
                for i, header in enumerate(header_row):
                    if header in header_indicators:
                        channel_id_col = i
                        break
                
                if channel_id_col is not None:
                    for row in rows[1:]:  # Skip header
                        if len(row) > channel_id_col and row[channel_id_col].strip():
                            channel_ids.append(row[channel_id_col].strip())
                else:
                    print("Could not find channel ID column in CSV header")
            else:
                # No header, assume first column contains channel IDs
                for row in rows:
                    if row and row[0].strip():
                        channel_ids.append(row[0].strip())
            
        print(f"Found {len(channel_ids)} channel IDs in {csv_filename}")
        for i, channel_id in enumerate(channel_ids, 1):
            print(f"  {i}. {channel_id}")
        
        return channel_ids
        
    except Exception as e:
        print(f"Error reading CSV file {csv_filename}: {e}")
        return channel_ids

if __name__ == "__main__":
    if API_KEY == "YOUR_API_KEY":
        print("Please set your API key.")
    else:
        youtube = get_youtube_service()
        if youtube:
            # Read channel IDs from CSV
            channel_ids = read_channel_ids_from_csv(CHANNEL_CSV)
            
            if not channel_ids:
                print(f"No channel IDs found in {CHANNEL_CSV}. Please add channel IDs to the CSV file.")
                exit(1)
            
            print(f"Processing {len(channel_ids)} channels...")
            
            # Process all channels
            all_videos = process_all_channels(channel_ids, youtube, MAX_RESULTS)
            
            if all_videos:
                write_to_csv(all_videos, CSV_FILENAME)
                print(f"\nTotal videos collected: {len(all_videos)}")
            else:
                print("No video metadata retrieved from any channel.")
        else:
            print("Failed to initialize YouTube service.")
