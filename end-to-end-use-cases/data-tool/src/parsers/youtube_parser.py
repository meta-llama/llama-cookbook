# Uses youtube_transcript_api and yt-dlp
import os
import re
import json
import subprocess
from datetime import datetime
from youtube_transcript_api import YouTubeTranscriptApi
from youtube_transcript_api.formatters import TextFormatter

class YouTubeParser:
    def __init__(self):
        self._check_dependencies()
            
    def _check_dependencies(self):
        try:
            subprocess.run(['yt-dlp', '--version'], capture_output=True, check=True)
        except (subprocess.SubprocessError, FileNotFoundError):
            print("WARNING: yt-dlp not found! Install with: pip install yt-dlp")
            
    def parse(self, url):
        if not self._is_valid_youtube_url(url):
            raise ValueError(f"That doesn't look like a YouTube URL: {url}")
        #meta-data
        vid_info = self._get_video_info(url)
        
        if not vid_info:
            return f"URL: {url}\n[Couldn't get video info - video private or unavailable?]"
        
        output = self._format_metadata(vid_info, url)
        
        # grab transcript
        video_id = self._extract_video_id(url)
        transcript = self._get_transcript(video_id)

        return output + transcript
    
    def _extract_video_id(self, url):
        if 'youtu.be/' in url:
            video_id = url.split('/')[-1].split('?')[0]
        elif 'v=' in url:
            video_id = url.split('v=')[1].split('&')[0]
        elif 'shorts/' in url: #lol since OG logic doesnt work on shorts-just incase someone wants to FT a LLM on shorts :)
            video_id = url.split('shorts/')[-1].split('?')[0]
        else:
            match = re.search(r'(?:embed|v|shorts)/([a-zA-Z0-9_-]{11})', url)
            if match:
                video_id = match.group(1)
            else:
                video_id = url.split('/')[-1].split('?')[0]
                if len(video_id) != 11:  # YouTube IDs are usually 11 chars
                    raise ValueError(f"Could not extract video ID from URL: {url}")
        
        return video_id
    
    def _get_video_info(self, url):
        try:
            cmd = ['yt-dlp', '--dump-json', '--no-playlist', '--skip-download', url]
            result = subprocess.run(cmd, capture_output=True, text=True)
            if result.returncode != 0:
                return None
            return json.loads(result.stdout)
        except Exception as e:
            print(f"Oops! Couldn't get video info: {e}")
            return None
    
    def _format_metadata(self, vid_info, url):
        output = "# YOUTUBE VIDEO TRANSCRIPT\n\n"
        output += f"Title: {vid_info.get('title', '?')}\n"
        output += f"Channel: {vid_info.get('channel', vid_info.get('uploader', '?'))}\n"
        output += f"URL: {url}\n"
        if 'description' in vid_info and vid_info['description']:
            desc = vid_info['description']
            if len(desc) > 512:
                desc = desc[:512] + "..."
            output += f"Description: {desc}\n"

        if 'duration' in vid_info:
            secs = int(vid_info['duration'])
            mins, secs = divmod(secs, 60)
            hrs, mins = divmod(mins, 60)
            
            if hrs > 0:
                duration = f"{hrs}:{mins:02d}:{secs:02d}"
            else:
                duration = f"{mins}:{secs:02d}"
                
            output += f"Duration: {duration}\n"

        if 'view_count' in vid_info:
            views = vid_info['view_count']
            output += f"Views: {views:,}\n"

        if 'upload_date' in vid_info:
            raw_date = vid_info['upload_date']
            try:
                date_obj = datetime.strptime(raw_date, "%Y%m%d")
                formatted_date = date_obj.strftime("%B %d, %Y")
                output += f"Upload Date: {formatted_date}\n"
            except:
                # Fallback if date parsing fails
                output += f"Upload Date: {raw_date}\n"
            
        output += "\n----- TRANSCRIPT -----\n\n"
        return output
    
    def _get_transcript(self, video_id):
        try:
            transcript_list = YouTubeTranscriptApi.list_transcripts(video_id)
            transcript = None
            try:
                transcript = transcript_list.find_transcript(['en'])
            except:
                try:
                    transcript = transcript_list.find_manually_created_transcript(transcript_list.transcript_data)
                except:
                    try:
                        transcript = transcript_list.find_generated_transcript(transcript_list.transcript_data)
                    except:
                        pass
            if not transcript:
                try:
                    transcript = list(transcript_list)[0]
                except:
                    return "[No transcript available for this video]"
                
            transcript_data = transcript.fetch()
            formatter = TextFormatter()
            formatted_transcript = formatter.format_transcript(transcript_data)
            transcript_type = "MANUALLY CREATED" if transcript.is_generated is False else "AUTO"
            lang = transcript.language
            note = f"[Using {transcript_type} captions in language: {lang}]\n\n"
            
            return note + formatted_transcript
            
        except Exception as e:
            return f"[Error retrieving transcript: {str(e)}]"
    
    def _is_valid_youtube_url(self, url):
        patterns = [
            r'youtube\.com/watch\?v=[\w-]+',
            r'youtu\.be/[\w-]+',
            r'youtube\.com/shorts/[\w-]+'
        ]
        
        for pattern in patterns:
            if re.search(pattern, url):
                return True
        return False
    
    def save(self, content, output_path):
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        header = f"# Downloaded: {timestamp}\n\n"
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(header + content)
        print(f"Saved transcript to {output_path} ({len(content)} chars)")
        return output_path