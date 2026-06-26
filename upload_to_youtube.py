import json
import os
from pathlib import Path
from dotenv import load_dotenv
from openai import OpenAI

import google_auth_oauthlib.flow
import googleapiclient.discovery
import googleapiclient.errors
from googleapiclient.http import MediaFileUpload

CLIENT_SECRETS_FILE = "client_secrets.json"
DEFAULT_PLAYLIST_ID = "PLl2SOtnm4g0EoRo7rUUh8NfqKdr_wRz5L"
SCOPES = ["https://www.googleapis.com/auth/youtube"]
TOKEN_FILE = "youtube_token.json"

STANDARD_TAGS = [
    "animated short story",
    "one hour story",
    "word of the day",
    "flash fiction",
    "storytelling",
    "Drew Baderson",
    "Drew Baderson Story Moment",
]

YOUTUBE_CATEGORIES = {
    "1":  "Film & Animation",
    "2":  "Autos & Vehicles",
    "10": "Music",
    "22": "People & Blogs",
    "24": "Entertainment",
    "27": "Education",
    "28": "Science & Technology",
}


def get_authenticated_service():
    """Authenticate with YouTube API, reusing saved token if available."""
    from google.oauth2.credentials import Credentials
    from google.auth.transport.requests import Request

    creds = None
    if os.path.exists(TOKEN_FILE):
        creds = Credentials.from_authorized_user_file(TOKEN_FILE, SCOPES)

    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            flow = google_auth_oauthlib.flow.InstalledAppFlow.from_client_secrets_file(
                CLIENT_SECRETS_FILE, SCOPES
            )
            creds = flow.run_local_server(port=0)
        with open(TOKEN_FILE, "w") as f:
            f.write(creds.to_json())

    return googleapiclient.discovery.build("youtube", "v3", credentials=creds)


def generate_description_and_tags(transcript: str, episode_title: str, client: OpenAI) -> tuple:
    """Use GPT-4o to generate a YouTube description and story-specific tags."""
    print("  Generating description and tags...")

    response = client.chat.completions.create(
        model="gpt-5.5",
        messages=[{
            "role": "user",
            "content": (
                f"You are helping optimize a YouTube video upload for a short animated story.\n\n"
                f"The story is called '{episode_title}' and is part of 'The Drew Baderson Story Moment' series — "
                f"an animated short story written in one hour based on a word selected at random from the dictionary.\n\n"
                f"Please do two things:\n\n"
                f"1. Write a YouTube video description. Start with this exact fixed opening line:\n"
                f"'Welcome to The Drew Baderson Story Moment \u2014 an animated short story written in one hour based on a word selected at random from the dictionary.'\n"
                f"Then add 2-3 sentences that tease the story in an intriguing way — convey the mood, themes, and stakes "
                f"without giving away specific plot details or the ending. Think of it as a movie trailer in text form: "
                f"make the viewer curious without spoiling what happens. Follow YouTube best practices "
                f"(compelling, concise, uses natural keywords, under 200 words total). "
                f"Where natural, reference 'The Drew Baderson Story Moment' again within the description body. "
                f"End the description with this exact line on its own line: '(Writing and voice recording done by a human. Visuals produced using AI.)'\n\n"
                f"2. Generate 10-15 story-specific tags that reflect the themes, characters, and content of this story. "
                f"Tags should be lowercase, relevant, and suitable for YouTube search.\n\n"
                f"Return a JSON object with two fields:\n"
                f"- 'description': the full video description\n"
                f"- 'tags': array of story-specific tag strings\n\n"
                f"Return only the JSON object with no preamble or markdown formatting.\n\n"
                f"Story transcript:\n{transcript}"
            )
        }],
    )

    raw = response.choices[0].message.content.strip()
    if raw.startswith("```"):
        raw = raw.split("```")[1]
        if raw.startswith("json"):
            raw = raw[4:]
        raw = raw.strip()

    data = json.loads(raw)
    return data["description"], data["tags"], response.usage


def upload_to_youtube(word_directory: str, title: str = None, category_id: str = "1",
                      privacy_status: str = "public", thumbnail_filename: str = None,
                      made_for_kids: bool = False, playlist_id: str = DEFAULT_PLAYLIST_ID,
                      upload_captions: bool = True, cost_tracker=None):
    from pipeline_helpers import log_module_start, log_module_end, log, safe_write, story_path
    log_module_start("YouTube Upload")

    load_dotenv()
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OPENAI_API_KEY not found. Check your .env file.")

    client = OpenAI(api_key=api_key)

    output_dir = story_path(word_directory)
    word_of_day = word_directory.split("_", 1)[1].capitalize()

    # Load transcript
    txt_file = output_dir / f"{word_directory}.txt"
    if not txt_file.exists():
        raise FileNotFoundError(f"Transcript not found: {txt_file}")
    with open(txt_file, "r", encoding="utf-8") as f:
        transcript = f.read().strip()

    # Video file
    video_file = output_dir / f"{word_directory}.mp4"
    if not video_file.exists():
        raise FileNotFoundError(f"Video file not found: {video_file}")

    # Default title if not provided
    if not title:
        title = f"The Drew Baderson Story Moment - {word_of_day}"

    # Default thumbnail is image_001.png
    if thumbnail_filename:
        thumbnail_path = output_dir / thumbnail_filename
    else:
        thumbnail_path = output_dir / "images" / "image_001.png"
    if not thumbnail_path.exists():
        log(f"Thumbnail not found at {thumbnail_path}. YouTube will choose one.", level="WARNING")
        thumbnail_path = None

    # --- Generate or load description and story-specific tags ---
    metadata_file = output_dir / f"{word_directory}_metadata.json"

    if metadata_file.exists():
        log("Metadata file found, loading description and tags...")
        with open(metadata_file, "r", encoding="utf-8") as f:
            metadata = json.load(f)
        description = metadata["description"]
        story_tags = metadata["tags"]
        usage = None
    else:
        description, story_tags, usage = generate_description_and_tags(transcript, word_of_day, client)
        safe_write(
            lambda: open(metadata_file, "w", encoding="utf-8").write(
                json.dumps({"description": description, "tags": story_tags}, indent=2)),
            str(metadata_file)
        )
        log(f"Metadata saved: {metadata_file}")

    if cost_tracker and usage:
        from pricing import calc_text_cost
        cost = calc_text_cost("gpt-5.5", usage.prompt_tokens, usage.completion_tokens)
        cost_tracker.add("Description/tags generation (GPT-4o)", cost,
                         f"{usage.prompt_tokens} in / {usage.completion_tokens} out tokens")

    all_tags = STANDARD_TAGS + story_tags

    # --- Authenticate ---
    log("Authenticating with YouTube...")
    youtube = get_authenticated_service()

    # --- Check if already uploaded ---
    record_file = output_dir / f"{word_directory}_youtube.json"
    if record_file.exists():
        with open(record_file, "r", encoding="utf-8") as f:
            existing = json.load(f)
        video_id = existing["video_id"]
        log(f"Video already uploaded. Video ID: {video_id} — skipping upload, retrying thumbnail only.")
    else:
        log(f"Uploading '{title}'...")
        request_body = {
            "snippet": {
                "title": title,
                "description": description,
                "tags": all_tags,
                "categoryId": category_id,
            },
            "status": {
                "privacyStatus": privacy_status,
                "selfDeclaredMadeForKids": made_for_kids,
                "containsSyntheticMedia": False,
            }
        }

        media = MediaFileUpload(
            str(video_file),
            mimetype="video/mp4",
            resumable=True,
            chunksize=1024 * 1024 * 8,
        )

        upload_request = youtube.videos().insert(
            part="snippet,status",
            body=request_body,
            media_body=media,
        )

        response = None
        while response is None:
            status, response = upload_request.next_chunk()
            if status:
                log(f"Upload progress: {int(status.progress() * 100)}%")

        video_id = response["id"]
        log(f"Upload complete! Video ID: {video_id}")
        log(f"URL: https://www.youtube.com/watch?v={video_id}")

        # Add to playlist
        if playlist_id:
            try:
                youtube.playlistItems().insert(
                    part="snippet",
                    body={
                        "snippet": {
                            "playlistId": playlist_id,
                            "resourceId": {
                                "kind": "youtube#video",
                                "videoId": video_id,
                            }
                        }
                    }
                ).execute()
                log(f"Added to playlist: {playlist_id}")
            except Exception as e:
                log(f"Could not add to playlist: {e}", level="WARNING")

        # Save upload record immediately after upload succeeds
        # so a thumbnail failure can't prevent this from being written
        record_data = {
            "video_id": video_id,
            "title": title,
            "url": f"https://www.youtube.com/watch?v={video_id}",
            "privacy_status": privacy_status,
            "tags": all_tags,
        }
        safe_write(
            lambda: open(record_file, "w", encoding="utf-8").write(json.dumps(record_data, indent=2)),
            str(record_file)
        )
        log(f"Upload record saved: {record_file}")

    # Upload thumbnail if provided
    if thumbnail_path:
        log("Preparing thumbnail for upload...")
        import io
        from PIL import Image
        # YouTube thumbnail limit is 2MB — resize to fit
        img = Image.open(str(thumbnail_path))
        img.thumbnail((1280, 720), Image.LANCZOS)
        # Convert to RGB — JPEG doesn't support alpha channels (RGBA)
        if img.mode in ("RGBA", "LA", "P"):
            img = img.convert("RGB")
        thumb_buffer = io.BytesIO()
        img.save(thumb_buffer, format="JPEG", quality=85, optimize=True)
        thumb_buffer.seek(0)
        # Write to temp file for upload
        import tempfile
        with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
            tmp.write(thumb_buffer.read())
            tmp_path = tmp.name
        log("Uploading thumbnail...")
        youtube.thumbnails().set(
            videoId=video_id,
            media_body=MediaFileUpload(tmp_path, mimetype="image/jpeg")
        ).execute()
        os.unlink(tmp_path)
        log("Thumbnail uploaded.")


    # Upload captions
    if upload_captions:
        srt_file = output_dir / f"{word_directory}.srt"
        if srt_file.exists():
            log("Uploading captions...")
            try:
                youtube.captions().insert(
                    part="snippet",
                    body={
                        "snippet": {
                            "videoId": video_id,
                            "language": "en",
                            "name": "English",
                            "isDraft": False,
                        }
                    },
                    media_body=MediaFileUpload(str(srt_file), mimetype="application/octet-stream")
                ).execute()
                log("Captions uploaded.")
            except Exception as e:
                log(f"Caption upload failed: {e}", level="WARNING")
                log("You can upload captions manually in YouTube Studio.", level="WARNING")
        else:
            log(f"SRT file not found at {srt_file} — skipping caption upload.", level="WARNING")

    log_module_end("YouTube Upload")
    print(f"\n⚠️  Remember to complete these manual steps in YouTube Studio:")
    print(f"  1. Comments are not controllable via the API — set comment settings manually")
    print(f"     (comments are currently enabled by default on all uploads)")
    print(f"  2. Review captions in YouTube Studio if needed")


if __name__ == "__main__":
    from pipeline_helpers import prompt_with_found, load_word_of_day
    directory = input("Enter word directory: ").strip()
    if not directory:
        print("No directory entered. Exiting.")
    else:
        word = load_word_of_day(directory)
        default_title = f"The Drew Baderson Story Moment - {word}"

        # --- Transcript correction first, before any other prompts ---
        from correct_transcript import correct_transcript
        from pricing import pricing_status
        from cost_tracker import CostTracker
        available, status_msg = pricing_status()
        tracker = CostTracker(directory, pricing_updated=available, pricing_warning=None if available else status_msg)
        correct_transcript(directory, cost_tracker=tracker)
        print("\n(Press Enter to confirm found values or accept defaults)")
        yt_title = prompt_with_found("Video title", default_title)
        print(f"\n  YouTube categories:")
        for cat_id, cat_name in YOUTUBE_CATEGORIES.items():
            print(f"    {cat_id}: {cat_name}")
        yt_category = prompt_with_found("Category ID", None, default="1")
        print("\n  Privacy status: 1=Public (default), 2=Unlisted, 3=Private")
        privacy_choice = prompt_with_found("Privacy", None, default="1")
        privacy_map = {"1": "public", "2": "unlisted", "3": "private"}
        yt_privacy = privacy_map.get(privacy_choice, "public")
        print(f"\n  Thumbnail: image_001.png will be used by default.")
        yt_thumbnail = input("  Enter a different thumbnail filename (or press Enter to use image_001.png): ").strip() or None
        print("\n  Is this video made for kids?")
        kids_choice = input("  Enter 'yes' or press Enter for no (default: not made for kids): ").strip().lower()
        yt_made_for_kids = kids_choice == "yes"
        print(f"\n  Playlist ID (default: '{DEFAULT_PLAYLIST_ID}')")
        yt_playlist = input("  Press Enter to use default playlist, type a different ID, or type 'none' to skip: ").strip()
        if yt_playlist.lower() == "none":
            yt_playlist = None
        elif not yt_playlist:
            yt_playlist = DEFAULT_PLAYLIST_ID
        print("\n  Captions: upload the local .srt file to YouTube?")
        captions_choice = input("  Press Enter to upload captions (default), or type 'skip' to skip: ").strip().lower()
        yt_upload_captions = captions_choice != "skip"

        from pipeline_helpers import init_run_log, save_run_log, load_activity_log_text
        from notify import notify, prompt_notify_email
        from datetime import datetime
        notify_email = prompt_notify_email()
        init_run_log(directory)
        start_time = datetime.now()

        upload_to_youtube(directory,
                          title=yt_title,
                          category_id=yt_category,
                          privacy_status=yt_privacy,
                          thumbnail_filename=yt_thumbnail,
                          made_for_kids=yt_made_for_kids,
                          playlist_id=yt_playlist,
                          upload_captions=yt_upload_captions,
                          cost_tracker=tracker)

        end_time = datetime.now()
        tracker.print_total()
        tracker.save_to_log()
        save_run_log(
            word_directory=directory,
            style="N/A",
            total_cost=tracker.total_cost,
            start_time=start_time,
            end_time=end_time,
            pricing_warning=tracker.pricing_warning,
        )
        if notify_email:
            log_text = load_activity_log_text(directory)
            notify(
                subject=f"DBSM Upload To Youtube complete: {directory}",
                body=f"Duration: {(end_time - start_time)}\nCost: ${tracker.total_cost:.4f}\n\n{log_text}",
                to_address=notify_email,
            )
