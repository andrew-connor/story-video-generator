import os
import re
import json
from pathlib import Path
from dotenv import load_dotenv
from openai import OpenAI
try:
    from moviepy.editor import ImageClip, AudioFileClip, CompositeVideoClip
    MOVIEPY_V2 = False
except ImportError:
    from moviepy import ImageClip, AudioFileClip, CompositeVideoClip
    import moviepy.video.fx as vfx
    MOVIEPY_V2 = True
from pipeline_helpers import log, log_module_start, log_module_end, story_path


def parse_srt(srt_path: Path) -> list:
    """Parse SRT file into list of {index, start, end, text} dicts."""
    segments = []
    content = srt_path.read_text(encoding="utf-8").strip()
    blocks = re.split(r"\n\n+", content)
    for block in blocks:
        lines = block.strip().split("\n")
        if len(lines) < 3:
            continue
        index = int(lines[0].strip())
        times = lines[1].strip()
        text = " ".join(lines[2:]).strip()
        start, end = [srt_time_to_seconds(t) for t in times.split(" --> ")]
        segments.append({"index": index, "start": start, "end": end, "text": text})
    return segments


def srt_time_to_seconds(t: str) -> float:
    """Convert SRT timestamp HH:MM:SS,mmm to seconds."""
    t = t.strip().replace(",", ".")
    h, m, s = t.split(":")
    return int(h) * 3600 + int(m) * 60 + float(s)


def find_phrase_timestamp(srt_segments: list, phrase: str) -> float:
    """Find the start timestamp of the segment containing a phrase."""
    phrase = phrase.lower()
    for segment in srt_segments:
        if phrase in segment["text"].lower():
            return segment["start"]
    return None


def find_the_end_timestamp(srt_segments: list, end_anchor_phrase: str = None, min_timestamp: float = 0.0) -> float:
    """
    Find the timestamp for the end card.
    If end_anchor_phrase is provided, search for it in the SRT.
    min_timestamp: ignore any match before this time (prevents false matches in story dialogue).
    Otherwise, find the end of the last non-repeated spoken segment —
    ignoring any trailing duplicates that Whisper sometimes adds.
    """
    if end_anchor_phrase:
        phrase = end_anchor_phrase.lower().strip()
        for segment in srt_segments:
            if phrase in segment["text"].lower() and segment["start"] > min_timestamp:
                return segment["end"]
        print(f"  WARNING: Anchor phrase '{end_anchor_phrase}' not found in SRT after {min_timestamp:.1f}s. Falling back to end of spoken text.")

    # Find the last segment whose text differs from the one before it,
    # ignoring Whisper hallucinations like URLs at the end of recordings.
    import re as _re
    def _is_hallucinated(text: str) -> bool:
        t = text.strip().lower()
        return bool(_re.search(r'https?://|www[.]', t))

    last_unique = srt_segments[0]
    for i in range(1, len(srt_segments)):
        seg = srt_segments[i]
        if _is_hallucinated(seg["text"]):
            continue
        if seg["text"].strip().lower() != srt_segments[i-1]["text"].strip().lower():
            last_unique = seg

    return last_unique["end"]


def match_beats_to_srt(beats: list, srt_segments: list, client: OpenAI) -> list:
    """Use GPT-4o to match each beat to its starting SRT timestamp."""
    log("Matching beats to SRT timestamps...")

    srt_text = "\n".join([
        f"[{s['index']}] {s['start']:.2f}s: {s['text']}"
        for s in srt_segments
    ])
    beats_text = "\n".join([
        f"Beat {b['beat']}: {b['text']}"
        for b in beats
    ])

    prompt = f"""
You are matching transcript beats to SRT subtitle timestamps.

Each beat represents approximately 10 seconds of speech. Each SRT segment is a shorter chunk with a precise timestamp.

For each beat, find the SRT segment where that beat begins and return its start time in seconds.

SRT segments:
{srt_text}

Beats:
{beats_text}

Return a JSON array where each object has:
- "beat": the beat number
- "start_time": the start time in seconds (float) of the first SRT segment that corresponds to this beat

The start times must be in strictly ascending order. Double-check your work before returning.

Return only the JSON array with no preamble or markdown formatting.
"""

    response = client.chat.completions.create(
        model="gpt-5.5",
        messages=[{"role": "user", "content": prompt}],
    )

    raw = response.choices[0].message.content.strip()
    if raw.startswith("```"):
        raw = raw.split("```")[1]
        if raw.startswith("json"):
            raw = raw[4:]
        raw = raw.strip()

    timings = json.loads(raw)

    # Enforce strictly ascending order as a safety net
    for i in range(1, len(timings)):
        if timings[i]["start_time"] <= timings[i-1]["start_time"]:
            timings[i]["start_time"] = timings[i-1]["start_time"] + 1.0

    return timings, response.usage


def make_clip(img_path, start_time, peak_time, end_time, fade_in_from_black=False,
              fade_in_override=None, fade_out_override=None):
    """
    Create an ImageClip positioned at start_time in the timeline.
    - peak_time: when the image is fully visible
    - fade_in_override: force a specific fade in duration
    - fade_out_override: force a specific fade out duration
    """
    duration = end_time - start_time
    fade_in_dur = fade_in_override if fade_in_override is not None else max(peak_time - 1.5 - start_time, 0.1)
    fade_out_dur = fade_out_override if fade_out_override is not None else max(end_time - (peak_time + 1.5), 0.1)
    duration = max(duration, fade_in_dur + 0.1 + fade_out_dur)

    if MOVIEPY_V2:
        clip = ImageClip(str(img_path)).with_duration(duration).with_start(start_time)
        if fade_in_from_black:
            clip = clip.with_effects([vfx.FadeIn(fade_in_dur)])
        else:
            clip = clip.with_effects([vfx.CrossFadeIn(fade_in_dur)])
        clip = clip.with_effects([vfx.CrossFadeOut(fade_out_dur)])
    else:
        clip = ImageClip(str(img_path)).set_duration(duration).set_start(start_time)
        if fade_in_from_black:
            clip = clip.fadein(fade_in_dur)
        else:
            clip = clip.crossfadein(fade_in_dur)
        clip = clip.crossfadeout(fade_out_dur)

    return clip


def assemble_video(word_directory: str, anchor_phrase: str = "this is what i wrote", end_anchor_phrase: str = "the end", cost_tracker=None):
    log_module_start("Video Assembly")

    load_dotenv()
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OPENAI_API_KEY not found. Check your .env file.")

    client = OpenAI(api_key=api_key)

    output_dir = story_path(word_directory)
    images_dir = output_dir / "images"
    srt_path = output_dir / f"{word_directory}.srt"
    audio_path = output_dir / f"{word_directory}.wav"
    beats_file = output_dir / f"{word_directory}_beats.json"
    output_video = output_dir / f"{word_directory}.mp4"

    # --- Check for existing video before doing any work ---
    if output_video.exists():
        overwrite = input(f"\n  ⚠️  Video already exists: {output_video.name}\n  Overwrite it? Press Enter to overwrite, or type 'cancel' to exit: ").strip().lower()
        if overwrite == "cancel":
            log("Video assembly cancelled by user — existing video preserved.", level="WARNING")
            return

    # --- Load assets ---
    for path in [srt_path, audio_path, beats_file]:
        if not path.exists():
            raise FileNotFoundError(f"Required file not found: {path}")

    with open(beats_file, "r", encoding="utf-8") as f:
        data = json.load(f)
    beats = data["beats"]

    srt_segments = parse_srt(srt_path)
    audio = AudioFileClip(str(audio_path))
    total_duration = audio.duration

    log(f"Audio duration: {total_duration:.2f}s")
    log(f"Total beats: {len(beats)}")

    # --- Find intro anchor timestamp ---
    this_is_what_i_wrote_time = find_phrase_timestamp(srt_segments, anchor_phrase.lower())
    if this_is_what_i_wrote_time is None:
        print(f"WARNING: Could not find '{anchor_phrase}' in SRT. Defaulting image 001 peak to t=10.")
        this_is_what_i_wrote_time = 10.0

    print(f"'This is what I wrote' timestamp: {this_is_what_i_wrote_time:.2f}s")

    # --- Match beats to timestamps ---
    timing_file = output_dir / f"{word_directory}_timing.json"
    if timing_file.exists():
        print("Timing file already exists, loading...")
        with open(timing_file, "r", encoding="utf-8") as f:
            beat_timings = json.load(f)
        usage = None
        timing_was_cached = True
    else:
        beat_timings, usage = match_beats_to_srt(beats, srt_segments, client)
        with open(timing_file, "w", encoding="utf-8") as f:
            json.dump(beat_timings, f, indent=2)
        print(f"Timing saved: {timing_file}")
        if cost_tracker and usage:
            from pricing import calc_text_cost
            cost = calc_text_cost("gpt-5.5", usage.prompt_tokens, usage.completion_tokens)
            cost_tracker.add("SRT timestamp matching (GPT-4o)", cost, f"{usage.prompt_tokens} in / {usage.completion_tokens} out tokens")
        timing_was_cached = False

    # --- Find end timestamp ---
    # When timing is cached, use the last beat's start time as a minimum bound
    # to prevent false matches on "the end" appearing in story dialogue
    last_beat_start = max(t["start_time"] for t in beat_timings) if timing_was_cached else 0.0
    the_end_time = find_the_end_timestamp(
        srt_segments,
        end_anchor_phrase if end_anchor_phrase.strip() else None,
        min_timestamp=last_beat_start
    )
    print(f"'The End' timestamp: {the_end_time:.2f}s")

    timing_map = {t["beat"]: t["start_time"] for t in beat_timings}

    # --- Collect and sort story images numerically ---
    story_images = sorted(
        images_dir.glob("image_*.png"),
        key=lambda p: int(p.stem.split("_")[1])
    )
    the_end_path = images_dir / "the_end.png"
    credits_path = images_dir / "credits.png"

    # --- Compute peak times for each story image ---
    # Beat 1: peak at "this is what I wrote" timestamp
    # Beats 2+: peak at midpoint of their beat duration
    # Only include beats whose image files actually exist on disk —
    # this prevents extraneous timing entries from stretching the last real beat
    existing_beat_nums = set(
        int(p.stem.split("_")[1]) for p in story_images
    )
    filtered_timings = [t for t in beat_timings if t["beat"] in existing_beat_nums]
    if len(filtered_timings) < len(beat_timings):
        log(f"Timing filter: using {len(filtered_timings)} of {len(beat_timings)} beat timings (images on disk only).")

    beat_peaks = {}
    for i, timing in enumerate(filtered_timings):
        beat_num = timing["beat"]
        start = timing["start_time"]
        if i + 1 < len(filtered_timings):
            end = filtered_timings[i + 1]["start_time"]
        else:
            end = the_end_time
        midpoint = (start + end) / 2.0
        beat_peaks[beat_num] = midpoint

    # Override beat 1 peak
    beat_peaks[1] = this_is_what_i_wrote_time

    # --- Build all clip entries: (path, start, peak, end, fade_from_black) ---
    entries = []

    for img_path in story_images:
        beat_num = int(img_path.stem.split("_")[1])
        peak = beat_peaks.get(beat_num)
        if peak is None:
            continue

        # Skip beats whose peak falls at or after the end timestamp —
        # but only when timings were freshly generated, not when loaded from cache.
        # If _timing.json already existed it was validated on a prior run — trust it.
        if not timing_was_cached and beat_num != 1 and peak >= the_end_time:
            log(f"Skipping image {beat_num} — peak time ({peak:.1f}s) is at or after end timestamp ({the_end_time:.1f}s).", level="WARNING")
            continue

        # Start = peak - 1.5 - fade_in_duration
        # For beat 1: start at 0
        if beat_num == 1:
            clip_start = 0.0
        else:
            prev_peak = beat_peaks.get(beat_num - 1, peak - 10.0)
            clip_start = prev_peak + 1.5  # starts when previous image starts fading out

        last_beat_num = max(beat_peaks.keys())
        is_last_story_image = (beat_num == last_beat_num)

        # End = when next image reaches full visibility
        if is_last_story_image:
            # Last story image fades out over 1 second as The End card fades in
            clip_end = the_end_time + 1.0
        else:
            next_beat_peak = beat_peaks.get(beat_num + 1)
            if next_beat_peak is not None:
                clip_end = next_beat_peak - 1.5
            else:
                clip_end = the_end_time + 1.0

        entries.append({
            "path": img_path,
            "start": clip_start,
            "peak": peak,
            "end": clip_end,
            "fade_from_black": beat_num == 1,
            "is_last_story_image": is_last_story_image,
        })

    # --- The End image ---
    # Fades in over 1 second starting at the_end_time, overlapping with last story image
    the_end_start = the_end_time  # starts exactly when "The End" is spoken
    the_end_end = the_end_time + 1.0 + 10.0  # 1s fade in + 10s hold
    entries.append({
        "path": the_end_path,
        "start": the_end_start,
        "peak": the_end_time + 1.0,  # fully visible after 1-second fade
        "end": the_end_end,
        "fade_from_black": False,
        "the_end_card": True,
    })

    # --- Credits image ---
    credits_peak = the_end_end + 2.5  # midpoint of 5-second crossfade from The End
    credits_start = the_end_end  # starts when The End begins fading out
    credits_end = total_duration  # runs to end of audio
    entries.append({
        "path": credits_path,
        "start": credits_start,
        "peak": credits_peak,
        "end": credits_end,
        "fade_from_black": False,
    })

    # --- Build clips ---
    log("Building clips...")
    final_clips = []

    for entry in entries:
        is_the_end = entry.get("the_end_card", False)
        is_last_story = entry.get("is_last_story_image", False)

        clip = make_clip(
            img_path=entry["path"],
            start_time=entry["start"],
            peak_time=entry["peak"],
            end_time=entry["end"],
            fade_in_from_black=entry["fade_from_black"],
            fade_in_override=1.0 if is_the_end else None,
            fade_out_override=1.0 if is_last_story else None,
        )
        final_clips.append(clip)
        print(f"  {entry['path'].name}: start={entry['start']:.2f}s peak={entry['peak']:.2f}s end={entry['end']:.2f}s")

    # Apply final fadeout to credits clip
    credits_clip = final_clips[-1]
    if MOVIEPY_V2:
        credits_clip = credits_clip.with_effects([vfx.FadeOut(5.0)])
    else:
        credits_clip = credits_clip.fadeout(5.0)
    final_clips[-1] = credits_clip

    # --- Compose and export ---
    log("Compositing video...")
    video = CompositeVideoClip(final_clips, size=(1920, 1080))
    if MOVIEPY_V2:
        video = video.with_duration(total_duration)
        video = video.with_audio(audio)
    else:
        video = video.set_duration(total_duration)
        video = video.set_audio(audio)

    log(f"Exporting video to {output_video}...")
    video.write_videofile(
        str(output_video),
        fps=24,
        codec="libx264",
        audio_codec="aac",
        temp_audiofile="temp_audio.m4a",
        remove_temp=True,
        preset="slow",
        ffmpeg_params=["-crf", "18"],
    )

    log(f"Video saved: {output_video}")
    log_module_end("Video Assembly")


if __name__ == "__main__":
    from pipeline_helpers import prompt_with_found
    directory = input("Enter word directory: ").strip()
    if not directory:
        print("No directory entered. Exiting.")
    else:
        print("\n(Press Enter to confirm found values or accept defaults)")
        anchor = prompt_with_found("Anchor phrase for intro card", None, default="this is what I wrote")
        end_anchor = input("  Anchor phrase for end card (default: 'the end'): ").strip() or "the end"
        from pricing import pricing_status
        from cost_tracker import CostTracker
        from pipeline_helpers import init_run_log, save_run_log, load_activity_log_text
        from notify import notify, prompt_notify_email
        from datetime import datetime
        available, status_msg = pricing_status()
        if not available:
            print(f"  ⚠️  {status_msg}")
        tracker = CostTracker(directory, pricing_updated=available, pricing_warning=None if available else status_msg)
        notify_email = prompt_notify_email()
        init_run_log(directory)
        start_time = datetime.now()
        assemble_video(directory, anchor_phrase=anchor, end_anchor_phrase=end_anchor, cost_tracker=tracker)
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
                subject=f"DBSM Assemble Video complete: {directory}",
                body=f"Duration: {(end_time - start_time)}\nCost: ${tracker.total_cost:.4f}\n\n{log_text}",
                to_address=notify_email,
            )
