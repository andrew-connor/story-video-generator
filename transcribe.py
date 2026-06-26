import io
import os
import warnings
from pathlib import Path
from dotenv import load_dotenv
from openai import OpenAI

# Set ffmpeg path before importing pydub to suppress the PATH warning
os.environ["PATH"] += os.pathsep + r"C:\ffmpeg\bin"
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    from pydub import AudioSegment
    import pydub
    pydub.AudioSegment.converter = r"C:\ffmpeg\bin\ffmpeg.exe"
    pydub.AudioSegment.ffmpeg = r"C:\ffmpeg\bin\ffmpeg.exe"
    pydub.AudioSegment.ffprobe = r"C:\ffmpeg\bin\ffprobe.exe"


def format_timestamp(seconds: float) -> str:
    ms = int((seconds % 1) * 1000)
    s = int(seconds) % 60
    m = int(seconds // 60) % 60
    h = int(seconds // 3600)
    return f"{h:02}:{m:02}:{s:02},{ms:03}"


def transcribe(word_directory: str, cost_tracker=None):
    from pipeline_helpers import log_module_start, log_module_end, log, story_path
    log_module_start("Transcription")

    load_dotenv()
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OPENAI_API_KEY not found. Check your .env file.")

    client = OpenAI(api_key=api_key)

    output_dir = story_path(word_directory)
    audio_file = output_dir / f"{word_directory}.wav"

    if not audio_file.exists():
        raise FileNotFoundError(f"Audio file not found: {audio_file}")

    txt_file = output_dir / f"{word_directory}.txt"
    srt_file = output_dir / f"{word_directory}.srt"
    if txt_file.exists() and srt_file.exists():
        log("Transcript and subtitles already exist, skipping transcription.")
        log_module_end("Transcription")
        return txt_file.read_text(encoding="utf-8").strip()

    log("Preparing audio for upload...")
    audio = AudioSegment.from_wav(str(audio_file))
    duration_seconds = len(audio) / 1000.0
    compressed = io.BytesIO()
    audio.export(compressed, format="mp3", bitrate="64k")
    compressed.seek(0)
    compressed.name = f"{word_directory}.mp3"

    log(f"Transcribing {audio_file}...")
    response = client.audio.transcriptions.create(
        model="whisper-1",
        file=compressed,
        response_format="verbose_json",
    )

    txt_file = output_dir / f"{word_directory}.txt"
    with open(txt_file, "w", encoding="utf-8") as f:
        f.write(response.text.strip())
    log(f"Transcript saved: {txt_file}")

    srt_file = output_dir / f"{word_directory}.srt"
    with open(srt_file, "w", encoding="utf-8") as f:
        for i, segment in enumerate(response.segments, start=1):
            start = format_timestamp(segment.start)
            end = format_timestamp(segment.end)
            text = segment.text.strip()
            f.write(f"{i}\n{start} --> {end}\n{text}\n\n")
    log(f"Subtitles saved: {srt_file}")

    if cost_tracker:
        from pricing import calc_whisper_cost
        cost = calc_whisper_cost(duration_seconds)
        cost_tracker.add("Transcription (Whisper)", cost, f"{duration_seconds/60:.1f} min")

    log_module_end("Transcription")
    return response.text.strip()


if __name__ == "__main__":
    directory = input("Enter word directory: ").strip()
    if not directory:
        print("No directory entered. Exiting.")
    else:
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
        transcribe(directory, cost_tracker=tracker)
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
                subject=f"DBSM Transcribe complete: {directory}",
                body=f"Duration: {(end_time - start_time)}\nCost: ${tracker.total_cost:.4f}\n\n{log_text}",
                to_address=notify_email,
            )
