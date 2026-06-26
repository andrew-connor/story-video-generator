"""
Transcript correction module for the DBSM pipeline.

Compares the .docx script against the .srt and .txt transcript files,
identifies misspellings of proper nouns, made-up names, places, and
specific items, and applies corrections to both files.
"""

import json
import re
import os
from pathlib import Path
from dotenv import load_dotenv
from openai import OpenAI


def extract_docx_text(docx_path: Path) -> str:
    """Extract plain text from a .docx file using python-docx."""
    try:
        from docx import Document
    except ImportError:
        raise ImportError("python-docx is required. Run: pip install python-docx --break-system-packages")

    doc = Document(str(docx_path))
    paragraphs = [p.text for p in doc.paragraphs if p.text.strip()]
    return "\n".join(paragraphs)


def parse_srt(srt_path: Path) -> list:
    """Parse SRT file into a list of segment dicts with index, timestamp, and text."""
    content = srt_path.read_text(encoding="utf-8")
    segments = []
    blocks = re.split(r"\n\n+", content.strip())
    for block in blocks:
        lines = block.strip().splitlines()
        if len(lines) >= 3:
            idx = lines[0].strip()
            timestamp = lines[1].strip()
            text = " ".join(lines[2:]).strip()
            segments.append({"index": idx, "timestamp": timestamp, "text": text})
    return segments


def write_srt(segments: list, srt_path: Path):
    """Write segments back to SRT format."""
    lines = []
    for seg in segments:
        lines.append(seg["index"])
        lines.append(seg["timestamp"])
        lines.append(seg["text"])
        lines.append("")
    srt_path.write_text("\n".join(lines), encoding="utf-8")


def find_corrections(script_text: str, transcript_text: str, client: OpenAI) -> list:
    """
    Ask GPT-5.5 to identify proper noun misspellings in the transcript
    by comparing against the script. Returns a list of {wrong, right} dicts.
    """
    prompt = (
        "You are comparing a written script against an audio transcription of that script.\n\n"
        "The transcription was produced automatically by Whisper and may contain misspellings "
        "of proper nouns — character names, place names, made-up words, or specific invented "
        "items that appear in the story.\n\n"
        "Your job: identify any words or phrases in the TRANSCRIPT that are misspelled versions "
        "of proper nouns, names, places, or specific invented terms from the SCRIPT. "
        "Do NOT flag minor word choice differences, added or dropped filler words, or "
        "paraphrasing — only flag clear misspellings of specific named things.\n\n"
        "Return ONLY a JSON array of correction objects, with no preamble or explanation. "
        "Each object must have exactly two keys: 'wrong' (the incorrect version as it appears "
        "in the transcript) and 'right' (the correct spelling from the script). "
        "If there are no corrections needed, return an empty array: []\n\n"
        f"SCRIPT:\n{script_text}\n\n"
        f"TRANSCRIPT:\n{transcript_text}"
    )

    response = client.chat.completions.create(
        model="gpt-5.5",
        messages=[{"role": "user", "content": prompt}],
    )

    raw = response.choices[0].message.content.strip()
    # Strip markdown code fences if present
    raw = re.sub(r"^```[a-z]*\n?", "", raw)
    raw = re.sub(r"\n?```$", "", raw)

    try:
        corrections = json.loads(raw.strip())
    except json.JSONDecodeError:
        corrections = []

    return corrections, response.usage


def apply_corrections(text: str, corrections: list) -> str:
    """Apply a list of {wrong, right} corrections to a text string."""
    for c in corrections:
        wrong = c["wrong"]
        right = c["right"]
        # Case-insensitive replacement preserving original case pattern
        text = re.sub(re.escape(wrong), right, text, flags=re.IGNORECASE)
    return text


def correct_transcript(word_directory: str, cost_tracker=None):
    from pipeline_helpers import log_module_start, log_module_end, log, story_path, safe_write

    log_module_start("Transcript Correction")

    load_dotenv()
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OPENAI_API_KEY not found. Check your .env file.")

    client = OpenAI(api_key=api_key)
    output_dir = story_path(word_directory)

    docx_path = output_dir / f"{word_directory}.docx"
    srt_path = output_dir / f"{word_directory}.srt"
    txt_path = output_dir / f"{word_directory}.txt"

    # --- Check required files ---
    if not docx_path.exists():
        log(f"Script file not found: {docx_path} — skipping transcript correction.", level="WARNING")
        log_module_end("Transcript Correction")
        return

    if not srt_path.exists():
        log(f"SRT file not found: {srt_path} — skipping transcript correction.", level="WARNING")
        log_module_end("Transcript Correction")
        return

    # --- Check if correction pass already done ---
    flag_path = output_dir / f"{word_directory}_transcript_corrected.txt"
    if flag_path.exists():
        flag_date = flag_path.read_text(encoding="utf-8").strip()
        print(f"\n  ✅ Transcript correction already completed ({flag_date}).")
        redo = input("  Run again? Press Enter to skip, or type 'redo' to run again: ").strip().lower()
        if redo != "redo":
            log("Transcript correction skipped — already completed.")
            log_module_end("Transcript Correction")
            return
    else:
        print("\n  ⚠️  Transcript correction has not been run for this story.")
        run = input("  Run transcript correction now? Press Enter to proceed, or type 'skip' to skip: ").strip().lower()
        if run == "skip":
            log("Transcript correction skipped by user.")
            log_module_end("Transcript Correction")
            return

    # --- Extract script text ---
    log(f"Reading script: {docx_path.name}")
    script_text = extract_docx_text(docx_path)

    # --- Read transcript files ---
    log(f"Reading transcript: {srt_path.name}")
    srt_segments = parse_srt(srt_path)
    srt_text = " ".join(seg["text"] for seg in srt_segments)
    txt_text = txt_path.read_text(encoding="utf-8") if txt_path.exists() else srt_text

    # --- Find corrections ---
    log("Comparing script and transcript for proper noun misspellings...")
    corrections, usage = find_corrections(script_text, srt_text, client)

    if cost_tracker:
        from pricing import calc_text_cost
        cost = calc_text_cost("gpt-5.5", usage.prompt_tokens, usage.completion_tokens)
        cost_tracker.add("Transcript correction (gpt-5.5)", cost,
                         f"{usage.prompt_tokens} in / {usage.completion_tokens} out tokens")

    if not corrections:
        log("No corrections needed — transcript proper nouns match the script.")
        from datetime import datetime as _dt
        safe_write(
            lambda: flag_path.write_text(_dt.now().strftime("%Y-%m-%d %H:%M"), encoding="utf-8"),
            str(flag_path)
        )
        log_module_end("Transcript Correction")
        return

    # --- Show corrections to user for confirmation ---
    print(f"\n  Found {len(corrections)} proposed correction(s):")
    for i, c in enumerate(corrections, 1):
        print(f"    {i}. '{c['wrong']}' → '{c['right']}'")

    print()
    choice = input("  Apply all corrections? Press Enter to apply, 'r' to review individually, or 'skip' to skip: ").strip().lower()

    if choice == "skip":
        log("Transcript correction skipped by user.", level="WARNING")
        log_module_end("Transcript Correction")
        return

    if choice == "r":
        confirmed = []
        for c in corrections:
            ans = input(f"    Apply '{c['wrong']}' → '{c['right']}'? (Enter=yes, n=no): ").strip().lower()
            if ans != "n":
                confirmed.append(c)
        corrections = confirmed

    if not corrections:
        log("No corrections applied.")
        log_module_end("Transcript Correction")
        return

    # --- Apply corrections to SRT ---
    for seg in srt_segments:
        seg["text"] = apply_corrections(seg["text"], corrections)

    safe_write(lambda: write_srt(srt_segments, srt_path), str(srt_path))
    log(f"SRT updated: {srt_path.name}")

    # --- Apply corrections to .txt ---
    if txt_path.exists():
        corrected_txt = apply_corrections(txt_text, corrections)
        safe_write(lambda: txt_path.write_text(corrected_txt, encoding="utf-8"), str(txt_path))
        log(f"Transcript updated: {txt_path.name}")

    log(f"Applied {len(corrections)} correction(s).")
    from datetime import datetime as _dt
    safe_write(
        lambda: flag_path.write_text(_dt.now().strftime("%Y-%m-%d %H:%M"), encoding="utf-8"),
        str(flag_path)
    )
    log_module_end("Transcript Correction")


if __name__ == "__main__":
    from pipeline_helpers import init_run_log, save_run_log, load_activity_log_text, prompt_with_found
    from cost_tracker import CostTracker
    from notify import notify, prompt_notify_email
    from pricing import pricing_status
    from datetime import datetime

    load_dotenv()

    print("\nDBSM — Transcript Correction")
    print("=" * 40)

    available, status_msg = pricing_status()
    if not available:
        print(f"  ⚠️  {status_msg}")

    directory = input("Enter word directory: ").strip()
    if not directory:
        print("No directory entered. Exiting.")
        exit()

    notify_email = prompt_notify_email()
    tracker = CostTracker(directory, pricing_updated=available, pricing_warning=None if available else status_msg)
    init_run_log(directory)
    start_time = datetime.now()

    try:
        correct_transcript(directory, cost_tracker=tracker)
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        end_time = datetime.now()
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
                subject=f"DBSM Transcript Correction complete: {directory}",
                body=f"Duration: {end_time - start_time}\nCost: ${tracker.total_cost:.4f}\n\n{log_text}",
                to_address=notify_email,
            )
