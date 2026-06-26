"""
Shared helper functions for standalone pipeline script execution and logging.
Activity log entries are written immediately on every log() call.
"""
import json
from datetime import datetime
from pathlib import Path

LOG_TEXT_FILE = Path("pipeline_activity.log")


def safe_write(write_fn, filepath: str = None):
    """
    Execute a file write function, retrying if a PermissionError occurs.
    Prompts the user to close the file and press Enter to retry.
    write_fn: a callable that performs the write (no arguments)
    filepath: optional filename to show in the prompt
    """
    while True:
        try:
            write_fn()
            return
        except PermissionError:
            label = filepath or "a file"
            print(f"\n  ⚠️  Access denied writing to {label}.")
            print(f"  Please close it if it is open in another application.")
            retry = input("  Press Enter to retry, or type 'skip' to skip this write: ").strip().lower()
            if retry == "skip":
                print(f"  Skipping write to {label}.")
                return

# Stories subfolder — all word directories live here
STORIES_DIR = Path("stories")


def story_path(word_directory: str) -> Path:
    """Return the full path to a story directory, e.g. stories/20260406_compliance."""
    return STORIES_DIR / word_directory
LOG_JSON_FILE = Path("pipeline_activity.json")

# In-memory state for the current run
_current_run_log = []
_current_run_start = None
_current_run_directory = None
_run_timestamp = None


def init_run_log(word_directory: str):
    """Initialize the log for a new pipeline run."""
    global _current_run_log, _current_run_start, _current_run_directory, _run_timestamp
    _current_run_log = []
    _current_run_start = datetime.now()
    _current_run_directory = word_directory
    _run_timestamp = _current_run_start.strftime("%Y-%m-%d %H:%M:%S")

    # Write the run header to the text log immediately
    separator = "=" * 60
    header = (
        f"\n{separator}\n"
        f"RUN STARTED: {word_directory}\n"
        f"Started: {_run_timestamp}\n"
        f"{separator}\n"
    )
    safe_write(lambda: open(LOG_TEXT_FILE, "a", encoding="utf-8").write(header), str(LOG_TEXT_FILE))

    # Initialize JSON log entry
    _update_json_log(status="in_progress")


def log(message: str, level: str = "INFO"):
    """Log a message, print it, and write immediately to both log files."""
    global _current_run_log
    timestamp = datetime.now().strftime("%H:%M:%S")
    entry = {"time": timestamp, "level": level, "message": message}
    _current_run_log.append(entry)

    prefix = {"INFO": "  ", "WARNING": "  ⚠️  ", "ERROR": "  ❌ ", "START": "\n▶ ", "END": "✅ "}
    print(f"{prefix.get(level, '  ')}{message}")

    # Write to text log immediately
    if _run_timestamp:
        with open(LOG_TEXT_FILE, "a", encoding="utf-8") as f:
            f.write(f"  [{timestamp}] {level}: {message}\n")

        # Update JSON log
        _update_json_log()


def _update_json_log(status: str = "in_progress"):
    """Update the JSON activity log with current state."""
    if not _run_timestamp:
        return

    if LOG_JSON_FILE.exists():
        try:
            with open(LOG_JSON_FILE, "r", encoding="utf-8") as f:
                log_data = json.load(f)
        except Exception:
            log_data = {"runs": []}
    else:
        log_data = {"runs": []}

    # Find or create this run's entry
    run_entry = None
    for run in log_data["runs"]:
        if run.get("run_timestamp") == _run_timestamp and run.get("word_directory") == _current_run_directory:
            run_entry = run
            break

    if run_entry is None:
        run_entry = {
            "run_timestamp": _run_timestamp,
            "word_directory": _current_run_directory,
            "status": status,
            "events": [],
        }
        log_data["runs"].append(run_entry)

    run_entry["events"] = _current_run_log
    run_entry["status"] = status

    with open(LOG_JSON_FILE, "w", encoding="utf-8") as f:
        json.dump(log_data, f, indent=2)


def log_module_start(module_name: str):
    """Log the start of a pipeline module."""
    log(f"{module_name} started...", level="START")


def log_module_end(module_name: str):
    """Log the successful completion of a pipeline module."""
    log(f"{module_name} complete.", level="END")


def load_activity_log_text(word_directory: str) -> str:
    """Return the log entries for the current run from the in-memory log."""
    entries = []
    for entry in _current_run_log:
        entries.append(f"[{entry['time']}] {entry['level']}: {entry['message']}")
    return "\n".join(entries) if entries else "(no log entries)"


def save_run_log(word_directory: str, style: str, total_cost: float,
                 start_time: datetime, end_time: datetime,
                 pricing_warning: str = None, status: str = "complete"):
    """Finalize the run log with summary info."""
    duration = end_time - start_time
    hours, remainder = divmod(int(duration.total_seconds()), 3600)
    minutes, seconds = divmod(remainder, 60)
    duration_str = f"{hours:02d}:{minutes:02d}:{seconds:02d}"

    # Write closing summary to text log
    summary = (
        f"  Style: {style}\n"
        f"  Total cost: ${total_cost:.4f}\n"
        f"  Duration: {duration_str}\n"
        f"  Status: {status}\n"
        f"  Ended: {end_time.strftime('%Y-%m-%d %H:%M:%S')}\n"
    )
    if pricing_warning:
        summary += f"  ⚠️  PRICING WARNING: {pricing_warning}\n"
    summary += f"{'=' * 60}\n"

    safe_write(lambda: open(LOG_TEXT_FILE, "a", encoding="utf-8").write(summary), str(LOG_TEXT_FILE))

    # Update JSON log with final status
    if LOG_JSON_FILE.exists():
        try:
            with open(LOG_JSON_FILE, "r", encoding="utf-8") as f:
                log_data = json.load(f)
            for run in log_data["runs"]:
                if run.get("run_timestamp") == _run_timestamp and run.get("word_directory") == word_directory:
                    run["status"] = status
                    run["style"] = style
                    run["total_cost"] = round(total_cost, 6)
                    run["duration"] = duration_str
                    run["ended"] = end_time.strftime("%Y-%m-%d %H:%M:%S")
                    if pricing_warning:
                        run["pricing_warning"] = pricing_warning
                    break
            with open(LOG_JSON_FILE, "w", encoding="utf-8") as f:
                json.dump(log_data, f, indent=2)
        except Exception:
            pass

    print(f"\n  Activity log updated: {LOG_TEXT_FILE}")
    print(f"  Activity log (JSON) updated: {LOG_JSON_FILE}")
    print(f"\n--- Run Summary ---")
    print(f"  Started:  {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"  Ended:    {end_time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"  Duration: {duration_str}")
    print(f"  Status:   {status}")


def prompt_with_found(label: str, found_value: str, default: str = None) -> str:
    """
    If a value was found automatically, show it and ask to confirm or override.
    If not found, prompt the user to enter it, with an optional default.
    """
    if found_value is not None:
        response = input(f"  {label}: Found '{found_value}'. Press Enter to use it, or type a replacement: ").strip()
        return response if response else found_value
    elif default is not None:
        response = input(f"  {label}: Not found. Press Enter for default ('{default}'), or type a value: ").strip()
        return response if response else default
    else:
        response = input(f"  {label}: Not found. Please enter a value: ").strip()
        return response


def load_style(word_directory: str) -> str:
    """Try to load style from the _style.txt file."""
    style_file = story_path(word_directory) / f"{word_directory}_style.txt"
    if style_file.exists():
        return style_file.read_text(encoding="utf-8").strip()
    return None


def load_word_of_day(word_directory: str) -> str:
    """Derive word of day from directory name."""
    return word_directory.split("_", 1)[1].capitalize()
