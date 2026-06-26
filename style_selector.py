import csv
import random
from pathlib import Path
from pipeline_helpers import story_path, safe_write

STYLES_FILE = Path("styles.csv")


def _load_rows() -> list:
    """Load all rows from styles.csv."""
    with open(STYLES_FILE, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        return list(reader)


def _save_rows(rows: list):
    """Save rows back to styles.csv."""
    def _write():
        with open(STYLES_FILE, "w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["style", "used", "directories"])
            writer.writeheader()
            writer.writerows(rows)
    safe_write(_write, "styles.csv")


def _migrate_csv_if_needed():
    """Add 'directories' column if it doesn't exist yet."""
    with open(STYLES_FILE, "r", encoding="utf-8") as f:
        header = f.readline().strip()
    if "directories" not in header:
        rows = []
        with open(STYLES_FILE, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                rows.append({
                    "style": row["style"],
                    "used": row["used"],
                    "directories": "",
                })
        _save_rows(rows)
        print("  styles.csv updated to new format.")


def select_style(word_directory: str) -> str:
    """
    Randomly select an unused style (used=0).
    Does NOT mark it yet — call confirm_style() after user confirmation.
    Returns the selected style name.
    """
    if not STYLES_FILE.exists():
        raise FileNotFoundError(f"Styles file not found: {STYLES_FILE}")

    _migrate_csv_if_needed()
    rows = _load_rows()
    unused = [row for row in rows if row["used"] == "0"]

    if len(unused) == 0:
        raise RuntimeError(
            "No unused styles remaining. "
            "Please reset styles.csv manually by setting all 'used' values back to 0."
        )

    if len(unused) <= 5:
        print(f"  ⚠️  WARNING: Only {len(unused)} unused style(s) remaining. Please reset styles.csv soon.")

    chosen = random.choice(unused)
    return chosen["style"]


def confirm_style(style: str, word_directory: str):
    """
    Mark the confirmed style as used in styles.csv and write _style.txt.
    - Increments the 'used' count
    - Appends word_directory to the directories column
    - Adds the style as a new row if it doesn't exist
    - Warns if the style has already been used (used > 0)
    - Writes _style.txt
    """
    if not STYLES_FILE.exists():
        raise FileNotFoundError(f"Styles file not found: {STYLES_FILE}")

    _migrate_csv_if_needed()
    rows = _load_rows()

    # Find the style row
    style_row = None
    for row in rows:
        if row["style"].lower() == style.lower():
            style_row = row
            break

    # If not found, add it
    if style_row is None:
        style_row = {"style": style, "used": "0", "directories": ""}
        rows.append(style_row)
        print(f"  Style '{style}' not found in styles.csv — adding it.")

    current_count = int(style_row["used"])

    # Warn if already used
    if current_count > 0:
        print(f"  ⚠️  WARNING: Style '{style}' has already been used {current_count} time(s).")
        confirm = input("  Type 'yes' to proceed anyway, or press Enter to cancel: ").strip().lower()
        if confirm != "yes":
            raise RuntimeError("Style selection cancelled by user.")

    # Increment count
    style_row["used"] = str(current_count + 1)

    # Append word_directory to directories
    existing_dirs = style_row.get("directories", "").strip()
    if existing_dirs:
        style_row["directories"] = existing_dirs + "," + word_directory
    else:
        style_row["directories"] = word_directory

    _save_rows(rows)

    # Write _style.txt
    output_dir = story_path(word_directory)
    output_dir.mkdir(parents=True, exist_ok=True)
    style_file = output_dir / f"{word_directory}_style.txt"
    safe_write(lambda: open(style_file, "w", encoding="utf-8").write(style), str(style_file))

    print(f"  Style confirmed: '{style}'")
    print(f"  Style file saved: {style_file}")


if __name__ == "__main__":
    from pipeline_helpers import prompt_with_found, load_word_of_day, story_path
    directory = input("Enter word directory: ").strip()
    if not directory:
        print("No directory entered. Exiting.")
    else:
        print("\n(Press Enter to confirm found values or accept defaults)")
        style = select_style(directory)
        while True:
            print(f"  Randomly selected style: '{style}'")
            choice = input("  Press Enter to accept, 'r' for another random pick, or type a style name: ").strip()
            if choice == "":
                break
            elif choice.lower() == "r":
                style = select_style(directory)
            else:
                style = choice
                break
        confirm_style(style, directory)
        print(f"Style for image generation: {style}")
