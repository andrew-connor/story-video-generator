from datetime import datetime
from pathlib import Path
from pricing import pricing_status
from cost_tracker import CostTracker
from transcribe import transcribe
from style_selector import select_style, confirm_style
from image_generator import generate_images
from assemble_video import assemble_video
from pipeline_helpers import init_run_log, log, log_module_start, log_module_end, save_run_log, story_path, STORIES_DIR
from notify import notify, prompt_notify_email
import os, requests
from dotenv import load_dotenv


def get_openai_usage_today() -> float:
    """
    Fetch today's usage cost from the OpenAI Usage API.
    Returns cost in USD, or None if unavailable.
    Note: data may be delayed by up to a few hours.
    """
    try:
        load_dotenv()
        api_key = os.getenv("OPENAI_API_KEY")
        today = datetime.now().strftime("%Y-%m-%d")
        url = f"https://api.openai.com/v1/organization/costs?start_time={today}T00:00:00Z&limit=1"
        response = requests.get(url, headers={"Authorization": f"Bearer {api_key}"}, timeout=10)
        if response.status_code == 200:
            data = response.json()
            # Sum all cost buckets for today
            total = sum(
                item.get("amount", {}).get("value", 0)
                for item in data.get("data", [])
            )
            return round(total, 4)
        else:
            return None
    except Exception:
        return None

# --- Current model versions (last checked: April 2026) ---
# Text model:  gpt-5.5
# Image model: gpt-image-2
# Model strings are set in image_generator.py, assemble_video.py, and upload_to_youtube.py.

if __name__ == "__main__":
    print("\n=== The Drew Baderson Story Moment Pipeline ===")
    print("  Text model:  gpt-5.5")
    print("  Image model: gpt-image-2")
    proceed = input("\nProceed with these models? Press Enter to continue, or type 'no' to exit: ").strip().lower()
    if proceed == "no":
        print("Exiting. Update model strings in image_generator.py, assemble_video.py, and upload_to_youtube.py.")
        exit()

    # --- Check OpenAI account usage before run ---
    usage_before = get_openai_usage_today()
    if usage_before is not None:
        print(f"  OpenAI usage so far today: ${usage_before:.4f} (note: may be delayed)")
    else:
        print(f"  OpenAI usage check unavailable.")

    # Ensure stories subfolder exists
    STORIES_DIR.mkdir(exist_ok=True)

    directory = input("\nEnter word directory (inside the 'stories' folder): ").strip()
    if not directory:
        print("No directory entered. Exiting.")
    else:
        word = directory.split("_", 1)[1].capitalize()
        default_title = f"The Drew Baderson Story Moment - {word}"

        # --- Collect ALL inputs upfront ---
        print("\n--- Configuration (press Enter to accept any default) ---")

        anchor = input(f"  Anchor phrase for intro card (default: 'this is what I wrote'): ").strip() or "this is what i wrote"
        end_anchor = input(f"  Anchor phrase for end card (press Enter to use end of spoken text, or type a specific phrase): ").strip()
        show = input(f"  Show name (default: 'The Drew Baderson Story Moment'): ").strip() or "The Drew Baderson Story Moment"
        episode = input(f"  Episode title (default: '{word}'): ").strip() or word
        end_top = input(f"  Top text for end card (default: 'The End'): ").strip() or "The End"
        end_bottom = input(f"  Bottom text for end card (default: '{show}'): ").strip() or show
        credits_top = input(f"  Top text for credits (default: 'All Music by Casey Connor'): ").strip() or "All Music by Casey Connor"
        credits_bottom = input(f"  Bottom text for credits (default: 'caseyconnor.org'): ").strip() or "caseyconnor.org"
        notify_email = prompt_notify_email()

        # Style selection — check for existing style first, then random pick
        print(f"\n  --- Style Selection ---")
        from pipeline_helpers import load_style
        existing_style = load_style(directory)
        if existing_style:
            style = existing_style
            while True:
                print(f"  Found existing style: '{style}'")
                choice = input("  Press Enter to accept, 'r' for a random pick, or type a style name: ").strip()
                if choice == "":
                    print(f"  Using existing style: '{style}'")
                    break
                elif choice.lower() == "r":
                    style = select_style(directory)
                    # Loop again to show the new random pick and offer options
                else:
                    style = choice
                    break
            # Only call confirm_style if style changed from the existing one
            if style != existing_style:
                confirm_style(style, directory)
        else:
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

        # --- Initialize cost tracker ---
        available, status_msg = pricing_status()
        if available:
            print(f"\n  💰 litellm pricing active.")
        else:
            print(f"\n  ⚠️  {status_msg}")
        tracker = CostTracker(directory, pricing_updated=available, pricing_warning=None if available else status_msg)

        # --- Initialize run log ---
        init_run_log(directory)

        # --- Run pipeline ---
        start_time = datetime.now()
        log(f"Pipeline started for '{directory}' with style '{style}'", level="INFO")
        print(f"\n--- Starting pipeline at {start_time.strftime('%Y-%m-%d %H:%M:%S')} ---")

        pipeline_status = "complete"
        try:
            transcribe(directory, cost_tracker=tracker)
            generate_images(directory, style,
                            show_name=show,
                            episode_title=episode,
                            end_top_text=end_top,
                            end_bottom_text=end_bottom,
                            credits_top_text=credits_top,
                            credits_bottom_text=credits_bottom,
                            cost_tracker=tracker)
            assemble_video(directory, anchor_phrase=anchor, end_anchor_phrase=end_anchor,
                           cost_tracker=tracker)

        except Exception as e:
            pipeline_status = "failed"
            log(f"Pipeline failed: {e}", level="ERROR")
            print(f"\n❌ Pipeline error: {e}")
        finally:
            # --- Final summary — always runs even on error ---
            end_time = datetime.now()
            duration = end_time - start_time
            hours, remainder = divmod(int(duration.total_seconds()), 3600)
            minutes, seconds = divmod(remainder, 60)

        log(f"Pipeline complete. Duration: {hours:02d}:{minutes:02d}:{seconds:02d}", level="INFO")

        tracker.print_total()
        tracker.save_to_log()

        save_run_log(
            word_directory=directory,
            style=style,
            total_cost=tracker.total_cost,
            start_time=start_time,
            end_time=end_time,
            pricing_warning=tracker.pricing_warning,
        )

        print(f"\n--- Pipeline complete! ---")
        print(f"Style used:  {style}")
        print(f"Started:     {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"Ended:       {end_time.strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"Duration:    {hours:02d}:{minutes:02d}:{seconds:02d}")
        print(f"\n⚠️  Remember to handle manual YouTube steps: comment settings, caption upload, etc.")

        # --- Notify ---
        if notify_email:
            from pipeline_helpers import load_activity_log_text
            log_text = load_activity_log_text(directory)
            notify(
                subject=f"DBSM Pipeline {'Complete' if pipeline_status == 'complete' else 'Failed'}: {directory}",
                body=f"Status: {pipeline_status}\nDuration: {hours:02d}:{minutes:02d}:{seconds:02d}\nCost: ${tracker.total_cost:.4f}\n\n{log_text}",
                to_address=notify_email,
            )
