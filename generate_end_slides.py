import os
from pathlib import Path
from dotenv import load_dotenv
from openai import OpenAI
from image_generator import save_image


def generate_end_slides_standalone(word_directory: str, style: str):
    load_dotenv()
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OPENAI_API_KEY not found. Check your .env file.")

    client = OpenAI(api_key=api_key)

    images_dir = story_path(word_directory) / "images"
    images_dir.mkdir(exist_ok=True)

    style_ref_path = story_path(word_directory) / "references" / "style_reference.png"
    first_image_path = images_dir / "image_001.png"

    print(f"\n--- Generating end slides ---")

    # --- The End slide ---
    the_end_path = images_dir / "the_end.png"
    if the_end_path.exists():
        print(f"The End slide already exists, skipping.")
    else:
        print(f"Generating The End slide...")
        the_end_prompt = (
            f"A professional show end card in the style of {style}. "
            f"'The End' is displayed prominently at the top, with 'The Drew Baderson Story Moment' below it. "
            f"The design should feel like a closing title card — warm, elegant, and conclusive. "
            f"No other text or imagery."
        )
        ref_images = [p for p in [first_image_path, style_ref_path] if p.exists()]
        if ref_images:
            image_files = [open(p, "rb") for p in ref_images]
            try:
                response = client.images.edit(
                    model="gpt-image-2",
                    image=image_files,
                    prompt=the_end_prompt,
                    size="1536x1024",
                    quality="medium",
                    input_fidelity="high",
                    n=1,
                )
            finally:
                for f in image_files:
                    f.close()
        else:
            response = client.images.generate(
                model="gpt-image-2",
                prompt=the_end_prompt,
                size="1536x1024",
                quality="medium",
                n=1,
            )
        save_image(response.data[0].b64_json, the_end_path)
        print(f"The End slide saved: {the_end_path}")

    # --- Credits slide ---
    credits_path = images_dir / "credits.png"
    if credits_path.exists():
        print(f"Credits slide already exists, skipping.")
    else:
        print(f"Generating credits slide...")
        credits_prompt = (
            f"A clean, elegant credits slide in the style of {style}. "
            f"Display the text 'All Music by Casey Connor' and 'caseyconnor.org' clearly and prominently. "
            f"Simple, uncluttered design with no other text or imagery."
        )
        ref_images = [p for p in [style_ref_path] if p.exists()]
        if ref_images:
            image_files = [open(p, "rb") for p in ref_images]
            try:
                response = client.images.edit(
                    model="gpt-image-2",
                    image=image_files,
                    prompt=credits_prompt,
                    size="1536x1024",
                    quality="medium",
                    input_fidelity="high",
                    n=1,
                )
            finally:
                for f in image_files:
                    f.close()
        else:
            response = client.images.generate(
                model="gpt-image-2",
                prompt=credits_prompt,
                size="1536x1024",
                quality="medium",
                n=1,
            )
        save_image(response.data[0].b64_json, credits_path)
        print(f"Credits slide saved: {credits_path}")

    print(f"\n--- End slides complete! ---")


if __name__ == "__main__":
    from pipeline_helpers import prompt_with_found, load_style, load_word_of_day, story_path
    directory = input("Enter word directory: ").strip()
    if not directory:
        print("No directory entered. Exiting.")
    else:
        print("\n(Press Enter to confirm found values or accept defaults)")
        found_style = load_style(directory)
        style = prompt_with_found("Style", found_style)
        if not style:
            print("No style entered. Exiting.")
        else:
            generate_end_slides_standalone(directory, style)
