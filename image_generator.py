import base64
import json
import os
from io import BytesIO
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI, BadRequestError, APIConnectionError
from PIL import Image


# Rich style descriptions to reinforce stylistic fidelity in every prompt
STYLE_DETAILS = {
    "1970s airbrush": "rendered in a 1970s airbrush illustration style with smooth color gradients, soft glowing edges, warm amber and orange tones, velvety surface texture, and the dreamy nostalgic quality of 1970s commercial art",
    "pencil sketch": "rendered as a hand-drawn pencil sketch with visible cross-hatching, rough expressive lines, warm sepia tones, areas of dense layered shadow, and the organic imperfect quality of traditional graphite drawing",
    "watercolor": "rendered in loose expressive watercolor with soft wet-on-wet washes, visible paper texture, bleeding edges, transparent layered color, and the delicate spontaneous quality of traditional watercolor painting",
    "oil painting": "rendered in the style of a classical oil painting with rich saturated color, visible brushwork, impasto texture, dramatic chiaroscuro lighting, and the depth and warmth of traditional oil on canvas",
    "woodblock print": "rendered in the style of a traditional woodblock print with bold flat color areas, strong black outlines, visible grain texture, and the graphic high-contrast quality of Japanese ukiyo-e printmaking",
    "charcoal": "rendered as a charcoal drawing with bold smudged strokes, high contrast black and white tones, soft blended shadows, and the raw expressive quality of traditional charcoal on textured paper",
}

def get_style_detail(style: str) -> str:
    """Return a rich style description for known styles, or a generic one for unknown styles."""
    style_lower = style.lower().strip()
    for key, detail in STYLE_DETAILS.items():
        if key in style_lower or style_lower in key:
            return detail
    return f"rendered in the distinctive visual style of {style}, with all the characteristic textures, colors, linework, and techniques that define that style"


def save_image(b64_data: str, output_path: Path):
    """Decode base64 image, upscale to 1920x1080, and save."""
    image_bytes = base64.b64decode(b64_data)
    image = Image.open(BytesIO(image_bytes))
    image = image.resize((1920, 1080), Image.LANCZOS)
    image.save(output_path, format="PNG")


def rewrite_prompt(client: OpenAI, prompt: str, violation_type: str = None) -> str:
    """
    Ask GPT-5.4 to rewrite a blocked prompt, tailoring the rewrite to the violation type.
    violation_type: e.g. "sexual", "violence", "hate", or None if unknown.
    """
    print(f"    Rewriting prompt to avoid moderation filter (violation: {violation_type or 'unknown'})...")

    if violation_type:
        guidance = (
            f"The prompt was flagged for a '{violation_type}' violation. "
            f"Rewrite it to address that specific issue while keeping the characters, "
            f"setting, illustration style, and emotional tone as intact as possible. "
            f"If this appears to be a false positive (e.g. innocent language that superficially "
            f"resembles problematic content), rephrase the triggering language in neutral "
            f"visual and compositional terms."
        )
    else:
        guidance = (
            "Rewrite to avoid any content that could trigger moderation — remove or soften "
            "any violence, conflict, dark themes, sexual language, or sensitive imagery "
            "while keeping the illustration style, characters, and general scene intact."
        )

    response = client.chat.completions.create(
        model="gpt-5.5",
        messages=[{
            "role": "user",
            "content": (
                f"The following image generation prompt was blocked by a content moderation system.\n\n"
                f"{guidance}\n\n"
                f"Original prompt:\n{prompt}\n\n"
                "Return only the rewritten prompt with no explanation or preamble."
            )
        }],
    )
    return response.choices[0].message.content.strip()


def build_bible_context(beat: dict, story_bible: dict, all_characters: list = None) -> str:
    """
    Build a context string from the story bible for a given beat.
    Prepends a mandatory scene header with location and character presence/absence.
    """
    beat_settings = beat.get("settings", [])
    beat_objects = beat.get("objects", [])
    beat_characters = beat.get("characters", [])

    # --- Build mandatory scene header ---
    location = beat_settings[0] if beat_settings else "Unspecified"

    # Build characters present list with brief description reminder to prevent species/type drift
    char_lookup = {c["name"]: c.get("description", "") for c in (all_characters or [])}

    def char_with_hint(name: str) -> str:
        desc = char_lookup.get(name, "")
        if not desc:
            return name
        # Extract first sentence or up to 60 chars as a brief reminder
        hint = desc.split(".")[0].strip()
        if len(hint) > 60:
            hint = hint[:60].rsplit(" ", 1)[0]
        return f"{name} ({hint})"

    present_parts = [char_with_hint(c) for c in beat_characters] if beat_characters else ["None"]
    present = ", ".join(present_parts)

    not_present = []
    if all_characters:
        not_present = [c["name"] for c in all_characters if c["name"] not in beat_characters]
    absent = ", ".join(not_present) if not_present else "None"

    header = (
        f"MANDATORY SCENE REQUIREMENTS (must be followed exactly):\n"
        f"  LOCATION: {location}\n"
        f"  CHARACTERS PRESENT: {present}\n"
        f"  CHARACTERS NOT PRESENT: {absent}\n"
        f"  The image must match these requirements exactly. Any character listed as not present\n"
        f"  must not appear in the image under any circumstances.\n"
        f"  Each character's appearance must match their description exactly — do not change species, "
        f"age, or defining physical traits."
    )

    if not story_bible:
        return "\n\n" + header

    context_parts = []

    # Add relevant setting constraints
    # Match against both the beat's settings list AND the location from the scene header
    settings_to_check = set(s.lower() for s in beat_settings)
    if location and location.lower() != "unspecified":
        settings_to_check.add(location.lower())
    for setting in story_bible.get("settings", []):
        setting_name_lower = setting["name"].lower()
        if any(s in setting_name_lower or setting_name_lower in s for s in settings_to_check):
            context_parts.append(f"SETTING — {setting['name']}: {setting['description']}")
            for constraint in setting.get("constraints", []):
                context_parts.append(f"  MUST BE TRUE: {constraint}")

    # Add relevant object constraints
    for obj in story_bible.get("objects", []):
        if any(o.lower() in obj["name"].lower() or obj["name"].lower() in o.lower()
               for o in beat_objects):
            context_parts.append(f"OBJECT — {obj['name']}: {obj['description']}")
            for constraint in obj.get("constraints", []):
                context_parts.append(f"  MUST BE TRUE: {constraint}")

    # Always include key facts and relationships
    for fact in story_bible.get("key_facts", []):
        context_parts.append(f"KEY FACT: {fact}")
    for rel in story_bible.get("relationships", []):
        context_parts.append(f"RELATIONSHIP: {rel}")

    bible_section = ""
    if context_parts:
        bible_section = "\n\nSTORY BIBLE CONSTRAINTS (these must also be followed exactly):\n" + "\n".join(context_parts)

    return "\n\n" + header + bible_section


def generate_end_slides(client: OpenAI, images_dir: Path, style: str, word_of_day: str,
                        show_name: str = "The Drew Baderson Story Moment",
                        end_top_text: str = "The End",
                        end_bottom_text: str = "The Drew Baderson Story Moment",
                        credits_top_text: str = "All Music by Casey Connor",
                        credits_bottom_text: str = "caseyconnor.org",
                        story_themes: str = ""):
    """Generate the end card and credits slide."""
    from pipeline_helpers import log_module_start, log_module_end, log, story_path, safe_write
    log_module_start("End Slides Generation")

    style_ref_path = images_dir.parent / "references" / "style_reference.png"
    first_image_path = images_dir / "image_001.png"

    # --- The End slide ---
    the_end_path = images_dir / "the_end.png"
    if the_end_path.exists():
        print(f"The End slide already exists, skipping.")
    else:
        print(f"Generating The End slide...")
        themes_context = f" The atmosphere and color palette should reflect the world of this story: {story_themes}." if story_themes else ""
        the_end_prompt = (
            f"STYLE REQUIREMENT (highest priority): This image must be {get_style_detail(style)}. "
            f"The style must be immediately and unmistakably present. "
            f"A professional show end card in the style of {style}."
            f"{themes_context} "
            f"The text '{end_top_text}' is displayed prominently at the top — these are two separate words "
            f"with a space between them, rendered clearly and legibly with no letters merged or dropped. "
            f"Below it, '{end_bottom_text}' appears in a smaller, understated typographic treatment. "
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
            f"STYLE REQUIREMENT (highest priority): This image must be {get_style_detail(style)}. "
            f"The style must be immediately and unmistakably present — do not default to plain white background with black text. "
            f"A credits slide in the style of {style}."
            f"{themes_context} "
            f"Display the text '{credits_top_text}' prominently, with '{credits_bottom_text}' below it. "
            f"The typography and layout should feel like a natural extension of the story's visual world. "
            f"No other text or imagery."
        )
        ref_images = [p for p in [first_image_path, style_ref_path] if p.exists()]
        if ref_images:
            image_files = [open(p, "rb") for p in ref_images]
            try:
                response = client.images.edit(
                    model="gpt-image-2",
                    image=image_files,
                    prompt=credits_prompt,
                    size="1536x1024",
                    quality="medium",
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
        log(f"Credits slide saved: {credits_path}")
    log_module_end("End Slides Generation")


def generate_images(word_directory: str, style: str,
                    show_name: str = "The Drew Baderson Story Moment",
                    episode_title: str = None,
                    end_top_text: str = "The End",
                    end_bottom_text: str = "The Drew Baderson Story Moment",
                    credits_top_text: str = "All Music by Casey Connor",
                    credits_bottom_text: str = "caseyconnor.org",
                    anchor_phrase: str = "this is what i wrote",
                    cost_tracker=None):
    from pipeline_helpers import log_module_start, log_module_end, log, story_path, safe_write
    log_module_start("Image Generation")

    load_dotenv()
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OPENAI_API_KEY not found. Check your .env file.")

    client = OpenAI(api_key=api_key)

    output_dir = story_path(word_directory)
    images_dir = output_dir / "images"
    references_dir = output_dir / "references"
    images_dir.mkdir(exist_ok=True)
    references_dir.mkdir(exist_ok=True)

    txt_file = output_dir / f"{word_directory}.txt"
    if not txt_file.exists():
        raise FileNotFoundError(f"Transcript not found: {txt_file}")

    with open(txt_file, "r", encoding="utf-8") as f:
        transcript = f.read().strip()

    word_of_day = episode_title if episode_title else word_directory.split("_", 1)[1].capitalize()

    # --- Step 1: Generate or load beats ---
    beats_file = output_dir / f"{word_directory}_beats.json"

    if beats_file.exists():
        log(f"Beats file already exists, loading from {beats_file}")
        with open(beats_file, "r", encoding="utf-8") as f:
            data = json.load(f)
        beats = data["beats"]
        characters = data["characters"]
        story_bible = data.get("story_bible", {})
    else:
        log("Analyzing transcript and generating image prompts...")

        # Calculate target beat count from SRT file
        srt_file = output_dir / f"{word_directory}.srt"
        target_beats = None
        if srt_file.exists():
            try:
                import re as _re
                srt_content = srt_file.read_text(encoding="utf-8").strip()
                blocks = _re.split(r"\n\n+", srt_content)
                segments = []
                for block in blocks:
                    lines = block.strip().split("\n")
                    if len(lines) >= 2 and "-->" in lines[1]:
                        end_str = lines[1].split("-->")[1].strip().replace(",", ".")
                        h, m, s = end_str.split(":")
                        end_sec = int(h) * 3600 + int(m) * 60 + float(s)
                        text = " ".join(lines[2:]).strip() if len(lines) > 2 else ""
                        segments.append({"end": end_sec, "text": text})
                # Find last non-duplicate segment for total duration
                last_unique_end = segments[0]["end"] if segments else 0
                for i in range(1, len(segments)):
                    if segments[i]["text"].strip().lower() != segments[i-1]["text"].strip().lower():
                        last_unique_end = segments[i]["end"]

                # Find intro end time using anchor phrase
                anchor_end = None
                anchor_lower = anchor_phrase.lower().strip()
                for seg in segments:
                    if anchor_lower in seg["text"].lower():
                        anchor_end = seg["end"]
                        break
                if anchor_end:
                    log(f"Anchor phrase '{anchor_phrase}' found at {anchor_end:.1f}s — using as intro end.")
                    story_duration = max(last_unique_end - anchor_end, 0)
                else:
                    log(f"Anchor phrase '{anchor_phrase}' not found in SRT — falling back to 30s estimate.", level="WARNING")
                    story_duration = max(last_unique_end - 30, 0)

                target_beats = max(int(story_duration / 10), 5) + 1  # +1 for beat 1 (intro)
                log(f"Story duration: {story_duration:.0f}s — target beats: {target_beats}")
            except Exception as e:
                log(f"Could not calculate beat count from SRT: {e}", level="WARNING")

        target_beats_instruction = ""
        if target_beats:
            target_beats_instruction = f"IMPORTANT: The story runs approximately {story_duration:.0f} seconds. At one beat per 10 seconds, you should generate approximately {target_beats} beats total (including beat 1 for the intro). Do not generate significantly fewer than this number. Every distinct story moment deserves its own beat — do not collapse multiple moments into one beat."

        prompt_request = f"""
You are a visual director preparing image prompts for a video.

The transcript always begins with a fixed introduction that reads something like:
"Welcome to the {show_name}. One hour ago, I chose a random word from the dictionary and used it to inspire a story. The word I chose was [word], and this is what I wrote."

Read the ENTIRE transcript carefully before writing anything. Then do three things:

PART 1 - CHARACTER LIST:
Identify all major characters that appear in the story (not the introduction). For each, provide a detailed visual description for consistent image generation.

IMPORTANT — Include ONLY defining traits in the character description:
- Physical appearance: skin tone, face, hair, body type, age
- Regular clothing: what they typically wear throughout the story
- Permanent accessories or distinguishing features that are always present

Do NOT include in the character description:
- Scene-specific props or equipment (e.g. a hard hat worn only in a mine, a weapon carried only in one scene)
- Items a character carries only sometimes or in specific situations
- Costumes or disguises worn temporarily

Situational items belong in the STORY BIBLE (PART 2) under objects or key facts, with a note about when they appear.

When designing characters, follow these casting guidelines:
- Strive for a racially and ethnically diverse cast. Unless the story specifies otherwise, deliberately vary skin tones, facial features, hair textures, and cultural backgrounds. Do not default to a homogenous cast.
- Vary body types, ages, clothing styles, and overall appearance. Characters should look like real people from different walks of life — some might be heavyset, slight, or athletic; some elderly with grey hair and weathered skin, middle-aged, or young adults; some tall, some short. Avoid defaulting to a cast of young, slim adults.
- If a character is portrayed as villainous, cruel, immoral, or engaged in harmful behavior, do not depict that character as a person of color. Assign those roles to white or ambiguously raced characters instead.

PART 2 - STORY BIBLE:
Create a comprehensive visual reference document covering everything that appears more than once or is established early and likely to recur. Do NOT include the title card or beat 1 as a setting — it is not a story location. Be exhaustive — err on the side of too much detail rather than too little. Include:

- SETTINGS: Every location in the story. For each, describe its visual properties in detail. Include spatial relationships (what is above, below, inside, outside, attached to what). Flag any constraints that must always be true (e.g. "the cages are ALWAYS suspended from tree branches by rope — never on the ground, never inside buildings").
- OBJECTS: Every significant object that recurs. Describe its appearance, size, material, color, and any spatial or positional constraints that must always be maintained.
- KEY FACTS: A list of visual facts that must remain consistent throughout — things that are easy to get wrong if not explicitly stated. Phrase these as hard rules (e.g. "The tower is always visible on the horizon in outdoor scenes", "The scar is always on the LEFT cheek").
- RELATIONSHIPS: Any spatial or physical relationships between characters or objects that recur (e.g. "Character A is always taller than Character B", "The dog always stays close to Elena").

{target_beats_instruction}

PART 3 - BEATS:
1. Identify the introductory section (everything up to and including "this is what I wrote" or similar) and treat it as beat 1.
2. Break the remainder of the transcript into beats of approximately 10 seconds of speech each (roughly 25-30 words per beat). CRITICAL: You must only generate beats for content that is explicitly present in the transcript. Stop generating beats when you reach "The End" or equivalent closing phrase. Do not generate beats for any content beyond that point, do not summarise the story, do not add reflective or thematic commentary beats, and do not invent content that was not spoken. Every beat's text field must be a direct quote or close paraphrase of words actually present in the transcript.
2a. When writing each beat's image prompt, consider the two preceding and two following beats to ensure visual flow. Vary composition and framing naturally across beats (wide shots, medium shots, close-ups, overhead views, etc.). Avoid repeating the same framing back to back. Create smooth emotional and visual transitions — each image should feel like part of a coherent sequence, not a series of isolated illustrations.
3. Before writing the beat 1 prompt, read the full transcript and identify 2-3 dominant visual themes — the mood, setting, key motifs, or emotional arc of the story. These themes should inform the background, color palette, and overall atmosphere of the title card, grounding it in the world of this specific story rather than being a generic treatment of the word alone. For example: if the story is about a medieval village oppressed by a boy king and a tailor who defies him with colorful clothing, the title card might evoke a dark stone castle, swatches of vivid color breaking through drab surroundings, and a sense of quiet rebellion. Then generate a branded title card image prompt with this exact visual hierarchy: (1) the COMPLETE show title "{show_name}" — every single word of it, spelled exactly as written, with no words dropped or shortened — appears at the TOP of the image in a clean, understated typographic treatment — elegant and readable but not the visual focus; (2) the word "{word_of_day}" appears below it as the dominant visual element — larger, more stylized, and designed to fully embrace the chosen illustration style in a way that feels expressive and distinctive, and visually connected to the story's themes. Both texts are absolutely mandatory — if either is missing or incomplete from the generated image, the image must be considered a failure and regenerated. Do not omit, shorten, or alter either text, and do not swap their positions. The complete show title is always understated at the top; the story word is always the bold, stylized focal point below it. Style: {style}
4. For all subsequent beats, write a detailed image generation prompt that visually represents the content of that beat. Every beat prompt MUST begin with a mandatory scene header in this exact format:
   LOCATION: [the exact setting name from PART 2, or "Unspecified" if no specific location]
   CHARACTERS PRESENT: [comma-separated list of character names who are in this scene, or "None"]
   CHARACTERS NOT PRESENT: [comma-separated list of all other characters who do NOT appear in this scene]
   This header is non-negotiable. After the header, write the visual prompt. The image must exactly match the header — no characters from the "not present" list may appear in the image, and the location must match.
5. Maintain consistency — use the exact character descriptions from PART 1 and the constraints from PART 2 verbatim whenever relevant elements appear.
6. Every beat prompt MUST begin with the style instruction as the very first sentence, before any scene description: "In the style of {style} — [rich sensory description of what that style looks and feels like]: ..." This ensures the style is weighted heavily. Do not bury the style at the end of the prompt.
7. Every prompt must be self-contained and descriptive enough to generate a compelling image.
8. For beats 2 onwards, ABSOLUTELY NO TEXT OR WORDS in the images. This is a strict rule — no labels, captions, signs, titles, banners, books with visible text, newspapers, letters, or any written words whatsoever. Every prompt you write must explicitly end with: "No text, letters, words, or written language anywhere in the image, unless it is absolutely necessary for clarity." Tell the story purely through visual imagery, composition, color, and mood.
9. IMPORTANT: Write all prompts to be visually safe and suitable for all audiences. Avoid any content that could trigger content moderation — no graphic violence, no explicit conflict, no disturbing imagery. Convey dramatic moments through mood, color, and composition rather than explicit depictions. When writing prompts involving child characters and animals, describe their relationship in purely visual and compositional terms — avoid any language about physical proximity, contact, pocket-riding, sleeping arrangements, or touch, as these phrases can trigger false moderation flags even in innocent contexts. Instead describe what is visually visible: e.g. "a small lizard perched on the wooden floor of the cage near Shonda" rather than "Mitzi rides in her pocket".
10. For each beat, list which characters, settings, and key objects from PARTS 1 and 2 appear in that beat.

Return your response as a single JSON object with this structure:
{{
  "characters": [
    {{
      "name": "character name",
      "description": "detailed visual description for image generation"
    }}
  ],
  "story_bible": {{
    "settings": [
      {{
        "name": "setting name",
        "description": "detailed visual description",
        "constraints": ["must always be true fact 1", "must always be true fact 2"]
      }}
    ],
    "objects": [
      {{
        "name": "object name",
        "description": "detailed visual description",
        "constraints": ["must always be true fact 1"]
      }}
    ],
    "key_facts": ["visual fact that must always be consistent"],
    "relationships": ["spatial or physical relationship that recurs"]
  }},
  "beats": [
    {{
      "beat": 1,
      "text": "transcript text for this beat",
      "prompt": "image generation prompt",
      "characters": ["character name 1"],
      "settings": ["setting name 1"],
      "objects": ["object name 1"]
    }}
  ]
}}

Return only the JSON object with no preamble, explanation, or markdown formatting.

Transcript:
{transcript}
"""

        gpt_response = client.chat.completions.create(
            model="gpt-5.5",
            messages=[{"role": "user", "content": prompt_request}],
        )

        raw = gpt_response.choices[0].message.content.strip()

        # Strip markdown code fences if present
        if raw.startswith("```"):
            raw = raw.split("```")[1]
            if raw.startswith("json"):
                raw = raw[4:]
            raw = raw.strip()

        if not raw:
            raise ValueError("GPT-4 returned an empty response. Please try again.")

        data = json.loads(raw)
        beats = data["beats"]
        characters = data["characters"]
        story_bible = data.get("story_bible", {})

        if cost_tracker:
            from pricing import calc_text_cost
            usage = gpt_response.usage
            cost = calc_text_cost("gpt-5.5", usage.prompt_tokens, usage.completion_tokens)
            cost_tracker.add("Beat/character analysis (GPT-4o)", cost, f"{usage.prompt_tokens} in / {usage.completion_tokens} out tokens")

        # --- Verification pass 1: Fact-check story bible constraints ---
        log("Verifying story bible constraints against transcript...")
        bible_verify_prompt = f"""
You are a careful story editor reviewing a visual story bible for accuracy.

Below is a story transcript and a story bible that was generated from it. Your job is to fact-check every constraint in the story bible against the transcript and correct any inaccuracies.

Common errors to look for:
- Constraints that say something "always" happens when it only happens sometimes
- Constraints that describe a character possessing or being near an object when they don't always have it
- Constraints that describe a location or relationship that changes over the course of the story
- Any constraint that would cause a visual inconsistency if applied rigidly to every scene

For each incorrect constraint, rewrite it to be accurate. Replace absolute rules ("always", "never", "must") with conditional ones ("when X", "during scenes where Y") where appropriate.

Return the corrected story bible as a JSON object with the exact same structure as the input. Do not add or remove entries — only correct inaccurate constraints. Return only the JSON object with no preamble or markdown formatting.

Story bible:
{json.dumps(story_bible, indent=2)}

Transcript:
{transcript}
"""
        bible_verify_response = client.chat.completions.create(
            model="gpt-5.5",
            messages=[{"role": "user", "content": bible_verify_prompt}],
        )
        bible_raw = bible_verify_response.choices[0].message.content.strip()
        if bible_raw.startswith("```"):
            bible_raw = bible_raw.split("```")[1]
            if bible_raw.startswith("json"):
                bible_raw = bible_raw[4:]
            bible_raw = bible_raw.strip()
        if bible_raw:
            try:
                story_bible = json.loads(bible_raw)
                data["story_bible"] = story_bible
                log("Story bible verified and corrected.")
            except Exception as e:
                log(f"Story bible verification parse error: {e}. Using original.", level="WARNING")
        if cost_tracker:
            usage = bible_verify_response.usage
            cost = calc_text_cost("gpt-5.5", usage.prompt_tokens, usage.completion_tokens)
            cost_tracker.add("Story bible verification (GPT-4o)", cost, f"{usage.prompt_tokens} in / {usage.completion_tokens} out tokens")

        # --- Verification pass 2: Check beat prompts for story inconsistencies ---
        log("Verifying beat prompts for story consistency...")
        beats_for_verify = json.dumps(
            [{"beat": b["beat"], "text": b["text"], "prompt": b["prompt"]} for b in beats],
            indent=2
        )
        prompt_verify_prompt = (
            "You are a careful story editor reviewing image generation prompts for a video.\n\n"
            "Below is a story transcript and a list of image prompts — one per story beat. "
            "Your job is to check each prompt against the transcript and identify any inconsistencies, such as:\n"
            "- A character appearing in a location they haven't reached yet or have already left\n"
            "- An object being shown that a character doesn't have in that part of the story\n"
            "- A visual detail that contradicts what the story describes at that moment\n"
            "- A flashback or memory being depicted as present reality\n"
            "- Any anachronism or continuity error\n\n"
            "For each inconsistent prompt, rewrite only the problematic portion to align with the story. "
            "Do not change the style, composition, or overall visual intent — only fix factual inconsistencies.\n\n"
            "Return a JSON array with the same structure as the input beats, with corrected prompts where needed. "
            "Return only the JSON array with no preamble or markdown formatting.\n\n"
            f"Beats (prompts to verify):\n{beats_for_verify}\n\n"
            f"Transcript:\n{transcript}"
        )
        prompt_verify_response = client.chat.completions.create(
            model="gpt-5.5",
            messages=[{"role": "user", "content": prompt_verify_prompt}],
        )
        prompt_raw = prompt_verify_response.choices[0].message.content.strip()
        if prompt_raw.startswith("```"):
            prompt_raw = prompt_raw.split("```")[1]
            if prompt_raw.startswith("json"):
                prompt_raw = prompt_raw[4:]
            prompt_raw = prompt_raw.strip()
        if prompt_raw:
            try:
                verified_beats = json.loads(prompt_raw)
                verified_map = {b["beat"]: b["prompt"] for b in verified_beats}
                for beat in beats:
                    if beat["beat"] in verified_map:
                        beat["prompt"] = verified_map[beat["beat"]]
                data["beats"] = beats
                log("Beat prompts verified and corrected.")
            except Exception as e:
                log(f"Prompt verification parse error: {e}. Using original prompts.", level="WARNING")
        if cost_tracker:
            usage = prompt_verify_response.usage
            cost = calc_text_cost("gpt-5.5", usage.prompt_tokens, usage.completion_tokens)
            cost_tracker.add("Prompt verification (GPT-4o)", cost, f"{usage.prompt_tokens} in / {usage.completion_tokens} out tokens")

        # --- Save verified data ---
        safe_write(
            lambda: open(beats_file, "w", encoding="utf-8").write(json.dumps(data, indent=2)),
            str(beats_file)
        )
        log(f"Beats, character list, and verified story bible saved: {beats_file}")

    log(f"Transcript broken into {len(beats)} beats.")
    log(f"Characters identified: {[c['name'] for c in characters]}")

    # --- Step 2: Generate style reference image ---
    style_ref_path = references_dir / "style_reference.png"
    if style_ref_path.exists():
        log("Style reference already exists, skipping.")
    else:
        log("Generating style reference image...")
        style_prompt = (
            f"An establishing scene in the style of {style}. "
            f"A warm, inviting storytelling atmosphere with rich colors and textures. "
            f"No characters, no text, no letters, no words, no written language of any kind. Pure environment and mood."
        )
        response = client.images.generate(
            model="gpt-image-2",
            prompt=style_prompt,
            size="1536x1024",
            quality="medium",
            n=1,
        )
        save_image(response.data[0].b64_json, style_ref_path)
        print(f"Style reference saved: {style_ref_path}")
        if cost_tracker:
            from pricing import calc_image_cost_from_usage
            usage = getattr(response, 'usage', None)
            cost = calc_image_cost_from_usage(usage) if usage else 0.042
            cost_tracker.add("Style reference image (gpt-image-2)", cost)

    # --- Step 3: Generate character reference images ---
    character_image_paths = {}
    for character in characters:
        name = character["name"]
        safe_name = name.lower().replace(" ", "_")
        char_path = references_dir / f"character_{safe_name}.png"
        character_image_paths[name] = char_path

        if char_path.exists():
            log(f"Character reference for '{name}' already exists, skipping.")
            continue

        log(f"Generating character reference for '{name}'...")
        # Use description only in prompt — avoid using the character name directly
        # as it may contain historically sensitive terms that trigger moderation
        char_prompt = (
            f"A character sheet. {character['description']}. "
            f"Full body view, neutral pose, clear lighting. Style: {style}. "
            f"No text, no labels, no letters, no words, no written language of any kind in the image."
        )

        # Try generating with moderation retry
        import time as _time
        char_response = None
        for char_attempt in range(2):
            try:
                if style_ref_path.exists():
                    image_files = [open(style_ref_path, "rb")]
                    try:
                        char_response = client.images.edit(
                            model="gpt-image-2",
                            image=image_files,
                            prompt=char_prompt,
                            size="1536x1024",
                            quality="medium",
                            n=1,
                        )
                    finally:
                        for f in image_files:
                            f.close()
                else:
                    char_response = client.images.generate(
                        model="gpt-image-2",
                        prompt=char_prompt,
                        size="1536x1024",
                        quality="medium",
                        n=1,
                    )
                break
            except BadRequestError as e:
                error_str = str(e)
                if "moderation_blocked" in error_str and char_attempt == 0:
                    violation_type = None
                    try:
                        import re as _re
                        match = _re.search(r'safety_violations=\[([^\]]+)\]', error_str)
                        if match:
                            violation_type = match.group(1).strip()
                    except Exception:
                        pass
                    log(f"Character sheet for '{name}' blocked (type: {violation_type or 'unknown'}), rewriting prompt...", level="WARNING")
                    char_prompt = rewrite_prompt(client, char_prompt, violation_type=violation_type)
                else:
                    log(f"Character sheet for '{name}' failed after retry: {e}", level="ERROR")
                    log(f"Skipping character sheet for '{name}'...", level="WARNING")
                    break
            except APIConnectionError as e:
                log(f"Connection error generating character sheet for '{name}', retrying...", level="WARNING")
                _time.sleep(10)

        if char_response:
            save_image(char_response.data[0].b64_json, char_path)
            log(f"Character reference saved: {char_path}")
            if cost_tracker:
                from pricing import calc_image_cost_from_usage
                usage = getattr(char_response, 'usage', None)
                cost = calc_image_cost_from_usage(usage) if usage else 0.042
                cost_tracker.add(f"Character sheet: {name} (gpt-image-2)", cost)

    # --- Step 3b: Generate location reference images ---
    # Find all locations that appear in more than one beat
    location_beat_counts = {}
    for beat in beats:
        for setting in beat.get("settings", []):
            if setting and setting.lower() != "unspecified":
                location_beat_counts[setting] = location_beat_counts.get(setting, 0) + 1

    recurring_locations = {name: count for name, count in location_beat_counts.items() if count > 1}
    log(f"Recurring locations for reference images: {list(recurring_locations.keys())}")

    location_image_paths = {}
    for loc_name in recurring_locations:
        safe_name = loc_name.lower().replace(" ", "_").replace("'", "")
        loc_path = references_dir / f"location_{safe_name}.png"
        location_image_paths[loc_name] = loc_path

        if loc_path.exists():
            log(f"Location reference for '{loc_name}' already exists, skipping.")
            continue

        # Look up description from story bible
        loc_description = loc_name
        for setting in story_bible.get("settings", []):
            if setting["name"].lower() == loc_name.lower():
                loc_description = setting["description"]
                constraints = setting.get("constraints", [])
                if constraints:
                    loc_description += " " + " ".join(constraints)
                break

        log(f"Generating location reference for '{loc_name}'...")
        loc_prompt = (
            f"An empty establishing shot of this location: {loc_description}. "
            f"Style: {style}. No characters or people present. "
            f"No text, letters, words, or written language anywhere in the image."
        )
        if style_ref_path.exists():
            image_files = [open(style_ref_path, "rb")]
            try:
                response = client.images.edit(
                    model="gpt-image-2",
                    image=image_files,
                    prompt=loc_prompt,
                    size="1536x1024",
                    quality="medium",
                    n=1,
                )
            finally:
                for f in image_files:
                    f.close()
        else:
            response = client.images.generate(
                model="gpt-image-2",
                prompt=loc_prompt,
                size="1536x1024",
                quality="medium",
                n=1,
            )
        save_image(response.data[0].b64_json, loc_path)
        log(f"Location reference saved: {loc_path}")
        if cost_tracker:
            from pricing import calc_image_cost_from_usage
            usage = getattr(response, 'usage', None)
            cost = calc_image_cost_from_usage(usage) if usage else 0.042
            cost_tracker.add(f"Location reference: {loc_name} (gpt-image-2)", cost)

    # --- Step 3c: Reference image review loop ---
    # Check for any missing character or location references and offer to retry
    import time as _time
    while True:
        missing_refs = []

        # Check character sheets
        for character in characters:
            name = character["name"]
            char_path = character_image_paths.get(name)
            if char_path and not char_path.exists():
                missing_refs.append(("character", name, character, char_path))

        # Check location references
        for loc_name, loc_path in location_image_paths.items():
            if not loc_path.exists():
                missing_refs.append(("location", loc_name, None, loc_path))

        if not missing_refs:
            log("All reference images accounted for.")
            break

        missing_names = [f"{rtype}: {name}" for rtype, name, _, _ in missing_refs]
        print(f"\n  ⚠️  The following reference images were not generated:")
        for n in missing_names:
            print(f"    - {n}")
        retry = input(f"  Retry generating these {len(missing_refs)} reference image(s)? Press Enter to retry, or type 'skip' to continue: ").strip().lower()
        if retry == "skip":
            log(f"User skipped reference image retry. Missing: {missing_names}", level="WARNING")
            break

        log(f"Retrying {len(missing_refs)} missing reference image(s)...")
        for rtype, name, character, ref_path in missing_refs:
            if rtype == "character":
                char_prompt = (
                    f"A character sheet. {character['description']}. "
                    f"Full body view, neutral pose, clear lighting. Style: {style}. "
                    f"No text, no labels, no letters, no words, no written language of any kind in the image."
                )
                for attempt in range(2):
                    try:
                        if style_ref_path.exists():
                            image_files = [open(style_ref_path, "rb")]
                            try:
                                ref_response = client.images.edit(
                                    model="gpt-image-2",
                                    image=image_files,
                                    prompt=char_prompt,
                                    size="1536x1024",
                                    quality="medium",
                                    n=1,
                                )
                            finally:
                                for f in image_files:
                                    f.close()
                        else:
                            ref_response = client.images.generate(
                                model="gpt-image-2",
                                prompt=char_prompt,
                                size="1536x1024",
                                quality="medium",
                                n=1,
                            )
                        save_image(ref_response.data[0].b64_json, ref_path)
                        log(f"Character reference retry saved: {ref_path}")
                        if cost_tracker:
                            from pricing import calc_image_cost_from_usage
                            usage = getattr(ref_response, "usage", None)
                            cost = calc_image_cost_from_usage(usage) if usage else 0.042
                            cost_tracker.add(f"Character sheet retry: {name} (gpt-image-2)", cost)
                        break
                    except BadRequestError as e:
                        error_str = str(e)
                        if "moderation_blocked" in error_str and attempt == 0:
                            violation_type = None
                            try:
                                import re as _re
                                match = _re.search(r'safety_violations=\[([^\]]+)\]', error_str)
                                if match:
                                    violation_type = match.group(1).strip()
                            except Exception:
                                pass
                            log(f"Character sheet retry for '{name}' blocked ({violation_type or 'unknown'}), rewriting...", level="WARNING")
                            char_prompt = rewrite_prompt(client, char_prompt, violation_type=violation_type)
                        else:
                            log(f"Character sheet retry for '{name}' failed: {e}", level="ERROR")
                            break
                    except APIConnectionError:
                        log(f"Connection error on character sheet retry for '{name}'", level="WARNING")
                        _time.sleep(10)

            elif rtype == "location":
                loc_description = name
                for setting in story_bible.get("settings", []):
                    if setting["name"].lower() == name.lower():
                        loc_description = setting["description"]
                        constraints = setting.get("constraints", [])
                        if constraints:
                            loc_description += " " + " ".join(constraints)
                        break
                loc_prompt = (
                    f"An empty establishing shot of this location: {loc_description}. "
                    f"Style: {style}. No characters or people present. "
                    f"No text, letters, words, or written language anywhere in the image."
                )
                for attempt in range(2):
                    try:
                        if style_ref_path.exists():
                            image_files = [open(style_ref_path, "rb")]
                            try:
                                ref_response = client.images.edit(
                                    model="gpt-image-2",
                                    image=image_files,
                                    prompt=loc_prompt,
                                    size="1536x1024",
                                    quality="medium",
                                    n=1,
                                )
                            finally:
                                for f in image_files:
                                    f.close()
                        else:
                            ref_response = client.images.generate(
                                model="gpt-image-2",
                                prompt=loc_prompt,
                                size="1536x1024",
                                quality="medium",
                                n=1,
                            )
                        save_image(ref_response.data[0].b64_json, ref_path)
                        log(f"Location reference retry saved: {ref_path}")
                        if cost_tracker:
                            from pricing import calc_image_cost_from_usage
                            usage = getattr(ref_response, "usage", None)
                            cost = calc_image_cost_from_usage(usage) if usage else 0.042
                            cost_tracker.add(f"Location reference retry: {name} (gpt-image-2)", cost)
                        break
                    except BadRequestError as e:
                        error_str = str(e)
                        if "moderation_blocked" in error_str and attempt == 0:
                            violation_type = None
                            try:
                                import re as _re
                                match = _re.search(r'safety_violations=\[([^\]]+)\]', error_str)
                                if match:
                                    violation_type = match.group(1).strip()
                            except Exception:
                                pass
                            log(f"Location reference retry for '{name}' blocked ({violation_type or 'unknown'}), rewriting...", level="WARNING")
                            loc_prompt = rewrite_prompt(client, loc_prompt, violation_type=violation_type)
                        else:
                            log(f"Location reference retry for '{name}' failed: {e}", level="ERROR")
                            break
                    except APIConnectionError:
                        log(f"Connection error on location reference retry for '{name}'", level="WARNING")
                        _time.sleep(10)

    # --- Step 4: Generate story images ---
    previous_image_path = None

    for beat in beats:
        beat_num = beat["beat"]
        prompt = beat["prompt"]
        beat_characters = beat.get("characters", [])
        output_path = images_dir / f"image_{beat_num:03d}.png"

        # Resume: skip if image already exists
        if output_path.exists():
            log(f"Image {beat_num} already exists, skipping.")
            previous_image_path = output_path
            continue

        log(f"Generating image {beat_num} of {len(beats)}...")

        # Beat 1 is the title card — skip bible context entirely, use prompt as-is
        if beat_num == 1:
            enriched_prompt = prompt
        else:
            # Enrich prompt with story bible context and mandatory scene header
            bible_context = build_bible_context(beat, story_bible, all_characters=characters)
            # Prepend rich style reinforcement to the front of the prompt
            style_reinforcement = (
                f"STYLE REQUIREMENT (highest priority): This image must be {get_style_detail(style)}. "
                f"The style must be immediately recognizable and must not drift toward generic AI illustration. "
            )
            enriched_prompt = style_reinforcement + prompt + bible_context

        # Build reference image list:
        # Order: style reference, location reference(s), character sheets, previous image last
        reference_images = []
        reference_images.append(style_ref_path)
        # Add location reference if this beat has a named location
        beat_settings = beat.get("settings", [])
        for setting_name in beat_settings:
            if setting_name in location_image_paths:
                loc_path = location_image_paths[setting_name]
                if loc_path.exists():
                    reference_images.append(loc_path)
        # Add character references
        for char_name in beat_characters:
            if char_name in character_image_paths:
                char_path = character_image_paths[char_name]
                if char_path.exists():
                    reference_images.append(char_path)
        if previous_image_path:
            reference_images.append(previous_image_path)

        # Cap at 10 images (API limit)
        reference_images = reference_images[:10]

        # Beat 1 is always generated fresh — no reference images — for reliable text rendering
        is_title_card = (beat_num == 1)

        # Try generating — retry on connection errors, rewrite on moderation blocks
        MAX_RETRIES = 3
        RETRY_DELAY = 10  # seconds between connection error retries
        import time as _time

        for attempt in range(MAX_RETRIES + 1):  # +1 for initial moderation rewrite
            try:
                if is_title_card or not reference_images:
                    response = client.images.generate(
                        model="gpt-image-2",
                        prompt=enriched_prompt,
                        size="1536x1024",
                        quality="medium",
                        n=1,
                    )
                    image_b64 = response.data[0].b64_json
                else:
                    image_files = [open(p, "rb") for p in reference_images]
                    try:
                        response = client.images.edit(
                            model="gpt-image-2",
                            image=image_files,
                            prompt=enriched_prompt,
                            size="1536x1024",
                            quality="medium",
                            n=1,
                        )
                        image_b64 = response.data[0].b64_json
                    finally:
                        for f in image_files:
                            f.close()

                save_image(image_b64, output_path)
                log(f"Image saved: {output_path}")
                previous_image_path = output_path
                if cost_tracker:
                    from pricing import calc_image_cost_from_usage
                    usage = getattr(response, 'usage', None)
                    cost = calc_image_cost_from_usage(usage) if usage else 0.042
                    cost_tracker.add(f"Story image {beat_num:03d} (gpt-image-2)", cost)
                break

            except APIConnectionError as e:
                if attempt < MAX_RETRIES:
                    log(f"Beat {beat_num} — connection error, retrying in {RETRY_DELAY}s (attempt {attempt + 1}/{MAX_RETRIES})...", level="WARNING")
                    _time.sleep(RETRY_DELAY)
                else:
                    log(f"Beat {beat_num} — connection error after {MAX_RETRIES} retries, skipping.", level="ERROR")
                    break

            except BadRequestError as e:
                error_str = str(e)
                if "moderation_blocked" in error_str and attempt == 0:
                    violation_type = None
                    try:
                        import re as _re
                        match = _re.search(r'safety_violations=\[([^\]]+)\]', error_str)
                        if match:
                            violation_type = match.group(1).strip()
                    except Exception:
                        pass
                    log(f"Beat {beat_num} blocked by moderation (type: {violation_type or 'unknown'}), rewriting prompt...")
                    prompt = rewrite_prompt(client, enriched_prompt, violation_type=violation_type)
                else:
                    log(f"ERROR on beat {beat_num}: {e}", level="ERROR")
                    log(f"Skipping beat {beat_num} and continuing...", level="WARNING")
                    break

    # --- Post-generation review: retry any missing images ---
    while True:
        missing = []
        for beat in beats:
            output_path = images_dir / f"image_{beat['beat']:03d}.png"
            if not output_path.exists():
                missing.append(beat["beat"])

        if not missing:
            log("All story images accounted for.")
            break

        print(f"\n  ⚠️  The following images were not generated: {missing}")
        retry = input(f"  Retry generating these {len(missing)} image(s)? Press Enter to retry, or type 'skip' to finish: ").strip().lower()
        if retry == "skip":
            log(f"User skipped retry. Missing images: {missing}", level="WARNING")
            break

        # Retry missing images
        log(f"Retrying {len(missing)} missing image(s)...")
        for beat in beats:
            beat_num = beat["beat"]
            if beat_num not in missing:
                continue

            prompt = beat["prompt"]
            beat_characters = beat.get("characters", [])
            output_path = images_dir / f"image_{beat_num:03d}.png"

            if beat_num == 1:
                enriched_prompt = prompt
            else:
                bible_context = build_bible_context(beat, story_bible, all_characters=characters)
                style_reinforcement = (
                    f"STYLE REQUIREMENT (highest priority): This image must be {get_style_detail(style)}. "
                    f"The style must be immediately recognizable and must not drift toward generic AI illustration. "
                )
                enriched_prompt = style_reinforcement + prompt + bible_context

            reference_images = []
            reference_images.append(style_ref_path)
            beat_settings = beat.get("settings", [])
            for setting_name in beat_settings:
                if setting_name in location_image_paths:
                    loc_path = location_image_paths[setting_name]
                    if loc_path.exists():
                        reference_images.append(loc_path)
            for char_name in beat_characters:
                if char_name in character_image_paths:
                    char_path = character_image_paths[char_name]
                    if char_path.exists():
                        reference_images.append(char_path)
            if previous_image_path and previous_image_path.exists():
                reference_images.append(previous_image_path)
            reference_images = reference_images[:10]

            is_title_card = (beat_num == 1)

            import time as _time
            for attempt in range(MAX_RETRIES + 1):
                try:
                    if is_title_card or not reference_images:
                        response = client.images.generate(
                            model="gpt-image-2",
                            prompt=enriched_prompt,
                            size="1536x1024",
                            quality="medium",
                            n=1,
                        )
                        image_b64 = response.data[0].b64_json
                    else:
                        image_files = [open(p, "rb") for p in reference_images]
                        try:
                            response = client.images.edit(
                                model="gpt-image-2",
                                image=image_files,
                                prompt=enriched_prompt,
                                size="1536x1024",
                                quality="medium",
                                n=1,
                            )
                            image_b64 = response.data[0].b64_json
                        finally:
                            for f in image_files:
                                f.close()

                    save_image(image_b64, output_path)
                    log(f"Retry: image {beat_num} saved.")
                    previous_image_path = output_path
                    if cost_tracker:
                        from pricing import calc_image_cost_from_usage
                        usage = getattr(response, "usage", None)
                        cost = calc_image_cost_from_usage(usage) if usage else 0.042
                        cost_tracker.add(f"Story image {beat_num:03d} retry (gpt-image-2)", cost)
                    break

                except APIConnectionError as e:
                    if attempt < MAX_RETRIES:
                        log(f"Beat {beat_num} retry — connection error, waiting {RETRY_DELAY}s...", level="WARNING")
                        _time.sleep(RETRY_DELAY)
                    else:
                        log(f"Beat {beat_num} retry — connection error after {MAX_RETRIES} attempts, skipping.", level="ERROR")
                        break

                except BadRequestError as e:
                    error_str = str(e)
                    if "moderation_blocked" in error_str and attempt == 0:
                        violation_type = None
                        try:
                            import re as _re
                            match = _re.search(r'safety_violations=\[([^\]]+)\]', error_str)
                            if match:
                                violation_type = match.group(1).strip()
                        except Exception:
                            pass
                        log(f"Beat {beat_num} retry blocked (type: {violation_type or 'unknown'}), rewriting...", level="WARNING")
                        enriched_prompt = rewrite_prompt(client, enriched_prompt, violation_type=violation_type)
                    else:
                        log(f"Beat {beat_num} retry failed: {e}", level="ERROR")
                        break

    log(f"Story image phase complete.")
    log_module_end("Image Generation")

    # --- Step 5: Generate end slides ---
    the_end_path = images_dir / "the_end.png"
    credits_path = images_dir / "credits.png"
    themes_file = output_dir / f"{word_directory}_themes.txt"

    # Extract story themes — skip if both end slides already exist
    story_themes = ""
    if the_end_path.exists() and credits_path.exists():
        if themes_file.exists():
            story_themes = themes_file.read_text(encoding="utf-8").strip()
            log(f"Story themes loaded from cache: {story_themes}")
        else:
            log("End slides already exist — skipping story themes extraction.")
    else:
        # Load from cache if available, otherwise extract
        if themes_file.exists():
            story_themes = themes_file.read_text(encoding="utf-8").strip()
            log(f"Story themes loaded from cache: {story_themes}")
        elif transcript:
            try:
                themes_response = client.chat.completions.create(
                    model="gpt-5.5",
                    messages=[{
                        "role": "user",
                        "content": (
                            f"Read this story transcript and describe in one sentence (20-30 words) "
                            f"the 2-3 dominant visual themes — the mood, setting, and key motifs — "
                            f"that should inform the atmosphere and color palette of illustrated title cards for this story. "
                            f"Be specific and visual. Return only the sentence, no preamble.\n\n{transcript}"
                        )
                    }],
                )
                story_themes = themes_response.choices[0].message.content.strip()
                log(f"Story themes for end slides: {story_themes}")
                safe_write(
                    lambda: themes_file.write_text(story_themes, encoding="utf-8"),
                    str(themes_file)
                )
                if cost_tracker:
                    from pricing import calc_text_cost
                    usage = themes_response.usage
                    cost = calc_text_cost("gpt-5.5", usage.prompt_tokens, usage.completion_tokens)
                    cost_tracker.add("Story themes extraction (gpt-5.5)", cost,
                                     f"{usage.prompt_tokens} in / {usage.completion_tokens} out tokens")
            except Exception as e:
                log(f"Could not extract story themes: {e}", level="WARNING")

    generate_end_slides(client, images_dir, style, word_of_day,
                        show_name=show_name,
                        end_top_text=end_top_text,
                        end_bottom_text=end_bottom_text,
                        credits_top_text=credits_top_text,
                        credits_bottom_text=credits_bottom_text,
                        story_themes=story_themes)


if __name__ == "__main__":
    from style_selector import select_style
    from pipeline_helpers import prompt_with_found, load_style, load_word_of_day
    directory = input("Enter word directory: ").strip()
    if not directory:
        print("No directory entered. Exiting.")
    else:
        # --- Option to regenerate specific images ---
        from pathlib import Path as _Path
        from pipeline_helpers import story_path as _sp
        images_dir = _sp(directory) / "images"
        if images_dir.exists():
            print("  You can enter image numbers (e.g. 1,5,12), 'end' for the end card, 'credits' for the credits slide, or any combination (e.g. 1,end,credits).")
            regen_input = input("  Regenerate specific images? Enter values or press Enter to skip: ").strip()
            if regen_input:
                named_map = {
                    "end": images_dir / "the_end.png",
                    "credits": images_dir / "credits.png",
                }
                for token in [t.strip() for t in regen_input.split(",")]:
                    if token.lower() in named_map:
                        img_path = named_map[token.lower()]
                        if img_path.exists():
                            img_path.unlink()
                            print(f"  Deleted {img_path.name} — will regenerate.")
                        else:
                            print(f"  {img_path.name} not found, skipping.")
                    else:
                        try:
                            num = int(token)
                            img_path = images_dir / f"image_{num:03d}.png"
                            if img_path.exists():
                                img_path.unlink()
                                print(f"  Deleted image_{num:03d}.png — will regenerate.")
                            else:
                                print(f"  image_{num:03d}.png already absent — will regenerate.")
                        except ValueError:
                            print(f"  '{token}' not recognized — skipping.")

        word = load_word_of_day(directory)
        print("\n(Press Enter to accept the default for any option)")
        found_style = load_style(directory)
        selected_style = prompt_with_found("Style", found_style, default=None)
        if not selected_style:
            selected_style = select_style(directory)
        show = prompt_with_found("Show name", None, default="The Drew Baderson Story Moment")
        episode = prompt_with_found("Episode title", word, default=word)
        end_top = prompt_with_found("Top text for end card", None, default="The End")
        end_bottom = prompt_with_found("Bottom text for end card", None, default=show)
        credits_top = prompt_with_found("Top text for credits", None, default="All Music by Casey Connor")
        credits_bottom = prompt_with_found("Bottom text for credits", None, default="caseyconnor.org")
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
        anchor = prompt_with_found("Intro anchor phrase", None, default="this is what I wrote")
        generate_images(directory, selected_style,
                        show_name=show,
                        episode_title=episode,
                        end_top_text=end_top,
                        end_bottom_text=end_bottom,
                        credits_top_text=credits_top,
                        credits_bottom_text=credits_bottom,
                        anchor_phrase=anchor,
                        cost_tracker=tracker)
        end_time = datetime.now()
        tracker.print_total()
        tracker.save_to_log()
        save_run_log(
            word_directory=directory,
            style=selected_style,
            total_cost=tracker.total_cost,
            start_time=start_time,
            end_time=end_time,
            pricing_warning=tracker.pricing_warning,
        )
        if notify_email:
            log_text = load_activity_log_text(directory)
            notify(
                subject=f"DBSM Image Generator complete: {directory}",
                body=f"Duration: {end_time - start_time}\nCost: ${tracker.total_cost:.4f}\n\n{log_text}",
                to_address=notify_email,
            )
