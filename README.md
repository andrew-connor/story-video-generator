# Story Video Generator

An automated pipeline that turns a recorded spoken-word story into a fully edited, illustrated video and uploads it to YouTube.

It was built for **The Drew Baderson Story Moment** — a show where the host has one hour to write and record a short story sparked by a random dictionary word — but the defaults are configurable, so it adapts to any similar narrated-story format.

Give it a `.wav` of the recording and it transcribes the audio, picks an art style, generates a sequence of illustrations timed to the narration, assembles everything into an MP4 with crossfades, and (as a separate, reviewed step) publishes to YouTube with a description, tags, thumbnail, and captions.

## How it works

The run is split in two on purpose. `main.py` takes a recording through assembly and then **stops** so you can watch the video before anything goes public. Publishing is a separate command.

**`main.py` — steps 1–4:**

1. **Transcription** — Compresses the WAV to MP3 and runs it through Whisper. Produces a `.txt` transcript and an `.srt` with timings. Skips if both already exist.
2. **Style selection** — Picks an unused style at random from `styles.csv`. You can accept it, roll again, or type a specific style name. Usage counts are tracked so styles don't repeat until the set is exhausted.
3. **Image generation** — The heart of the pipeline. It reads the transcript to work out how many "beats" the story needs (roughly one image per ten seconds), then uses the text model to build a character list, a story bible, and a prompt per beat. Before generating the story images it produces a style reference, character sheets, and location references, and runs two verification passes (a bible fact-check and a beat-prompt consistency check). Story images are generated style-first, with reference images fed back in for consistency. Missing references or beats can be retried through a review loop.
4. **Video assembly** — Matches each beat to its place in the narration using the SRT timings, drops any beat whose image isn't on disk, and renders an MP4 (H.264/AAC) with crossfades and a short static hold at each image's peak moment.

**`upload_to_youtube.py` — steps 5–6, run after you've reviewed the video:**

5. **Transcript correction** — Compares the original written script (`.docx`) against the Whisper transcript and flags likely misspellings of names, places, and invented terms. You approve corrections individually or in bulk; they're written back to the `.srt` and `.txt`. Skipped if there's no `.docx`.
6. **YouTube upload** — Generates a teaser description and tags, uploads the video with its thumbnail, adds it to a playlist, and uploads the captions. The YouTube record is saved immediately after upload so a re-run never double-publishes.

Every stage is resumable. Intermediate files (`_beats.json`, `_timing.json`, `_metadata.json`, `_youtube.json`, and the rest) are cached, so re-running a story picks up where it left off rather than regenerating work.

## Project layout

```
.
├── main.py                  # Steps 1–4: transcribe → style → images → assemble
├── transcribe.py            # Whisper transcription
├── style_selector.py        # Random/forced style selection from styles.csv
├── image_generator.py       # Character list, story bible, beat prompts, image generation
├── assemble_video.py        # Beat-to-timestamp matching and MP4 render
├── generate_end_slides.py   # "The End" card and credits
├── correct_transcript.py    # Compares the .docx script against the transcript
├── upload_to_youtube.py     # Steps 5–6: transcript correction + YouTube upload
├── pricing.py               # Per-model price table
├── cost_tracker.py          # Per-run API cost accounting
├── pipeline_helpers.py      # Shared paths, logging, safe file writes
├── notify.py                # End-of-run email via the Gmail API
├── styles.csv               # The art-style library
└── stories/
    └── <word>/              # One folder per episode, named for the story's word
        ├── <word>.wav        # The recording (input)
        ├── <word>.docx       # The written script (optional, used for correction)
        ├── <word>.txt / .srt # Transcript
        ├── <word>.mp4        # Final video
        ├── images/           # Generated illustrations + end card + credits
        └── references/       # Style, character, and location reference images
```

## Setup

**Python dependencies:**

```bash
pip install openai python-dotenv pydub litellm moviepy pillow requests python-docx
pip install google-api-python-client google-auth-httplib2 google-auth-oauthlib
```

**ffmpeg** is required for audio compression and video export. The pipeline expects it at `C:\ffmpeg\bin\ffmpeg.exe` and adds it to `PATH` automatically at the start of a run.

**Credentials** (see `.env.example` for the full list — none of these files are committed):

- `.env` — holds `OPENAI_API_KEY`.
- `client_secrets.json` / `youtube_token.json` — Google OAuth for YouTube uploads (scope `https://www.googleapis.com/auth/youtube`). Delete the token after any scope change to force re-auth.
- `gmail_client_secret.json` / `gmail_token.json` — a separate Google OAuth pair for the end-of-run notification email, independent of the YouTube credentials.

## Running it

A full episode:

```bash
python main.py        # transcribe, style, illustrate, assemble — then stops
# review stories/<word>/<word>.mp4
python upload_to_youtube.py   # correct transcript, then publish
```

Each stage can also be run on its own — `transcribe.py`, `style_selector.py`, `image_generator.py`, `assemble_video.py`, and `correct_transcript.py` all take a word/episode folder and skip any work that's already cached. Every script tracks its own API costs and prints a run summary on completion.

## Defaults

Configurable in the scripts and prompts; current values:

- Intro anchor phrase: `this is what I wrote` · End anchor: `the end`
- End card: **The End** over the show name; credits: *All Music by Casey Connor* / *caseyconnor.org*
- Upload: category *Film & Animation*, public, not made for kids; thumbnail taken from the first generated image
- Models: `gpt-5.5` for text, `gpt-image-2` for images (set in `image_generator.py`, `assemble_video.py`, and `upload_to_youtube.py`)

## Notes

- The image step asks the model to identify two or three dominant themes before designing the title card, and feeds character/location reference images back into later prompts to keep the cast and settings visually consistent across beats.
- Beat generation stops hard at "The End" — the model is instructed never to invent post-story epilogue scenes.
- YouTube comment settings can't be set through the API; that's a manual step in YouTube Studio after upload.
- The OpenAI usage figures shown at the start of a run can lag the real total by a few hours, so treat in-run cost comparisons as approximate.
