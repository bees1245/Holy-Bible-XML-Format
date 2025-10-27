# Bible in XML
Welcome. Here you will find XML Bibles from various languages, created from the pasted 15 years.
200+ Languages and 1000+ Bible Versions.
Any questions or comments: andrey@beblia.com

Use at your own discretion, no need to ask for permission, no warranty’s.

Author: Proud Slave of Christ

Visit our site:
https://beblia.com
God Bless.
Thank you.

## Phonetic trailing song toolkit

This repository now also ships a lightweight phonetic analysis toolkit that can
generate deterministic numeric signatures for any passage of text.  The core
entry point is `phonetic_trailing_song_model.py` which exposes both a Python API
and a command-line interface.

### Command-line usage

```
python phonetic_trailing_song_model.py --title "Sample" samples.txt \
  --compare-with reference.txt --compare-window-size 3 --emit-json
```

Key options:

* `--seed-from` – supply a deterministic seed string (for example a title).
* `--select-token` – print the slider/seed metrics for a specific token index.
* `--no-report` and `--emit-json` – suppress the textual report and emit a JSON payload instead.
* `--emit-data-bundle PATH` – create a machine-learning friendly bundle with prompts and completions.
* `--language-code CODE` – force a specific language profile (otherwise the model auto-detects per token).
* `--list-languages` – print the available language profiles (covering Latin, Greek, Cyrillic, Arabic, Hebrew, Devanagari, Hangul, and Kana scripts) and exit.

### Knowledge base output

Every run now produces a structured knowledge base that captures per-token
signatures, repeating clusters, and aggregated "wisdom" insights.  The CLI
report surfaces the top entries, while JSON exports and data bundles expose the
full dataset for downstream storage engines or retrieval workflows.

### HTTP API

`phonetic_api.py` wraps the model in a tiny WSGI application.  Run it with the
built-in server:

```
python phonetic_api.py --port 8080
```

POST JSON requests to `/analyze` with the following schema:

```
{
  "title": "My Analysis",
  "text": "Amazing grace how sweet the sound",
  "reference_text": "Amazing grace a song of sound",
  "language_code": "el",
  "options": {
    "include_bundle": true,
    "window_size": 2,
    "language_code": "el"
  }
}
```

The response contains the structured analysis and, if requested, a reusable data
bundle.  The root metadata endpoint (`GET /`) now lists every built-in language
profile so clients can select a compatible transliteration strategy ahead of
time, and advertises the available capabilities (structured analysis,
knowledge-base export, and bundle generation).

### Language profiles

The new `language_profiles.py` module centralises lightweight transliteration
and detection heuristics for common scripts without relying on external
packages.  Profiles currently cover English, Spanish, French, German,
Portuguese, Greek, Russian/Ukrainian (Cyrillic), Arabic, Hebrew, Hindi
(Devanagari), Korean (Hangul), and Japanese Kana.  Custom code can instantiate
`LanguageProfileRegistry` directly or extend it with additional scripts before
passing it into `PhoneticTrailingSongModel`.

### Repository chunk planner

`repo_chunk_planner.py` creates 3000-line iteration plans that cover the XML
collection without leaving holes.  Example usage:

```
python repo_chunk_planner.py --include "*.xml" --chunk-size 3000 --export-json plan.json
```

The exported JSON documents the contiguous chunks, filler segments, and file
statistics which can be shared alongside GitHub issues or pull requests when
organising large review efforts.
