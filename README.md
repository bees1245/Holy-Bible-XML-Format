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

## Phonetic analysis helpers

This repository now also contains a Python implementation of the "phonetic trailing
song" analysis model together with a small HTTP interface.  The tools live next to
the XML sources so that they can be versioned alongside the texts they operate on.

## Repository chunk planner

For contributors that need to work with the very large XML corpus in smaller
installments, the repository ships with a helper script that breaks files into
deterministic 3000-line chunks (configurable) and can optionally emit the plan as
JSON for automation.

```bash
python repo_chunk_planner.py --chunk-size 3000 --export-json chunk_plan.json
```

The script can be pointed at any directory, filtered with `--include`/`--exclude`
glob patterns, and asked to repeat the resulting chunk list multiple times to
support multi-iteration workflows.

### Command line usage

```
python phonetic_trailing_song_model.py --help
```

Examples:

* Analyse standard input and print the full textual report.

  ```bash
  python phonetic_trailing_song_model.py <<'EOF'
  Amazing grace how sweet the sound
  That saved a wretch like me
  I once was lost but now am found
  Was blind but now I see
  EOF
  ```

* Compare two files and emit a machine-readable bundle:

  ```bash
  python phonetic_trailing_song_model.py first.txt \
      --compare-with second.txt \
      --compare-window-size 3 \
      --no-report --emit-data-bundle bundle.json
  ```

### HTTP API

`phonetic_api.py` exposes the same functionality through a minimal WSGI app that
ships with a convenience launcher:

```
python phonetic_api.py --host 0.0.0.0 --port 8000
```

Once running you can:

* `GET /` or `GET /metadata` for endpoint discovery.
* `GET /health` for a readiness check.
* `POST /analyze` with a JSON body containing `text`, optional
  `reference_text`, `title`, `reference_title`, and the various options for
  reports, bundles, and comparability windows.
* `POST /data-bundle` to export only the prepared data generation payload.

All responses are UTF-8 encoded JSON and CORS headers are emitted so that the
API can be queried directly from in-browser prototypes.
