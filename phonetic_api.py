"""WSGI wrapper exposing the phonetic trailing song model over HTTP."""

from __future__ import annotations

import json
from typing import Callable, Dict, Iterable

from phonetic_trailing_song_model import (
    PhoneticTrailingSongModel,
    prepare_data_generation,
)


class PhoneticAPI:
    """Minimal WSGI application for phonetic analysis."""

    def __init__(self) -> None:
        self._model = PhoneticTrailingSongModel()

    # The WSGI application interface -------------------------------------------------
    def __call__(self, environ: Dict[str, object], start_response: Callable) -> Iterable[bytes]:
        method = environ.get("REQUEST_METHOD", "GET")
        path = environ.get("PATH_INFO", "/") or "/"

        if method == "GET" and path == "/health":
            return self._respond_json(start_response, {"status": "ok"})
        if method == "GET" and path == "/":
            metadata = {
                "name": "phonetic-trailing-song-api",
                "endpoints": [
                    {"path": "/", "method": "GET", "description": "API metadata"},
                    {"path": "/health", "method": "GET", "description": "Liveness check"},
                    {
                        "path": "/analyze",
                        "method": "POST",
                        "description": "Run phonetic analysis and optional bundle export",
                    },
                ],
                "capabilities": [
                    "structured_analysis",
                    "silent_language_titles",
                    "read_aloud_codex",
                    "knowledge_base",
                    "data_bundle_export",
                ],
                "languages": [
                    {
                        "code": profile.code,
                        "name": profile.name,
                        "description": profile.description,
                    }
                    for profile in self._model.language_registry.list_profiles()
                ],
            }
            return self._respond_json(start_response, metadata)
        if method == "POST" and path == "/analyze":
            return self._handle_analyze(environ, start_response)

        return self._respond_json(start_response, {"error": "Not found"}, status="404 Not Found")

    # ------------------------------------------------------------------
    def _handle_analyze(self, environ: Dict[str, object], start_response: Callable) -> Iterable[bytes]:
        try:
            length = int(environ.get("CONTENT_LENGTH") or 0)
        except (ValueError, TypeError):
            length = 0
        wsgi_input = environ.get("wsgi.input")
        if hasattr(wsgi_input, "read"):
            body_bytes = wsgi_input.read(length)
        else:
            body_bytes = b""
        payload = body_bytes.decode("utf-8") if body_bytes else "{}"
        try:
            data = json.loads(payload or "{}")
        except json.JSONDecodeError:
            return self._respond_json(
                start_response,
                {"error": "Invalid JSON payload"},
                status="400 Bad Request",
            )

        text = data.get("text")
        reference_text = data.get("reference_text")
        options = data.get("options", {}) if isinstance(data.get("options"), dict) else {}
        title = data.get("title") or options.get("title")

        if not text:
            return self._respond_json(
                start_response,
                {"error": "Missing 'text' field"},
                status="400 Bad Request",
            )

        include_bundle = bool(options.get("include_bundle"))
        window_size = int(options.get("window_size") or 4)
        include_report = options.get("include_report", True)
        include_comparability = options.get("include_comparability", bool(reference_text))
        language_code = data.get("language_code") or options.get("language_code")

        analysis = self._model.analyze(
            text,
            title=title,
            reference_text=reference_text,
            seed_from=data.get("seed_from"),
            window_size=window_size,
            compare=bool(reference_text and include_comparability),
            language_code=language_code,
        )

        response = {
            "analysis": analysis.as_dict(),
        }

        if include_bundle:
            bundle = prepare_data_generation(
                text,
                title=title,
                reference_text=reference_text,
                window_size=window_size,
                include_report=include_report,
                include_comparability=include_comparability,
                language_code=language_code,
            )
            response["bundle"] = bundle.as_dict()

        return self._respond_json(start_response, response)

    # ------------------------------------------------------------------
    def _respond_json(
        self,
        start_response: Callable,
        payload: Dict[str, object],
        *,
        status: str = "200 OK",
    ) -> Iterable[bytes]:
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        start_response(status, [("Content-Type", "application/json"), ("Content-Length", str(len(body)))])
        return [body]


def make_app() -> PhoneticAPI:
    return PhoneticAPI()


if __name__ == "__main__":  # pragma: no cover
    from wsgiref.simple_server import make_server
    import argparse

    parser = argparse.ArgumentParser(description="Run the phonetic API")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()

    app = PhoneticAPI()
    with make_server(args.host, args.port, app) as httpd:
        print(f"Serving on http://{args.host}:{args.port}")
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            print("Stopping server…")
