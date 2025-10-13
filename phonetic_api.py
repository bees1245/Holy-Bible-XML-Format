"""Minimal WSGI API exposing the phonetic trailing song analysis model."""

from __future__ import annotations

import json
from dataclasses import dataclass
from http import HTTPStatus
from typing import Any, Callable, Dict, Iterable, Tuple
from wsgiref.simple_server import make_server

from phonetic_trailing_song_model import (
    AnalysisResult,
    ComparabilityResult,
    DataGenerationBundle,
    PhoneticTrailingSongModel,
)


JsonIterable = Iterable[bytes]
StartResponse = Callable[[str, list[tuple[str, str]]], None]


@dataclass
class ApiOptions:
    """Options derived from the request payload for analysis endpoints."""

    include_report: bool = True
    include_reference_report: bool | None = None
    include_comparability: bool = True
    include_comparability_report: bool | None = None
    include_reference_analysis: bool = True
    include_bundle: bool = False
    bundle_include_reports: bool | None = None
    window_size: int = 5

    def normalise(self) -> None:
        """Apply default fallbacks where dependent options were omitted."""

        self.include_report = bool(self.include_report)
        self.include_comparability = bool(self.include_comparability)
        self.include_reference_analysis = bool(self.include_reference_analysis)
        self.include_bundle = bool(self.include_bundle)
        if self.include_reference_report is not None:
            self.include_reference_report = bool(self.include_reference_report)
        if self.include_comparability_report is not None:
            self.include_comparability_report = bool(self.include_comparability_report)
        if self.bundle_include_reports is not None:
            self.bundle_include_reports = bool(self.bundle_include_reports)

        if self.include_reference_report is None:
            self.include_reference_report = self.include_report
        if self.include_comparability_report is None:
            self.include_comparability_report = self.include_report
        if self.bundle_include_reports is None:
            self.bundle_include_reports = self.include_report

        try:
            self.window_size = int(self.window_size)
        except (TypeError, ValueError):
            self.window_size = 5

        if self.window_size <= 0:
            self.window_size = 1


class PhoneticAPI:
    """WSGI application providing JSON endpoints for the phonetic model."""

    _CORS_HEADERS: Tuple[Tuple[str, str], ...] = (
        ("Access-Control-Allow-Origin", "*"),
        ("Access-Control-Allow-Methods", "GET, POST, OPTIONS"),
        ("Access-Control-Allow-Headers", "Content-Type"),
    )
    _SERVICE_TITLE = "Phonetic Trailing Song API"
    _SERVICE_DESCRIPTION = (
        "JSON endpoints for the phonetic trailing song analysis and data bundle generation"
    )

    def __init__(self, model: PhoneticTrailingSongModel | None = None) -> None:
        self._model = model or PhoneticTrailingSongModel()

    # ------------------------------------------------------------------
    # WSGI plumbing
    # ------------------------------------------------------------------
    def __call__(self, environ: Dict[str, Any], start_response: StartResponse) -> JsonIterable:
        method = environ.get("REQUEST_METHOD", "GET").upper()
        path = environ.get("PATH_INFO", "") or "/"

        if method == "OPTIONS":
            return self._options_response(start_response)

        if path in ("/", "/metadata") and method == "GET":
            return self._json_response(
                start_response,
                HTTPStatus.OK,
                self._service_metadata(),
            )

        if path == "/health" and method == "GET":
            return self._json_response(start_response, HTTPStatus.OK, {"status": "ok"})

        if path == "/analyze" and method == "POST":
            return self._handle_analyze(environ, start_response)

        if path == "/data-bundle" and method == "POST":
            return self._handle_bundle(environ, start_response)

        return self._json_response(
            start_response,
            HTTPStatus.NOT_FOUND,
            {"error": "Unknown endpoint", "path": path},
        )

    # ------------------------------------------------------------------
    # Request helpers
    # ------------------------------------------------------------------
    def _options_response(self, start_response: StartResponse) -> JsonIterable:
        headers = list(self._CORS_HEADERS)
        headers.append(("Content-Length", "0"))
        headers.append(("Content-Type", "application/json"))
        start_response(f"{HTTPStatus.NO_CONTENT.value} {HTTPStatus.NO_CONTENT.phrase}", headers)
        return [b""]

    def _read_json(self, environ: Dict[str, Any]) -> Tuple[Dict[str, Any], str | None]:
        try:
            length = int(environ.get("CONTENT_LENGTH") or "0")
        except ValueError:
            length = 0

        body = b""
        stream = environ.get("wsgi.input")
        if stream is not None and length != 0:
            body = stream.read(length) if length > 0 else stream.read() or b""

        if not body:
            return {}, None

        try:
            payload = json.loads(body.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:  # pragma: no cover - defensive
            return {}, f"Invalid JSON payload: {exc}"
        if not isinstance(payload, dict):
            return {}, "JSON body must be an object"
        return payload, None

    def _json_response(
        self,
        start_response: StartResponse,
        status: HTTPStatus,
        payload: Dict[str, Any],
    ) -> JsonIterable:
        data = json.dumps(
            payload,
            ensure_ascii=False,
            separators=(",", ":"),
        ).encode("utf-8")
        headers = [
            ("Content-Type", "application/json; charset=utf-8"),
            ("Content-Length", str(len(data))),
        ]
        headers.extend(self._CORS_HEADERS)
        start_response(f"{status.value} {status.phrase}", headers)
        return [data]

    def _service_metadata(self) -> Dict[str, Any]:
        """Return descriptive metadata for clients discovering the API."""

        return {
            "title": self._SERVICE_TITLE,
            "description": self._SERVICE_DESCRIPTION,
            "endpoints": {
                "metadata": {"method": "GET", "path": "/"},
                "health": {"method": "GET", "path": "/health"},
                "analyze": {"method": "POST", "path": "/analyze"},
                "data_bundle": {"method": "POST", "path": "/data-bundle"},
            },
        }

    # ------------------------------------------------------------------
    # Endpoint handlers
    # ------------------------------------------------------------------
    def _handle_analyze(self, environ: Dict[str, Any], start_response: StartResponse) -> JsonIterable:
        payload, error = self._read_json(environ)
        if error:
            return self._json_response(start_response, HTTPStatus.BAD_REQUEST, {"error": error})

        text = payload.get("text")
        if not isinstance(text, str) or not text.strip():
            return self._json_response(
                start_response,
                HTTPStatus.BAD_REQUEST,
                {"error": "Field 'text' must be a non-empty string"},
            )

        reference_text = payload.get("reference_text")
        if reference_text is not None and not isinstance(reference_text, str):
            return self._json_response(
                start_response,
                HTTPStatus.BAD_REQUEST,
                {"error": "Field 'reference_text' must be a string when provided"},
            )

        title = payload.get("title")
        if title is not None and not isinstance(title, str):
            return self._json_response(
                start_response,
                HTTPStatus.BAD_REQUEST,
                {"error": "Field 'title' must be a string when provided"},
            )

        reference_title = payload.get("reference_title")
        if reference_title is not None and not isinstance(reference_title, str):
            return self._json_response(
                start_response,
                HTTPStatus.BAD_REQUEST,
                {"error": "Field 'reference_title' must be a string when provided"},
            )

        options_dict = payload.get("options", {})
        if not isinstance(options_dict, dict):
            return self._json_response(
                start_response,
                HTTPStatus.BAD_REQUEST,
                {"error": "Field 'options' must be an object when provided"},
            )

        valid_keys = set(ApiOptions.__dataclass_fields__.keys())
        filtered_options = {k: options_dict[k] for k in options_dict.keys() & valid_keys}
        options = ApiOptions(**filtered_options)
        options.normalise()

        analysis = self._model.analyze(text, title=title)
        response: Dict[str, Any] = {"analysis": analysis.as_dict()}

        if options.include_report:
            response["report"] = self._model.generate_report(text, analysis=analysis)

        reference_analysis: AnalysisResult | None = None
        comparability: ComparabilityResult | None = None

        if reference_text is not None:
            reference_analysis = self._model.analyze(reference_text, title=reference_title)
            if options.include_reference_analysis:
                response["reference_analysis"] = reference_analysis.as_dict()
            if options.include_reference_report:
                response["reference_report"] = self._model.generate_report(
                    reference_text,
                    analysis=reference_analysis,
                )
            if options.include_comparability:
                comparability = self._model.compare_analyses(
                    analysis,
                    reference_analysis,
                    window_size=options.window_size,
                )
                response["comparability"] = comparability.as_dict()
                if options.include_comparability_report:
                    response["comparability_report"] = self._model.format_comparability_report(
                        comparability,
                        heading="Comparability report",
                    )

        if options.include_bundle:
            bundle = self._model.prepare_data_generation(
                text,
                analysis=analysis,
                reference_text=reference_text,
                reference_analysis=reference_analysis,
                title=title,
                reference_title=reference_title,
                window_size=options.window_size,
                include_reports=bool(options.bundle_include_reports),
                comparability=comparability,
            )
            response["data_bundle"] = bundle.as_dict()

        return self._json_response(start_response, HTTPStatus.OK, response)

    def _handle_bundle(self, environ: Dict[str, Any], start_response: StartResponse) -> JsonIterable:
        payload, error = self._read_json(environ)
        if error:
            return self._json_response(start_response, HTTPStatus.BAD_REQUEST, {"error": error})

        text = payload.get("text")
        if not isinstance(text, str) or not text.strip():
            return self._json_response(
                start_response,
                HTTPStatus.BAD_REQUEST,
                {"error": "Field 'text' must be a non-empty string"},
            )

        reference_text = payload.get("reference_text")
        if reference_text is not None and not isinstance(reference_text, str):
            return self._json_response(
                start_response,
                HTTPStatus.BAD_REQUEST,
                {"error": "Field 'reference_text' must be a string when provided"},
            )

        title = payload.get("title")
        if title is not None and not isinstance(title, str):
            return self._json_response(
                start_response,
                HTTPStatus.BAD_REQUEST,
                {"error": "Field 'title' must be a string when provided"},
            )

        reference_title = payload.get("reference_title")
        if reference_title is not None and not isinstance(reference_title, str):
            return self._json_response(
                start_response,
                HTTPStatus.BAD_REQUEST,
                {"error": "Field 'reference_title' must be a string when provided"},
            )

        window_size = payload.get("window_size", 5)
        if not isinstance(window_size, int):
            return self._json_response(
                start_response,
                HTTPStatus.BAD_REQUEST,
                {"error": "Field 'window_size' must be an integer"},
            )
        if window_size <= 0:
            window_size = 1

        include_reports = payload.get("include_reports", True)
        if not isinstance(include_reports, bool):
            include_reports = bool(include_reports)

        bundle: DataGenerationBundle = self._model.prepare_data_generation(
            text,
            analysis=None,
            reference_text=reference_text,
            reference_analysis=None,
            title=title,
            reference_title=reference_title,
            window_size=window_size,
            include_reports=include_reports,
            comparability=None,
        )
        return self._json_response(start_response, HTTPStatus.OK, bundle.as_dict())

    # ------------------------------------------------------------------
    # Convenience helpers
    # ------------------------------------------------------------------
    def run(self, host: str = "127.0.0.1", port: int = 8000) -> None:
        """Run the API using :func:`wsgiref.simple_server.make_server`."""

        with make_server(host, port, self) as server:  # pragma: no cover - runtime helper
            print(f"Serving phonetic API on http://{host}:{port}")
            server.serve_forever()


def main(argv: list[str] | None = None) -> int:  # pragma: no cover - CLI entry point
    import argparse

    parser = argparse.ArgumentParser(description="Run the phonetic trailing song API server")
    parser.add_argument("--host", default="127.0.0.1", help="Interface to bind the server to")
    parser.add_argument("--port", type=int, default=8000, help="Port to listen on")
    args = parser.parse_args(argv)

    api = PhoneticAPI()
    api.run(args.host, args.port)
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    raise SystemExit(main())
