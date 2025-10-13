"""Repository chunk planning utilities.

This module helps contributors work with very large text repositories by
producing deterministic chunking plans that split source files into evenly
measured segments.  The default configuration follows the request to emit
"3000 code lined bits", but callers can tweak the chunk size, target file
patterns, and iteration strategy as needed.

The toolkit is intentionally self-contained and standard-library only so it
can run anywhere that Python 3.9+ is available (matching the baseline version
used by most of the tooling in this repository).
"""

from __future__ import annotations

import argparse
import dataclasses
import fnmatch
import json
import os
import sys
import textwrap
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, Iterator, List, Optional, Sequence, Tuple


DEFAULT_CHUNK_SIZE = 3000
DEFAULT_PREVIEW_LINES = 3
DEFAULT_ENCODING = "utf-8"


class PlannerError(RuntimeError):
    """Raised when chunk planning encounters an unrecoverable error."""


@dataclasses.dataclass(frozen=True)
class FileStats:
    """Metadata about a source file detected during planning."""

    path: Path
    line_count: int
    size_bytes: int

    @property
    def is_empty(self) -> bool:
        return self.line_count == 0

    def to_dict(self) -> Dict[str, object]:
        """Return a JSON-serialisable representation of the file stats."""

        return {
            "path": str(self.path),
            "line_count": self.line_count,
            "size_bytes": self.size_bytes,
        }


@dataclasses.dataclass(frozen=True)
class FileChunk:
    """A 1-based chunk designation for a specific source file."""

    path: Path
    index: int
    start_line: int
    end_line: int
    preview: Tuple[str, ...]
    is_filler: bool = False

    @property
    def line_count(self) -> int:
        return self.end_line - self.start_line + 1

    def to_dict(self) -> Dict[str, object]:
        return {
            "path": str(self.path),
            "index": self.index,
            "start_line": self.start_line,
            "end_line": self.end_line,
            "line_count": self.line_count,
            "preview": self.preview,
            "is_filler": self.is_filler,
        }


@dataclasses.dataclass(frozen=True)
class HolePatch:
    """Represents a gap that was filled during chunk construction."""

    path: Path
    start_line: int
    end_line: int
    preview: Tuple[str, ...]

    @property
    def line_count(self) -> int:
        return self.end_line - self.start_line + 1

    def to_dict(self) -> Dict[str, object]:
        return {
            "path": str(self.path),
            "start_line": self.start_line,
            "end_line": self.end_line,
            "line_count": self.line_count,
            "preview": self.preview,
        }


@dataclasses.dataclass(frozen=True)
class ChunkIteration:
    """Represents a full pass over the repository chunks."""

    iteration_index: int
    chunks: Tuple[FileChunk, ...]

    def to_dict(self) -> Dict[str, object]:
        return {
            "iteration_index": self.iteration_index,
            "chunks": [chunk.to_dict() for chunk in self.chunks],
        }


@dataclasses.dataclass
class ChunkPlan:
    """Aggregates the chunks discovered during planning."""

    root: Path
    files: Tuple[FileStats, ...]
    chunk_size: int
    preview_lines: int
    chunks_by_file: Dict[Path, Tuple[FileChunk, ...]]
    hole_patches_by_file: Dict[Path, Tuple[HolePatch, ...]]

    def iterations(self, repeat: int = 1) -> Iterator[ChunkIteration]:
        if repeat <= 0:
            raise PlannerError("repeat must be positive")

        all_chunks: Tuple[FileChunk, ...] = tuple(
            chunk
            for stats in self.files
            for chunk in self.chunks_by_file.get(stats.path, ())
        )
        for iteration_index in range(1, repeat + 1):
            yield ChunkIteration(iteration_index=iteration_index, chunks=all_chunks)

    @property
    def total_chunks(self) -> int:
        return sum(len(chunks) for chunks in self.chunks_by_file.values())

    @property
    def total_lines(self) -> int:
        return sum(stats.line_count for stats in self.files)

    @property
    def total_filler_chunks(self) -> int:
        return sum(
            1
            for chunks in self.chunks_by_file.values()
            for chunk in chunks
            if chunk.is_filler
        )

    def to_dict(self, repeat: int = 1) -> Dict[str, object]:
        return {
            "root": str(self.root),
            "chunk_size": self.chunk_size,
            "preview_lines": self.preview_lines,
            "files": [stats.to_dict() for stats in self.files],
            "iterations": [iteration.to_dict() for iteration in self.iterations(repeat)],
            "hole_patches": {
                str(path): [patch.to_dict() for patch in patches]
                for path, patches in self.hole_patches_by_file.items()
                if patches
            },
        }


class RepositoryChunkPlanner:
    """Utility class that scans a repository and builds chunk plans."""

    def __init__(
        self,
        root: Path,
        *,
        chunk_size: int = DEFAULT_CHUNK_SIZE,
        preview_lines: int = DEFAULT_PREVIEW_LINES,
        include_patterns: Optional[Sequence[str]] = None,
        exclude_patterns: Optional[Sequence[str]] = None,
        encoding: str = DEFAULT_ENCODING,
    ) -> None:
        if chunk_size <= 0:
            raise PlannerError("chunk_size must be positive")
        if preview_lines < 0:
            raise PlannerError("preview_lines must be non-negative")

        self.root = root
        self.chunk_size = chunk_size
        self.preview_lines = preview_lines
        self.encoding = encoding
        self.include_patterns = tuple(include_patterns or ("*.xml", "*.py", "*.md"))
        self.exclude_patterns = tuple(exclude_patterns or ())

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def build_plan(self) -> ChunkPlan:
        files = tuple(self._scan_files())
        chunks_by_file: Dict[Path, Tuple[FileChunk, ...]] = {}
        hole_patches: Dict[Path, Tuple[HolePatch, ...]] = {}
        for stats in files:
            chunks, patches = self._build_chunks(stats)
            chunks_by_file[stats.path] = chunks
            hole_patches[stats.path] = patches
        return ChunkPlan(
            root=self.root,
            files=files,
            chunk_size=self.chunk_size,
            preview_lines=self.preview_lines,
            chunks_by_file=chunks_by_file,
            hole_patches_by_file=hole_patches,
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _scan_files(self) -> Iterable[FileStats]:
        for path in sorted(self.root.rglob("*")):
            if path.is_dir():
                continue
            relative = path.relative_to(self.root)
            if not self._matches(relative):
                continue
            line_count, size_bytes = self._count_lines(path)
            yield FileStats(path=relative, line_count=line_count, size_bytes=size_bytes)

    def _matches(self, relative_path: Path) -> bool:
        path_str = str(relative_path)
        if self.exclude_patterns and any(
            fnmatch.fnmatch(path_str, pattern) for pattern in self.exclude_patterns
        ):
            return False
        return any(fnmatch.fnmatch(path_str, pattern) for pattern in self.include_patterns)

    def _count_lines(self, path: Path) -> Tuple[int, int]:
        try:
            with path.open("r", encoding=self.encoding) as handle:
                line_count = sum(1 for _ in handle)
        except UnicodeDecodeError as exc:
            raise PlannerError(f"Failed to decode {path}: {exc}") from exc
        size_bytes = path.stat().st_size
        return line_count, size_bytes

    def _build_chunks(self, stats: FileStats) -> Tuple[Tuple[FileChunk, ...], Tuple[HolePatch, ...]]:
        if stats.is_empty:
            empty_chunk = FileChunk(
                path=stats.path,
                index=1,
                start_line=1,
                end_line=0,
                preview=(),
            )
            return (empty_chunk,), ()

        with (self.root / stats.path).open("r", encoding=self.encoding) as handle:
            lines = [line.rstrip("\n") for line in handle]

        raw_segments: List[Tuple[int, int, Tuple[str, ...], bool]] = []
        for start in range(0, stats.line_count, self.chunk_size):
            end = min(start + self.chunk_size, stats.line_count)
            preview_slice = tuple(lines[start : start + self.preview_lines])
            raw_segments.append((start + 1, end, preview_slice, False))

        segments, patches = self._fill_holes(stats, raw_segments, lines)

        file_chunks: List[FileChunk] = []
        for index, (start_line, end_line, preview, is_filler) in enumerate(segments, start=1):
            file_chunks.append(
                FileChunk(
                    path=stats.path,
                    index=index,
                    start_line=start_line,
                    end_line=end_line,
                    preview=preview,
                    is_filler=is_filler,
                )
            )

        return tuple(file_chunks), patches

    def _fill_holes(
        self,
        stats: FileStats,
        segments: Sequence[Tuple[int, int, Tuple[str, ...], bool]],
        lines: Sequence[str],
    ) -> Tuple[Tuple[Tuple[int, int, Tuple[str, ...], bool], ...], Tuple[HolePatch, ...]]:
        if not segments:
            if stats.line_count == 0:
                return tuple(), tuple()
            preview_slice = tuple(lines[: self.preview_lines])
            if not preview_slice:
                preview_slice = ("<filled gap>",)
            filler = (
                (1, stats.line_count, preview_slice, True),
            )
            patch = HolePatch(
                path=stats.path,
                start_line=1,
                end_line=stats.line_count,
                preview=preview_slice,
            )
            return filler, (patch,)

        filled_segments: List[Tuple[int, int, Tuple[str, ...], bool]] = []
        hole_patches: List[HolePatch] = []
        expected_start = 1

        for start_line, end_line, preview, is_filler in segments:
            if start_line > end_line:
                continue
            if start_line > expected_start:
                filler_preview = tuple(
                    lines[expected_start - 1 : min(expected_start - 1 + self.preview_lines, start_line - 1)]
                )
                if not filler_preview:
                    filler_preview = ("<filled gap>",)
                hole_patches.append(
                    HolePatch(
                        path=stats.path,
                        start_line=expected_start,
                        end_line=start_line - 1,
                        preview=filler_preview,
                    )
                )
                filled_segments.append(
                    (expected_start, start_line - 1, filler_preview, True)
                )
            elif start_line < expected_start:
                raise PlannerError(
                    f"Overlapping chunks detected in {stats.path}: "
                    f"chunk starting at {start_line} overlaps expected line {expected_start}"
                )

            filled_segments.append((start_line, end_line, preview, is_filler))
            expected_start = end_line + 1

        if expected_start <= stats.line_count:
            filler_preview = tuple(
                lines[expected_start - 1 : min(expected_start - 1 + self.preview_lines, stats.line_count)]
            )
            if not filler_preview:
                filler_preview = ("<filled gap>",)
            hole_patches.append(
                HolePatch(
                    path=stats.path,
                    start_line=expected_start,
                    end_line=stats.line_count,
                    preview=filler_preview,
                )
            )
            filled_segments.append(
                (expected_start, stats.line_count, filler_preview, True)
            )

        return tuple(filled_segments), tuple(hole_patches)


def _format_summary(plan: ChunkPlan, repeat: int) -> str:
    wrapper = textwrap.TextWrapper(width=100)
    root_display = str(plan.root)
    lines = [
        "Repository chunk plan summary",
        "=" * 32,
        f"Root: {root_display}",
        f"Chunk size: {plan.chunk_size} lines",
        f"Preview lines per chunk: {plan.preview_lines}",
        f"Total files matched: {len(plan.files)}",
        f"Total lines spanned: {plan.total_lines}",
        f"Total chunks per iteration: {plan.total_chunks}",
        f"Total filler chunks inserted: {plan.total_filler_chunks}",
        f"Iterations requested: {repeat}",
        "",
    ]
    if plan.files:
        for stats in plan.files:
            lines.append(f"- {stats.path} ({stats.line_count} lines, {stats.size_bytes} bytes)")
    else:
        lines.append("- No files matched the provided include/exclude patterns.")
    lines.append("")

    if plan.total_chunks == 0:
        lines.append("No chunks were generated.")
    else:
        for iteration in plan.iterations(repeat):
            lines.append(f"Iteration {iteration.iteration_index} ({len(iteration.chunks)} chunks)")
            lines.append("-" * 16)
            for chunk in iteration.chunks:
                preview = " | ".join(chunk.preview) if chunk.preview else "<empty>"
                filler_marker = " (filler)" if chunk.is_filler else ""
                lines.append(
                    wrapper.fill(
                        f"  {chunk.path} [chunk {chunk.index}{filler_marker}] lines "
                        f"{chunk.start_line}-{chunk.end_line}: {preview}"
                    )
                )
            lines.append("")
    if plan.total_filler_chunks:
        lines.append("Filled gaps")
        lines.append("-" * 11)
        for stats in plan.files:
            patches = plan.hole_patches_by_file.get(stats.path, ())
            for patch in patches:
                preview = " | ".join(patch.preview) if patch.preview else "<empty>"
                lines.append(
                    wrapper.fill(
                        f"  {patch.path} lines {patch.start_line}-{patch.end_line} "
                        f"({patch.line_count} lines): {preview}"
                    )
                )
        lines.append("")
    else:
        lines.append("No gaps detected; all files were already contiguous.")
    return "\n".join(lines)


def _parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Plan repository contributions by splitting files into deterministic "
            "3000-line (configurable) chunks."
        )
    )
    parser.add_argument(
        "root",
        nargs="?",
        default=".",
        help="Repository root to scan (defaults to current directory).",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=DEFAULT_CHUNK_SIZE,
        help="Number of lines per chunk (default: %(default)s).",
    )
    parser.add_argument(
        "--preview-lines",
        type=int,
        default=DEFAULT_PREVIEW_LINES,
        help="Number of preview lines to capture for each chunk.",
    )
    parser.add_argument(
        "--include",
        action="append",
        help=(
            "Glob pattern to include (can be repeated). Defaults to '*.xml', '*.py', '*.md'."
        ),
    )
    parser.add_argument(
        "--exclude",
        action="append",
        help="Glob pattern to exclude (can be repeated).",
    )
    parser.add_argument(
        "--repeat",
        type=int,
        default=1,
        help="Number of times to repeat the chunk list for iterative workflows.",
    )
    parser.add_argument(
        "--export-json",
        type=Path,
        help="Optional path to write the plan as JSON.",
    )
    parser.add_argument(
        "--encoding",
        default=DEFAULT_ENCODING,
        help="File encoding to use when reading sources (default: %(default)s).",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress the human-readable summary and only emit JSON if requested.",
    )
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = _parse_args(argv)
    root = Path(args.root).resolve()
    if not root.exists():
        raise PlannerError(f"Root path does not exist: {root}")

    planner = RepositoryChunkPlanner(
        root=root,
        chunk_size=args.chunk_size,
        preview_lines=args.preview_lines,
        include_patterns=args.include,
        exclude_patterns=args.exclude,
        encoding=args.encoding,
    )
    plan = planner.build_plan()

    if args.export_json:
        json_payload = json.dumps(plan.to_dict(args.repeat), indent=2, ensure_ascii=False)
        args.export_json.write_text(json_payload, encoding="utf-8")

    if not args.quiet:
        summary = _format_summary(plan, args.repeat)
        print(summary)

    return 0


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    try:
        raise SystemExit(main())
    except PlannerError as error:
        print(f"error: {error}", file=sys.stderr)
        raise SystemExit(1) from error
