"""Repository chunk planner utility.

This tool helps contributors carve large corpora into approximately
`chunk_size`-sized windows.  It collects file statistics, guarantees that each
chunk is contiguous, and records filler segments when the natural file coverage
would otherwise leave holes.
"""

from __future__ import annotations

import argparse
import fnmatch
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence


@dataclass
class FileStat:
    path: str
    line_count: int

    def as_dict(self) -> Dict[str, object]:
        return {"path": self.path, "line_count": self.line_count}


@dataclass
class Chunk:
    index: int
    file: str
    start_line: int
    end_line: int
    preview: str
    filler: bool = False

    def as_dict(self) -> Dict[str, object]:
        return {
            "index": self.index,
            "file": self.file,
            "start_line": self.start_line,
            "end_line": self.end_line,
            "preview": self.preview,
            "filler": self.filler,
        }


@dataclass
class ChunkPlan:
    root: str
    chunk_size: int
    chunks: List[Chunk]
    file_stats: List[FileStat]

    def filler_statistics(self) -> Dict[str, int]:
        filler_chunks = [chunk for chunk in self.chunks if chunk.filler]
        return {
            "total_chunks": len(self.chunks),
            "filler_chunks": len(filler_chunks),
        }

    def as_dict(self) -> Dict[str, object]:
        return {
            "root": self.root,
            "chunk_size": self.chunk_size,
            "chunks": [chunk.as_dict() for chunk in self.chunks],
            "file_stats": [stat.as_dict() for stat in self.file_stats],
            "filler_statistics": self.filler_statistics(),
        }


def iter_matching_files(
    root: Path,
    includes: Sequence[str],
    excludes: Sequence[str],
) -> Iterable[Path]:
    for path in root.rglob("*"):
        if not path.is_file():
            continue
        relative = path.relative_to(root)
        if includes and not any(fnmatch.fnmatch(str(relative), pattern) for pattern in includes):
            continue
        if excludes and any(fnmatch.fnmatch(str(relative), pattern) for pattern in excludes):
            continue
        yield path


def count_lines(path: Path) -> int:
    try:
        return sum(1 for _ in path.open("r", encoding="utf-8", errors="ignore"))
    except OSError:
        return 0


def build_chunk_plan(
    root: Path,
    *,
    chunk_size: int,
    includes: Sequence[str],
    excludes: Sequence[str],
    preview_lines: int,
) -> ChunkPlan:
    chunks: List[Chunk] = []
    file_stats: List[FileStat] = []

    matched_files = list(iter_matching_files(root, includes, excludes))
    if not matched_files:
        chunks.append(
            Chunk(
                index=0,
                file="(no matches)",
                start_line=0,
                end_line=chunk_size,
                preview="",
                filler=True,
            )
        )
        return ChunkPlan(root=str(root), chunk_size=chunk_size, chunks=chunks, file_stats=[])

    chunk_index = 0
    for file_path in matched_files:
        lines = file_path.read_text(encoding="utf-8", errors="ignore").splitlines()
        file_stats.append(FileStat(path=str(file_path.relative_to(root)), line_count=len(lines)))
        if not lines:
            chunks.append(
                Chunk(
                    index=chunk_index,
                    file=str(file_path.relative_to(root)),
                    start_line=1,
                    end_line=1,
                    preview="",
                    filler=True,
                )
            )
            chunk_index += 1
            continue
        for start in range(0, len(lines), chunk_size):
            end = min(len(lines), start + chunk_size)
            preview = "\n".join(lines[start : min(end, start + preview_lines)]) if preview_lines else ""
            chunks.append(
                Chunk(
                    index=chunk_index,
                    file=str(file_path.relative_to(root)),
                    start_line=start + 1,
                    end_line=end,
                    preview=preview,
                    filler=False,
                )
            )
            chunk_index += 1
        remainder = len(lines) % chunk_size
        if remainder and remainder < chunk_size // 2:
            filler_preview = lines[-preview_lines:] if preview_lines else []
            chunks.append(
                Chunk(
                    index=chunk_index,
                    file=str(file_path.relative_to(root)),
                    start_line=len(lines) + 1,
                    end_line=len(lines) + (chunk_size - remainder),
                    preview="\n".join(filler_preview),
                    filler=True,
                )
            )
            chunk_index += 1

    return ChunkPlan(
        root=str(root),
        chunk_size=chunk_size,
        chunks=chunks,
        file_stats=file_stats,
    )


def _parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate chunk plans for a repository")
    parser.add_argument("root", nargs="?", default=".", help="Root directory")
    parser.add_argument("--chunk-size", type=int, default=3000, help="Target number of lines per chunk")
    parser.add_argument("--include", action="append", default=[], help="Glob of files to include")
    parser.add_argument("--exclude", action="append", default=[], help="Glob of files to exclude")
    parser.add_argument("--preview-lines", type=int, default=5, help="Number of preview lines to display")
    parser.add_argument("--export-json", help="Write the chunk plan to this JSON file")
    parser.add_argument("--quiet", action="store_true", help="Suppress chunk listing")
    return parser.parse_args(argv)


def _main(argv: Optional[Sequence[str]] = None) -> int:
    args = _parse_args(argv)
    root = Path(args.root).resolve()
    plan = build_chunk_plan(
        root,
        chunk_size=args.chunk_size,
        includes=args.include,
        excludes=args.exclude,
        preview_lines=args.preview_lines,
    )

    if not args.quiet:
        print(f"Chunk plan for {plan.root} (chunk size {plan.chunk_size})")
        for chunk in plan.chunks:
            status = "FILL" if chunk.filler else "DATA"
            print(
                f"[{chunk.index:04d}] {status} {chunk.file}:{chunk.start_line}-{chunk.end_line}"
                + (f" -> {chunk.preview.splitlines()[0]}" if chunk.preview else "")
            )
        stats = plan.filler_statistics()
        print(f"Total chunks: {stats['total_chunks']} | filler: {stats['filler_chunks']}")

    if args.export_json:
        Path(args.export_json).write_text(
            json.dumps(plan.as_dict(), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(_main())
