"""Phonetic trailing song analysis toolkit.

This module exposes a high level :class:`PhoneticTrailingSongModel` that can be
used to derive deterministic numeric signatures, phonetic summaries, and
navigational metadata for arbitrary text.  The implementation is intentionally
self-contained so it can operate in restricted execution environments where
third‑party phonetic libraries are not available.

The design mirrors the feature set that evolved throughout the previous chat
iteration:

* title sound fragments and base-36 numeric seeds
* numeric equivalence groupings and repetition clusters
* binary bookmark/codex markers with absence detection
* cursor friendly table-of-contents and title language records
* universal similarity groupings across scripts
* slider-ready per-token seed metrics and language layering metadata
* comparability utilities for sliding window inspection
* data-generation bundle helpers for ML workflows

The functions in this module favour deterministic hashing so that multiple runs
with the same inputs always yield the same outputs.
"""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import statistics
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterator, List, Optional, Sequence, Tuple
import unicodedata
import re

from language_profiles import (
    LanguageProfile,
    LanguageProfileRegistry,
    default_language_registry,
)

__all__ = [
    "PhoneticTrailingSongModel",
    "AnalysisResult",
    "LanguageUsage",
    "KnowledgeBase",
    "prepare_data_generation",
    "analyze_text",
]

_WORD_RE = re.compile(r"\b\w+\b", re.UNICODE)
_BASE36_ALPHABET = "0123456789abcdefghijklmnopqrstuvwxyz"


def _strip_accents(value: str) -> str:
    normalized = unicodedata.normalize("NFKD", value)
    return "".join(ch for ch in normalized if unicodedata.category(ch) != "Mn")


def _ascii_fallback(value: str) -> str:
    ascii_bytes = _strip_accents(value).encode("ascii", "ignore")
    fallback = ascii_bytes.decode("ascii", "ignore")
    return fallback or value.lower()


def _base36(value: int) -> str:
    if value == 0:
        return "0"
    digits: List[str] = []
    abs_value = abs(value)
    while abs_value:
        abs_value, rem = divmod(abs_value, 36)
        digits.append(_BASE36_ALPHABET[rem])
    if value < 0:
        digits.append("-")
    return "".join(reversed(digits))


def _hash_int(seed: str, modulo: int = 36 ** 6) -> int:
    digest = hashlib.sha1(seed.encode("utf-8")).digest()
    value = int.from_bytes(digest[:8], "big", signed=False)
    return value % modulo


def _lengthen_base36(value: int) -> str:
    base = _base36(value)
    doubled = ''.join(ch * 2 for ch in base)
    return doubled or "00"


def _cubic_value(value: int) -> int:
    return value ** 3


def _slider_bar(value: int, maximum: int) -> str:
    if maximum <= 0:
        return "|"
    filled = max(1, round((value / maximum) * 20))
    return "|" + ("#" * min(filled, 20)).ljust(20, "-") + "|"


def _chunk_sequence(seq: Sequence, size: int) -> Iterator[Sequence]:
    for idx in range(0, len(seq), size):
        yield seq[idx : idx + size]


@dataclass
class LanguageLayer:
    kind: str
    value: str


@dataclass
class LanguageLayerGrouping:
    token_index: int
    layers: List[LanguageLayer]


@dataclass
class PhoneticToken:
    index: int
    text: str
    normalized: str
    phonetic: str
    ascii_value: str
    base36_value: int
    lengthened_value: str
    cubic_value: int
    seed: int
    language_code: str
    language_name: str
    language_layers: LanguageLayerGrouping

    def as_dict(self) -> Dict[str, object]:
        return {
            "index": self.index,
            "text": self.text,
            "normalized": self.normalized,
            "phonetic": self.phonetic,
            "ascii": self.ascii_value,
            "base36": self.base36_value,
            "lengthened": self.lengthened_value,
            "cubic": self.cubic_value,
            "seed": self.seed,
            "language_code": self.language_code,
            "language_name": self.language_name,
            "language_layers": [
                dataclasses.asdict(layer) for layer in self.language_layers.layers
            ],
        }


@dataclass
class TitleSoundSummary:
    line_number: int
    fragment: str
    combined_value: int

    def as_dict(self) -> Dict[str, object]:
        return dataclasses.asdict(self)


@dataclass
class NumericEquivalenceGroup:
    base36_value: int
    token_indices: List[int]

    def as_dict(self) -> Dict[str, object]:
        return dataclasses.asdict(self)


@dataclass
class LanguageUsage:
    code: str
    name: str
    description: str
    token_count: int
    coverage: float

    def as_dict(self) -> Dict[str, object]:
        return dataclasses.asdict(self)


@dataclass
class BinaryBookmark:
    marker_128bit: str
    codex_key: str
    has_absence: bool
    absent_indices: List[int]

    def as_dict(self) -> Dict[str, object]:
        return dataclasses.asdict(self)


@dataclass
class TokenSeedDetail:
    token_index: int
    base_seed: int
    max_seed: int
    slider: str

    def as_dict(self) -> Dict[str, object]:
        return dataclasses.asdict(self)


@dataclass
class NumericRepetitionCluster:
    base36_value: int
    occurrences: List[int]

    def as_dict(self) -> Dict[str, object]:
        return dataclasses.asdict(self)


@dataclass
class TableOfContentsMarker:
    anchor: str
    title: str
    token_range: Tuple[int, int]

    def as_dict(self) -> Dict[str, object]:
        return {
            "anchor": self.anchor,
            "title": self.title,
            "token_range": list(self.token_range),
        }


@dataclass
class UniversalSimilarityGroup:
    signature: str
    token_indices: List[int]

    def as_dict(self) -> Dict[str, object]:
        return dataclasses.asdict(self)


@dataclass
class TitleLanguageRecord:
    order: int
    anchor: str
    narrative: str

    def as_dict(self) -> Dict[str, object]:
        return dataclasses.asdict(self)


@dataclass
class CodeGapTitle:
    token_index: int
    reason: str

    def as_dict(self) -> Dict[str, object]:
        return dataclasses.asdict(self)


@dataclass
class NameTimeMarker:
    order: int
    label: str
    summary: str

    def as_dict(self) -> Dict[str, object]:
        return dataclasses.asdict(self)


@dataclass
class KnowledgeEntry:
    token_index: int
    surface: str
    normalized: str
    base36: str
    lengthened: str
    cubic: int
    seed: int
    language_code: str
    language_name: str
    universal_signature: Optional[str]
    numeric_group_size: int
    repetition_count: int

    def as_dict(self) -> Dict[str, object]:
        return dataclasses.asdict(self)


@dataclass
class KnowledgeCluster:
    kind: str
    label: str
    token_indices: List[int]
    summary: str

    def as_dict(self) -> Dict[str, object]:
        return dataclasses.asdict(self)


@dataclass
class KnowledgeBase:
    entries: List[KnowledgeEntry]
    clusters: List[KnowledgeCluster]
    insights: List[str]

    def as_dict(self) -> Dict[str, object]:
        return {
            "entries": [entry.as_dict() for entry in self.entries],
            "clusters": [cluster.as_dict() for cluster in self.clusters],
            "insights": list(self.insights),
        }


@dataclass
class ComparabilityWindow:
    window_index: int
    current_tokens: List[str]
    reference_tokens: List[str]
    numeric_overlap: float
    similarity_overlap: float

    def as_dict(self) -> Dict[str, object]:
        return dataclasses.asdict(self)


@dataclass
class ComparabilityResult:
    window_size: int
    aggregate_numeric_similarity: float
    aggregate_similarity_overlap: float
    windows: List[ComparabilityWindow]

    def as_dict(self) -> Dict[str, object]:
        return {
            "window_size": self.window_size,
            "aggregate_numeric_similarity": self.aggregate_numeric_similarity,
            "aggregate_similarity_overlap": self.aggregate_similarity_overlap,
            "windows": [w.as_dict() for w in self.windows],
        }


@dataclass
class DataGenerationExample:
    prompt: str
    completion: str

    def as_dict(self) -> Dict[str, object]:
        return dataclasses.asdict(self)


@dataclass
class DataGenerationBundle:
    title: Optional[str]
    analysis: Dict[str, object]
    report: Optional[str]
    comparability: Optional[Dict[str, object]]
    examples: List[DataGenerationExample]

    def as_dict(self) -> Dict[str, object]:
        return {
            "title": self.title,
            "analysis": self.analysis,
            "report": self.report,
            "comparability": self.comparability,
            "examples": [ex.as_dict() for ex in self.examples],
        }


@dataclass
class AnalysisResult:
    title: Optional[str]
    tokens: List[PhoneticToken]
    language_usage: List[LanguageUsage]
    title_summaries: List[TitleSoundSummary]
    numeric_equivalences: List[NumericEquivalenceGroup]
    binary_bookmark: BinaryBookmark
    token_seeds: List[TokenSeedDetail]
    repetition_clusters: List[NumericRepetitionCluster]
    table_of_contents: List[TableOfContentsMarker]
    universal_similarity: List[UniversalSimilarityGroup]
    title_language_records: List[TitleLanguageRecord]
    code_gap_titles: List[CodeGapTitle]
    name_time_markers: List[NameTimeMarker]
    knowledge_base: KnowledgeBase
    analysis_notes: Dict[str, object]

    def as_dict(self) -> Dict[str, object]:
        return {
            "title": self.title,
            "tokens": [token.as_dict() for token in self.tokens],
            "language_usage": [usage.as_dict() for usage in self.language_usage],
            "title_summaries": [ts.as_dict() for ts in self.title_summaries],
            "numeric_equivalences": [g.as_dict() for g in self.numeric_equivalences],
            "binary_bookmark": self.binary_bookmark.as_dict(),
            "token_seeds": [seed.as_dict() for seed in self.token_seeds],
            "repetition_clusters": [rc.as_dict() for rc in self.repetition_clusters],
            "table_of_contents": [toc.as_dict() for toc in self.table_of_contents],
            "universal_similarity": [g.as_dict() for g in self.universal_similarity],
            "title_language_records": [r.as_dict() for r in self.title_language_records],
            "code_gap_titles": [c.as_dict() for c in self.code_gap_titles],
            "name_time_markers": [m.as_dict() for m in self.name_time_markers],
            "knowledge_base": self.knowledge_base.as_dict(),
            "analysis_notes": self.analysis_notes,
        }


class PhoneticTrailingSongModel:
    """Perform phonetic and numeric trailing song analysis."""

    def __init__(
        self,
        base_seed: int = 0xC0FFEE,
        *,
        language_registry: Optional[LanguageProfileRegistry] = None,
        default_language_code: str = "en",
    ):
        self.base_seed = base_seed
        self.language_registry = language_registry or default_language_registry()
        self.default_language_code = default_language_code

    # ------------------------------------------------------------------
    # Public helpers
    def analyze(
        self,
        text: str,
        *,
        title: Optional[str] = None,
        reference_text: Optional[str] = None,
        seed_from: Optional[str] = None,
        window_size: int = 4,
        compare: bool = False,
        language_code: Optional[str] = None,
    ) -> AnalysisResult:
        base_profile, detection_notes = self.language_registry.detect(
            text,
            preferred=language_code,
            fallback=self.default_language_code,
        )
        allow_mixed = language_code is None
        detection_notes["allow_mixed"] = allow_mixed

        tokens, usage_counts = self._tokenize(
            text,
            base_profile=base_profile,
            allow_mixed=allow_mixed,
        )
        language_usage = self._summarize_language_usage(usage_counts)
        title_summaries = self._build_title_sound_summaries(text, tokens)
        numeric_groups = self._group_numeric_equivalents(tokens)
        binary_bookmark = self._build_binary_bookmark(tokens, seed_from or title)
        token_seeds = self._build_token_seeds(tokens)
        repetitions = self._build_repetition_clusters(tokens)
        universal = self._build_universal_similarity(tokens)
        toc = self._build_table_of_contents(tokens, title_summaries, universal)
        title_language = self._build_title_language_records(toc)
        code_gaps = self._build_code_gap_titles(tokens, numeric_groups, binary_bookmark)
        name_markers = self._build_name_time_markers(title_language)
        knowledge_base = self._build_knowledge_base(
            tokens,
            numeric_groups,
            universal,
            repetitions,
        )

        notes: Dict[str, object] = {
            "token_count": len(tokens),
            "seed_source": seed_from or title,
            "base_seed": self.base_seed,
            "language_detection": detection_notes,
        }

        if compare and reference_text:
            comparability = self.compare_texts(text, reference_text, window_size=window_size)
            notes["comparability"] = comparability.as_dict()
        notes["language_usage"] = [usage.as_dict() for usage in language_usage]
        notes["knowledge"] = {
            "entries": len(knowledge_base.entries),
            "clusters": len(knowledge_base.clusters),
        }

        return AnalysisResult(
            title=title,
            tokens=tokens,
            language_usage=language_usage,
            title_summaries=title_summaries,
            numeric_equivalences=numeric_groups,
            binary_bookmark=binary_bookmark,
            token_seeds=token_seeds,
            repetition_clusters=repetitions,
            table_of_contents=toc,
            universal_similarity=universal,
            title_language_records=title_language,
            code_gap_titles=code_gaps,
            name_time_markers=name_markers,
            knowledge_base=knowledge_base,
            analysis_notes=notes,
        )

    def compare_texts(
        self,
        current_text: str,
        reference_text: str,
        *,
        window_size: int = 4,
    ) -> ComparabilityResult:
        current_profile, _ = self.language_registry.detect(
            current_text,
            fallback=self.default_language_code,
        )
        reference_profile, _ = self.language_registry.detect(
            reference_text,
            fallback=self.default_language_code,
        )
        current_tokens, _ = self._tokenize(
            current_text,
            base_profile=current_profile,
            allow_mixed=True,
        )
        reference_tokens, _ = self._tokenize(
            reference_text,
            base_profile=reference_profile,
            allow_mixed=True,
        )

        current_base36 = [token.base36_value for token in current_tokens]
        reference_base36 = [token.base36_value for token in reference_tokens]
        current_signatures = [token.ascii_value for token in current_tokens]
        reference_signatures = [token.ascii_value for token in reference_tokens]

        windows: List[ComparabilityWindow] = []
        numeric_overlaps: List[float] = []
        similarity_overlaps: List[float] = []

        if window_size <= 0:
            window_size = 1

        for idx, chunk in enumerate(_chunk_sequence(current_tokens, window_size)):
            current_slice = current_base36[idx * window_size : idx * window_size + len(chunk)]
            ref_slice = reference_base36[idx * window_size : idx * window_size + len(chunk)]
            cur_sig = current_signatures[idx * window_size : idx * window_size + len(chunk)]
            ref_sig = reference_signatures[idx * window_size : idx * window_size + len(chunk)]

            if not chunk:
                continue

            numeric_overlap = _sequence_overlap(current_slice, ref_slice)
            similarity_overlap = _sequence_overlap(cur_sig, ref_sig)

            windows.append(
                ComparabilityWindow(
                    window_index=idx,
                    current_tokens=[t.text for t in chunk],
                    reference_tokens=ref_tokens_slice(reference_tokens, idx, window_size),
                    numeric_overlap=numeric_overlap,
                    similarity_overlap=similarity_overlap,
                )
            )
            numeric_overlaps.append(numeric_overlap)
            similarity_overlaps.append(similarity_overlap)

        aggregate_numeric = statistics.fmean(numeric_overlaps) if numeric_overlaps else 0.0
        aggregate_similarity = statistics.fmean(similarity_overlaps) if similarity_overlaps else 0.0

        return ComparabilityResult(
            window_size=window_size,
            aggregate_numeric_similarity=aggregate_numeric,
            aggregate_similarity_overlap=aggregate_similarity,
            windows=windows,
        )

    # ------------------------------------------------------------------
    # Internal builders
    def _tokenize(
        self,
        text: str,
        *,
        base_profile: LanguageProfile,
        allow_mixed: bool,
    ) -> Tuple[List[PhoneticToken], Dict[str, int]]:
        tokens: List[PhoneticToken] = []
        usage: Dict[str, int] = {}
        for index, match in enumerate(_WORD_RE.finditer(text)):
            raw = match.group(0)
            if allow_mixed:
                profile, _ = self.language_registry.detect(
                    raw,
                    preferred=None,
                    fallback=base_profile.code,
                )
            else:
                profile = base_profile

            romanized = profile.romanize(raw)
            ascii_candidate = _ascii_fallback(romanized)
            ascii_value = ascii_candidate or _ascii_fallback(raw) or romanized.lower()
            normalized_source = romanized if romanized else raw
            normalized = profile.normalize(normalized_source)
            phonetic = normalized.lower()
            base36_value = _hash_int(phonetic)
            lengthened = _lengthen_base36(base36_value)
            cubic = _cubic_value(base36_value)
            seed = _hash_int(f"{self.base_seed}:{phonetic}:{index}", modulo=2 ** 31)
            layers = LanguageLayerGrouping(
                token_index=index,
                layers=[
                    LanguageLayer("original", raw),
                    LanguageLayer("normalized", normalized),
                    LanguageLayer("romanized", romanized),
                    LanguageLayer("ascii", ascii_value),
                    LanguageLayer("language", profile.code),
                    LanguageLayer("language_name", profile.name),
                ],
            )
            tokens.append(
                PhoneticToken(
                    index=index,
                    text=raw,
                    normalized=normalized,
                    phonetic=phonetic,
                    ascii_value=ascii_value,
                    base36_value=base36_value,
                    lengthened_value=lengthened,
                    cubic_value=cubic,
                    seed=seed,
                    language_code=profile.code,
                    language_name=profile.name,
                    language_layers=layers,
                )
            )
            usage[profile.code] = usage.get(profile.code, 0) + 1
        return tokens, usage

    def _build_title_sound_summaries(
        self, text: str, tokens: Sequence[PhoneticToken]
    ) -> List[TitleSoundSummary]:
        summaries: List[TitleSoundSummary] = []
        lines = text.splitlines()
        offset = 0
        for idx, line in enumerate(lines, start=1):
            line_tokens = [t for t in tokens if offset <= t.index < offset + len(_WORD_RE.findall(line))]
            fragment = " ".join(t.text for t in line_tokens[-3:]) if line_tokens else ""
            combined = sum(t.base36_value for t in line_tokens)
            summaries.append(TitleSoundSummary(line_number=idx, fragment=fragment, combined_value=combined))
            offset += len(_WORD_RE.findall(line))
        return summaries

    def _group_numeric_equivalents(self, tokens: Sequence[PhoneticToken]) -> List[NumericEquivalenceGroup]:
        groups: Dict[int, List[int]] = {}
        for token in tokens:
            groups.setdefault(token.base36_value, []).append(token.index)
        return [
            NumericEquivalenceGroup(base36_value=value, token_indices=indices)
            for value, indices in sorted(groups.items())
            if len(indices) > 1 or value == 0
        ]

    def _summarize_language_usage(self, usage_counts: Dict[str, int]) -> List[LanguageUsage]:
        if not usage_counts:
            return []
        total = sum(usage_counts.values())
        summaries: List[LanguageUsage] = []
        for code, count in sorted(usage_counts.items(), key=lambda item: (-item[1], item[0])):
            profile = self.language_registry.get(code)
            name = profile.name if profile else code
            description = profile.description if profile else ""
            coverage = count / total if total else 0.0
            summaries.append(
                LanguageUsage(
                    code=code,
                    name=name,
                    description=description,
                    token_count=count,
                    coverage=round(coverage, 6),
                )
            )
        return summaries

    def _build_binary_bookmark(
        self, tokens: Sequence[PhoneticToken], seed_source: Optional[str]
    ) -> BinaryBookmark:
        combined_seed = self.base_seed
        absent_indices: List[int] = []
        for token in tokens:
            combined_seed ^= token.seed
            if token.base36_value == 0:
                absent_indices.append(token.index)
        if seed_source:
            combined_seed ^= _hash_int(seed_source, modulo=2 ** 64)
        marker = combined_seed & ((1 << 128) - 1)
        codex = hashlib.sha256(str(combined_seed).encode("utf-8")).hexdigest()[:32]
        has_absence = bool(absent_indices)
        marker_hex = f"{marker:032x}"
        return BinaryBookmark(
            marker_128bit=marker_hex,
            codex_key=codex,
            has_absence=has_absence,
            absent_indices=absent_indices,
        )

    def _build_token_seeds(self, tokens: Sequence[PhoneticToken]) -> List[TokenSeedDetail]:
        max_seed = max((token.seed for token in tokens), default=1)
        details: List[TokenSeedDetail] = []
        for token in tokens:
            slider = _slider_bar(token.seed, max_seed)
            details.append(
                TokenSeedDetail(
                    token_index=token.index,
                    base_seed=token.seed,
                    max_seed=max_seed,
                    slider=slider,
                )
            )
        return details

    def _build_repetition_clusters(
        self, tokens: Sequence[PhoneticToken]
    ) -> List[NumericRepetitionCluster]:
        occurrences: Dict[int, List[int]] = {}
        for token in tokens:
            occurrences.setdefault(token.base36_value, []).append(token.index)
        clusters = [
            NumericRepetitionCluster(base36_value=value, occurrences=indices)
            for value, indices in sorted(occurrences.items())
            if len(indices) > 1
        ]
        return clusters

    def _build_universal_similarity(
        self, tokens: Sequence[PhoneticToken]
    ) -> List[UniversalSimilarityGroup]:
        groups: Dict[str, List[int]] = {}
        for token in tokens:
            groups.setdefault(token.ascii_value, []).append(token.index)
        return [
            UniversalSimilarityGroup(signature=sig, token_indices=indices)
            for sig, indices in sorted(groups.items())
            if len(indices) > 1
        ]

    def _build_table_of_contents(
        self,
        tokens: Sequence[PhoneticToken],
        summaries: Sequence[TitleSoundSummary],
        universal: Sequence[UniversalSimilarityGroup],
    ) -> List[TableOfContentsMarker]:
        markers: List[TableOfContentsMarker] = []
        sections = [
            (f"line-{summary.line_number}", f"Line {summary.line_number}: {summary.fragment or '∅'}")
            for summary in summaries
        ]
        sections += [
            (f"similarity-{group.signature}", f"Similarity: {group.signature}")
            for group in universal[:10]
        ]
        for idx, (anchor, title) in enumerate(sections):
            start = idx * max(1, len(tokens) // max(1, len(sections)))
            end = min(len(tokens), start + max(1, len(tokens) // max(1, len(sections))))
            markers.append(
                TableOfContentsMarker(
                    anchor=anchor,
                    title=title,
                    token_range=(start, end),
                )
            )
        return markers

    def _build_title_language_records(
        self, toc: Sequence[TableOfContentsMarker]
    ) -> List[TitleLanguageRecord]:
        records: List[TitleLanguageRecord] = []
        for order, marker in enumerate(sorted(toc, key=lambda m: m.anchor, reverse=True)):
            narrative = f"{marker.title} (tokens {marker.token_range[0]}-{marker.token_range[1]})"
            records.append(
                TitleLanguageRecord(order=order + 1, anchor=marker.anchor, narrative=narrative)
            )
        return records

    def _build_code_gap_titles(
        self,
        tokens: Sequence[PhoneticToken],
        groups: Sequence[NumericEquivalenceGroup],
        bookmark: BinaryBookmark,
    ) -> List[CodeGapTitle]:
        gaps: List[CodeGapTitle] = []
        zero_tokens = [token for token in tokens if token.base36_value == 0]
        for token in zero_tokens:
            gaps.append(CodeGapTitle(token_index=token.index, reason="zero-value token"))
        if bookmark.has_absence:
            for index in bookmark.absent_indices:
                gaps.append(CodeGapTitle(token_index=index, reason="bookmark absence coverage"))
        covered_indices = {idx for group in groups for idx in group.token_indices}
        for token in tokens:
            if token.index not in covered_indices and token.base36_value != 0:
                gaps.append(CodeGapTitle(token_index=token.index, reason="unique numeric signature"))
        return gaps

    def _build_name_time_markers(
        self, records: Sequence[TitleLanguageRecord]
    ) -> List[NameTimeMarker]:
        markers: List[NameTimeMarker] = []
        for record in records:
            label = f"#{record.order} {record.anchor}"
            summary = record.narrative
            markers.append(NameTimeMarker(order=record.order, label=label, summary=summary))
        return markers

    def _build_knowledge_base(
        self,
        tokens: Sequence[PhoneticToken],
        numeric_groups: Sequence[NumericEquivalenceGroup],
        universal_groups: Sequence[UniversalSimilarityGroup],
        repetitions: Sequence[NumericRepetitionCluster],
    ) -> KnowledgeBase:
        numeric_membership: Dict[int, int] = {}
        for group in numeric_groups:
            for idx in group.token_indices:
                numeric_membership[idx] = len(group.token_indices)

        universal_membership: Dict[int, str] = {}
        for group in universal_groups:
            for idx in group.token_indices:
                universal_membership.setdefault(idx, group.signature)

        repetition_membership: Dict[int, int] = {}
        for cluster in repetitions:
            for idx in cluster.occurrences:
                repetition_membership[idx] = len(cluster.occurrences)

        entries: List[KnowledgeEntry] = []
        for token in tokens:
            entries.append(
                KnowledgeEntry(
                    token_index=token.index,
                    surface=token.text,
                    normalized=token.normalized,
                    base36=_base36(token.base36_value),
                    lengthened=token.lengthened_value,
                    cubic=token.cubic_value,
                    seed=token.seed,
                    language_code=token.language_code,
                    language_name=token.language_name,
                    universal_signature=universal_membership.get(token.index),
                    numeric_group_size=numeric_membership.get(token.index, 1),
                    repetition_count=repetition_membership.get(token.index, 1),
                )
            )

        clusters: List[KnowledgeCluster] = []
        for group in numeric_groups:
            base36_value = _base36(group.base36_value)
            clusters.append(
                KnowledgeCluster(
                    kind="numeric",
                    label=base36_value,
                    token_indices=group.token_indices,
                    summary=f"{len(group.token_indices)} tokens share numeric signature {base36_value}",
                )
            )
        for group in universal_groups:
            clusters.append(
                KnowledgeCluster(
                    kind="universal",
                    label=group.signature,
                    token_indices=group.token_indices,
                    summary=f"{len(group.token_indices)} tokens share universal signature '{group.signature}'",
                )
            )
        for cluster in repetitions:
            base36_value = _base36(cluster.base36_value)
            clusters.append(
                KnowledgeCluster(
                    kind="repetition",
                    label=base36_value,
                    token_indices=cluster.occurrences,
                    summary=f"{len(cluster.occurrences)} repeated occurrences of {base36_value}",
                )
            )

        insights: List[str] = []
        if tokens:
            unique_signatures = len({token.base36_value for token in tokens})
            insights.append(
                f"{unique_signatures} unique base36 signatures across {len(tokens)} tokens"
            )
            strongest_seed = max(tokens, key=lambda token: token.seed)
            insights.append(
                f"Highest seed token #{strongest_seed.index} '{strongest_seed.text}' (seed {strongest_seed.seed})"
            )
            language_counts: Dict[str, int] = {}
            for token in tokens:
                language_counts[token.language_name] = language_counts.get(token.language_name, 0) + 1
            if language_counts:
                primary_language, count = max(language_counts.items(), key=lambda item: item[1])
                insights.append(f"Primary language {primary_language} covers {count} tokens")
        if numeric_groups:
            top_numeric = max(numeric_groups, key=lambda group: len(group.token_indices))
            insights.append(
                f"Numeric signature {_base36(top_numeric.base36_value)} repeats {len(top_numeric.token_indices)} times"
            )
        if universal_groups:
            top_universal = max(universal_groups, key=lambda group: len(group.token_indices))
            insights.append(
                f"Universal signature '{top_universal.signature}' shared by {len(top_universal.token_indices)} tokens"
            )

        deduped_insights = list(dict.fromkeys(insights))
        return KnowledgeBase(entries=entries, clusters=clusters, insights=deduped_insights)

    # ------------------------------------------------------------------
    # Reporting helpers
    def build_report(
        self,
        analysis: AnalysisResult,
        *,
        include_comparability: bool = True,
    ) -> str:
        lines: List[str] = []
        if analysis.title:
            lines.append(f"Analysis Title: {analysis.title}")
        lines.append(f"Token count: {len(analysis.tokens)}")
        if analysis.language_usage:
            lines.append("Language Usage:")
            for usage in analysis.language_usage:
                percentage = f"{usage.coverage * 100:.2f}%"
                lines.append(
                    f"  - {usage.code}: {usage.name} ({usage.token_count} tokens, {percentage})"
                )
            lines.append("")
        lines.append("Table of Contents:")
        for marker in analysis.table_of_contents:
            lines.append(f"  - [{marker.anchor}] {marker.title}")
        lines.append("")
        lines.append("Title Language Records (descending):")
        for record in analysis.title_language_records:
            lines.append(f"  {record.order}. {record.narrative}")
        lines.append("")
        lines.append("Binary Bookmark:")
        lines.append(f"  Marker: {analysis.binary_bookmark.marker_128bit}")
        lines.append(f"  Codex:  {analysis.binary_bookmark.codex_key}")
        if analysis.binary_bookmark.has_absence:
            lines.append(f"  Absences at tokens: {analysis.binary_bookmark.absent_indices}")
        lines.append("")
        lines.append("Numeric Equivalences:")
        for group in analysis.numeric_equivalences[:10]:
            lines.append(f"  value={group.base36_value} indices={group.token_indices}")
        if not analysis.numeric_equivalences:
            lines.append("  (none)")
        lines.append("")
        lines.append("Repetition Clusters:")
        for cluster in analysis.repetition_clusters[:10]:
            lines.append(f"  value={cluster.base36_value} occurrences={cluster.occurrences}")
        if not analysis.repetition_clusters:
            lines.append("  (none)")
        lines.append("")
        lines.append("Token Seeds:")
        for seed in analysis.token_seeds[:10]:
            lines.append(f"  idx={seed.token_index} seed={seed.base_seed} {seed.slider}")
        if not analysis.token_seeds:
            lines.append("  (none)")
        lines.append("")
        lines.append("Universal Similarity Groups:")
        for group in analysis.universal_similarity[:10]:
            lines.append(f"  signature={group.signature} indices={group.token_indices}")
        if not analysis.universal_similarity:
            lines.append("  (none)")
        lines.append("")
        lines.append("Code Gap Titles:")
        for gap in analysis.code_gap_titles[:10]:
            lines.append(f"  idx={gap.token_index} reason={gap.reason}")
        if not analysis.code_gap_titles:
            lines.append("  (none)")
        lines.append("")
        lines.append("Name Order Timeline:")
        for marker in analysis.name_time_markers:
            lines.append(f"  {marker.label}: {marker.summary}")
        lines.append("")
        lines.append("Knowledge Base Insights:")
        for insight in analysis.knowledge_base.insights[:10]:
            lines.append(f"  - {insight}")
        if not analysis.knowledge_base.insights:
            lines.append("  (none)")
        lines.append("")
        lines.append("Knowledge Entries (first 5):")
        for entry in analysis.knowledge_base.entries[:5]:
            universal = entry.universal_signature or "∅"
            lines.append(
                f"  #{entry.token_index} '{entry.surface}' base36={entry.base36} universal={universal} group={entry.numeric_group_size}"
            )
        if not analysis.knowledge_base.entries:
            lines.append("  (none)")
        lines.append("")
        lines.append("Knowledge Clusters (first 5):")
        for cluster in analysis.knowledge_base.clusters[:5]:
            lines.append(f"  [{cluster.kind}] {cluster.summary} -> {cluster.token_indices}")
        if not analysis.knowledge_base.clusters:
            lines.append("  (none)")
        if include_comparability and "comparability" in analysis.analysis_notes:
            lines.append("")
            lines.append("Comparability Summary:")
            comp = analysis.analysis_notes["comparability"]
            lines.append(
                f"  Window Size: {comp['window_size']} | Numeric: {comp['aggregate_numeric_similarity']:.3f} | "
                f"Similarity: {comp['aggregate_similarity_overlap']:.3f}"
            )
        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Data generation
    def prepare_data_generation(
        self,
        text: str,
        *,
        title: Optional[str] = None,
        reference_text: Optional[str] = None,
        window_size: int = 4,
        include_report: bool = True,
        include_comparability: bool = True,
        language_code: Optional[str] = None,
    ) -> DataGenerationBundle:
        analysis = self.analyze(
            text,
            title=title,
            reference_text=reference_text,
            seed_from=title,
            window_size=window_size,
            compare=bool(reference_text and include_comparability),
            language_code=language_code,
        )
        report = self.build_report(analysis, include_comparability=include_comparability) if include_report else None
        comparability_dict = (
            analysis.analysis_notes.get("comparability") if include_comparability else None
        )
        examples = self._build_examples(analysis, report)
        return DataGenerationBundle(
            title=title,
            analysis=analysis.as_dict(),
            report=report,
            comparability=comparability_dict,
            examples=examples,
        )

    def _build_examples(
        self, analysis: AnalysisResult, report: Optional[str]
    ) -> List[DataGenerationExample]:
        examples: List[DataGenerationExample] = []
        summary_prompt = "Summarise the binary bookmark significance"
        summary_completion = json.dumps(analysis.binary_bookmark.as_dict())
        examples.append(DataGenerationExample(prompt=summary_prompt, completion=summary_completion))
        if analysis.knowledge_base.insights:
            insights_prompt = "List the primary knowledge base insights"
            insights_completion = json.dumps(
                analysis.knowledge_base.insights[:5], ensure_ascii=False
            )
            examples.append(
                DataGenerationExample(prompt=insights_prompt, completion=insights_completion)
            )
        if report:
            examples.append(
                DataGenerationExample(
                    prompt="Provide a concise navigation summary",
                    completion="\n".join(report.splitlines()[:10]),
                )
            )
        return examples


# ----------------------------------------------------------------------
# Utility helpers outside the class

def ref_tokens_slice(tokens: Sequence[PhoneticToken], idx: int, window_size: int) -> List[str]:
    start = idx * window_size
    return [token.text for token in tokens[start : start + window_size]]


def _sequence_overlap(seq_a: Sequence[object], seq_b: Sequence[object]) -> float:
    if not seq_a and not seq_b:
        return 1.0
    if not seq_a or not seq_b:
        return 0.0
    set_a = set(seq_a)
    set_b = set(seq_b)
    intersection = set_a & set_b
    union = set_a | set_b
    if not union:
        return 0.0
    return len(intersection) / len(union)


def analyze_text(
    text: str,
    *,
    title: Optional[str] = None,
    reference_text: Optional[str] = None,
    window_size: int = 4,
    language_code: Optional[str] = None,
) -> AnalysisResult:
    model = PhoneticTrailingSongModel()
    return model.analyze(
        text,
        title=title,
        reference_text=reference_text,
        window_size=window_size,
        compare=bool(reference_text),
        language_code=language_code,
    )


def prepare_data_generation(
    text: str,
    *,
    title: Optional[str] = None,
    reference_text: Optional[str] = None,
    window_size: int = 4,
    include_report: bool = True,
    include_comparability: bool = True,
    language_code: Optional[str] = None,
) -> DataGenerationBundle:
    model = PhoneticTrailingSongModel()
    return model.prepare_data_generation(
        text,
        title=title,
        reference_text=reference_text,
        window_size=window_size,
        include_report=include_report,
        include_comparability=include_comparability,
        language_code=language_code,
    )


# ----------------------------------------------------------------------
# Command line interface

def _load_text_argument(path_or_dash: str) -> str:
    if path_or_dash == "-":
        return sys.stdin.read()
    return Path(path_or_dash).read_text(encoding="utf-8")


def _emit_json(data: Dict[str, object]) -> None:
    json.dump(data, sys.stdout, ensure_ascii=False, indent=2)
    sys.stdout.write("\n")


def _parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Phonetic trailing song analyzer")
    parser.add_argument("text", help="Path to the text file or '-' for stdin")
    parser.add_argument("--title", help="Optional analysis title")
    parser.add_argument("--seed-from", dest="seed_from", help="Seed marker derived from supplied string")
    parser.add_argument("--select-token", type=int, help="Show slider data for the given token index")
    parser.add_argument("--compare-with", dest="compare_with", help="Reference text for comparability")
    parser.add_argument("--compare-window-size", type=int, default=4, help="Sliding window size for comparisons")
    parser.add_argument("--no-report", action="store_true", help="Suppress textual report output")
    parser.add_argument("--emit-json", action="store_true", help="Emit the analysis as JSON")
    parser.add_argument("--emit-data-bundle", metavar="PATH", help="Write a data generation bundle to PATH")
    parser.add_argument("--reference-title", help="Optional title for the reference analysis")
    parser.add_argument(
        "--language-code",
        help="Force a specific language profile (defaults to auto-detection)",
    )
    parser.add_argument(
        "--list-languages",
        action="store_true",
        help="List available language profile codes and exit",
    )
    return parser.parse_args(argv)


def _main(argv: Optional[Sequence[str]] = None) -> int:
    args = _parse_args(argv)
    model = PhoneticTrailingSongModel()

    if args.list_languages:
        for profile in model.language_registry.list_profiles():
            print(f"{profile.code}\t{profile.name} - {profile.description}")
        return 0

    text = _load_text_argument(args.text)
    reference_text = _load_text_argument(args.compare_with) if args.compare_with else None

    analysis = model.analyze(
        text,
        title=args.title,
        reference_text=reference_text,
        seed_from=args.seed_from,
        window_size=args.compare_window_size,
        compare=bool(reference_text),
        language_code=args.language_code,
    )

    if args.select_token is not None:
        selected = next((seed for seed in analysis.token_seeds if seed.token_index == args.select_token), None)
        if selected:
            print(f"Token {selected.token_index}: seed={selected.base_seed} {selected.slider}")
        else:
            print(f"Token {args.select_token} not found", file=sys.stderr)

    if not args.no_report:
        report = model.build_report(analysis)
        print(report)
    else:
        report = None

    if args.emit_json:
        _emit_json(analysis.as_dict())

    if args.emit_data_bundle:
        bundle = model.prepare_data_generation(
            text,
            title=args.title,
            reference_text=reference_text,
            window_size=args.compare_window_size,
            include_report=not args.no_report,
            include_comparability=bool(reference_text),
            language_code=args.language_code,
        )
        Path(args.emit_data_bundle).write_text(
            json.dumps(bundle.as_dict(), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(_main())
