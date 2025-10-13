"""Utility module for phonetic normalization, trailing rhyme grouping, and
title-sound numeric seeding.

The implementation purposely avoids optional third-party dependencies so it can
be executed in constrained environments.  The phoneticization logic is
rule-based and aims for stability rather than linguistic perfection.

Example
-------
>>> model = PhoneticTrailingSongModel()
>>> model.generate_report('''
... Amazing grace how sweet the sound
... That saved a wretch like me
... I once was lost but now am found
... Was blind but now I see
... ''')
"""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field, replace
from typing import Any, Dict, Iterator, List, Sequence, Tuple
import json
import unicodedata
import textwrap

__all__ = [
    "PhoneticToken",
    "LanguageLayer",
    "LanguageLayerGrouping",
    "TitleSoundSummary",
    "NumericEquivalenceGroup",
    "NumericRepetitionCluster",
    "TokenSeedDetail",
    "BinaryBookmark",
    "TableOfContentsMarker",
    "NameTimeMarker",
    "UniversalSimilarityGroup",
    "CursorTranslation",
    "TitleLanguageRecord",
    "CodeGapTitle",
    "ComparabilityWindow",
    "ComparabilityResult",
    "DataGenerationExample",
    "DataGenerationBundle",
    "AnalysisResult",
    "PhoneticTrailingSongModel",
    "analyze_text",
    "prepare_data_generation",
]


def _romanize_letter(character: str) -> str | None:
    """Return a Latin fallback for ``character`` when possible.

    The helper inspects the Unicode name to recover the base letter after the
    ``LETTER`` keyword.  If no explicit base can be derived a best-effort
    fallback is returned by scanning the remaining ASCII letters in the name.
    """

    if "a" <= character <= "z":
        return character

    category = unicodedata.category(character)
    if not category.startswith("L"):
        return None

    name = unicodedata.name(character, "")
    if "LETTER " in name:
        suffix = name.split("LETTER ", 1)[1]
        base = suffix.split()[0]
        ascii_letters = [ch for ch in base.lower() if "a" <= ch <= "z"]
        if ascii_letters:
            return ascii_letters[0]

    ascii_letters = [ch for ch in name.lower() if "a" <= ch <= "z"]
    if ascii_letters:
        return ascii_letters[0]

    return None


def _strip_accents(text: str) -> str:
    """Return ``text`` case-folded with script-aware Latin fallbacks."""

    normalized = unicodedata.normalize("NFKD", text.casefold())
    characters: List[str] = []
    for character in normalized:
        if unicodedata.combining(character):
            continue

        if "0" <= character <= "9":
            characters.append(character)
            continue

        fallback = _romanize_letter(character)
        if fallback:
            characters.append(fallback)

    return "".join(characters)


def _iter_word_spans(text: str) -> Iterator[Tuple[str, int, int]]:
    """Yield ``(word, start, end)`` tuples for Unicode letter sequences."""

    start: int | None = None
    seen_letter = False

    for index, character in enumerate(text):
        category = unicodedata.category(character)
        if category.startswith("L"):
            if start is None:
                start = index
            seen_letter = True
            continue

        if character == "'" and start is not None and seen_letter:
            continue

        if start is not None:
            yield (text[start:index], start, index)
            start = None
            seen_letter = False

    if start is not None:
        yield (text[start:], start, len(text))


@dataclass(frozen=True)
class LanguageLayer:
    """Representation of a token rendered through a specific language lens."""

    #: Name describing the layer role (e.g. ``original`` or ``romanized``).
    layer_name: str
    #: Script hint associated with ``value`` (``latin``, ``universal``, etc.).
    script_hint: str
    #: Value rendered according to the layer rules.
    value: str


@dataclass(frozen=True)
class LanguageLayerGrouping:
    """Aggregate view of shared layers for language-compatible navigation."""

    #: Layer identifier corresponding to :class:`LanguageLayer.layer_name`.
    layer_name: str
    #: Script hint covering the values grouped in this layer.
    script_hint: str
    #: Token positions contributing to the layer.
    positions: Tuple[int, ...]
    #: Layer values recorded for each contributing token (mirrors ``positions``).
    values: Tuple[str, ...]
    #: Canonical value representative for the layer (first recorded entry).
    canonical_value: str


@dataclass(frozen=True)
class PhoneticToken:
    """Container describing a token and its phonetic signature."""

    text: str
    code: str
    layers: Tuple[LanguageLayer, ...] = field(default_factory=tuple)


@dataclass(frozen=True)
class TitleSoundSummary:
    """Aggregate data used for "title sound" and numeric seeding."""

    #: Highlighted combination (first/middle/last) of phonetic fragments.
    title_sound: str
    #: Canonical code created by combining the raw phonetic fragments.
    combined_code: str
    #: Largest base-36 number expressible from the combined code.
    max_numeric_value: int
    #: Deterministic 64-bit seed derived from the numeric value and text stats.
    seed_value: int


@dataclass(frozen=True)
class NumericEquivalenceGroup:
    """Group of tokens sharing the same numeric interpretation."""

    #: Base-36 numeric value derived from the phonetic code minus its unit digit.
    numeric_value: int
    #: Canonical phonetic code that produced the numeric value.
    code: str
    #: Words observed in program (input) order that map to ``numeric_value``.
    words: Tuple[str, ...]
    #: Zero-based token positions at which the words occurred.
    positions: Tuple[int, ...]
    #: Total number of occurrences contributing to the group.
    occurrence_count: int


@dataclass(frozen=True)
class NumericRepetitionCluster:
    """Cluster of tokens sharing a numeric value regardless of phonetic code."""

    #: Base-36 numeric value shared across the cluster.
    numeric_value: int
    #: Ordered members stored as ``(token_position, text, code)``.
    members: Tuple[Tuple[int, str, str], ...]


@dataclass(frozen=True)
class TokenSeedDetail:
    """Per-token numeric and seed metadata for slider-style inspection."""

    #: One-based token index preserving the input order.
    position: int
    #: Original token text as it appeared in the input.
    text: str
    #: Phonetic code generated for ``text``.
    code: str
    #: Raw base-36 value derived from ``code`` (maximal numeric extent).
    max_numeric_value: int
    #: Trimmed numeric value after removing the base-36 unit digit.
    trimmed_numeric_value: int
    #: Doubled-width interpretation of ``trimmed_numeric_value``.
    lengthened_numeric_value: int
    #: Cubic interpretation of ``trimmed_numeric_value``.
    cubic_numeric_value: int
    #: Deterministic 64-bit seed blending trimmed, lengthened, and cubic values.
    seed_value: int
    #: Deterministic 64-bit seed derived solely from ``max_numeric_value``.
    max_seed_value: int
    #: Ratio between trimmed and maximal values for slider visualisation.
    slider_ratio: float
    #: Base-36 unit digit removed during trimming (0 if ``max_numeric_value`` is zero).
    unit_digit: int
    #: Flag noting that ``max_numeric_value`` was zero (token absent numerically).
    is_zero_token: bool
    #: Flag noting that the token only contributed a single base-36 unit digit.
    is_unit_token: bool


@dataclass(frozen=True)
class BinaryBookmark:
    """Combined binary marker capturing title and trailing numeric context."""

    #: 128-bit binary marker derived from title and numeric groupings.
    marker: str
    #: Highlighted trailing codes left behind for each processed line.
    trailing_highlights: Tuple[Tuple[int, str], ...]
    #: Hexadecimal key suitable for API or codex integration.
    codex_key: str
    #: Underlying 128-bit integer used to derive the marker and key.
    constant_value: int
    #: Tokens that failed to contribute a numeric signature (value of zero).
    absent_tokens: Tuple[Tuple[int, str], ...]
    #: Tokens deliberately excluded for being base-36 units (value < 36).
    #: Stored as ``(token_position, text, code, unit_digit)``.
    excluded_unit_tokens: Tuple[Tuple[int, str, str, int], ...]
    #: Flag indicating that the bookmark does not add information over inputs.
    is_absent_cover: bool
    #: Optional explanation for why ``is_absent_cover`` is true.
    absence_reason: str | None


@dataclass(frozen=True)
class UniversalSimilarityGroup:
    """Language-agnostic cluster derived from universal similarity signatures."""

    #: Canonical similarity signature shared across all grouped tokens.
    signature: str
    #: Most representative script hint detected for the grouped tokens.
    script_hint: str
    #: Tokens belonging to the group as ``(position, text, code, ascii_equivalent)``.
    tokens: Tuple[Tuple[int, str, str, str], ...]


@dataclass(frozen=True)
class CursorTranslation:
    """Cursor-friendly translation data derived from similarity signatures."""

    #: One-based token index where the translation applies.
    position: int
    #: Original token text captured from the input.
    text: str
    #: Universal similarity signature backing the translation.
    signature: str
    #: Dominant script hint driving transliteration choices.
    script_hint: str
    #: ASCII-friendly approximation of the token for downstream APIs.
    ascii_equivalent: str
    #: Short explanation describing how the approximation was derived.
    explanation: str


@dataclass(frozen=True)
class TitleLanguageRecord:
    """Textual view treating code constructs as language-titled entries."""

    #: Human-facing title corresponding to the code construct.
    title_name: str
    #: Canonical code or identifier that the title represents.
    source_name: str
    #: Narrative text describing how the code name is interpreted as language.
    language_text: str
    #: Metric used to order titles from highest to lowest significance.
    order_value: int
    #: Concise summary extracted from live analysis data.
    summary: str


@dataclass(frozen=True)
class NameTimeMarker:
    """Timeline view ordering titled sections by language-centric priority."""

    #: Human-facing title mirrored from :class:`TitleLanguageRecord`.
    title_name: str
    #: Canonical source name the title originates from.
    source_name: str
    #: Sequential index representing the time slot in name order.
    time_index: int
    #: Readable label for ``time_index`` keeping exports stable.
    timestamp_label: str
    #: Ordering weight inherited from the originating record.
    order_value: int
    #: Identifier referencing the table-of-contents anchor when available.
    anchor_identifier: str
    #: One-based start index pointing at the anchored span (0 when absent).
    start_index: int
    #: One-based end index pointing at the anchored span (0 when absent).
    end_index: int
    #: Narrative summary aligned with the originating record.
    summary: str

    def as_dict(self) -> Dict[str, Any]:
        """Return a JSON-friendly representation of the time marker."""

        return {
            "title_name": self.title_name,
            "source_name": self.source_name,
            "time_index": self.time_index,
            "timestamp_label": self.timestamp_label,
            "order_value": self.order_value,
            "anchor_identifier": self.anchor_identifier,
            "start_index": self.start_index,
            "end_index": self.end_index,
            "summary": self.summary,
        }


@dataclass(frozen=True)
class CodeGapTitle:
    """Named view describing uncovered or zero-value code regions."""

    #: Machine-friendly identifier for the gap description.
    identifier: str
    #: Human-readable title exposing the gap to user interfaces.
    title_name: str
    #: Positions of the tokens contributing to the gap (1-based).
    positions: Tuple[int, ...]
    #: Token texts corresponding to ``positions``.
    tokens: Tuple[str, ...]
    #: Explanation detailing why the gap exists.
    description: str
    #: Guidance describing how the gap can be filled or acknowledged.
    coverage_hint: str


@dataclass(frozen=True)
class TableOfContentsMarker:
    """Navigational pointer covering a section of the analysis report."""

    #: Machine-friendly identifier for the section.
    identifier: str
    #: Human-friendly heading that can be rendered in reports.
    title: str
    #: One-based starting index of the section (0 if not applicable).
    start_index: int
    #: One-based ending index of the section (0 if not applicable).
    end_index: int
    #: Short description explaining what the section contains.
    description: str
    #: Flag indicating whether the cursor should begin on this section.
    is_cursor: bool


@dataclass(frozen=True)
class ComparabilityWindow:
    """Snapshot comparing numeric and similarity signals across two analyses."""

    #: One-based window index in processing order.
    window_index: int
    #: Inclusive token range for the current analysis (start, end).
    current_range: Tuple[int, int]
    #: Inclusive token range for the reference analysis (start, end).
    reference_range: Tuple[int, int]
    #: Ordered token texts from the current analysis within the window.
    current_tokens: Tuple[str, ...]
    #: Ordered token texts from the reference analysis within the window.
    reference_tokens: Tuple[str, ...]
    #: Numeric values common to both windows after trimming unit digits.
    numeric_overlap: Tuple[int, ...]
    #: Numeric values observed across either window.
    numeric_union: Tuple[int, ...]
    #: Universal similarity signatures shared across both windows.
    universal_overlap: Tuple[str, ...]
    #: Universal similarity signatures seen in either window.
    universal_union: Tuple[str, ...]
    #: Ratio of ``numeric_overlap`` to ``numeric_union`` (1.0 if union is empty).
    numeric_similarity: float
    #: Ratio of ``universal_overlap`` to ``universal_union`` (1.0 if union empty).
    universal_similarity: float
    #: Human-friendly commentary describing the overlap conditions.
    commentary: str

    def as_dict(self) -> Dict[str, Any]:
        """Return a JSON-friendly view of the window."""

        return {
            "window_index": self.window_index,
            "current_range": list(self.current_range),
            "reference_range": list(self.reference_range),
            "current_tokens": list(self.current_tokens),
            "reference_tokens": list(self.reference_tokens),
            "numeric_overlap": list(self.numeric_overlap),
            "numeric_union": list(self.numeric_union),
            "universal_overlap": list(self.universal_overlap),
            "universal_union": list(self.universal_union),
            "numeric_similarity": self.numeric_similarity,
            "universal_similarity": self.universal_similarity,
            "commentary": self.commentary,
        }


@dataclass(frozen=True)
class ComparabilityResult:
    """Aggregate comparability metrics derived from two analyses."""

    #: Size of the windows evaluated during comparison.
    window_size: int
    #: Windows covering the shared content span between both analyses.
    windows: Tuple[ComparabilityWindow, ...]
    #: Mean numeric similarity across all windows (1.0 if no windows).
    aggregate_numeric_similarity: float
    #: Mean universal similarity across all windows (1.0 if no windows).
    aggregate_universal_similarity: float

    def as_dict(self) -> Dict[str, Any]:
        """Return a JSON-serialisable representation of the comparability output."""

        return {
            "window_size": self.window_size,
            "windows": [window.as_dict() for window in self.windows],
            "aggregate_numeric_similarity": self.aggregate_numeric_similarity,
            "aggregate_universal_similarity": self.aggregate_universal_similarity,
        }


@dataclass(frozen=True)
class AnalysisResult:
    """Structured container holding all model outputs for programmatic use."""

    #: Optional descriptive title associated with the analysed text.
    title: str | None
    raw_text: str
    tokens: Tuple[PhoneticToken, ...]
    trailing_rhyme_groups: Dict[str, Tuple[int, ...]]
    title_sound: TitleSoundSummary
    numeric_groups: Tuple[NumericEquivalenceGroup, ...]
    repeated_numeric_groups: Tuple[NumericEquivalenceGroup, ...]
    numeric_repetition_clusters: Tuple[NumericRepetitionCluster, ...]
    token_seed_details: Tuple[TokenSeedDetail, ...]
    interpretation_sequence: Tuple[int, ...]
    #: Sequence where each numeric value is followed by its doubled-width variant.
    lengthened_interpretation_sequence: Tuple[int, ...]
    #: Cubed sequence derived after unit filtering.
    cubic_interpretation_sequence: Tuple[int, ...]
    #: Trimmed numeric values repeated for every contributing token.
    repetition_interpretation_sequence: Tuple[int, ...]
    #: Lengthened numeric values repeated for every contributing token.
    repetition_lengthened_sequence: Tuple[int, ...]
    #: Cubic numeric values repeated for every contributing token.
    repetition_cubic_sequence: Tuple[int, ...]
    binary_bookmark: BinaryBookmark
    bookmark_input_order: Tuple[Tuple[int, str, str, int], ...]
    bookmark_numeric_order: Tuple[Tuple[int, str, str, int], ...]
    #: Tokens filtered out for being base-36 units (position, text, code, unit_digit).
    excluded_unit_tokens: Tuple[Tuple[int, str, str, int], ...]
    #: Named descriptions of gaps where numeric coverage is absent.
    code_gap_titles: Tuple[CodeGapTitle, ...]
    #: Layered language views derived from each token across scripts.
    language_layer_groupings: Tuple[LanguageLayerGrouping, ...]
    #: Navigational markers describing how to traverse the report.
    table_of_contents: Tuple[TableOfContentsMarker, ...]
    #: Universal language-agnostic similarity clusters derived from tokens.
    universal_similarity_groups: Tuple[UniversalSimilarityGroup, ...]
    #: Cursor-oriented translations providing ASCII fallbacks for every token.
    cursor_translations: Tuple[CursorTranslation, ...]
    #: Title-oriented language records treating code constructs as named prose.
    title_language_records: Tuple[TitleLanguageRecord, ...]
    #: Descending order of titles ranked by their computed ``order_value``.
    descending_title_language_order: Tuple[str, ...]
    #: Timeline describing name-based ordering independent from token positions.
    name_time_markers: Tuple[NameTimeMarker, ...]

    def as_dict(self) -> Dict[str, Any]:
        """Return a JSON-serialisable representation of the analysis."""

        return {
            "title": self.title,
            "raw_text": self.raw_text,
            "tokens": [
                {
                    "text": token.text,
                    "code": token.code,
                    "layers": [
                        {
                            "layer_name": layer.layer_name,
                            "script_hint": layer.script_hint,
                            "value": layer.value,
                        }
                        for layer in token.layers
                    ],
                }
                for token in self.tokens
            ],
            "trailing_rhyme_groups": {
                code: list(indices)
                for code, indices in self.trailing_rhyme_groups.items()
            },
            "title_sound": {
                "title_sound": self.title_sound.title_sound,
                "combined_code": self.title_sound.combined_code,
                "max_numeric_value": self.title_sound.max_numeric_value,
                "seed_value": self.title_sound.seed_value,
            },
            "numeric_groups": [
                {
                    "numeric_value": group.numeric_value,
                    "code": group.code,
                    "words": list(group.words),
                    "positions": list(group.positions),
                    "occurrence_count": group.occurrence_count,
                }
                for group in self.numeric_groups
            ],
            "repeated_numeric_groups": [
                {
                    "numeric_value": group.numeric_value,
                    "code": group.code,
                    "words": list(group.words),
                    "positions": list(group.positions),
                    "occurrence_count": group.occurrence_count,
                }
                for group in self.repeated_numeric_groups
            ],
            "numeric_repetition_clusters": [
                {
                    "numeric_value": cluster.numeric_value,
                    "members": [list(member) for member in cluster.members],
                }
                for cluster in self.numeric_repetition_clusters
            ],
            "token_seed_details": [
                {
                    "position": detail.position,
                    "text": detail.text,
                    "code": detail.code,
                    "max_numeric_value": detail.max_numeric_value,
                    "trimmed_numeric_value": detail.trimmed_numeric_value,
                    "lengthened_numeric_value": detail.lengthened_numeric_value,
                    "cubic_numeric_value": detail.cubic_numeric_value,
                    "seed_value": detail.seed_value,
                    "max_seed_value": detail.max_seed_value,
                    "slider_ratio": detail.slider_ratio,
                    "unit_digit": detail.unit_digit,
                    "is_zero_token": detail.is_zero_token,
                    "is_unit_token": detail.is_unit_token,
                }
                for detail in self.token_seed_details
            ],
            "interpretation_sequence": list(self.interpretation_sequence),
            "lengthened_interpretation_sequence": list(
                self.lengthened_interpretation_sequence
            ),
            "cubic_interpretation_sequence": list(
                self.cubic_interpretation_sequence
            ),
            "repetition_interpretation_sequence": list(
                self.repetition_interpretation_sequence
            ),
            "repetition_lengthened_sequence": list(
                self.repetition_lengthened_sequence
            ),
            "repetition_cubic_sequence": list(
                self.repetition_cubic_sequence
            ),
            "code_gap_titles": [
                {
                    "identifier": record.identifier,
                    "title_name": record.title_name,
                    "positions": list(record.positions),
                    "tokens": list(record.tokens),
                    "description": record.description,
                    "coverage_hint": record.coverage_hint,
                }
                for record in self.code_gap_titles
            ],
            "language_layer_groupings": [
                {
                    "layer_name": group.layer_name,
                    "script_hint": group.script_hint,
                    "positions": list(group.positions),
                    "values": list(group.values),
                    "canonical_value": group.canonical_value,
                }
                for group in self.language_layer_groupings
            ],
            "binary_bookmark": {
                "marker": self.binary_bookmark.marker,
                "trailing_highlights": [
                    list(item) for item in self.binary_bookmark.trailing_highlights
                ],
                "codex_key": self.binary_bookmark.codex_key,
                "constant_value": self.binary_bookmark.constant_value,
                "absent_tokens": [
                    list(item) for item in self.binary_bookmark.absent_tokens
                ],
                "excluded_unit_tokens": [
                    list(item)
                    for item in self.binary_bookmark.excluded_unit_tokens
                ],
                "is_absent_cover": self.binary_bookmark.is_absent_cover,
                "absence_reason": self.binary_bookmark.absence_reason,
            },
            "bookmark_input_order": [list(item) for item in self.bookmark_input_order],
            "bookmark_numeric_order": [
                list(item) for item in self.bookmark_numeric_order
            ],
            "excluded_unit_tokens": [list(item) for item in self.excluded_unit_tokens],
            "table_of_contents": [
                {
                    "identifier": marker.identifier,
                    "title": marker.title,
                    "start_index": marker.start_index,
                    "end_index": marker.end_index,
                    "description": marker.description,
                    "is_cursor": marker.is_cursor,
                }
                for marker in self.table_of_contents
            ],
            "universal_similarity_groups": [
                {
                    "signature": group.signature,
                    "script_hint": group.script_hint,
                    "tokens": [list(item) for item in group.tokens],
                }
                for group in self.universal_similarity_groups
            ],
            "cursor_translations": [
                {
                    "position": entry.position,
                    "text": entry.text,
                    "signature": entry.signature,
                    "script_hint": entry.script_hint,
                    "ascii_equivalent": entry.ascii_equivalent,
                    "explanation": entry.explanation,
                }
                for entry in self.cursor_translations
            ],
            "title_language_records": [
                {
                    "title_name": record.title_name,
                    "source_name": record.source_name,
                    "language_text": record.language_text,
                    "order_value": record.order_value,
                    "summary": record.summary,
                }
                for record in self.title_language_records
            ],
            "descending_title_language_order": list(
                self.descending_title_language_order
            ),
            "name_time_markers": [marker.as_dict() for marker in self.name_time_markers],
        }


@dataclass(frozen=True)
class DataGenerationExample:
    """Prompt/completion pair suitable for supervised data generation."""

    prompt: str
    completion: str
    metadata: Dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> Dict[str, Any]:
        """Return a JSON-serialisable mapping for this example."""

        payload = {"prompt": self.prompt, "completion": self.completion}
        if self.metadata:
            payload["metadata"] = self.metadata
        return payload


@dataclass(frozen=True)
class DataGenerationBundle:
    """Container grouping analysis outputs with data generation examples."""

    analysis: AnalysisResult
    reference_analysis: AnalysisResult | None
    comparability: ComparabilityResult | None
    analysis_report: str | None
    reference_report: str | None
    comparability_report: str | None
    examples: Tuple[DataGenerationExample, ...]
    analysis_title: str | None
    reference_title: str | None

    def as_dict(self) -> Dict[str, Any]:
        """Return a mapping suitable for JSON serialisation."""

        payload: Dict[str, Any] = {
            "analysis": self.analysis.as_dict(),
            "analysis_report": self.analysis_report,
            "examples": [example.as_dict() for example in self.examples],
        }
        if self.analysis_title is not None:
            payload["analysis_title"] = self.analysis_title
        if self.reference_analysis is not None:
            payload["reference_analysis"] = self.reference_analysis.as_dict()
        if self.reference_report is not None:
            payload["reference_report"] = self.reference_report
        if self.comparability is not None:
            payload["comparability"] = self.comparability.as_dict()
        if self.comparability_report is not None:
            payload["comparability_report"] = self.comparability_report
        if self.reference_title is not None:
            payload["reference_title"] = self.reference_title
        return payload


class PhoneticTrailingSongModel:
    """Compute phonetic approximations and trailing rhyme groupings.

    The algorithm approximates English phonetics using a deterministic mapping
    inspired by Soundex and Metaphone.  The mapping favours reproducibility
    across diverse scripts included in the repository rather than perfect
    phonological accuracy.  Each word is normalised, stripped of accents, and
    converted into a compact consonant/vowel code.  That code is subsequently
    used to cluster the trailing words of lines, which provides a lightweight
    "song typing" or rhyme detection heuristic.
    """

    #: Basic consonant-to-phoneme grouping that balances fidelity and
    #: predictability.  Groups follow a rough approximation of articulatory
    #: similarity.
    _CONSONANT_GROUPS: Tuple[Tuple[str, str], ...] = (
        ("bfpv", "1"),
        ("cgjksqxz", "2"),
        ("dt", "3"),
        ("l", "4"),
        ("mn", "5"),
        ("r", "6"),
        ("ywh", "7"),
    )

    #: Vowel canonical symbol used in the signature.  Collapsing all vowels to
    #: a single marker improves resilience to dialect differences.
    _VOWEL_CODE = "0"

    #: Script keywords mapped to friendly hints used for universal similarity.
    _SCRIPT_HINTS: Tuple[Tuple[str, str], ...] = (
        ("LATIN", "latin"),
        ("CYRILLIC", "cyrillic"),
        ("GREEK", "greek"),
        ("HEBREW", "hebrew"),
        ("ARABIC", "arabic"),
        ("DEVANAGARI", "devanagari"),
        ("HIRAGANA", "hiragana"),
        ("KATAKANA", "katakana"),
        ("HANGUL", "hangul"),
        ("CJK UNIFIED", "cjk"),
        ("CJK COMPATIBILITY", "cjk"),
        ("ARMENIAN", "armenian"),
        ("GEORGIAN", "georgian"),
        ("ETHIOPIC", "ethiopic"),
        ("THAI", "thai"),
        ("LAO", "lao"),
        ("TIBETAN", "tibetan"),
        ("CANADIAN SYLLABICS", "canadian"),
        ("CHEROKEE", "cherokee"),
        ("TELUGU", "telugu"),
        ("MALAYALAM", "malayalam"),
        ("TAMIL", "tamil"),
        ("BENGALI", "bengali"),
        ("GUJARATI", "gujarati"),
        ("GURMUKHI", "gurmukhi"),
        ("SINHALA", "sinhala"),
    )

    #: Short markers representing scripts in universal similarity signatures.
    _SCRIPT_MARKERS: Dict[str, str] = {
        "latin": "a",
        "cyrillic": "c",
        "greek": "g",
        "hebrew": "h",
        "arabic": "r",
        "devanagari": "d",
        "hiragana": "j",
        "katakana": "k",
        "hangul": "n",
        "cjk": "z",
        "armenian": "m",
        "georgian": "e",
        "ethiopic": "p",
        "thai": "t",
        "lao": "l",
        "tibetan": "b",
        "canadian": "q",
        "cherokee": "o",
        "telugu": "u",
        "malayalam": "v",
        "tamil": "y",
        "bengali": "w",
        "gujarati": "x",
        "gurmukhi": "s",
        "sinhala": "f",
    }

    def __init__(self, max_code_length: int = 6) -> None:
        self.max_code_length = max_code_length
        self._consonant_lookup: Dict[str, str] = self._build_consonant_lookup()

    def _build_consonant_lookup(self) -> Dict[str, str]:
        lookup: Dict[str, str] = {}
        for chars, code in self._CONSONANT_GROUPS:
            for char in chars:
                lookup[char] = code
        return lookup

    # ------------------------------------------------------------------
    # Phonetic encoding helpers
    # ------------------------------------------------------------------
    def phonetic_code(self, word: str) -> str:
        """Return a deterministic phonetic code for ``word``.

        The procedure consists of the following steps:

        1. Remove accents and convert the word to lower case.
        2. Remove non alphabetic characters.
        3. Encode the first letter explicitly to preserve word onsets.
        4. Convert subsequent letters using the consonant grouping table.
        5. Collapse consecutive duplicates and vowels.
        6. Pad or trim the output so that all codes have uniform length.
        """

        clean = _strip_accents(word)
        letters = [ch for ch in clean if ch.isalpha()]
        if not letters:
            return "".ljust(self.max_code_length, "0")

        first_letter = letters[0]
        signature = [first_letter]
        previous_code = None

        for letter in letters[1:]:
            if letter in "aeiou":
                code = self._VOWEL_CODE
            else:
                code = self._consonant_lookup.get(letter, letter)

            if code == previous_code or code == self._VOWEL_CODE:
                previous_code = code
                continue

            signature.append(code)
            previous_code = code

        combined = "".join(signature)[: self.max_code_length]
        return combined.ljust(self.max_code_length, "0")

    def phoneticize(self, text: str) -> List[PhoneticToken]:
        """Return ``PhoneticToken`` objects for every recognised word."""

        tokens: List[PhoneticToken] = []
        for word, _, _ in _iter_word_spans(text):
            signature, script_hint, ascii_equivalent = self._similarity_components(word)
            layers = self._build_language_layers(
                word=word,
                signature=signature,
                script_hint=script_hint,
                ascii_equivalent=ascii_equivalent,
            )
            tokens.append(
                PhoneticToken(
                    text=word,
                    code=self.phonetic_code(word),
                    layers=layers,
                )
            )
        return tokens

    # ------------------------------------------------------------------
    # Universal similarity helpers
    # ------------------------------------------------------------------
    def _similarity_components(self, word: str) -> Tuple[str, str, str]:
        """Return ``(signature, script_hint, ascii_equivalent)`` for ``word``."""

        if not word:
            return ("_", "unknown", "")

        normalized = unicodedata.normalize("NFKD", word)
        signature: List[str] = []
        ascii_chars: List[str] = []
        script_counter: Counter[str] = Counter()

        for character in normalized:
            category = unicodedata.category(character)
            if category.startswith("M"):
                # Skip combining marks, they are represented by the base letter.
                continue

            lower = character.lower()
            if "a" <= lower <= "z":
                signature.append(lower)
                ascii_chars.append(lower)
                script_counter["latin"] += 1
                continue

            if "0" <= lower <= "9":
                signature.append("#")
                ascii_chars.append(lower)
                continue

            script_hint = "unicode"
            name = unicodedata.name(character, "")
            for keyword, hint in self._SCRIPT_HINTS:
                if keyword in name:
                    script_hint = hint
                    break

            script_counter[script_hint] += 1
            marker = self._SCRIPT_MARKERS.get(script_hint, "?")
            signature.append(marker)
            ascii_chars.append(f"u{ord(character):04x}")

        if not signature:
            signature.append("_")

        script_hint = (
            script_counter.most_common(1)[0][0] if script_counter else "unknown"
        )
        ascii_equivalent = "".join(ascii_chars)
        if not ascii_equivalent:
            ascii_equivalent = word

        return ("".join(signature), script_hint, ascii_equivalent)

    def _build_language_layers(
        self,
        *,
        word: str,
        signature: str,
        script_hint: str,
        ascii_equivalent: str,
    ) -> Tuple[LanguageLayer, ...]:
        """Return layered renderings for ``word`` to enhance language compatibility."""

        layers: List[LanguageLayer] = []
        seen: set[Tuple[str, str, str]] = set()

        def add_layer(name: str, hint: str, value: str) -> None:
            if not value:
                return
            key = (name, hint, value)
            if key in seen:
                return
            seen.add(key)
            layers.append(
                LanguageLayer(layer_name=name, script_hint=hint, value=value)
            )

        add_layer("signature", "universal", signature or "_")
        add_layer("original", script_hint or "unknown", word)

        normalized = unicodedata.normalize("NFKC", word)
        if normalized and normalized != word:
            add_layer("normalized", script_hint or "unknown", normalized)

        romanized = _strip_accents(word)
        if romanized:
            add_layer("romanized", "latin", romanized)

        ascii_value = ascii_equivalent or romanized or normalized or signature or word
        add_layer("ascii", "latin", ascii_value)

        return tuple(layers)

    def token_similarity_signatures(
        self, tokens: Sequence[PhoneticToken]
    ) -> List[Tuple[str, str, str]]:
        """Return similarity components for each token in ``tokens``."""

        components: List[Tuple[str, str, str]] = []
        for token in tokens:
            components.append(self._similarity_components(token.text))
        return components

    def universal_similarity_groups(
        self,
        tokens: Sequence[PhoneticToken],
        signatures: Sequence[Tuple[str, str, str]] | None = None,
    ) -> List[UniversalSimilarityGroup]:
        """Cluster tokens using universal similarity signatures."""

        if signatures is None:
            signatures = self.token_similarity_signatures(tokens)

        buckets: Dict[str, List[Tuple[int, str, str, str, str]]] = {}
        for index, (token, component) in enumerate(zip(tokens, signatures)):
            signature, script_hint, ascii_equivalent = component
            bucket = buckets.setdefault(signature, [])
            bucket.append(
                (
                    index + 1,
                    token.text,
                    token.code,
                    ascii_equivalent,
                    script_hint,
                )
            )

        groups: List[UniversalSimilarityGroup] = []
        for signature, items in buckets.items():
            script_counter: Counter[str] = Counter(item[4] for item in items)
            script_hint = (
                script_counter.most_common(1)[0][0] if script_counter else "unknown"
            )
            ordered = tuple(
                (position, text, code, ascii_equivalent)
                for position, text, code, ascii_equivalent, _ in items
            )
            groups.append(
                UniversalSimilarityGroup(
                    signature=signature,
                    script_hint=script_hint,
                    tokens=ordered,
                )
            )

        groups.sort(key=lambda group: (len(group.tokens) * -1, group.signature))
        return groups

    def cursor_translations(
        self,
        tokens: Sequence[PhoneticToken],
        signatures: Sequence[Tuple[str, str, str]] | None = None,
    ) -> List[CursorTranslation]:
        """Return cursor-friendly translation entries for ``tokens``."""

        if signatures is None:
            signatures = self.token_similarity_signatures(tokens)

        translations: List[CursorTranslation] = []
        for index, (token, component) in enumerate(zip(tokens, signatures)):
            signature, script_hint, ascii_equivalent = component
            explanation = (
                "Derived from phonetic code {code} and script hint '{hint}'."
            ).format(code=token.code, hint=script_hint)
            translations.append(
                CursorTranslation(
                    position=index + 1,
                    text=token.text,
                    signature=signature,
                    script_hint=script_hint,
                    ascii_equivalent=ascii_equivalent,
                    explanation=explanation,
                )
            )

        return translations

    # ------------------------------------------------------------------
    # Trailing rhyme ("song typing") helpers
    # ------------------------------------------------------------------
    def trailing_rhyme_key(self, line: str) -> str:
        """Return the phonetic code for the final word of ``line``."""

        words = [word for word, _, _ in _iter_word_spans(line)]
        if not words:
            return "".ljust(self.max_code_length, "0")
        return self.phonetic_code(words[-1])

    def group_by_trailing_rhyme(self, lines: Sequence[str]) -> Dict[str, List[int]]:
        """Group line numbers by their trailing phonetic code."""

        groups: Dict[str, List[int]] = {}
        for index, line in enumerate(lines):
            key = self.trailing_rhyme_key(line)
            groups.setdefault(key, []).append(index)
        return groups

    # ------------------------------------------------------------------
    # Title sound and numeric seed helpers
    # ------------------------------------------------------------------
    def line_title_sound(self, line: str) -> str:
        """Return the title sound fragment for an individual line.

        The fragment is composed by pairing the leading and trailing words so
        that melodic similarity and onset are both captured.
        """

        words = [word for word, _, _ in _iter_word_spans(line)]
        if not words:
            return "".ljust(self.max_code_length * 2, "0")

        first = self.phonetic_code(words[0])
        last = self.phonetic_code(words[-1])
        fragment_width = max(1, self.max_code_length // 2)
        return f"{first[:fragment_width]}{last[-fragment_width:]}"

    def _code_to_number(self, code: str) -> int:
        """Translate an alphanumeric code to a base-36 integer."""

        value = 0
        for char in code.lower():
            if char.isdigit():
                digit = int(char)
            elif "a" <= char <= "z":
                digit = 10 + (ord(char) - ord("a"))
            else:
                continue
            value = value * 36 + digit
        return value

    def _number_to_base36(self, value: int) -> str:
        """Return the base-36 string representation of ``value``."""

        if value <= 0:
            return "0"

        digits: List[str] = []
        while value:
            value, remainder = divmod(value, 36)
            if remainder < 10:
                digits.append(str(remainder))
            else:
                digits.append(chr(ord("a") + (remainder - 10)))
        return "".join(reversed(digits))

    def _lengthen_numeric_value(self, value: int) -> int:
        """Return a deterministic lengthened number with doubled base-36 width."""

        base36 = self._number_to_base36(value)
        if base36 == "0":
            return 0
        doubled = base36 + base36
        return self._code_to_number(doubled)

    def _trim_unit_digit(self, value: int) -> int:
        """Remove the base-36 unit digit from ``value``."""

        return value // 36

    def _collect_numeric_details(
        self, tokens: Sequence[PhoneticToken]
    ) -> Tuple[
        List[NumericEquivalenceGroup],
        List[Tuple[int, str]],
        List[Tuple[int, str, str, int]],
    ]:
        """Return numeric groups, zero-value tokens, and excluded unit tokens.

        Excluded unit entries record the one-digit remainder that was removed
        from each phonetic code.
        """

        bucket: Dict[Tuple[int, str], Dict[str, List[Any]]] = {}
        ordered_keys: List[Tuple[int, str]] = []
        zero_tokens: List[Tuple[int, str]] = []
        excluded_units: List[Tuple[int, str, str, int]] = []

        for index, token in enumerate(tokens):
            numeric_value = self._code_to_number(token.code)
            if numeric_value == 0:
                zero_tokens.append((index + 1, token.text))
                continue

            trimmed_value = self._trim_unit_digit(numeric_value)
            if trimmed_value == 0:
                unit_digit = numeric_value % 36
                excluded_units.append(
                    (index + 1, token.text, token.code, unit_digit)
                )
                continue

            key = (trimmed_value, token.code)
            if key not in bucket:
                bucket[key] = {"words": [], "positions": []}
                ordered_keys.append(key)
            bucket[key]["words"].append(token.text)
            bucket[key]["positions"].append(index)

        groups: List[NumericEquivalenceGroup] = []
        for trimmed_value, code in ordered_keys:
            data = bucket[(trimmed_value, code)]
            occurrence_count = len(data["positions"])
            groups.append(
                NumericEquivalenceGroup(
                    numeric_value=trimmed_value,
                    code=code,
                    words=tuple(data["words"]),
                    positions=tuple(data["positions"]),
                    occurrence_count=occurrence_count,
                )
            )

        return groups, zero_tokens, excluded_units

    def token_seed_details(
        self, tokens: Sequence[PhoneticToken]
    ) -> List[TokenSeedDetail]:
        """Return per-token numeric and seed metadata for slider inspection."""

        details: List[TokenSeedDetail] = []
        mask = 0xFFFFFFFFFFFFFFFF

        for index, token in enumerate(tokens):
            position = index + 1
            max_value = self._code_to_number(token.code)
            trimmed_value = self._trim_unit_digit(max_value) if max_value else 0
            lengthened_value = (
                self._lengthen_numeric_value(trimmed_value) if trimmed_value else 0
            )
            cubic_value = pow(trimmed_value, 3) if trimmed_value else 0
            unit_digit = max_value % 36 if max_value else 0
            ratio = (trimmed_value / max_value) if max_value else 0.0
            ratio = max(0.0, min(1.0, ratio))

            seed_value = (
                (trimmed_value * 0x9E3779B185EBCA87)
                ^ (lengthened_value & mask)
                ^ (cubic_value & mask)
                ^ ((position) * 0xBF58476D1CE4E5B9)
            ) & mask
            max_seed_value = (
                (max_value * 0xD6E8FEB86659FD93)
                ^ (position * 0x94D049BB133111EB)
            ) & mask

            details.append(
                TokenSeedDetail(
                    position=position,
                    text=token.text,
                    code=token.code,
                    max_numeric_value=max_value,
                    trimmed_numeric_value=trimmed_value,
                    lengthened_numeric_value=lengthened_value,
                    cubic_numeric_value=cubic_value,
                    seed_value=seed_value,
                    max_seed_value=max_seed_value,
                    slider_ratio=ratio,
                    unit_digit=unit_digit,
                    is_zero_token=max_value == 0,
                    is_unit_token=max_value != 0 and trimmed_value == 0,
                )
            )

        return details

    def title_sound_summary(
        self, text: str, tokens: Sequence[PhoneticToken] | None = None
    ) -> TitleSoundSummary:
        """Return combined title sound information for ``text``.

        The algorithm:

        1. Computes line-level fragments with :meth:`line_title_sound`.
        2. Concatenates the fragments into a canonical code.
        3. Converts the canonical code into its maximum base-36 value.
        4. Derives a deterministic 64-bit seed from the numeric value and text
           statistics.  The seed can be used for pseudo-random scheduling,
           dataset sharding, or any component that requires reproducible
           selection.
        """

        lines = [line.strip() for line in text.strip().splitlines() if line.strip()]
        if not lines:
            zero_code = "".ljust(self.max_code_length * 2, "0")
            return TitleSoundSummary(
                title_sound=zero_code,
                combined_code=zero_code,
                max_numeric_value=0,
                seed_value=0,
            )

        fragments: List[str] = [self.line_title_sound(line) for line in lines]
        combined_code = "-".join(fragments)
        numeric_value = self._code_to_number(combined_code)

        if tokens is None:
            tokens = self.phoneticize(text)
        else:
            tokens = list(tokens)
        token_count = len(tokens)
        line_count = len(lines)
        seed_value = (
            (numeric_value * 0x9E3779B185EBCA87)
            ^ (token_count * 0xC2B2AE3D27D4EB4F)
            ^ line_count
        ) & 0xFFFFFFFFFFFFFFFF

        focus_indices = sorted({0, len(fragments) // 2, len(fragments) - 1})
        highlighted = [fragments[index] for index in focus_indices]
        title_sound = " / ".join(highlighted)

        return TitleSoundSummary(
            title_sound=title_sound,
            combined_code=combined_code,
            max_numeric_value=numeric_value,
            seed_value=seed_value,
        )

    def seed_from_name(self, name: str) -> int:
        """Return a deterministic seed computed from ``name``.

        This helper enables "factory" style selection or batching mechanisms
        where a name or identifier should map to a reproducible numeric seed.
        """

        summary = self.title_sound_summary(name)
        return summary.seed_value

    def numeric_equivalence_groups(
        self, tokens: Sequence[PhoneticToken]
    ) -> List[NumericEquivalenceGroup]:
        """Return tokens grouped by identical numeric interpretations.

        The unit digit of each base-36 number is removed prior to grouping, and
        any token that consists solely of that unit digit is excluded entirely.
        """

        groups, _, _ = self._collect_numeric_details(tokens)
        return groups

    def repeated_numeric_groups(
        self, tokens: Sequence[PhoneticToken]
    ) -> List[NumericEquivalenceGroup]:
        """Return numeric groups that contain more than one contributing token."""

        return [
            group
            for group in self.numeric_equivalence_groups(tokens)
            if group.occurrence_count > 1
        ]

    def numeric_repetition_clusters(
        self, tokens: Sequence[PhoneticToken]
    ) -> List[NumericRepetitionCluster]:
        """Return clusters sharing a numeric value irrespective of phonetic code."""

        buckets: Dict[int, List[Tuple[int, str, str]]] = {}
        for index, token in enumerate(tokens):
            numeric_value = self._code_to_number(token.code)
            if not numeric_value:
                continue
            trimmed_value = self._trim_unit_digit(numeric_value)
            if not trimmed_value:
                continue
            members = buckets.setdefault(trimmed_value, [])
            members.append((index + 1, token.text, token.code))

        clusters: List[NumericRepetitionCluster] = []
        for numeric_value, members in buckets.items():
            if len(members) < 2:
                continue
            clusters.append(
                NumericRepetitionCluster(
                    numeric_value=numeric_value,
                    members=tuple(members),
                )
            )
        return clusters

    def interpretation_sequence(self, tokens: Sequence[PhoneticToken]) -> List[int]:
        """Return numeric equivalence identifiers in their appearance order."""

        return [group.numeric_value for group in self.numeric_equivalence_groups(tokens)]

    def lengthened_interpretation_sequence(
        self,
        tokens: Sequence[PhoneticToken],
        groups: Sequence[NumericEquivalenceGroup] | None = None,
    ) -> List[int]:
        """Return a sequence with each numeric value followed by its lengthened form."""

        if groups is None:
            groups = self.numeric_equivalence_groups(tokens)

        sequence: List[int] = []
        for group in groups:
            sequence.append(group.numeric_value)
            sequence.append(self._lengthen_numeric_value(group.numeric_value))
        return sequence

    def cubic_interpretation_sequence(
        self,
        tokens: Sequence[PhoneticToken],
        groups: Sequence[NumericEquivalenceGroup] | None = None,
    ) -> List[int]:
        """Return the cubed value for each numeric group (unit-filtered)."""

        if groups is None:
            groups = self.numeric_equivalence_groups(tokens)

        return [pow(group.numeric_value, 3) for group in groups]

    def repetition_interpretation_sequence(
        self, tokens: Sequence[PhoneticToken]
    ) -> List[int]:
        """Return trimmed numeric values repeated for each contributing token."""

        repeated: List[int] = []
        for token in tokens:
            numeric_value = self._code_to_number(token.code)
            if not numeric_value:
                continue
            trimmed_value = self._trim_unit_digit(numeric_value)
            if not trimmed_value:
                continue
            repeated.append(trimmed_value)
        return repeated

    def repetition_lengthened_sequence(
        self, tokens: Sequence[PhoneticToken]
    ) -> List[int]:
        """Return lengthened numeric values repeated for each contributing token."""

        repeated: List[int] = []
        for value in self.repetition_interpretation_sequence(tokens):
            repeated.append(self._lengthen_numeric_value(value))
        return repeated

    def repetition_cubic_sequence(
        self, tokens: Sequence[PhoneticToken]
    ) -> List[int]:
        """Return cubic numeric values repeated for each contributing token."""

        repeated: List[int] = []
        for value in self.repetition_interpretation_sequence(tokens):
            repeated.append(pow(value, 3))
        return repeated

    # ------------------------------------------------------------------
    # Binary marker helpers
    # ------------------------------------------------------------------
    def binary_constant_marker(
        self, text: str, tokens: Sequence[PhoneticToken] | None = None
    ) -> BinaryBookmark:
        """Combine title and numeric signatures into a binary bookmark."""

        if tokens is None:
            tokens = self.phoneticize(text)
        else:
            tokens = list(tokens)

        lines = [line.strip() for line in text.strip().splitlines() if line.strip()]
        trailing_highlights: List[Tuple[int, str]] = [
            (index + 1, self.trailing_rhyme_key(line))
            for index, line in enumerate(lines)
        ]

        title_summary = self.title_sound_summary(text, tokens)
        numeric_groups, zero_tokens, excluded_units = self._collect_numeric_details(tokens)
        absent_tokens = tuple(zero_tokens)

        mask = 0xFFFFFFFFFFFFFFFF
        low_value = title_summary.seed_value & mask
        high_value = (~title_summary.seed_value) & mask
        changed = False

        contributions: List[Tuple[int, int, int, int]] = []
        for index, token in enumerate(tokens):
            numeric_value = self._code_to_number(token.code)
            if not numeric_value:
                continue
            trimmed_value = self._trim_unit_digit(numeric_value)
            if not trimmed_value:
                continue
            lengthened_value = self._lengthen_numeric_value(trimmed_value)
            cube_value = pow(trimmed_value, 3)
            contributions.append(
                (index + 1, trimmed_value, lengthened_value, cube_value)
            )

        for offset, (position, trimmed_value, lengthened_value, cube_value) in enumerate(
            contributions, start=1
        ):
            numeric_value = trimmed_value & mask
            lengthened_low = lengthened_value & mask
            lengthened_high = (lengthened_value >> 64) & mask
            cube_low = cube_value & mask
            cube_high = (cube_value >> 64) & mask

            low_rot = ((low_value << 17) | (low_value >> (64 - 17))) & mask
            high_rot = ((high_value << 23) | (high_value >> (64 - 23))) & mask

            position_mix_low = ((position * 0xA24BAED4963EE407) & mask)
            position_mix_high = ((position * 0x9E3779B97F4A7C15) & mask)

            new_low = (
                low_rot
                ^ numeric_value
                ^ ((offset * 0x9E3779B185EBCA87) & mask)
                ^ position_mix_low
                ^ lengthened_high
                ^ cube_high
            ) & mask
            new_high = (
                high_rot
                ^ lengthened_low
                ^ ((offset * 0xC2B2AE3D27D4EB4F) & mask)
                ^ position_mix_high
                ^ cube_low
            ) & mask

            if new_low != low_value or new_high != high_value:
                changed = True

            low_value = new_low
            high_value = new_high

        combined_value = (high_value << 64) | low_value
        marker = format(combined_value, "0128b")
        codex_key = format(combined_value, "032x")

        is_absent_cover = False
        absence_reason: str | None = None
        if not tokens:
            is_absent_cover = True
            absence_reason = "no tokens available"
        elif not numeric_groups:
            is_absent_cover = True
            if excluded_units:
                absence_reason = "all numeric values were unit-length"
            else:
                absence_reason = "no numeric equivalence groups"
        elif not changed:
            is_absent_cover = True
            absence_reason = "numeric groups did not influence bookmark"

        return BinaryBookmark(
            marker=marker,
            trailing_highlights=tuple(trailing_highlights),
            codex_key=codex_key,
            constant_value=combined_value,
            absent_tokens=absent_tokens,
            excluded_unit_tokens=tuple(excluded_units),
            is_absent_cover=is_absent_cover,
            absence_reason=absence_reason,
        )

    # ------------------------------------------------------------------
    # Reporting helpers
    # ------------------------------------------------------------------
    def code_gap_titles(
        self,
        *,
        token_seed_details: Sequence[TokenSeedDetail],
        binary_bookmark: BinaryBookmark,
        excluded_unit_tokens: Sequence[Tuple[int, str, str, int]],
    ) -> Tuple[CodeGapTitle, ...]:
        """Return titled descriptions for uncovered numeric gaps."""

        zero_contributors: List[Tuple[int, str, str]] = []
        unit_contributors: List[Tuple[int, str, str, int]] = []
        for detail in token_seed_details:
            if detail.is_zero_token:
                zero_contributors.append((detail.position, detail.text, detail.code))
            if detail.is_unit_token:
                unit_contributors.append(
                    (detail.position, detail.text, detail.code, detail.unit_digit)
                )

        unit_lookup = {
            position: (text, code, digit)
            for position, text, code, digit in excluded_unit_tokens
        }

        records: List[CodeGapTitle] = []

        if zero_contributors:
            positions = tuple(position for position, _, _ in zero_contributors)
            tokens = tuple(
                f"{text} ({code})" for _position, text, code in zero_contributors
            )
            description = (
                "Tokens resolved to zero numeric values, leaving visible gaps "
                "within the interpretation sequence."
            )
            absent_count = len(binary_bookmark.absent_tokens)
            coverage_hint = (
                "Review phonetic rules for these tokens or mark them as "
                "intentional separators."
            )
            if absent_count:
                coverage_hint += (
                    f" Binary bookmark also lists {absent_count} absent token"
                    f"{'s' if absent_count != 1 else ''}."
                )
            records.append(
                CodeGapTitle(
                    identifier="gap-zero-contributors",
                    title_name="Zero Contribution Tokens",
                    positions=positions,
                    tokens=tokens,
                    description=description,
                    coverage_hint=coverage_hint,
                )
            )

        if unit_contributors:
            positions = tuple(position for position, _, _, _ in unit_contributors)
            tokens = []
            for position, text, code, digit in unit_contributors:
                lookup_text, lookup_code, lookup_digit = unit_lookup.get(
                    position, (text, code, digit)
                )
                digit_char = self._number_to_base36(lookup_digit)
                tokens.append(f"{lookup_text} ({lookup_code}) unit {digit_char}")
            description = (
                "Tokens contributed only base-36 unit digits and were held "
                "outside trimmed numeric mixing."
            )
            coverage_hint = (
                "Pair these tokens with neighbours or archive their unit "
                "values via excluded_unit_tokens for complete coverage."
            )
            records.append(
                CodeGapTitle(
                    identifier="gap-unit-contributors",
                    title_name="Unit Digit Tokens",
                    positions=positions,
                    tokens=tuple(tokens),
                    description=description,
                    coverage_hint=coverage_hint,
                )
            )

        if binary_bookmark.is_absent_cover:
            absence_reason = binary_bookmark.absence_reason or "unspecified condition"
            positions: List[int] = []
            tokens: List[str] = []
            for position, text in binary_bookmark.absent_tokens:
                lookup = unit_lookup.get(position)
                if lookup is not None:
                    _, lookup_code, _ = lookup
                else:
                    lookup_code = next(
                        (
                            detail.code
                            for detail in token_seed_details
                            if detail.position == position
                        ),
                        "",
                    )
                tokens.append(f"{text} ({lookup_code or 'n/a'})")
                positions.append(position)
            description = (
                "Binary bookmark collapsed into an absence cover, indicating "
                "numeric material is missing by design or pending definition."
            )
            coverage_hint = (
                f"Reason recorded: {absence_reason}. Add numeric equivalence "
                "groups or confirm that absence is deliberate."
            )
            records.append(
                CodeGapTitle(
                    identifier="gap-bookmark-absence",
                    title_name="Bookmark Absence Cover",
                    positions=tuple(positions),
                    tokens=tuple(tokens),
                    description=description,
                    coverage_hint=coverage_hint,
                )
            )

        return tuple(records)

    def _build_language_layer_groupings(
        self, tokens: Sequence[PhoneticToken]
    ) -> Tuple[LanguageLayerGrouping, ...]:
        """Aggregate token language layers for compatibility-focused browsing."""

        buckets: Dict[Tuple[str, str], List[Tuple[int, str]]] = {}
        for index, token in enumerate(tokens, start=1):
            for layer in token.layers:
                key = (layer.layer_name, layer.script_hint)
                buckets.setdefault(key, []).append((index, layer.value))

        groupings: List[LanguageLayerGrouping] = []
        for (layer_name, script_hint), entries in buckets.items():
            positions = tuple(position for position, _ in entries)
            values = tuple(value for _, value in entries)
            canonical_value = entries[0][1] if entries else ""
            groupings.append(
                LanguageLayerGrouping(
                    layer_name=layer_name,
                    script_hint=script_hint,
                    positions=positions,
                    values=values,
                    canonical_value=canonical_value,
                )
            )

        groupings.sort(key=lambda item: (item.layer_name, item.script_hint))
        return tuple(groupings)

    def _build_table_of_contents(
        self,
        *,
        tokens: Sequence[PhoneticToken],
        language_layers: Sequence[LanguageLayerGrouping],
        token_seed_details: Sequence[TokenSeedDetail],
        trailing_groups: Dict[str, Tuple[int, ...]],
        title_sound: TitleSoundSummary,
        numeric_groups: Sequence[NumericEquivalenceGroup],
        repeated_numeric_groups: Sequence[NumericEquivalenceGroup],
        repetition_clusters: Sequence[NumericRepetitionCluster],
        code_gap_titles: Sequence[CodeGapTitle],
        universal_groups: Sequence[UniversalSimilarityGroup],
        cursor_translations: Sequence[CursorTranslation],
        excluded_unit_tokens: Sequence[Tuple[int, str, str, int]],
        interpretation_sequence: Sequence[int],
        lengthened_sequence: Sequence[int],
        cubic_sequence: Sequence[int],
        repetition_interpretation_sequence: Sequence[int],
        repetition_lengthened_sequence: Sequence[int],
        repetition_cubic_sequence: Sequence[int],
        bookmark_input_order: Sequence[Tuple[int, str, str, int]],
        bookmark_numeric_order: Sequence[Tuple[int, str, str, int]],
        binary_bookmark: BinaryBookmark,
    ) -> Tuple[TableOfContentsMarker, ...]:
        """Create navigational markers for the human-readable report."""

        def span(count: int) -> Tuple[int, int]:
            return (1, count) if count > 0 else (0, 0)

        entries: List[TableOfContentsMarker] = []
        cursor_assigned = False

        def add_entry(
            identifier: str,
            title: str,
            start: int,
            end: int,
            description: str,
            *,
            prefer_cursor: bool = False,
        ) -> None:
            nonlocal cursor_assigned
            is_cursor = False
            if not cursor_assigned and (prefer_cursor or not entries):
                is_cursor = True
                cursor_assigned = True
            entries.append(
                TableOfContentsMarker(
                    identifier=identifier,
                    title=title,
                    start_index=start,
                    end_index=end,
                    description=description,
                    is_cursor=is_cursor,
                )
            )

        token_count = len(tokens)
        start, end = span(token_count)
        add_entry(
            "tokens",
            "Phonetic Tokens",
            start,
            end,
            f"{token_count} token{'s' if token_count != 1 else ''} recognised for cursor navigation.",
            prefer_cursor=True,
        )

        layer_count = len(language_layers)
        start, end = span(layer_count)
        add_entry(
            "language-layers",
            "Language Compatibility Layers",
            start,
            end,
            (
                f"{layer_count} layer{'s' if layer_count != 1 else ''} blending original, "
                "normalized, and ASCII renderings across scripts."
            ),
        )

        universal_count = len(universal_groups)
        start, end = span(universal_count)
        add_entry(
            "universal-similarity",
            "Universal Similarity Groups",
            start,
            end,
            (
                f"{universal_count} group{'s' if universal_count != 1 else ''}"
                " covering language-agnostic matches."
            ),
        )

        translation_count = len(cursor_translations)
        start, end = span(translation_count)
        add_entry(
            "cursor-translations",
            "Cursor Translator",
            start,
            end,
            (
                f"{translation_count} cursor translation {'entries' if translation_count != 1 else 'entry'}"
                " exposing ASCII fallbacks."
            ),
        )

        seed_count = len(token_seed_details)
        start, end = span(seed_count)
        add_entry(
            "token-seeds",
            "Token Seed Metrics",
            start,
            end,
            f"Slider-ready seed details for {seed_count} token{'s' if seed_count != 1 else ''}.",
        )

        group_count = len(trailing_groups)
        start, end = span(group_count)
        add_entry(
            "trailing-groups",
            "Trailing Rhyme Groups",
            start,
            end,
            f"{group_count} trailing rhyme grouping{'s' if group_count != 1 else ''} detected.",
        )

        add_entry(
            "title-sound",
            "Title Sound Summary",
            0,
            0,
            f"Combined code {title_sound.combined_code} with seed {title_sound.seed_value}.",
        )

        numeric_count = len(numeric_groups)
        start, end = span(numeric_count)
        add_entry(
            "numeric-groups",
            "Numeric Equivalence Groups",
            start,
            end,
            f"{numeric_count} equivalence group{'s' if numeric_count != 1 else ''} sharing trimmed values.",
        )

        repeated_count = len(repeated_numeric_groups)
        start, end = span(repeated_count)
        add_entry(
            "repeated-groups",
            "Repeated Numeric Groups",
            start,
            end,
            f"{repeated_count} group{'s' if repeated_count != 1 else ''} with multiple occurrences.",
        )

        cluster_count = len(repetition_clusters)
        start, end = span(cluster_count)
        add_entry(
            "repetition-clusters",
            "Numeric Repetition Clusters",
            start,
            end,
            f"{cluster_count} cluster{'s' if cluster_count != 1 else ''} retaining every repeated contribution.",
        )

        gap_count = len(code_gap_titles)
        start, end = span(gap_count)
        add_entry(
            "code-gap-titles",
            "Code Gap Titles",
            start,
            end,
            f"{gap_count} titled gap{'s' if gap_count != 1 else ''} recorded for numeric coverage.",
        )

        excluded_count = len(excluded_unit_tokens)
        start, end = span(excluded_count)
        add_entry(
            "excluded-units",
            "Excluded Unit Tokens",
            start,
            end,
            f"{excluded_count} unit token{'s' if excluded_count != 1 else ''} parked outside numeric mixing.",
        )

        interpretation_count = len(interpretation_sequence)
        start, end = span(interpretation_count)
        add_entry(
            "interpretation",
            "Interpretation Sequence",
            start,
            end,
            f"{interpretation_count} trimmed numeric step{'s' if interpretation_count != 1 else ''}.",
        )

        lengthened_count = len(lengthened_sequence)
        start, end = span(lengthened_count)
        add_entry(
            "lengthened",
            "Lengthened Interpretation Sequence",
            start,
            end,
            f"{lengthened_count} doubled-width numeric marker{'s' if lengthened_count != 1 else ''}.",
        )

        cubic_count = len(cubic_sequence)
        start, end = span(cubic_count)
        add_entry(
            "cubic",
            "Cubic Interpretation Sequence",
            start,
            end,
            f"{cubic_count} cubic transformation{'s' if cubic_count != 1 else ''} applied after unit exclusion.",
        )

        rep_interp_count = len(repetition_interpretation_sequence)
        start, end = span(rep_interp_count)
        add_entry(
            "repetition-interpretation",
            "Repetition Interpretation Sequence",
            start,
            end,
            f"{rep_interp_count} repeated trimmed contribution{'s' if rep_interp_count != 1 else ''}.",
        )

        rep_lengthened_count = len(repetition_lengthened_sequence)
        start, end = span(rep_lengthened_count)
        add_entry(
            "repetition-lengthened",
            "Repetition Lengthened Sequence",
            start,
            end,
            f"{rep_lengthened_count} repeated lengthened contribution{'s' if rep_lengthened_count != 1 else ''}.",
        )

        rep_cubic_count = len(repetition_cubic_sequence)
        start, end = span(rep_cubic_count)
        add_entry(
            "repetition-cubic",
            "Repetition Cubic Sequence",
            start,
            end,
            f"{rep_cubic_count} repeated cubic contribution{'s' if rep_cubic_count != 1 else ''}.",
        )

        bookmark_input_count = len(bookmark_input_order)
        start, end = span(bookmark_input_count)
        add_entry(
            "bookmark-input",
            "Bookmark Coverage (Input Order)",
            start,
            end,
            f"{bookmark_input_count} token{'s' if bookmark_input_count != 1 else ''} traced in capture order.",
        )

        bookmark_numeric_count = len(bookmark_numeric_order)
        start, end = span(bookmark_numeric_count)
        add_entry(
            "bookmark-numeric",
            "Bookmark Coverage (Numeric Order)",
            start,
            end,
            f"{bookmark_numeric_count} token{'s' if bookmark_numeric_count != 1 else ''} arranged by trimmed value.",
        )

        add_entry(
            "name-order-timeline",
            "Name Order Timeline",
            0,
            0,
            "Name-indexed timeline integrating title order independent of token flow.",
        )

        highlight_count = len(binary_bookmark.trailing_highlights)
        absent_count = len(binary_bookmark.absent_tokens)
        add_entry(
            "binary-bookmark",
            "Binary Constant Marker",
            0,
            0,
            (
                f"Marker {binary_bookmark.marker} with {highlight_count} highlight{'s' if highlight_count != 1 else ''} "
                f"and {absent_count} absent token{'s' if absent_count != 1 else ''}."
            ),
        )

        add_entry(
            "report-end",
            "End of Report",
            0,
            0,
            "Terminal marker for instant movement to the analysis tail.",
        )

        return tuple(entries)

    def _build_title_language_records(
        self,
        *,
        table_of_contents: Sequence[TableOfContentsMarker],
        tokens: Sequence[PhoneticToken],
        trailing_groups: Dict[str, Tuple[int, ...]],
        title_sound: TitleSoundSummary,
        numeric_groups: Sequence[NumericEquivalenceGroup],
        repeated_numeric_groups: Sequence[NumericEquivalenceGroup],
        repetition_clusters: Sequence[NumericRepetitionCluster],
        code_gap_titles: Sequence[CodeGapTitle],
        excluded_unit_tokens: Sequence[Tuple[int, str, str, int]],
        interpretation_sequence: Sequence[int],
        lengthened_sequence: Sequence[int],
        cubic_sequence: Sequence[int],
        repetition_interpretation_sequence: Sequence[int],
        repetition_lengthened_sequence: Sequence[int],
        repetition_cubic_sequence: Sequence[int],
        token_seed_details: Sequence[TokenSeedDetail],
        universal_groups: Sequence[UniversalSimilarityGroup],
        cursor_translations: Sequence[CursorTranslation],
        bookmark_input_order: Sequence[Tuple[int, str, str, int]],
        bookmark_numeric_order: Sequence[Tuple[int, str, str, int]],
        binary_bookmark: BinaryBookmark,
    ) -> Tuple[TitleLanguageRecord, ...]:
        """Translate code-centric constructs into language-titled records."""

        def max_or_zero(values: Sequence[int]) -> int:
            return max(values) if values else 0

        source_name_map = {
            "Phonetic Tokens": "PhoneticToken",
            "Universal Similarity Groups": "UniversalSimilarityGroup",
            "Cursor Translator": "CursorTranslation",
            "Token Seed Metrics": "TokenSeedDetail",
            "Trailing Rhyme Groups": "trailing_rhyme_groups",
            "Title Sound Summary": "TitleSoundSummary",
            "Numeric Equivalence Groups": "NumericEquivalenceGroup",
            "Repeated Numeric Groups": "NumericEquivalenceGroup",
            "Numeric Repetition Clusters": "NumericRepetitionCluster",
            "Code Gap Titles": "CodeGapTitle",
            "Excluded Unit Tokens": "excluded_unit_tokens",
            "Interpretation Sequence": "interpretation_sequence",
            "Lengthened Interpretation Sequence": "lengthened_interpretation_sequence",
            "Cubic Interpretation Sequence": "cubic_interpretation_sequence",
            "Repetition Interpretation Sequence": "repetition_interpretation_sequence",
            "Repetition Lengthened Sequence": "repetition_lengthened_sequence",
            "Repetition Cubic Sequence": "repetition_cubic_sequence",
            "Bookmark Coverage (Input Order)": "bookmark_input_order",
            "Bookmark Coverage (Numeric Order)": "bookmark_numeric_order",
            "Binary Constant Marker": "BinaryBookmark",
            "Name Order Timeline": "NameTimeMarker",
            "End of Report": "analysis_tail",
        }

        metrics_map: Dict[str, int] = {
            "Phonetic Tokens": len(tokens),
            "Universal Similarity Groups": len(universal_groups),
            "Cursor Translator": len(cursor_translations),
            "Token Seed Metrics": max_or_zero(
                [detail.seed_value for detail in token_seed_details]
            ),
            "Trailing Rhyme Groups": len(trailing_groups),
            "Title Sound Summary": title_sound.max_numeric_value,
            "Numeric Equivalence Groups": max_or_zero(
                [group.numeric_value for group in numeric_groups]
            ),
            "Repeated Numeric Groups": sum(
                group.occurrence_count for group in repeated_numeric_groups
            ),
            "Numeric Repetition Clusters": len(repetition_clusters),
            "Code Gap Titles": len(code_gap_titles),
            "Excluded Unit Tokens": len(excluded_unit_tokens),
            "Interpretation Sequence": max_or_zero(interpretation_sequence),
            "Lengthened Interpretation Sequence": max_or_zero(lengthened_sequence),
            "Cubic Interpretation Sequence": max_or_zero(cubic_sequence),
            "Repetition Interpretation Sequence": max_or_zero(
                repetition_interpretation_sequence
            ),
            "Repetition Lengthened Sequence": max_or_zero(
                repetition_lengthened_sequence
            ),
            "Repetition Cubic Sequence": max_or_zero(repetition_cubic_sequence),
            "Bookmark Coverage (Input Order)": len(bookmark_input_order),
            "Bookmark Coverage (Numeric Order)": len(bookmark_numeric_order),
            "Binary Constant Marker": binary_bookmark.constant_value,
            "Name Order Timeline": len(table_of_contents),
            "End of Report": 0,
        }

        def summarise(title: str) -> str:
            if title == "Phonetic Tokens":
                preview_tokens = ", ".join(
                    f"{token.text}/{token.code}" for token in tokens[:3]
                )
                if len(tokens) > 3:
                    preview_tokens += ", …"
                return (
                    "PhoneticToken is voiced as \"Phonetic Tokens\" with "
                    f"{len(tokens)} captured word{'s' if len(tokens) != 1 else ''}."
                    + (f" Preview: {preview_tokens}." if preview_tokens else "")
                )
            if title == "Universal Similarity Groups":
                return (
                    "UniversalSimilarityGroup speaks without language bias, "
                    f"forming {len(universal_groups)} universal cluster"
                    f"{'s' if len(universal_groups) != 1 else ''} that avoid alternation."
                )
            if title == "Cursor Translator":
                ascii_preview = ", ".join(
                    entry.ascii_equivalent for entry in cursor_translations[:3]
                )
                if len(cursor_translations) > 3:
                    ascii_preview += ", …"
                return (
                    "CursorTranslation is rendered as \"Cursor Translator\" with "
                    f"{len(cursor_translations)} single-pass ASCII fallback"
                    f"{'s' if len(cursor_translations) != 1 else ''}."
                    " Alternating translations remain intentionally absent."
                    + (f" Preview: {ascii_preview}." if ascii_preview else "")
                )
            if title == "Token Seed Metrics":
                if token_seed_details:
                    max_seed = max(detail.seed_value for detail in token_seed_details)
                else:
                    max_seed = 0
                return (
                    "TokenSeedDetail narrates slider seeds under \"Token Seed Metrics\" "
                    f"with peak blend {max_seed}."
                )
            if title == "Trailing Rhyme Groups":
                return (
                    "trailing_rhyme_groups becomes prose as \"Trailing Rhyme Groups\" "
                    f"holding {len(trailing_groups)} group"
                    f"{'s' if len(trailing_groups) != 1 else ''}."
                )
            if title == "Title Sound Summary":
                return (
                    "TitleSoundSummary echoes as \"Title Sound Summary\" with combined code "
                    f"{title_sound.combined_code} and seed {title_sound.seed_value}."
                )
            if title == "Numeric Equivalence Groups":
                if numeric_groups:
                    top_group = max(numeric_groups, key=lambda g: g.numeric_value)
                    detail = (
                        f" peak numeric value {top_group.numeric_value} from code {top_group.code}"
                    )
                else:
                    detail = " no numeric matches"
                return (
                    "NumericEquivalenceGroup speaks as \"Numeric Equivalence Groups\" with"
                    + detail
                    + "."
                )
            if title == "Repeated Numeric Groups":
                total_occurrences = sum(
                    group.occurrence_count for group in repeated_numeric_groups
                )
                return (
                    "NumericEquivalenceGroup repeats under \"Repeated Numeric Groups\" "
                    f"tracking {total_occurrences} occurrence"
                    f"{'s' if total_occurrences != 1 else ''}."
                )
            if title == "Numeric Repetition Clusters":
                return (
                    "NumericRepetitionCluster is written as \"Numeric Repetition Clusters\" "
                    f"with {len(repetition_clusters)} ordered bundle"
                    f"{'s' if len(repetition_clusters) != 1 else ''}."
                )
            if title == "Code Gap Titles":
                return (
                    "CodeGapTitle surfaces as \"Code Gap Titles\" "
                    f"highlighting {len(code_gap_titles)} numeric gap"
                    f"{'s' if len(code_gap_titles) != 1 else ''}."
                )
            if title == "Excluded Unit Tokens":
                return (
                    "excluded_unit_tokens is described as \"Excluded Unit Tokens\" "
                    f"listing {len(excluded_unit_tokens)} base-36 unit placeholder"
                    f"{'s' if len(excluded_unit_tokens) != 1 else ''}."
                )
            if title == "Interpretation Sequence":
                return (
                    "interpretation_sequence is retold as \"Interpretation Sequence\" "
                    f"with {len(interpretation_sequence)} trimmed step"
                    f"{'s' if len(interpretation_sequence) != 1 else ''}."
                )
            if title == "Lengthened Interpretation Sequence":
                return (
                    "lengthened_interpretation_sequence is phrased as \"Lengthened Interpretation Sequence\" "
                    f"spanning {len(lengthened_sequence)} doubled marker"
                    f"{'s' if len(lengthened_sequence) != 1 else ''}."
                )
            if title == "Cubic Interpretation Sequence":
                return (
                    "cubic_interpretation_sequence is cast as \"Cubic Interpretation Sequence\" "
                    f"with {len(cubic_sequence)} cubic measure"
                    f"{'s' if len(cubic_sequence) != 1 else ''}."
                )
            if title == "Repetition Interpretation Sequence":
                return (
                    "repetition_interpretation_sequence is narrated as \"Repetition Interpretation Sequence\" "
                    f"covering {len(repetition_interpretation_sequence)} repeated trimmed value"
                    f"{'s' if len(repetition_interpretation_sequence) != 1 else ''}."
                )
            if title == "Repetition Lengthened Sequence":
                return (
                    "repetition_lengthened_sequence is voiced as \"Repetition Lengthened Sequence\" "
                    f"with {len(repetition_lengthened_sequence)} repeated double marker"
                    f"{'s' if len(repetition_lengthened_sequence) != 1 else ''}."
                )
            if title == "Repetition Cubic Sequence":
                return (
                    "repetition_cubic_sequence is rendered as \"Repetition Cubic Sequence\" "
                    f"with {len(repetition_cubic_sequence)} repeated cubic contribution"
                    f"{'s' if len(repetition_cubic_sequence) != 1 else ''}."
                )
            if title == "Bookmark Coverage (Input Order)":
                return (
                    "bookmark_input_order is declared as \"Bookmark Coverage (Input Order)\" "
                    f"listing {len(bookmark_input_order)} capture trace"
                    f"{'s' if len(bookmark_input_order) != 1 else ''}."
                )
            if title == "Bookmark Coverage (Numeric Order)":
                return (
                    "bookmark_numeric_order is presented as \"Bookmark Coverage (Numeric Order)\" "
                    f"sorting {len(bookmark_numeric_order)} numeric trace"
                    f"{'s' if len(bookmark_numeric_order) != 1 else ''}."
                )
            if title == "Name Order Timeline":
                return (
                    "NameTimeMarker arranges \"Name Order Timeline\" so every titled section "
                    "receives a sequential slot independent from word order."
                )
            if title == "Binary Constant Marker":
                return (
                    "BinaryBookmark becomes \"Binary Constant Marker\" and holds constant "
                    f"value {binary_bookmark.constant_value} with marker {binary_bookmark.marker}."
                )
            if title == "End of Report":
                return (
                    "analysis_tail closes as \"End of Report\" marking the navigational finish."
                )
            return f"{title} is preserved as titled language without additional detail."

        records: List[TitleLanguageRecord] = []
        total_entries = len(table_of_contents)
        for index, entry in enumerate(table_of_contents):
            source_name = source_name_map.get(entry.title, entry.identifier)
            order_value = metrics_map.get(entry.title, total_entries - index)
            summary = summarise(entry.title)
            language_text = f"{source_name} is read as \"{entry.title}\""
            records.append(
                TitleLanguageRecord(
                    title_name=entry.title,
                    source_name=source_name,
                    language_text=language_text,
                    order_value=order_value,
                    summary=summary,
                )
            )

        return tuple(records)

    def _format_slider(self, ratio: float, width: int = 20) -> str:
        """Return an ASCII slider visualisation for ``ratio``."""

        clamped = max(0.0, min(1.0, ratio))
        filled = int(round(clamped * width))
        filled = max(0, min(width, filled))
        return "[{}{}]".format("#" * filled, "-" * (width - filled))

    def _build_name_time_markers(
        self,
        *,
        title_language_records: Sequence[TitleLanguageRecord],
        table_of_contents: Sequence[TableOfContentsMarker],
    ) -> Tuple[NameTimeMarker, ...]:
        """Return timeline markers ordered by title language priority."""

        if not title_language_records:
            return ()

        toc_lookup = {entry.title: entry for entry in table_of_contents}
        ordered_records = sorted(
            title_language_records,
            key=lambda record: (-record.order_value, record.title_name.lower()),
        )

        markers: List[NameTimeMarker] = []
        for index, record in enumerate(ordered_records, start=1):
            toc_entry = toc_lookup.get(record.title_name)
            if toc_entry:
                anchor_identifier = toc_entry.identifier
                start_index = toc_entry.start_index
                end_index = toc_entry.end_index
            else:
                anchor_identifier = record.source_name
                start_index = 0
                end_index = 0

            markers.append(
                NameTimeMarker(
                    title_name=record.title_name,
                    source_name=record.source_name,
                    time_index=index,
                    timestamp_label=f"T{index:03d}",
                    order_value=record.order_value,
                    anchor_identifier=anchor_identifier,
                    start_index=start_index,
                    end_index=end_index,
                    summary=record.summary,
                )
            )

        return tuple(markers)

    def analyze(self, text: str, *, title: str | None = None) -> AnalysisResult:
        """Return a structured :class:`AnalysisResult` for ``text``."""

        tokens = tuple(self.phoneticize(text))
        lines = [line.strip() for line in text.strip().splitlines() if line.strip()]
        trailing_groups_raw = self.group_by_trailing_rhyme(lines)
        trailing_groups = {
            code: tuple(index + 1 for index in indices)
            for code, indices in trailing_groups_raw.items()
        }

        title_sound = self.title_sound_summary(text, tokens)
        numeric_groups_list, _zero_tokens, excluded_units = self._collect_numeric_details(tokens)
        numeric_groups = tuple(numeric_groups_list)
        repeated_numeric_groups = tuple(self.repeated_numeric_groups(tokens))
        repetition_clusters = tuple(self.numeric_repetition_clusters(tokens))
        token_seed_details = tuple(self.token_seed_details(tokens))
        similarity_components = tuple(self.token_similarity_signatures(tokens))
        universal_groups = tuple(
            self.universal_similarity_groups(tokens, similarity_components)
        )
        cursor_translations = tuple(
            self.cursor_translations(tokens, similarity_components)
        )
        interpretation = tuple(self.interpretation_sequence(tokens))
        lengthened_interpretation = tuple(
            self.lengthened_interpretation_sequence(tokens, numeric_groups)
        )
        cubic_interpretation = tuple(
            self.cubic_interpretation_sequence(tokens, numeric_groups)
        )
        repetition_sequence = tuple(self.repetition_interpretation_sequence(tokens))
        repetition_lengthened = tuple(self.repetition_lengthened_sequence(tokens))
        repetition_cubic = tuple(self.repetition_cubic_sequence(tokens))
        binary_bookmark = self.binary_constant_marker(text, tokens)
        code_gap_titles = tuple(
            self.code_gap_titles(
                token_seed_details=token_seed_details,
                binary_bookmark=binary_bookmark,
                excluded_unit_tokens=tuple(excluded_units),
            )
        )

        bookmark_input_order = tuple(
            (
                position + 1,
                token.text,
                token.code,
                self._trim_unit_digit(self._code_to_number(token.code)),
            )
            for position, token in enumerate(tokens)
        )
        bookmark_numeric_order = tuple(
            sorted(bookmark_input_order, key=lambda item: (item[3], item[0]))
        )

        language_layers = self._build_language_layer_groupings(tokens)

        table_of_contents = self._build_table_of_contents(
            tokens=tokens,
            language_layers=language_layers,
            token_seed_details=token_seed_details,
            trailing_groups=trailing_groups,
            title_sound=title_sound,
            numeric_groups=numeric_groups,
            repeated_numeric_groups=repeated_numeric_groups,
            repetition_clusters=repetition_clusters,
            code_gap_titles=code_gap_titles,
            universal_groups=universal_groups,
            cursor_translations=cursor_translations,
            excluded_unit_tokens=tuple(excluded_units),
            interpretation_sequence=interpretation,
            lengthened_sequence=lengthened_interpretation,
            cubic_sequence=cubic_interpretation,
            repetition_interpretation_sequence=repetition_sequence,
            repetition_lengthened_sequence=repetition_lengthened,
            repetition_cubic_sequence=repetition_cubic,
            bookmark_input_order=bookmark_input_order,
            bookmark_numeric_order=bookmark_numeric_order,
            binary_bookmark=binary_bookmark,
        )

        title_language_records = self._build_title_language_records(
            table_of_contents=table_of_contents,
            tokens=tokens,
            trailing_groups=trailing_groups,
            title_sound=title_sound,
            numeric_groups=numeric_groups,
            repeated_numeric_groups=repeated_numeric_groups,
            repetition_clusters=repetition_clusters,
            code_gap_titles=code_gap_titles,
            excluded_unit_tokens=tuple(excluded_units),
            interpretation_sequence=interpretation,
            lengthened_sequence=lengthened_interpretation,
            cubic_sequence=cubic_interpretation,
            repetition_interpretation_sequence=repetition_sequence,
            repetition_lengthened_sequence=repetition_lengthened,
            repetition_cubic_sequence=repetition_cubic,
            token_seed_details=token_seed_details,
            universal_groups=universal_groups,
            cursor_translations=cursor_translations,
            bookmark_input_order=bookmark_input_order,
            bookmark_numeric_order=bookmark_numeric_order,
            binary_bookmark=binary_bookmark,
        )

        descending_order = tuple(
            record.title_name
            for record in sorted(
                title_language_records, key=lambda item: item.order_value, reverse=True
            )
        )

        name_time_markers = self._build_name_time_markers(
            title_language_records=title_language_records,
            table_of_contents=table_of_contents,
        )

        return AnalysisResult(
            title=title,
            raw_text=text,
            tokens=tokens,
            trailing_rhyme_groups=trailing_groups,
            title_sound=title_sound,
            numeric_groups=numeric_groups,
            repeated_numeric_groups=repeated_numeric_groups,
            numeric_repetition_clusters=repetition_clusters,
            token_seed_details=token_seed_details,
            interpretation_sequence=interpretation,
            lengthened_interpretation_sequence=lengthened_interpretation,
            cubic_interpretation_sequence=cubic_interpretation,
            repetition_interpretation_sequence=repetition_sequence,
            repetition_lengthened_sequence=repetition_lengthened,
            repetition_cubic_sequence=repetition_cubic,
            binary_bookmark=binary_bookmark,
            bookmark_input_order=bookmark_input_order,
            bookmark_numeric_order=bookmark_numeric_order,
            excluded_unit_tokens=tuple(excluded_units),
            code_gap_titles=code_gap_titles,
            language_layer_groupings=language_layers,
            table_of_contents=table_of_contents,
            universal_similarity_groups=universal_groups,
            cursor_translations=cursor_translations,
            title_language_records=title_language_records,
            descending_title_language_order=descending_order,
            name_time_markers=name_time_markers,
        )

    def _signature_lookup(self, analysis: AnalysisResult) -> Dict[int, str]:
        """Return a mapping of token position to universal signature."""

        lookup: Dict[int, str] = {}
        for entry in analysis.cursor_translations:
            lookup[entry.position] = entry.signature
        return lookup

    def compare_analyses(
        self,
        current: AnalysisResult,
        reference: AnalysisResult,
        *,
        window_size: int = 5,
    ) -> ComparabilityResult:
        """Compare two analyses and return windowed similarity metrics."""

        if window_size <= 0:
            window_size = 1

        current_details = current.token_seed_details
        reference_details = reference.token_seed_details
        signature_current = self._signature_lookup(current)
        signature_reference = self._signature_lookup(reference)

        limit = max(len(current_details), len(reference_details))
        windows: List[ComparabilityWindow] = []

        for index, start in enumerate(range(0, limit, window_size), start=1):
            current_slice = current_details[start : start + window_size]
            reference_slice = reference_details[start : start + window_size]

            current_range = (
                current_slice[0].position,
                current_slice[-1].position,
            ) if current_slice else (0, 0)
            reference_range = (
                reference_slice[0].position,
                reference_slice[-1].position,
            ) if reference_slice else (0, 0)

            current_tokens = tuple(detail.text for detail in current_slice)
            reference_tokens = tuple(detail.text for detail in reference_slice)

            numeric_current = {
                detail.trimmed_numeric_value
                for detail in current_slice
                if detail.trimmed_numeric_value > 0
            }
            numeric_reference = {
                detail.trimmed_numeric_value
                for detail in reference_slice
                if detail.trimmed_numeric_value > 0
            }

            numeric_union = sorted(numeric_current | numeric_reference)
            numeric_overlap = sorted(numeric_current & numeric_reference)
            if numeric_union:
                numeric_similarity = len(numeric_overlap) / len(numeric_union)
            else:
                numeric_similarity = 1.0

            universal_current = {
                signature_current.get(detail.position)
                for detail in current_slice
                if signature_current.get(detail.position)
            }
            universal_reference = {
                signature_reference.get(detail.position)
                for detail in reference_slice
                if signature_reference.get(detail.position)
            }

            universal_union = sorted(universal_current | universal_reference)
            universal_overlap = sorted(universal_current & universal_reference)
            if universal_union:
                universal_similarity = len(universal_overlap) / len(universal_union)
            else:
                universal_similarity = 1.0

            if not numeric_union and not universal_union:
                commentary = "No shared numeric or similarity signals in this window."
            elif numeric_similarity == 1.0 and universal_similarity == 1.0:
                commentary = "Complete alignment across numeric and similarity signals."
            else:
                commentary = (
                    "Shared {num_overlap}/{num_total} numeric values and "
                    "{sim_overlap}/{sim_total} similarity signatures."
                ).format(
                    num_overlap=len(numeric_overlap),
                    num_total=len(numeric_union),
                    sim_overlap=len(universal_overlap),
                    sim_total=len(universal_union),
                )

            windows.append(
                ComparabilityWindow(
                    window_index=index,
                    current_range=current_range,
                    reference_range=reference_range,
                    current_tokens=current_tokens,
                    reference_tokens=reference_tokens,
                    numeric_overlap=tuple(numeric_overlap),
                    numeric_union=tuple(numeric_union),
                    universal_overlap=tuple(universal_overlap),
                    universal_union=tuple(universal_union),
                    numeric_similarity=numeric_similarity,
                    universal_similarity=universal_similarity,
                    commentary=commentary,
                )
            )

        if windows:
            numeric_mean = sum(window.numeric_similarity for window in windows) / len(windows)
            universal_mean = (
                sum(window.universal_similarity for window in windows) / len(windows)
            )
        else:
            numeric_mean = 1.0
            universal_mean = 1.0

        return ComparabilityResult(
            window_size=window_size,
            windows=tuple(windows),
            aggregate_numeric_similarity=numeric_mean,
            aggregate_universal_similarity=universal_mean,
        )

    def compare_texts(
        self,
        current_text: str,
        reference_text: str,
        *,
        current_title: str | None = None,
        reference_title: str | None = None,
        window_size: int = 5,
    ) -> ComparabilityResult:
        """Convenience wrapper returning comparability between two texts."""

        current = self.analyze(current_text, title=current_title)
        reference = self.analyze(reference_text, title=reference_title)
        return self.compare_analyses(
            current,
            reference,
            window_size=window_size,
        )

    def format_comparability_report(
        self,
        comparability: ComparabilityResult,
        *,
        heading: str = "Comparability windows",
    ) -> str:
        """Return a textual representation of ``comparability`` suitable for CLI output."""

        builder: List[str] = []
        builder.append(f"{heading}:")
        builder.append(f"  window size: {comparability.window_size}")
        builder.append(
            "  aggregate numeric similarity: {value:.4f}".format(
                value=comparability.aggregate_numeric_similarity
            )
        )
        builder.append(
            "  aggregate similarity signature match: {value:.4f}".format(
                value=comparability.aggregate_universal_similarity
            )
        )

        if not comparability.windows:
            builder.append("  <no windows>")
            return "\n".join(builder)

        def _format_range(token_range: Tuple[int, int]) -> str:
            start, end = token_range
            if start == 0 and end == 0:
                return "-"
            if start == end:
                return str(start)
            return f"{start}-{end}"

        for window in comparability.windows:
            builder.append(
                "- window {index}: current {current_range} vs reference {reference_range}".format(
                    index=window.window_index,
                    current_range=_format_range(window.current_range),
                    reference_range=_format_range(window.reference_range),
                )
            )
            builder.append(
                "    current tokens: {tokens}".format(
                    tokens=", ".join(window.current_tokens) if window.current_tokens else "<none>"
                )
            )
            builder.append(
                "    reference tokens: {tokens}".format(
                    tokens=", ".join(window.reference_tokens)
                    if window.reference_tokens
                    else "<none>"
                )
            )
            builder.append(
                "    numeric overlap: {overlap}".format(
                    overlap=", ".join(str(value) for value in window.numeric_overlap)
                    if window.numeric_overlap
                    else "<none>"
                )
            )
            builder.append(
                "    numeric union: {union}".format(
                    union=", ".join(str(value) for value in window.numeric_union)
                    if window.numeric_union
                    else "<none>"
                )
            )
            builder.append(
                "    signature overlap: {overlap}".format(
                    overlap=", ".join(window.universal_overlap)
                    if window.universal_overlap
                    else "<none>"
                )
            )
            builder.append(
                "    signature union: {union}".format(
                    union=", ".join(window.universal_union)
                    if window.universal_union
                    else "<none>"
                )
            )
            builder.append(
                "    numeric similarity: {value:.4f}".format(
                    value=window.numeric_similarity
                )
            )
            builder.append(
                "    signature similarity: {value:.4f}".format(
                    value=window.universal_similarity
                )
            )
            builder.append(f"    note: {window.commentary}")

        return "\n".join(builder)

    def prepare_data_generation(
        self,
        text: str,
        *,
        analysis: AnalysisResult | None = None,
        reference_text: str | None = None,
        reference_analysis: AnalysisResult | None = None,
        title: str | None = None,
        reference_title: str | None = None,
        window_size: int = 5,
        include_reports: bool = True,
        comparability: ComparabilityResult | None = None,
    ) -> DataGenerationBundle:
        """Bundle analysis outputs and prompts for downstream data generation."""

        if analysis is None:
            analysis = self.analyze(text, title=title)
        elif title is not None and analysis.title != title:
            analysis = replace(analysis, title=title)

        analysis_report = (
            self.generate_report(text, analysis=analysis) if include_reports else None
        )

        if reference_text is None and reference_analysis is not None:
            reference_text = reference_analysis.raw_text

        if reference_analysis is None and reference_text is not None:
            reference_analysis = self.analyze(reference_text, title=reference_title)
        elif (
            reference_title is not None
            and reference_analysis is not None
            and reference_analysis.title != reference_title
        ):
            reference_analysis = replace(reference_analysis, title=reference_title)

        reference_report = None
        if (
            include_reports
            and reference_text is not None
            and reference_analysis is not None
        ):
            reference_report = self.generate_report(
                reference_text,
                analysis=reference_analysis,
            )

        if reference_analysis is not None and comparability is None:
            comparability = self.compare_analyses(
                analysis,
                reference_analysis,
                window_size=window_size,
            )

        comparability_report = None
        if include_reports and comparability is not None:
            comparability_report = self.format_comparability_report(
                comparability,
                heading="Comparability report",
            )

        analysis_payload = analysis.as_dict()
        analysis_json = json.dumps(
            analysis_payload,
            ensure_ascii=False,
            sort_keys=True,
            indent=2,
        )

        examples: List[DataGenerationExample] = []
        text_block = text.strip()
        if not text_block:
            text_block = "<empty>"
        analysis_prompt = textwrap.dedent(
            f"""
            Analyse the supplied text and emit the phonetic trailing song analysis JSON.
            Provide the exact JSON structure with canonical key ordering.
            <BEGIN_TEXT>
            {text_block}
            <END_TEXT>
            """
        ).strip()
        examples.append(
            DataGenerationExample(
                prompt=analysis_prompt,
                completion=analysis_json,
                metadata={
                    "example_type": "analysis-json",
                    "keys": sorted(analysis_payload.keys()),
                    "has_reference": reference_analysis is not None,
                },
            )
        )

        reference_json: str | None = None
        if reference_analysis is not None:
            reference_json = json.dumps(
                reference_analysis.as_dict(),
                ensure_ascii=False,
                sort_keys=True,
                indent=2,
            )

        if include_reports and analysis_report is not None:
            report_prompt = textwrap.dedent(
                f"""
                Convert the phonetic trailing song analysis JSON into the canonical
                human-readable report used by the CLI.
                <BEGIN_JSON>
                {analysis_json}
                <END_JSON>
                """
            ).strip()
            examples.append(
                DataGenerationExample(
                    prompt=report_prompt,
                    completion=analysis_report,
                    metadata={
                        "example_type": "analysis-report",
                        "has_reference": reference_analysis is not None,
                    },
                )
            )

        if include_reports and reference_report is not None and reference_json is not None:
            reference_report_prompt = textwrap.dedent(
                f"""
                Convert the reference analysis JSON into the canonical
                human-readable report used by the CLI.
                <BEGIN_JSON>
                {reference_json}
                <END_JSON>
                """
            ).strip()
            examples.append(
                DataGenerationExample(
                    prompt=reference_report_prompt,
                    completion=reference_report,
                    metadata={
                        "example_type": "reference-report",
                        "window_size": window_size,
                    },
                )
            )

        if reference_json is not None and comparability is not None:
            comparability_json = json.dumps(
                comparability.as_dict(),
                ensure_ascii=False,
                sort_keys=True,
                indent=2,
            )
            comparability_prompt = textwrap.dedent(
                f"""
                Using the current and reference phonetic analyses, compute the
                comparability metrics with window size {window_size}.
                <BEGIN_CURRENT_ANALYSIS>
                {analysis_json}
                <END_CURRENT_ANALYSIS>
                <BEGIN_REFERENCE_ANALYSIS>
                {reference_json}
                <END_REFERENCE_ANALYSIS>
                """
            ).strip()
            examples.append(
                DataGenerationExample(
                    prompt=comparability_prompt,
                    completion=comparability_json,
                    metadata={
                        "example_type": "comparability-json",
                        "window_size": window_size,
                    },
                )
            )

            if include_reports and comparability_report is not None:
                comparability_report_prompt = textwrap.dedent(
                    f"""
                    Produce the narrative comparability report from the supplied
                    comparability JSON payload.
                    <BEGIN_JSON>
                    {comparability_json}
                    <END_JSON>
                    """
                ).strip()
                examples.append(
                    DataGenerationExample(
                        prompt=comparability_report_prompt,
                        completion=comparability_report,
                        metadata={
                            "example_type": "comparability-report",
                            "window_size": window_size,
                        },
                    )
                )

        return DataGenerationBundle(
            analysis=analysis,
            reference_analysis=reference_analysis,
            comparability=comparability,
            analysis_report=analysis_report,
            reference_report=reference_report,
            comparability_report=comparability_report,
            examples=tuple(examples),
            analysis_title=analysis.title,
            reference_title=(
                reference_analysis.title if reference_analysis is not None else reference_title
            ),
        )

    def generate_report(
        self, text: str, *, analysis: AnalysisResult | None = None
    ) -> str:
        """Return a human-readable summary of tokens and trailing rhymes."""

        if analysis is None:
            analysis = self.analyze(text)
        tokens = analysis.tokens
        groups = analysis.trailing_rhyme_groups
        title_sound = analysis.title_sound
        numeric_groups = analysis.numeric_groups
        repeated_numeric_groups = analysis.repeated_numeric_groups
        repetition_clusters = analysis.numeric_repetition_clusters
        token_seed_details = analysis.token_seed_details
        universal_groups = analysis.universal_similarity_groups
        cursor_translations = analysis.cursor_translations
        language_layers = analysis.language_layer_groupings
        interpretation = analysis.interpretation_sequence
        lengthened = analysis.lengthened_interpretation_sequence
        cubic = analysis.cubic_interpretation_sequence
        repetition_sequence = analysis.repetition_interpretation_sequence
        repetition_lengthened = analysis.repetition_lengthened_sequence
        repetition_cubic = analysis.repetition_cubic_sequence
        binary_marker = analysis.binary_bookmark
        bookmark_orders = analysis.bookmark_input_order
        sorted_orders = analysis.bookmark_numeric_order
        excluded_units = analysis.excluded_unit_tokens
        code_gap_titles = analysis.code_gap_titles
        table_of_contents = analysis.table_of_contents
        title_language_records = analysis.title_language_records
        descending_titles = analysis.descending_title_language_order
        name_time_markers = analysis.name_time_markers

        builder: List[str] = []
        if analysis.title:
            builder.append(f"Analysis title: {analysis.title}")
            builder.append("")
        builder.append("Table of contents:")
        for entry in table_of_contents:
            if entry.start_index == 0 and entry.end_index == 0:
                span = "-"
            elif entry.start_index == entry.end_index:
                span = str(entry.start_index)
            else:
                span = f"{entry.start_index}..{entry.end_index}"
            cursor = ">>" if entry.is_cursor else "  "
            builder.append(
                f"{cursor} {entry.identifier}: {entry.title} [{span}] - {entry.description}"
            )

        builder.append("\nTitle language order (descending):")
        if descending_titles:
            builder.append("  " + " > ".join(descending_titles))
        else:
            builder.append("  <none>")

        builder.append("\nName order timeline:")
        if name_time_markers:
            for marker in name_time_markers:
                if marker.start_index == 0 and marker.end_index == 0:
                    span = "-"
                elif marker.start_index == marker.end_index:
                    span = str(marker.start_index)
                else:
                    span = f"{marker.start_index}..{marker.end_index}"
                builder.append(
                    "  - {label} {title} (source {source}) -> anchor {anchor} [{span}] order {order}".format(
                        label=marker.timestamp_label,
                        title=marker.title_name,
                        source=marker.source_name,
                        anchor=marker.anchor_identifier,
                        span=span,
                        order=marker.order_value,
                    )
                )
                builder.append(f"      summary: {marker.summary}")
        else:
            builder.append("  - <none>")

        builder.append("\nTitle language records:")
        if title_language_records:
            for record in title_language_records:
                builder.append(
                    "  - {source} => {title} (order {order}): {summary}".format(
                        source=record.source_name,
                        title=record.title_name,
                        order=record.order_value,
                        summary=record.summary,
                    )
                )
        else:
            builder.append("  - <none>")

        builder.append("\nPhonetic tokens:")
        for token in tokens:
            builder.append(f"  - {token.text} -> {token.code}")

        builder.append("\nLanguage compatibility layers:")
        if language_layers:
            for group in language_layers:
                position_list = ", ".join(str(pos) for pos in group.positions)
                sample_values = ", ".join(group.values[:4])
                if len(group.values) > 4:
                    sample_values += ", …"
                builder.append(
                    "  - {name} [{script}] canonical '{canonical}' :: positions {positions}".format(
                        name=group.layer_name,
                        script=group.script_hint,
                        canonical=group.canonical_value,
                        positions=position_list or "<none>",
                    )
                )
                builder.append(f"      values: {sample_values if sample_values else '<none>'}")
        else:
            builder.append("  - <none>")

        builder.append("\nUniversal similarity groups:")
        if universal_groups:
            for group in universal_groups:
                members = ", ".join(
                    f"{position}:{text} ({code}) ~ {ascii_equivalent}"
                    for position, text, code, ascii_equivalent in group.tokens
                )
                builder.append(
                    "  - signature {signature} [{hint}]: {members}".format(
                        signature=group.signature,
                        hint=group.script_hint,
                        members=members,
                    )
                )
        else:
            builder.append("  - <none>")

        builder.append("\nCursor translator entries:")
        if cursor_translations:
            for entry in cursor_translations:
                builder.append(
                    "  - token {pos}: {text} -> {ascii} [{hint}]".format(
                        pos=entry.position,
                        text=entry.text,
                        ascii=entry.ascii_equivalent,
                        hint=entry.script_hint,
                    )
                )
                builder.append(
                    "      signature: {signature}".format(
                        signature=entry.signature,
                    )
                )
                builder.append(f"      note: {entry.explanation}")
        else:
            builder.append("  - <none>")

        builder.append("\nToken seed metrics:")
        if token_seed_details:
            for detail in token_seed_details:
                slider = self._format_slider(detail.slider_ratio)
                builder.append(
                    f"  - token {detail.position}: {detail.text} ({detail.code})"
                )
                builder.append(
                    "      max: {max_val} trimmed: {trimmed} lengthened: {lengthened} "
                    "cubic: {cubic}".format(
                        max_val=detail.max_numeric_value,
                        trimmed=detail.trimmed_numeric_value,
                        lengthened=detail.lengthened_numeric_value,
                        cubic=detail.cubic_numeric_value,
                    )
                )
                builder.append(
                    "      seed: {seed} max-seed: {max_seed} slider: {slider} "
                    "({ratio:.4f})".format(
                        seed=detail.seed_value,
                        max_seed=detail.max_seed_value,
                        slider=slider,
                        ratio=detail.slider_ratio,
                    )
                )
                if detail.is_zero_token:
                    builder.append("      note: zero token (no numeric contribution)")
                elif detail.is_unit_token:
                    digit = self._number_to_base36(detail.unit_digit)
                    builder.append(
                        f"      note: unit token (digit {digit})"
                    )
        else:
            builder.append("  - <none>")

        builder.append("\nTrailing rhyme groups:")
        for code, indices in sorted(groups.items(), key=lambda item: item[0]):
            line_list = ", ".join(str(i) for i in indices)
            builder.append(f"  - {code}: lines {line_list}")

        builder.append("\nTitle sound summary:")
        builder.append(f"  - fragment: {title_sound.title_sound}")
        builder.append(f"  - combined code: {title_sound.combined_code}")
        builder.append(f"  - max numeric value: {title_sound.max_numeric_value}")
        builder.append(f"  - seed value: {title_sound.seed_value}")

        builder.append("\nNumeric equivalence groups:")
        if numeric_groups:
            for group in numeric_groups:
                words = ", ".join(group.words)
                positions = ", ".join(str(pos + 1) for pos in group.positions)
                builder.append(
                    "  - value {numeric} ({code}): words [{words}] at tokens {positions}".format(
                        numeric=group.numeric_value,
                        code=group.code,
                        words=words,
                        positions=positions,
                    )
                )
        else:
            builder.append("  - <no tokens>")

        builder.append("\nRepeated numeric groups:")
        if repeated_numeric_groups:
            for group in repeated_numeric_groups:
                words = ", ".join(group.words)
                positions = ", ".join(str(pos + 1) for pos in group.positions)
                builder.append(
                    "  - value {numeric} ({code}) repeated {count}x -> tokens {positions} [{words}]".format(
                        numeric=group.numeric_value,
                        code=group.code,
                        count=group.occurrence_count,
                        positions=positions,
                        words=words,
                    )
                )
        else:
            builder.append("  - <none>")

        builder.append("\nNumeric repetition clusters:")
        if repetition_clusters:
            for cluster in repetition_clusters:
                members = ", ".join(
                    f"{position}:{text} ({code})" for position, text, code in cluster.members
                )
                builder.append(
                    "  - value {numeric}: {members}".format(
                        numeric=cluster.numeric_value,
                        members=members,
                    )
                )
        else:
            builder.append("  - <none>")

        builder.append("\nCode gap titles:")
        if code_gap_titles:
            for record in code_gap_titles:
                builder.append(
                    f"  - {record.title_name} [{record.identifier}]"
                )
                if record.positions:
                    positions = ", ".join(str(pos) for pos in record.positions)
                    builder.append(f"      positions: {positions}")
                else:
                    builder.append("      positions: <none>")
                if record.tokens:
                    token_list = ", ".join(record.tokens)
                    builder.append(f"      tokens: {token_list}")
                else:
                    builder.append("      tokens: <none>")
                builder.append(f"      description: {record.description}")
                builder.append(f"      coverage: {record.coverage_hint}")
        else:
            builder.append("  - <none>")

        if excluded_units:
            builder.append("\nExcluded unit tokens:")
            for position, word, code, numeric in excluded_units:
                digit = self._number_to_base36(numeric)
                builder.append(
                    f"  - token {position}: {word} ({code}) -> {numeric} (unit {digit})"
                )
        else:
            builder.append("\nExcluded unit tokens: <none>")

        builder.append("\nInterpretation sequence:")
        if interpretation:
            builder.append(
                "  - " + " -> ".join(str(value) for value in interpretation)
            )
        else:
            builder.append("  - <empty>")

        builder.append("\nLengthened interpretation sequence:")
        if lengthened:
            builder.append("  - " + " -> ".join(str(value) for value in lengthened))
        else:
            builder.append("  - <empty>")

        builder.append("\nCubic interpretation sequence:")
        if cubic:
            builder.append("  - " + " -> ".join(str(value) for value in cubic))
        else:
            builder.append("  - <empty>")

        builder.append("\nRepetition interpretation sequence:")
        if repetition_sequence:
            builder.append(
                "  - " + " -> ".join(str(value) for value in repetition_sequence)
            )
        else:
            builder.append("  - <empty>")

        builder.append("\nRepetition lengthened sequence:")
        if repetition_lengthened:
            builder.append(
                "  - " + " -> ".join(str(value) for value in repetition_lengthened)
            )
        else:
            builder.append("  - <empty>")

        builder.append("\nRepetition cubic sequence:")
        if repetition_cubic:
            builder.append(
                "  - " + " -> ".join(str(value) for value in repetition_cubic)
            )
        else:
            builder.append("  - <empty>")

        builder.append("\nBinary constant marker:")
        builder.append(f"  - marker: {binary_marker.marker}")
        builder.append(f"  - codex key: {binary_marker.codex_key}")
        builder.append(f"  - constant value: {binary_marker.constant_value}")
        if binary_marker.is_absent_cover:
            reason = binary_marker.absence_reason or "unspecified"
            builder.append(f"  - absence cover: yes ({reason})")
        else:
            builder.append("  - absence cover: no")
        if binary_marker.trailing_highlights:
            builder.append("  - trailing highlights:")
            for line_number, code in binary_marker.trailing_highlights:
                builder.append(f"      line {line_number}: {code}")
        else:
            builder.append("  - trailing highlights: <none>")
        if binary_marker.absent_tokens:
            builder.append("  - absent tokens:")
            for position, word in binary_marker.absent_tokens:
                builder.append(f"      token {position}: {word}")
        else:
            builder.append("  - absent tokens: <none>")
        if binary_marker.excluded_unit_tokens:
            builder.append("  - excluded unit tokens:")
            for position, word, code, numeric in binary_marker.excluded_unit_tokens:
                digit = self._number_to_base36(numeric)
                builder.append(
                    f"      token {position}: {word} ({code}) -> {numeric} (unit {digit})"
                )
        else:
            builder.append("  - excluded unit tokens: <none>")

        if bookmark_orders:
            builder.append("\nBookmark coverage (input order):")
            for position, word, code, value in bookmark_orders:
                builder.append(
                    f"  - token {position}: {word} ({code}) -> {value}"
                )

            builder.append("\nBookmark coverage (numeric order):")
            for position, word, code, value in sorted_orders:
                builder.append(
                    f"  - token {position}: {word} ({code}) -> {value}"
                )

        else:
            builder.append("\nBookmark coverage: <no tokens>")

        return "\n".join(builder)


def _cli(argv: Sequence[str]) -> int:
    import argparse
    import sys

    parser = argparse.ArgumentParser(
        description=(
            "Analyse text for phonetic signatures and trailing rhyme groupings."
        )
    )
    parser.add_argument(
        "input",
        nargs="?",
        help="Optional path to a UTF-8 text file. If omitted, stdin is used.",
    )
    parser.add_argument(
        "--seed-from",
        dest="seed_from",
        help=(
            "Optional name or identifier to convert into a deterministic seed "
            "using the title sound summary."
        ),
    )
    parser.add_argument(
        "--title",
        dest="title",
        help="Optional title to associate with the primary analysis output.",
    )
    parser.add_argument(
        "--select-token",
        dest="select_token",
        type=int,
        help=(
            "Optional 1-based token index to spotlight in the slider metrics "
            "section."
        ),
    )
    parser.add_argument(
        "--compare-with",
        dest="compare_with",
        help=(
            "Optional path to a reference text whose analysis should be "
            "compared against the primary input."
        ),
    )
    parser.add_argument(
        "--reference-title",
        dest="reference_title",
        help="Optional title describing the reference analysis output.",
    )
    parser.add_argument(
        "--compare-window-size",
        dest="compare_window_size",
        type=int,
        default=5,
        help=(
            "Window size to use when computing comparability metrics (default: 5)."
        ),
    )
    parser.add_argument(
        "--no-report",
        dest="no_report",
        action="store_true",
        help=(
            "Skip printing the human-readable report. Useful when collecting JSON "
            "outputs for downstream data generation."
        ),
    )
    parser.add_argument(
        "--emit-json",
        dest="emit_json",
        help=(
            "Optional path where the analysis JSON should be written. Use '-' to "
            "emit the JSON to stdout."
        ),
    )
    parser.add_argument(
        "--emit-data-bundle",
        dest="emit_data_bundle",
        help=(
            "Optional path where the data generation bundle JSON should be written. "
            "Use '-' to emit the bundle to stdout."
        ),
    )
    args = parser.parse_args(argv[1:])

    if args.input:
        with open(args.input, "r", encoding="utf-8") as handle:
            contents = handle.read()
    else:
        contents = sys.stdin.read()

    model = PhoneticTrailingSongModel()
    analysis = model.analyze(contents, title=args.title)
    report = model.generate_report(contents, analysis=analysis)
    reference_text: str | None = None
    reference_analysis: AnalysisResult | None = None
    comparability: ComparabilityResult | None = None
    comparison_report: str | None = None

    if args.compare_with:
        with open(args.compare_with, "r", encoding="utf-8") as handle:
            reference_text = handle.read()
        reference_analysis = model.analyze(reference_text, title=args.reference_title)
        comparability = model.compare_analyses(
            analysis,
            reference_analysis,
            window_size=args.compare_window_size,
        )
        comparison_report = model.format_comparability_report(
            comparability,
            heading=f"Comparability against {args.compare_with}",
        )

    if not args.no_report:
        sys.stdout.write(report + "\n")
        if comparison_report is not None:
            sys.stdout.write("\n" + comparison_report + "\n")

    if args.select_token is not None:
        selected = next(
            (
                detail
                for detail in analysis.token_seed_details
                if detail.position == args.select_token
            ),
            None,
        )
        translation = next(
            (
                entry
                for entry in analysis.cursor_translations
                if entry.position == args.select_token
            ),
            None,
        )
        sys.stdout.write("\n")
        if selected is None:
            sys.stdout.write(
                f"Token {args.select_token} is outside the available range.\n"
            )
        else:
            slider = model._format_slider(selected.slider_ratio)
            sys.stdout.write(
                "Selected token {pos}: {text} ({code})\n".format(
                    pos=selected.position,
                    text=selected.text,
                    code=selected.code,
                )
            )
            sys.stdout.write(
                "  max: {max_val} trimmed: {trimmed} lengthened: {lengthened} "
                "cubic: {cubic}\n".format(
                    max_val=selected.max_numeric_value,
                    trimmed=selected.trimmed_numeric_value,
                    lengthened=selected.lengthened_numeric_value,
                    cubic=selected.cubic_numeric_value,
                )
            )
            sys.stdout.write(
                "  seed: {seed} max-seed: {max_seed} slider: {slider} "
                "({ratio:.4f})\n".format(
                    seed=selected.seed_value,
                    max_seed=selected.max_seed_value,
                    slider=slider,
                    ratio=selected.slider_ratio,
                )
            )
            if selected.is_zero_token:
                sys.stdout.write("  note: zero token (no numeric contribution)\n")
            elif selected.is_unit_token:
                digit = model._number_to_base36(selected.unit_digit)
                sys.stdout.write(
                    f"  note: unit token (digit {digit})\n"
                )
            if translation is not None:
                sys.stdout.write(
                    "  translator: {ascii} [{hint}] signature {signature}\n".format(
                        ascii=translation.ascii_equivalent,
                        hint=translation.script_hint,
                        signature=translation.signature,
                    )
                )
                sys.stdout.write(f"  translator note: {translation.explanation}\n")

    if args.seed_from is not None:
        seed = model.seed_from_name(args.seed_from)
        sys.stdout.write(f"\nSeed for '{args.seed_from}': {seed}\n")

    def _write_output(path: str, contents: str) -> None:
        if path == "-":
            if contents and not contents.endswith("\n"):
                contents += "\n"
            sys.stdout.write(contents)
        else:
            with open(path, "w", encoding="utf-8") as handle:
                handle.write(contents)
                if contents and not contents.endswith("\n"):
                    handle.write("\n")

    if args.emit_json:
        analysis_json = json.dumps(
            analysis.as_dict(),
            ensure_ascii=False,
            sort_keys=True,
            indent=2,
        )
        _write_output(args.emit_json, analysis_json)

    if args.emit_data_bundle:
        bundle = model.prepare_data_generation(
            contents,
            analysis=analysis,
            reference_text=reference_text,
            reference_analysis=reference_analysis,
            title=args.title,
            reference_title=args.reference_title,
            window_size=args.compare_window_size,
            include_reports=True,
            comparability=comparability,
        )
        bundle_json = json.dumps(
            bundle.as_dict(),
            ensure_ascii=False,
            sort_keys=True,
            indent=2,
        )
        _write_output(args.emit_data_bundle, bundle_json)
    return 0


def analyze_text(
    text: str,
    *,
    title: str | None = None,
    model: PhoneticTrailingSongModel | None = None,
) -> AnalysisResult:
    """Convenience wrapper returning :class:`AnalysisResult` for ``text``."""

    if model is None:
        model = PhoneticTrailingSongModel()
    return model.analyze(text, title=title)


def prepare_data_generation(
    text: str,
    *,
    reference_text: str | None = None,
    title: str | None = None,
    reference_title: str | None = None,
    window_size: int = 5,
    include_reports: bool = True,
    model: PhoneticTrailingSongModel | None = None,
) -> DataGenerationBundle:
    """Return a :class:`DataGenerationBundle` for ``text`` and optional reference."""

    if model is None:
        model = PhoneticTrailingSongModel()

    analysis = model.analyze(text, title=title)

    reference_analysis: AnalysisResult | None = None
    if reference_text is not None:
        reference_analysis = model.analyze(reference_text, title=reference_title)

    return model.prepare_data_generation(
        text,
        analysis=analysis,
        reference_text=reference_text,
        reference_analysis=reference_analysis,
        title=title,
        reference_title=reference_title,
        window_size=window_size,
        include_reports=include_reports,
    )


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    import sys

    raise SystemExit(_cli(sys.argv))
