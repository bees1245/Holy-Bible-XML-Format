"""Language profile registry for phonetic analysis.

This module provides lightweight transliteration and normalization helpers for a
handful of common writing systems so the phonetic tooling can operate across
multiple languages without external dependencies.  The goal is not to deliver
perfect linguistic accuracy but to expose deterministic, script-aware fallbacks
that cover the broad set of texts shipped with the repository.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, Iterable, List, Optional, Sequence, Tuple
import unicodedata

__all__ = [
    "LanguageProfile",
    "LanguageProfileRegistry",
    "default_language_registry",
]


# ---------------------------------------------------------------------------
# Utility helpers

def _strip_accents(value: str) -> str:
    normalized = unicodedata.normalize("NFKD", value)
    return "".join(ch for ch in normalized if unicodedata.category(ch) != "Mn")


def _ascii_fallback(value: str) -> str:
    ascii_bytes = _strip_accents(value).encode("ascii", "ignore")
    return ascii_bytes.decode("ascii", "ignore")


def _transliterate_with_mapping(value: str, mapping: Dict[str, str]) -> str:
    result: List[str] = []
    normalized = unicodedata.normalize("NFKD", value)
    for ch in normalized:
        if unicodedata.category(ch) == "Mn":
            continue
        lower = ch.lower()
        repl = mapping.get(lower)
        if repl is None:
            result.append(ch)
            continue
        if ch.isupper():
            result.append(repl.upper())
        else:
            result.append(repl)
    return "".join(result)


def _normalize_lower(value: str) -> str:
    return unicodedata.normalize("NFC", value).lower()


# Script ranges are approximate buckets used for detection heuristics.
_LATIN_RANGES: Tuple[Tuple[int, int], ...] = (
    (0x0041, 0x024F),  # Basic + Extended Latin
    (0x1E00, 0x1EFF),  # Latin Extended Additional
)
_GREEK_RANGES = ((0x0370, 0x03FF), (0x1F00, 0x1FFF))
_CYRILLIC_RANGES = ((0x0400, 0x04FF), (0x0500, 0x052F))
_ARABIC_RANGES = ((0x0600, 0x06FF), (0x0750, 0x077F))
_DEVANAGARI_RANGES = ((0x0900, 0x097F),)
_HEBREW_RANGES = ((0x0590, 0x05FF),)
_HANGUL_RANGES = ((0xAC00, 0xD7AF),)
_HIRAGANA_RANGES = ((0x3040, 0x309F),)
_KATAKANA_RANGES = ((0x30A0, 0x30FF),)


def _match_ratio(text: str, ranges: Sequence[Tuple[int, int]]) -> float:
    total = 0
    matches = 0
    for ch in text:
        if ch.isalpha():
            total += 1
            codepoint = ord(ch)
            if any(start <= codepoint <= end for start, end in ranges):
                matches += 1
    if total == 0:
        return 0.0
    return matches / total


@dataclass(frozen=True)
class LanguageProfile:
    code: str
    name: str
    description: str
    script_ranges: Sequence[Tuple[int, int]]
    romanizer: Callable[[str], str]
    normalizer: Callable[[str], str] = _normalize_lower

    def romanize(self, value: str) -> str:
        return self.romanizer(value)

    def normalize(self, value: str) -> str:
        return self.normalizer(value)

    def match_ratio(self, text: str) -> float:
        return _match_ratio(text, self.script_ranges)


class LanguageProfileRegistry:
    """Registry handling detection and lookup for language profiles."""

    def __init__(self, profiles: Optional[Iterable[LanguageProfile]] = None) -> None:
        self._profiles: Dict[str, LanguageProfile] = {}
        if profiles:
            for profile in profiles:
                self.register(profile)

    # Public API ---------------------------------------------------------
    def register(self, profile: LanguageProfile) -> None:
        self._profiles[profile.code] = profile

    def get(self, code: Optional[str]) -> Optional[LanguageProfile]:
        if code is None:
            return None
        return self._profiles.get(code)

    def list_profiles(self) -> List[LanguageProfile]:
        return [self._profiles[key] for key in sorted(self._profiles)]

    def detect(
        self,
        text: str,
        *,
        preferred: Optional[str] = None,
        fallback: Optional[str] = "en",
    ) -> Tuple[LanguageProfile, Dict[str, object]]:
        if preferred:
            profile = self.get(preferred)
            if profile:
                return profile, {
                    "strategy": "override",
                    "code": profile.code,
                    "name": profile.name,
                    "confidence": 1.0,
                }
        best_profile: Optional[LanguageProfile] = None
        best_ratio = 0.0
        for profile in self._profiles.values():
            ratio = profile.match_ratio(text)
            if ratio > best_ratio:
                best_profile = profile
                best_ratio = ratio
        if best_profile and best_ratio >= 0.2:
            return best_profile, {
                "strategy": "detected",
                "code": best_profile.code,
                "name": best_profile.name,
                "confidence": round(best_ratio, 3),
            }
        fallback_profile = self.get(fallback)
        if fallback_profile:
            return fallback_profile, {
                "strategy": "fallback",
                "code": fallback_profile.code,
                "name": fallback_profile.name,
                "confidence": round(best_ratio, 3),
            }
        # As a last resort pick the first available profile.
        first_profile = next(iter(self._profiles.values()))
        return first_profile, {
            "strategy": "fallback",
            "code": first_profile.code,
            "name": first_profile.name,
            "confidence": round(best_ratio, 3),
        }


# ---------------------------------------------------------------------------
# Default registry configuration

_GREEK_MAP = {
    "α": "a",
    "β": "b",
    "γ": "g",
    "δ": "d",
    "ε": "e",
    "ζ": "z",
    "η": "e",
    "θ": "th",
    "ι": "i",
    "κ": "k",
    "λ": "l",
    "μ": "m",
    "ν": "n",
    "ξ": "x",
    "ο": "o",
    "π": "p",
    "ρ": "r",
    "σ": "s",
    "ς": "s",
    "τ": "t",
    "υ": "y",
    "φ": "f",
    "χ": "ch",
    "ψ": "ps",
    "ω": "o",
}

_CYRILLIC_MAP = {
    "а": "a",
    "б": "b",
    "в": "v",
    "г": "g",
    "д": "d",
    "е": "e",
    "ё": "yo",
    "ж": "zh",
    "з": "z",
    "и": "i",
    "й": "y",
    "к": "k",
    "л": "l",
    "м": "m",
    "н": "n",
    "о": "o",
    "п": "p",
    "р": "r",
    "с": "s",
    "т": "t",
    "у": "u",
    "ф": "f",
    "х": "kh",
    "ц": "ts",
    "ч": "ch",
    "ш": "sh",
    "щ": "shch",
    "ъ": "",
    "ы": "y",
    "ь": "",
    "э": "e",
    "ю": "yu",
    "я": "ya",
}

_ARABIC_MAP = {
    "ا": "a",
    "أ": "a",
    "إ": "i",
    "آ": "a",
    "ب": "b",
    "ت": "t",
    "ث": "th",
    "ج": "j",
    "ح": "h",
    "خ": "kh",
    "د": "d",
    "ذ": "dh",
    "ر": "r",
    "ز": "z",
    "س": "s",
    "ش": "sh",
    "ص": "s",
    "ض": "d",
    "ط": "t",
    "ظ": "z",
    "ع": "a",
    "غ": "gh",
    "ف": "f",
    "ق": "q",
    "ك": "k",
    "ل": "l",
    "م": "m",
    "ن": "n",
    "ه": "h",
    "و": "w",
    "ي": "y",
    "ى": "a",
    "ة": "h",
    "ء": "",
}

_DEVANAGARI_MAP = {
    "अ": "a",
    "आ": "aa",
    "इ": "i",
    "ई": "ii",
    "उ": "u",
    "ऊ": "uu",
    "ऋ": "ri",
    "ए": "e",
    "ऐ": "ai",
    "ओ": "o",
    "औ": "au",
    "क": "k",
    "ख": "kh",
    "ग": "g",
    "घ": "gh",
    "ङ": "ng",
    "च": "ch",
    "छ": "chh",
    "ज": "j",
    "झ": "jh",
    "ञ": "ny",
    "ट": "t",
    "ठ": "th",
    "ड": "d",
    "ढ": "dh",
    "ण": "n",
    "त": "t",
    "थ": "th",
    "द": "d",
    "ध": "dh",
    "न": "n",
    "प": "p",
    "फ": "ph",
    "ब": "b",
    "भ": "bh",
    "म": "m",
    "य": "y",
    "र": "r",
    "ल": "l",
    "व": "v",
    "श": "sh",
    "ष": "sh",
    "स": "s",
    "ह": "h",
}

_HEBREW_MAP = {
    "א": "a",
    "ב": "b",
    "ג": "g",
    "ד": "d",
    "ה": "h",
    "ו": "v",
    "ז": "z",
    "ח": "h",
    "ט": "t",
    "י": "y",
    "כ": "k",
    "ל": "l",
    "מ": "m",
    "נ": "n",
    "ס": "s",
    "ע": "a",
    "פ": "p",
    "צ": "ts",
    "ק": "k",
    "ר": "r",
    "ש": "sh",
    "ת": "t",
}

_HANGUL_NORMALIZER = lambda value: unicodedata.normalize("NFKD", value)


def _hangul_romanize(value: str) -> str:
    decomposed = unicodedata.normalize("NFKD", value)
    ascii_text = _ascii_fallback(decomposed)
    return ascii_text or value


def _kana_romanize(value: str) -> str:
    # Katakana/Hiragana to ASCII via compatibility decomposition.
    decomposed = unicodedata.normalize("NFKD", value)
    ascii_text = _ascii_fallback(decomposed)
    return ascii_text or value


_DEFAULT_PROFILES: Tuple[LanguageProfile, ...] = (
    LanguageProfile(
        code="en",
        name="English",
        description="Latin script with accent stripping",
        script_ranges=_LATIN_RANGES,
        romanizer=_strip_accents,
    ),
    LanguageProfile(
        code="es",
        name="Spanish",
        description="Latin script with accent stripping",
        script_ranges=_LATIN_RANGES,
        romanizer=_strip_accents,
    ),
    LanguageProfile(
        code="fr",
        name="French",
        description="Latin script with accent stripping",
        script_ranges=_LATIN_RANGES,
        romanizer=_strip_accents,
    ),
    LanguageProfile(
        code="de",
        name="German",
        description="Latin script with umlaut handling",
        script_ranges=_LATIN_RANGES,
        romanizer=lambda value: _strip_accents(value.replace("ß", "ss")),
    ),
    LanguageProfile(
        code="pt",
        name="Portuguese",
        description="Latin script with accent stripping",
        script_ranges=_LATIN_RANGES,
        romanizer=_strip_accents,
    ),
    LanguageProfile(
        code="el",
        name="Greek",
        description="Greek to Latin transliteration",
        script_ranges=_GREEK_RANGES,
        romanizer=lambda value: _transliterate_with_mapping(value, _GREEK_MAP),
    ),
    LanguageProfile(
        code="ru",
        name="Russian",
        description="Cyrillic to Latin transliteration",
        script_ranges=_CYRILLIC_RANGES,
        romanizer=lambda value: _transliterate_with_mapping(value, _CYRILLIC_MAP),
    ),
    LanguageProfile(
        code="uk",
        name="Ukrainian",
        description="Cyrillic to Latin transliteration",
        script_ranges=_CYRILLIC_RANGES,
        romanizer=lambda value: _transliterate_with_mapping(value, _CYRILLIC_MAP),
    ),
    LanguageProfile(
        code="ar",
        name="Arabic",
        description="Arabic to Latin transliteration",
        script_ranges=_ARABIC_RANGES,
        romanizer=lambda value: _transliterate_with_mapping(value, _ARABIC_MAP),
    ),
    LanguageProfile(
        code="he",
        name="Hebrew",
        description="Hebrew to Latin transliteration",
        script_ranges=_HEBREW_RANGES,
        romanizer=lambda value: _transliterate_with_mapping(value, _HEBREW_MAP),
    ),
    LanguageProfile(
        code="hi",
        name="Hindi",
        description="Devanagari to Latin transliteration",
        script_ranges=_DEVANAGARI_RANGES,
        romanizer=lambda value: _transliterate_with_mapping(value, _DEVANAGARI_MAP),
    ),
    LanguageProfile(
        code="ko",
        name="Korean",
        description="Hangul compatibility decomposition",
        script_ranges=_HANGUL_RANGES,
        romanizer=_hangul_romanize,
        normalizer=_HANGUL_NORMALIZER,
    ),
    LanguageProfile(
        code="ja",
        name="Japanese",
        description="Kana compatibility decomposition",
        script_ranges=_HIRAGANA_RANGES + _KATAKANA_RANGES,
        romanizer=_kana_romanize,
        normalizer=_normalize_lower,
    ),
)


def default_language_registry() -> LanguageProfileRegistry:
    return LanguageProfileRegistry(_DEFAULT_PROFILES)
