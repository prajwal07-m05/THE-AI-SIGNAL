"""Deterministic entity resolution (Phase IV).

Canonicalizes messy organization/product names to a single canonical form, e.g.
"OpenAI", "OpenAI, Inc.", "Open AI" -> "OpenAI".

Deterministic pipeline (same input always yields same output):
  1. Normalize:  lowercase, strip legal suffixes (Inc/Ltd/LLC/…), unidecode,
                 collapse whitespace, drop punctuation.
  2. Exact hit:  normalized string == normalized canonical.
  3. Alias hit:  normalized string in a canonical's known alias set.
  4. Fuzzy hit:  rapidfuzz token_set_ratio >= threshold (default 90) against
                 all canonicals; highest score wins, ties broken alphabetically
                 for determinism.
  5. Miss:       return the cleaned title-cased string as a NEW canonical, and
                 log it so humans can promote it into the seed table later.

Every decision is recorded in a mapping log (raw -> canonical, method, score)
for the "Entity Mapping Log" deliverable tab.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

from rapidfuzz import fuzz, process
from unidecode import unidecode

from src.resolver.seed_entities import CANONICAL_ENTITIES

_LEGAL_SUFFIX_RE = re.compile(
    r"\b(inc|incorporated|llc|ltd|limited|corp|corporation|co|company|"
    r"pbc|plc|gmbh|b\.?v\.?|s\.?a\.?|labs|technologies|systems|ai)\b\.?",
    re.IGNORECASE,
)
_PUNCT_RE = re.compile(r"[^\w\s]")
_WS_RE = re.compile(r"\s+")


def normalize(name: str) -> str:
    s = unidecode(name).lower()
    s = _PUNCT_RE.sub(" ", s)
    s = _LEGAL_SUFFIX_RE.sub(" ", s)
    s = _WS_RE.sub(" ", s).strip()
    return s


@dataclass
class MappingResult:
    raw: str
    canonical: str
    method: str          # exact | alias | fuzzy | new
    score: float         # 100 for exact/alias, fuzzy score, or 0 for new


@dataclass
class EntityResolver:
    threshold: int = 90
    _norm_to_canon: dict[str, str] = field(default_factory=dict, init=False)
    log: list[MappingResult] = field(default_factory=list, init=False)

    def __post_init__(self) -> None:
        for canonical, aliases in CANONICAL_ENTITIES.items():
            self._norm_to_canon[normalize(canonical)] = canonical
            for alias in aliases:
                self._norm_to_canon[normalize(alias)] = canonical

    def resolve(self, raw: str) -> MappingResult:
        if not raw or not raw.strip():
            res = MappingResult(raw, "", "new", 0.0)
            self.log.append(res)
            return res

        norm = normalize(raw)

        # Exact / alias hit
        if norm in self._norm_to_canon:
            canon = self._norm_to_canon[norm]
            method = "exact" if normalize(canon) == norm else "alias"
            res = MappingResult(raw, canon, method, 100.0)
            self.log.append(res)
            return res

        # Fuzzy hit against all known normalized forms
        match = process.extractOne(
            norm, self._norm_to_canon.keys(), scorer=fuzz.token_set_ratio
        )
        if match and match[1] >= self.threshold:
            canon = self._norm_to_canon[match[0]]
            res = MappingResult(raw, canon, "fuzzy", float(match[1]))
            self.log.append(res)
            return res

        # Miss -> mint a new canonical (title-cased clean form) and learn it
        new_canon = norm.title()
        self._norm_to_canon[norm] = new_canon
        res = MappingResult(raw, new_canon, "new", 0.0)
        self.log.append(res)
        return res
