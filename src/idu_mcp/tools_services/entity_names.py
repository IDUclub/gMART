"""Match inflected Russian entity names to actual Urban API catalog entries."""

import re
from functools import lru_cache

from pymorphy3 import MorphAnalyzer


def _text(value: str) -> str:
    return " ".join(value.casefold().replace("ё", "е").split())


@lru_cache(maxsize=1)
def _morphology() -> MorphAnalyzer:
    return MorphAnalyzer(lang="ru")


@lru_cache(maxsize=4096)
def _forms(word: str) -> frozenset[str]:
    # Keep punctuation, numbers and unknown words literal. Do not infer synonyms
    # or discard qualifiers: a school gym is not a school.
    if not re.fullmatch(r"[а-я]+", word):
        return frozenset({word})
    known = [p for p in _morphology().parse(word) if p.is_known]
    return frozenset(_text(p.normal_form) for p in known) or frozenset({word})


def _words(value: str) -> tuple[frozenset[str], ...]:
    return tuple(_forms(word) for word in re.findall(r"\w+|[^\w\s]", _text(value)))


def resolve_catalog_names(
    requested: list[str], catalog: dict[str, int]
) -> dict[str, dict]:
    """Use an exact name first, then a unique full-phrase morphological match.

    Every token must match in order. Ambiguous matches stay unresolved, and the
    returned name/ID always comes from the catalog, never from generated text.
    """
    result = {}
    catalog_words = None
    for name in requested:
        exact = [item for item in catalog if _text(item) == _text(name)]
        candidates = exact
        if not candidates:
            words = _words(name)
            if catalog_words is None:
                catalog_words = {item: _words(item) for item in catalog}
            candidates = [
                item
                for item, tokens in catalog_words.items()
                if words
                and len(words) == len(tokens)
                and all(left & right for left, right in zip(words, tokens))
            ]
        canonical = candidates[0] if len(candidates) == 1 else None
        result[name] = {
            "found": canonical is not None,
            "canonical_name": canonical,
            "type_id": catalog[canonical] if canonical is not None else None,
        }
    return result
