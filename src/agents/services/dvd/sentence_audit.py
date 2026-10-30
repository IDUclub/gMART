"""Short critic references, restored to literal evidence before verdict assembly."""

from __future__ import annotations

import json
import re
from decimal import Decimal

from .answer_risk import _without_identifiers
from .document_reference import ADDRESS, DESIGNATION
from .dvd_context import SOURCE_SEPARATOR, source_records

_ABBREVIATION = re.compile(
    r"\b(?:т\.\s*е\.|т\.\s*д\.|т\.\s*п\.|пп?\.|ст\.|ч\.|м\.|г\.|гг\.|"
    r"ред\.|им\.|см\.|рис\.|табл\.|ул\.|св\.|мин\.|макс\.|га\.)|%\.",
    re.I,
)
_END = re.compile(r"[.!?]+[»”’\"')\]}]*(?=\s|$)")
_ITEM = re.compile(r"^\s*(?:\d+(?:\.\d+)*[.)]?|[а-яa-z][.)])\s+", re.I)
_HTML_ROW = re.compile(r"<tr\b[^>]*>.*?</tr\s*>", re.I | re.S)
_CITY_ADDRESS = re.compile(r"\b(?i:в|из|к|по|от)\s+г\.\s+[А-ЯЁ][а-яё-]+")
_NEXT_ITEM = re.compile(r"(?:[-*+•]|\d+[.)]|[а-яa-z][.)])\s+", re.I)
_OPENING = "«\"“„‘'([{"
_NUMBER = re.compile(r"(?<![\w\d])[-−]?\d+(?:[ \u00a0\u202f]\d{3})*(?:[.,]\d+)?")
_COMPARATOR = re.compile(
    r"\bне\s+(?P<direction>менее|более|ниже|выше)\s*(?P<number>"
    + _NUMBER.pattern.replace(r"(?<![\w\d])", "")
    + r")",
    re.I,
)
_BOUND = {"менее": "lower", "ниже": "lower", "более": "upper", "выше": "upper"}
_OPPOSITE = {"lower": "upper", "upper": "lower"}
_PROHIBITION = re.compile(r"\b(?:не\s+(?:допуска|разреша)\w*|запрещ\w*)", re.I)
_PERMISSION = re.compile(r"\b(?:допуска|разреша)\w*", re.I)
_MAX_ENUM_IDS = 400
_MAX_ENUM_BYTES = 8000


def _abbreviation_ends(text: str, end: int) -> bool:
    following = text[end:]
    token = following.lstrip()
    while token and token[0] in _OPENING:
        token = token[1:].lstrip()
    if DESIGNATION.match(token):
        return False
    return (
        not token
        or bool(re.match(r"[ \t]*(?:\r?\n|$)", following))
        or token[0].isupper()
        or bool(_NEXT_ITEM.match(token))
    )


def sentences(body: str) -> list[str]:
    """Split sentences and list/table rows; every result is an exact substring.

    Wrapped prose keeps its original newlines. Internal abbreviation dots and
    addresses are protected; a final abbreviation dot can end a sentence before
    a new sentence, list item or line ending. Designations stay intact.
    """
    if rows := list(_HTML_ROW.finditer(body)):
        result, start = [], 0
        for row in rows:
            result.extend(sentences(body[start : row.start()]))
            result.append(row[0])
            start = row.end()
        result.extend(sentences(body[start:]))
        return result
    result, prose = [], []

    def flush():
        text = "".join(prose)
        abbreviations = list(_ABBREVIATION.finditer(text))
        protected = [
            m.span()
            for pattern in (ADDRESS, DESIGNATION, _CITY_ADDRESS)
            for m in pattern.finditer(text)
        ]
        if item := _ITEM.match(text):
            protected.append(item.span())
        start = 0
        for match in _END.finditer(text):
            if any(left <= match.start() < right for left, right in protected):
                continue
            abbreviation = next(
                (m for m in abbreviations if m.start() <= match.start() < m.end()),
                None,
            )
            if abbreviation is not None and (
                match.start() != abbreviation.end() - 1
                or not _abbreviation_ends(text, match.end())
            ):
                continue
            if sentence := text[start : match.end()].strip():
                result.append(sentence)
            start = match.end()
        if sentence := text[start:].strip():
            result.append(sentence)
        prose.clear()

    for row in body.splitlines(keepends=True):
        if not row.strip():
            flush()
        elif "|" in row or "\t" in row.strip() or _NEXT_ITEM.match(row.lstrip()):
            flush()
            result.append(row.strip())
        else:
            prose.append(row)
    flush()
    return result


def numbered_context(
    context: str,
) -> tuple[str, dict[str, tuple[str, str]], tuple[str, str]]:
    """Number each labelled body once, preserving headers and source ownership."""
    opening, closing = "{", "}"
    while re.search(re.escape(opening) + r"\d+\.\d+" + re.escape(closing), context):
        opening, closing = "{" + opening, closing + "}"
    records, rendered = {}, []
    for label, (header, body) in source_records(context).items():
        if label == "unlabelled":
            rendered.append(body)
            continue
        rows = []
        for index, text in enumerate(sentences(body), 1):
            sentence_id = f"{label[1:-1]}.{index}"
            records[sentence_id] = (label, text)
            rows.append(f"{opening}{sentence_id}{closing} {text}")
        rendered.append(header + "\n" + "\n".join(rows))
    return SOURCE_SEPARATOR.join(rendered), records, (opening, closing)


def sentence_schema(schema: dict, records: dict[str, tuple[str, str]]) -> None:
    """Change only the compact LLM schema; the quote schema stays untouched."""
    ids = list(records)
    item = {"type": "string", "pattern": r"^[1-9]\d*\.[1-9]\d*$"}
    if (
        ids
        and len(ids) <= _MAX_ENUM_IDS
        and len(json.dumps(ids).encode()) <= _MAX_ENUM_BYTES
    ):
        item = {"type": "string", "enum": ids}
    for name in ("ClaimEvidence", "Correction"):
        definition = schema["$defs"][name]
        if "description" in definition:
            definition["description"] = definition["description"].replace(
                "``quote``", "``sentences``"
            )
        definition["properties"].pop("quote")
        definition["properties"]["sentences"] = {
            "type": "array",
            "items": item.copy(),
            "minItems": 1 if name == "ClaimEvidence" else 0,
            "maxItems": 3,
            "title": "Sentences",
        }
        definition["required"] = [
            field for field in definition.get("required", []) if field != "quote"
        ] + ["sentences"]


def restore_sentences(data: dict, records: dict[str, tuple[str, str]]) -> dict:
    """Validate IDs before Pydantic parsing, inside the malformed-audit retry."""

    if not isinstance(data, dict):
        raise ValueError("Audit must be an object")
    for field in ("claims", "corrections"):
        if not isinstance(data.get(field, []), list) or any(
            not isinstance(item, dict) for item in data.get(field, [])
        ):
            raise ValueError("Audit claims and corrections must be lists of objects")

    def quotes(item, *, correction=False):
        if not isinstance(item, dict):
            raise ValueError("Sentence evidence must be an object")
        if "quote" in item:
            raise ValueError("Sentence audit must use sentences, not quote")
        ids = item.get("sentences", []) if correction else item.get("sentences")
        if (
            not isinstance(ids, list)
            or not (0 if correction else 1) <= len(ids) <= 3
            or any(not isinstance(sid, str) for sid in ids)
        ):
            raise ValueError(
                "Evidence must cite 1-3 sentence IDs; corrections may cite none"
            )
        label = item.get("source_id", "")
        if any(sid not in records or records[sid][0] != label for sid in ids):
            raise ValueError(
                "Unknown sentence ID or sentence belongs to another source"
            )
        return [records[sid][1] for sid in ids]

    for claim in data.get("claims", []):
        if not isinstance(claim.get("evidence", []), list):
            raise ValueError("Claim evidence must be a list")
        evidence = []
        for item in claim.get("evidence", []):
            evidence.extend(
                {"source_id": item.get("source_id"), "quote": text}
                for text in quotes(item)
            )
        claim["evidence"] = evidence
    for item in data.get("corrections", []):
        item["quote"] = " … ".join(quotes(item, correction=True))
        item.pop("sentences", None)
    return data


def _normal_number(value: str) -> str:
    number = Decimal(re.sub(r"\s", "", value).replace(",", ".").replace("−", "-"))
    text = format(number, "f")
    return text.rstrip("0").rstrip(".") if "." in text else text


def _numbers(text: str) -> set[str]:
    return {_normal_number(m[0]) for m in _NUMBER.finditer(_without_identifiers(text))}


def match_defect(line: str, quotes: list[str]) -> str:
    """Reject missing quantities or explicit opposite polarity, conservatively.

    This checks occurrence, not arithmetic, unit conversion or legal entailment.
    An opposite comparator must be attached to the same number. A prohibition
    is contradicted only by explicit permission carrying a shared quantity.
    """
    cited = "\n".join(quotes)
    missing = _numbers(line) - _numbers(cited)
    if missing:
        return (
            "Число "
            + ", ".join(sorted(missing))
            + " не подтверждено предложениями источника."
        )
    comparators = {
        (_BOUND[m["direction"].lower()], _normal_number(m["number"]))
        for m in _COMPARATOR.finditer(_without_identifiers(cited))
    }
    for match in _COMPARATOR.finditer(_without_identifiers(line)):
        direction = _BOUND[match["direction"].lower()]
        number = _normal_number(match["number"])
        if (_OPPOSITE[direction], number) in comparators and (
            direction,
            number,
        ) not in comparators:
            return f"Для числа {match['number']} источник задаёт противоположное ограничение."
    if _PROHIBITION.search(line):
        for quote in quotes:
            if (
                _numbers(line) & _numbers(quote)
                and _PERMISSION.search(quote)
                and not _PROHIBITION.search(quote)
            ):
                return "Запрет противоречит разрешению для того же числа в предложении источника."
    return ""
