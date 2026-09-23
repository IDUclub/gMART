"""Topical search phrases, question intent and documents named inside fragments.

A vector query must describe the subject of the requirements, not the action the
user (or the orchestrator) asks for: «найти документы, содержащие требования к
школам» is closest to reference lists, while «требования к зданиям школ» is
closest to the requirements themselves.
"""

import re

from .document_reference import DESIGNATION

_ACTION = (
    r"(?:найд[иь]|найти|отыщи|определи(?:ть)?|получи(?:ть)?|предостав(?:ь|ить)|"
    r"приведи|привести|составь|составить|перечисли(?:ть)?|укажи|указать|скажи|"
    r"расскажи|рассказать|опиши|описать|проверь|проверить|сравни(?:ть)?|покажи|"
    r"показать|выдели(?:ть)?|дай|дать|объясни|объяснить|подбери|подобрать|"
    r"выпиши|выписать|процитируй|процитировать|подскажи(?:те)?|сформулируй)"
)
_LEAD = re.compile(
    r"^\s*(?:(?:пожалуйста|я\s+хочу\s+узнать|хочу\s+узнать|мне\s+нужно\s+знать|"
    r"мне\s+нужно|интересует)\s*,?\s*)*(?:" + _ACTION + r"\b\s*(?:мне\s+)?,?\s*)*",
    re.I,
)
# Tail clauses that describe the requested output, not the subject.
_TAIL = re.compile(
    r"(?:[,;]\s*|\s+)(?:и|а\s+также|а)\s+" + _ACTION + r"\b.*$",
    re.I | re.S,
)
_META = re.compile(
    r"^(?:(?:все|список|перечень)\s+)?(?:(?:нормативн\w*|правов\w*)\s+)?"
    r"(?:документ\w*|источник\w*|акт\w*)\s*,?\s*"
    r"(?:(?:котор\w+|где|в\s+которых)\s+)?"
    r"(?:(?:содерж\w*|есть|привод\w*|приведен\w*|описан\w*|установлен\w*|излож\w*)\s+)?",
    re.I,
)
_WHICH_DOCUMENTS = re.compile(
    r"^(?:в\s+)?(?:как\w+|котор\w+)\s+(?:(?:нормативн\w*|правов\w*)\s+)?"
    r"(?:документ\w*|источник\w*|акт\w*)\s*"
    r"(?:(?:содерж\w*|есть|привод\w*|приведен\w*|описан\w*|установлен\w*|излож\w*)\s+)?",
    re.I,
)
_INFO = re.compile(
    r"^(?:(?:вся\s+)?(?:информаци\w*|сведени\w*|данн\w*)\s+)?(?:о|об|по)\s+(?:том\s*,?\s*)?",
    re.I,
)

_DOCUMENT_LIST = re.compile(
    r"(?:\bв\s+как\w+\s+(?:нормативн\w*\s+)?(?:документ|источник|акт|сп\b|снип)"
    r"|\bкак\w+\s+(?:есть\s+|существуют\s+|действуют\s+)?(?:нормативн\w*\s+|градостроительн\w*\s+)?"
    r"(?:документ|регламент|норматив|нормы|свод\w*\s+правил|сп\b|снип|санпин|гост|правил)"
    r"|\b(?:перечень|список|перечисли\w*)\s+(?:\w+\s+){0,2}(?:документ|регламент|норматив|норм|сп\b|правил|источник)"
    r"|\bкакими\s+(?:документами|нормами|регламентами)"
    r"|\bдокумент\w*\s*,?\s*(?:котор\w+\s+|в\s+которых\s+|где\s+)?"
    r"(?:содерж\w*|есть|привод\w*|описан\w*|установлен\w*|излож\w*)"
    r"|\bгде\s+(?:содержатся|описаны|установлены|приведены)\s+(?:требовани|норм))",
    re.I,
)
_WHICH = re.compile(r"^как\w+\s+(?:есть\s+|существуют\s+|действуют\s+)?", re.I)
_MIN_WORDS = 2


def _sentences(text: str) -> list[str]:
    return [s for s in re.split(r"(?<=[.!?])\s+|\n+", text) if s.strip()]


def _clean(sentence: str) -> str:
    # The orchestrator labels its task wording; the label is not a subject.
    text = re.sub(r"^\s*задача\s*:\s*", "", sentence.strip(), flags=re.I)
    text = _TAIL.sub("", text)
    for pattern in (_LEAD, _WHICH_DOCUMENTS, _META, _INFO, _LEAD, _WHICH):
        text = pattern.sub("", text).strip(" ,;:")
    text = re.sub(
        r"\s+и\s+т\.\s*д\.?|\s+и\s+т\.\s*п\.?|\s+и\s+др\.?", "", text, flags=re.I
    )
    return " ".join(text.strip(" ,;:.?!«»\"'").split())


def topical_query(text: str, *, min_words: int = _MIN_WORDS) -> str:
    """Return the subject of ``text`` without request verbs and document meta-words.

    Returns an empty string when nothing topical remains, so the caller can fall
    back to another source instead of sending an instruction to the vector index.
    """

    parts = []
    for sentence in _sentences(text or ""):
        cleaned = _clean(sentence)
        if len(re.findall(r"[А-Яа-яЁёA-Za-z]{2,}", cleaned)) >= min_words:
            parts.append(cleaned)
    return "; ".join(dict.fromkeys(parts))


def is_document_list_question(text: str) -> bool:
    """Whether the user asks WHICH documents/regulations cover a subject."""

    return bool(_DOCUMENT_LIST.search(text or ""))


def mentioned_documents(hits: list[dict]) -> list[str]:
    """Normative designations a fragment refers to (resolved references first)."""

    names = []
    for hit in hits:
        for reference in hit.get("references") or []:
            if isinstance(reference, dict) and reference.get("scope") != "internal":
                if name := (reference.get("target_name") or "").strip():
                    names.append(name)
        text = hit.get("source_text") or hit.get("text") or ""
        names.extend(" ".join(m[0].split()) for m in DESIGNATION.finditer(text))
    return list(dict.fromkeys(names))
