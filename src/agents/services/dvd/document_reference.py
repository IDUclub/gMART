"""Literal document references, independent of model output and word order."""

import re
from dataclasses import dataclass

# Canonical names are selectors, not invented IDs. DVD resolves them against the
# scoped catalog (name / aliases / external IDs); zero matches never widens scope.
DOCUMENT_ALIASES = (
    (
        r"\b(?:грк\s*(?:рф)?|град(?:остроительн[а-я]+)?\s*кодекс[а-я]*)\b",
        "Градостроительный кодекс Российской Федерации",
    ),
    (
        r"\bконституци[а-я]+(?:\s+(?:рф|российск[а-я]+\s+федераци[а-я]+))?(?=\s*(?:$|[?,.;:]|стать|ст\.|част|ч\.|пункт|п\.|что))",
        "Конституция Российской Федерации",
    ),
)
DESIGNATION = re.compile(
    r"\b(?:ГОСТ(?:\s+Р)?|СП|СНиП|СанПиН|СН|ТСН|НПБ|ISO|EN)\s*\d+(?:[.\-]\d+){0,5}", re.I
)
ADDRESS = re.compile(
    r"(?<!\w)(?P<kind>подпункт[а-я]*|пункт[а-я]*|пп\.|п\.|раздел[а-я]*|глава|главы|главе|главу|стать[а-я]+|ст\.|част[а-я]*|ч\.|приложени[а-я]+|section|clause)(?=\s|\d|[А-ЯЁA-Z]\.)\s*"
    r"(?P<number>(?:[А-ЯЁA-Z]\.)?\d+(?:[._][\d*?]+)*(?:\s*[–—-]\s*\d+(?:\.\d+)*)?|[IVXLCDM]+|[А-ЯЁA-Z])(?=\s|[,;:?!.)]|$)",
    re.I,
)


@dataclass(frozen=True)
class DocumentReference:
    pattern: str | None
    document_names: list[str]


def parse_reference(query: str) -> DocumentReference:
    components = []
    for match in ADDRESS.finditer(query):
        kind, number = match["kind"].lower(), match["number"].replace("_", ".")
        # A full word cannot donate its final letter as a compact address:
        # "с подпунктами." is not "подпунктам и". Dotted abbreviations may
        # directly precede letter addresses ("п.А.1").
        if (
            match.start("number") == match.end("kind")
            and not kind.endswith(".")
            and not number[0].isdigit()
        ):
            continue
        number = re.sub(r"(?<=\d)\?$", "", number)
        if kind.startswith(("раздел", "section")):
            rank, label = 1, "раздел "
        elif kind.startswith("приложени"):
            rank, label = 0, "приложение "
        elif kind.startswith("глав"):
            rank, label = 2, "глава "
        elif kind.startswith(("стать", "ст.")):
            rank, label = 3, "статья "
        elif kind.startswith(("част", "ч.")):
            rank, label = 4, ""
        elif kind.startswith(("подпункт", "пп.")):
            rank, label = 6, ""
        else:
            rank, label = 5, ""
        components.append((rank, label + number))
    # Equal-level references (e.g. two explicitly named clauses) are left to the
    # model's range/union handling rather than fabricated into an ancestor path.
    ranks = [rank for rank, _ in components]
    pattern = (
        " / ".join(value for _, value in sorted(components))
        if len(ranks) == len(set(ranks)) and components
        else None
    )
    names = [m[0] for m in DESIGNATION.finditer(query)]
    names.extend(
        name
        for expression, name in DOCUMENT_ALIASES
        if re.search(expression, query, re.I)
    )
    return DocumentReference(pattern, list(dict.fromkeys(names)))


def wants_full_quote(query: str) -> bool:
    return bool(
        re.search(
            r"процитир|цитат|дословн|что\s+(?:написано|сказано|говорится|содержится)|текст\s+(?:пункт|стать|част)",
            query,
            re.I,
        )
    )


def quote_only(query: str) -> bool:
    explicit = re.search(
        r"процитир|только\s+(?:цитат|текст)|без\s+(?:объяснен|пояснен)", query, re.I
    )
    asks_text = re.search(
        r"что\s+(?:написано|сказано|говорится|содержится)", query, re.I
    )
    asks_explanation = re.search(
        r"объясн|поясн|разъясн|смысл|означа|примен|сравн|почему|зачем|кратк|резюм",
        query,
        re.I,
    )
    return bool(explicit or (asks_text and not asks_explanation))
