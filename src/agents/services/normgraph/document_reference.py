"""Literal document titles, separate from model-generated search filters."""

import re


def explicit_document_names(query: str) -> list[str]:
    names = []
    for match in re.finditer(
        r'\b(?i:документ(?:а|у|е|ом)?)\s+(?:«([^»]+)»|"([^"]+)"|([A-ZА-ЯЁ0-9][\w.-]*(?:[^\S\r\n]+[A-ZА-ЯЁ0-9][\w.-]*)*))',
        query,
    ):
        name = next(part.strip().rstrip(".") for part in match.groups() if part)
        if name not in names:
            names.append(name)
    return names
