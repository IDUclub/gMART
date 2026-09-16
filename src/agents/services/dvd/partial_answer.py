"""Keep audited evidence across rounds and render a closed set of verified claims."""

import re

from .dvd_context import source_records

EMPTY_ANSWER = (
    "Не удалось подтвердить ответ по найденным источникам. "
    "Уточните вопрос или укажите конкретный документ."
)


def normalize(text):
    return " ".join(text.split())


def statement(text):
    # Source labels are local to a retrieval; render new labels from the saved
    # evidence instead of accidentally attaching a later round's [1].
    return normalize(re.sub(r"\[\d+\]", "", text)).strip()


class PartialAnswerEvidence:
    def __init__(self, records=None):
        self.records = list(records or [])

    def add(self, claims, draft, context, literal_check):
        sources = source_records(context)
        for claim in claims:
            text = statement(claim.text)
            if not text or normalize(claim.text) not in normalize(draft):
                continue
            evidence = []
            for ref in claim.evidence:
                source = sources.get(ref.source_id)
                quote = normalize(ref.quote)
                if not source or not quote or quote not in normalize(source[1]):
                    evidence = []
                    break
                header = re.sub(r"^\[\d+\]\s*", "", source[0])
                evidence.append({"source": header, "quote": quote})
            status = claim.status
            if status == "supported" and (
                not evidence or literal_check(context, claim.text)
            ):
                status = "insufficient"
            self.records.append({"text": text, "status": status, "evidence": evidence})

    def candidates(self):
        contradicted = {
            r["text"].casefold() for r in self.records if r["status"] == "contradicted"
        }
        seen, result = set(), {}
        for i, record in enumerate(self.records):
            key = record["text"].casefold()
            if (
                record["status"] == "supported"
                and record["evidence"]
                and key not in contradicted
                and key not in seen
            ):
                result[i] = record
                seen.add(key)
        return result

    def render(self, approved_ids):
        candidates = self.candidates()
        if len(set(approved_ids)) != len(approved_ids) or any(
            i not in candidates for i in approved_ids
        ):
            raise ValueError("Partial answer selection contains an unverified claim")
        if not approved_ids:
            return EMPTY_ANSWER
        sources = {}
        for i in approved_ids:
            record = candidates[i]
            for evidence in record["evidence"]:
                key = (evidence["source"], evidence["quote"])
                sources.setdefault(key, len(sources) + 1)
        # The live critic can approve an overgeneralized paraphrase despite a
        # correct supporting quotation. After three rejected drafts, return only
        # literal evidence, never those draft claims or their generated metadata.
        references = [
            f"[{label}] {header}\n> {quote}"
            for (header, quote), label in sources.items()
        ]
        return (
            "По найденным источникам удалось подтвердить следующее:\n\n"
            + "Дословные выдержки из источников:\n\n"
            + "\n\n".join(references)
            + "\n\nЭто частичный ответ: остальные положения подтвердить не удалось."
        )
