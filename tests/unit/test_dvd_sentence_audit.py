"""Sentence references keep critic output short and literal evidence verifiable."""

import json
from copy import deepcopy
from unittest.mock import AsyncMock

import pytest

from src.agents.services.dvd import flags
from src.agents.services.dvd.answer_risk import AnswerRisk, critic_plan
from src.agents.services.dvd.context_reducer import PreparedContext
from src.agents.services.dvd.dvd_context import (
    SOURCE_SEPARATOR,
    TREE_PREFIX,
    source_records,
)
from src.agents.services.dvd.dvd_reasoning import AnswerCritic, CompactAudit
from src.agents.services.dvd.partial_answer import PartialAnswerEvidence
from src.agents.services.dvd.retry_policy import CriticResponseError
from src.agents.services.dvd.run_metrics import DvdRunMetrics
from src.agents.services.dvd.sentence_audit import (
    match_defect,
    numbered_context,
    restore_sentences,
    sentence_schema,
    sentences,
)

SOURCE = "Расстояние должно быть не менее 500 м. Доступ обеспечивается."
CONTEXT = f"[1] Правила\n{SOURCE}" + SOURCE_SEPARATOR + "[2] Условия\nПроезд открыт."
LINE = "Расстояние должно быть не менее 500 м [1]."


class SentenceLlm:
    def __init__(self, change=None):
        self.calls = []
        self.change = change

    async def model_input_tokens(self, *args, **kwargs):
        return 2000

    async def chat(self, **kwargs):
        self.calls.append(kwargs)
        schema = kwargs["format"]
        ids = schema["$defs"]["LineClaim"]["properties"]["line"].get("enum", [])
        data = {
            "claims": [
                {
                    "line": line,
                    "status": "supported",
                    "evidence": [{"source_id": "[1]", "sentences": ["1.1"]}],
                }
                for line in ids
            ],
            "unsupported_claims": [],
            "missing_requirements": [],
            "corrections": [],
            "satisfied": True,
            "critique": "",
            "refined_search_query": None,
        }
        if self.change:
            self.change(data)
        return {"message": {"content": json.dumps(data, ensure_ascii=False)}}


@pytest.fixture(autouse=True)
def sentence_settings(monkeypatch):
    monkeypatch.delenv("DVD_CRITIC_SENTENCE_IDS", raising=False)
    monkeypatch.delenv("DVD_CRITIC_FULL_AUDIT_REASONS", raising=False)
    monkeypatch.setenv("DVD_ADAPTIVE_CRITIC", "true")
    monkeypatch.setenv("DVD_PARALLEL_CRITIC", "true")
    monkeypatch.setenv("DVD_CRITIC_LINES_PER_CALL", "1")


@pytest.mark.parametrize(
    "text",
    [
        "Согласно п. 2.1.2 и ст. 5 расстояние 1,5 м предусмотрено. Приказ N 111 действует.",
        "В 2024 г. в г. Москве расстояние 2.5 м предусмотрено. Действует СП 42.13330.2016.",
        "Это условие, т. е. правило, сохраняется. Следующее предложение.",
    ],
)
def test_abbreviations_addresses_and_decimals_stay_in_their_sentence(text):
    assert len(sentences(text)) == 2
    assert all(part in text for part in sentences(text))


def test_list_items_table_rows_and_ids_are_stable():
    body = "1. Первый пункт\n2. Второй пункт\n| Вид | Размер |\n| А | 1.5 м |"
    context = "[3] Правила\n" + body
    numbered, records, markers = numbered_context(context)
    assert list(records) == ["3.1", "3.2", "3.3", "3.4"]
    assert [text for _, text in records.values()] == body.splitlines()
    assert markers == ("{", "}")
    assert numbered.startswith("[3] Правила\n{3.1} 1. Первый пункт")
    assert numbered_context(context) == (numbered, records, markers)


def test_short_unit_abbreviation_does_not_split_a_clause():
    text = "Расстояние 5 м. от линии в п. 2.1.2 СП 42.13330.2016 сохраняется."
    assert sentences(text) == [text]


@pytest.mark.parametrize(
    "first,following",
    [
        ("Расстояние должно быть не менее 500 м.", "Доступ обеспечивается."),
        ("500 м.", "Доступ обеспечивается."),
        ("30 мин.", "Доступ обеспечивается."),
        ("1.5 м.", "Доступ обеспечивается."),
        ("1,5 м.", "Доступ обеспечивается."),
        ("Ожидание составляет 30 мин.", "Далее следует проверка."),
        ("Площадь участка составляет 2 га.", "Доступ обеспечивается."),
        ("Доля составляет 15 %.", "Доступ обеспечивается."),
        ("Предусмотрены проезды и т. д.", "Доступ обеспечивается."),
        ("Редакция принята в 2024 г.", "Доступ обеспечивается."),
        ("Расстояние составляет 500 м.", "«Доступ обеспечивается»."),
        ("Расстояние составляет 500 м.", "(Доступ обеспечивается)."),
        ("Расстояние составляет 500 м.", "( «Доступ обеспечивается»)."),
        ("«Расстояние составляет 500 м.»", "Доступ обеспечивается."),
        ("“Расстояние составляет 500 м.”", "Доступ обеспечивается."),
        ("Расстояние составляет 500 м.", "- доступ обеспечивается."),
        ("Расстояние составляет 500 м.", "2) доступ обеспечивается."),
        ("Расстояние составляет 500 м.", "а) доступ обеспечивается."),
    ],
)
def test_abbreviation_can_end_a_sentence(first, following):
    text = first + " " + following
    assert sentences(text) == [first, following]
    assert all(sentence in text for sentence in sentences(text))


@pytest.mark.parametrize(
    "text",
    [
        "По п. 2.1.2 расстояние составляет 1.5 м.",
        "По ст. 51 и табл. 3 предусмотрен доступ.",
        "Учитывается приказ N 111 и СП 42.13330.2016.",
        "Расстояние составляет 500 м. при наличии проезда.",
        "Ожидание составляет 30 мин. после сигнала.",
        "Площадь составляет 2 га. при наличии проезда.",
        "Доля составляет 15 %. при наличии проезда.",
        "Предусмотрены проезды и т. д. при наличии доступа.",
        "Это условие, т. е. правило, сохраняется.",
        "Действует до 2024 г. при наличии доступа.",
        "Указано в г. Москве по п. 2.1.2.",
        "См. СП 42.13330.2016 и ГОСТ Р 123-2024.",
    ],
)
def test_abbreviation_continuations_and_identifiers_stay_intact(text):
    assert sentences(text) == [text]


@pytest.mark.parametrize("ending", ["", "\n", "\nдоступ обеспечивается."])
def test_abbreviation_ends_at_text_or_line_end(ending):
    first = "Расстояние составляет 500 м."
    assert sentences(first + ending) == [first] + (
        [ending.strip()] if ending.strip() else []
    )


def test_wrapped_prose_keeps_a_literal_sentence_and_newlines():
    text = "Расстояние от границы\nдолжно быть не менее 500 м при наличии\nпроезда. Доступ сохраняется."
    assert sentences(text) == [text.split(" Доступ", 1)[0], "Доступ сохраняется."]
    assert all(sentence in text for sentence in sentences(text))


@pytest.mark.parametrize(
    "body,rows",
    [
        ("Вид\tРазмер\nШкола\t500 м", ["Вид\tРазмер", "Школа\t500 м"]),
        (
            "<table><tr><th>Вид</th><th>Размер</th></tr><tr><td>Школа</td><td>500 м</td></tr></table>",
            [
                "<tr><th>Вид</th><th>Размер</th></tr>",
                "<tr><td>Школа</td><td>500 м</td></tr>",
            ],
        ),
    ],
)
def test_table_rows_are_separate_literal_units(body, rows):
    parts = sentences(body)
    assert all(row in parts for row in rows)
    assert all(part in body for part in parts)


def test_ids_are_stable_when_other_sources_are_removed():
    _, all_records, _ = numbered_context(CONTEXT)
    header, body = source_records(CONTEXT)["[2]"]
    _, selected, _ = numbered_context(header + "\n" + body)
    assert selected == {
        sid: value for sid, value in all_records.items() if value[0] == "[2]"
    }


def test_markers_do_not_collide_with_source_text_and_tree_headers_survive():
    context = TREE_PREFIX + "Правила\n  [3] п. 2\nИсходный маркер {3.1}."
    numbered, records, markers = numbered_context(context)
    assert markers == ("{{", "}}")
    assert numbered.startswith(source_records(context)["[3]"][0] + "\n")
    assert records["3.1"] == ("[3]", "Исходный маркер {3.1}.")


@pytest.mark.parametrize("count,enum", [(2, True), (400, True), (401, False)])
def test_schema_uses_bounded_sentence_enum_or_pattern(count, enum):
    schema = CompactAudit.model_json_schema()
    records = {f"1.{i}": ("[1]", "Текст.") for i in range(1, count + 1)}
    sentence_schema(schema, records)
    for name in ("ClaimEvidence", "Correction"):
        fields = schema["$defs"][name]["properties"]
        assert "quote" not in fields
        assert ("enum" in fields["sentences"]["items"]) is enum
        assert fields["sentences"]["maxItems"] == 3
        if not enum:
            assert "pattern" in fields["sentences"]["items"]


def test_sentence_enum_also_has_a_byte_limit():
    schema = CompactAudit.model_json_schema()
    records = {f"{'1' * 100}.{i}": ("[1]", "Текст.") for i in range(1, 100)}
    sentence_schema(schema, records)
    item = schema["$defs"]["ClaimEvidence"]["properties"]["sentences"]["items"]
    assert "enum" not in item and "pattern" in item


def test_sentence_prompt_keeps_the_tuned_hard_rules_verbatim():
    quote_prompt = AnswerCritic._prompt(CONTEXT)
    sentence_prompt = AnswerCritic._prompt(CONTEXT, markers=("{", "}"))
    hard_rules = quote_prompt.split("Hard rejection rules:", 1)[1].split(
        "Corrections:", 1
    )[0]
    assert hard_rules in sentence_prompt
    assert "evidence.quote" not in sentence_prompt
    assert "at most 25 words" not in sentence_prompt
    assert '"sentences": ["1.1"]' in sentence_prompt


async def test_default_schema_and_map_back_keep_exact_literal_evidence():
    assert flags.enabled(flags.CRITIC_SENTENCE_IDS)
    llm = SentenceLlm()
    metrics = DvdRunMetrics("sentences")
    verdict = await AnswerCritic(llm).review(
        "qwen", "Расстояние?", CONTEXT, LINE, metrics=metrics
    )
    assert verdict.satisfied
    [evidence] = verdict.claims[0].evidence
    assert evidence.source_id == "[1]"
    assert evidence.quote == "Расстояние должно быть не менее 500 м."
    assert evidence.quote in source_records(CONTEXT)["[1]"][1]
    schema = llm.calls[0]["format"]
    assert "quote" not in schema["$defs"]["ClaimEvidence"]["properties"]
    assert metrics.decisions["critic_schema_bytes"] == len(
        json.dumps(schema, ensure_ascii=False).encode()
    )
    evidence_set = PartialAnswerEvidence()
    evidence_set.add(verdict.claims, LINE, CONTEXT, AnswerCritic._literal_defects)
    assert len(evidence_set.candidates()) == 1
    assert evidence.quote in evidence_set.render([0])


async def test_flag_off_preserves_the_quote_schema_byte_for_byte(monkeypatch):
    monkeypatch.setenv("DVD_CRITIC_SENTENCE_IDS", "false")

    def quote(data):
        for claim in data["claims"]:
            claim["evidence"] = [{"source_id": "[1]", "quote": "не менее 500 м"}]

    llm = SentenceLlm(quote)
    verdict = await AnswerCritic(llm).review("qwen", "Вопрос", CONTEXT, LINE)
    assert verdict.satisfied
    expected = CompactAudit.model_json_schema()
    expected["$defs"]["LineClaim"]["properties"]["line"]["enum"] = ["L1"]
    expected["$defs"]["ClaimEvidence"]["properties"]["source_id"]["enum"] = [
        "[1]",
        "[2]",
    ]
    expected["$defs"]["Correction"]["properties"]["target"]["enum"] = ["L1", ""]
    expected["$defs"]["Correction"]["properties"]["source_id"]["enum"] = [
        "[1]",
        "[2]",
        "",
    ]
    expected["properties"]["unsupported_claims"]["items"]["enum"] = ["L1"]
    assert json.dumps(llm.calls[0]["format"]) == json.dumps(expected)
    assert "at most 25 words" in llm.calls[0]["messages"][0]["content"]
    assert "{1.1}" not in llm.calls[0]["messages"][1]["content"]


@pytest.mark.parametrize("field", ["evidence", "correction"])
@pytest.mark.parametrize("sid", ["99.1", "2.1"])
async def test_unknown_or_foreign_sentence_uses_the_bounded_retry_path(field, sid):
    def invalid(data):
        item = {"source_id": "[1]", "sentences": [sid]}
        if field == "evidence":
            data["claims"][0]["evidence"] = [item]
        else:
            data["corrections"] = [{"target": "L1", "instruction": "Исправь", **item}]

    llm = SentenceLlm(invalid)
    metrics = DvdRunMetrics("malformed")
    with pytest.raises(CriticResponseError):
        await AnswerCritic(llm).review("qwen", "Вопрос", CONTEXT, LINE, metrics=metrics)
    assert len(llm.calls) == metrics.counters["critic_calls"] == 3
    assert (
        "Твой предыдущий JSON нарушает схему" in llm.calls[1]["messages"][-1]["content"]
    )


def test_each_id_restores_one_evidence_item_and_correction_joins_sentences():
    _, records, _ = numbered_context(CONTEXT)
    data = {
        "claims": [{"evidence": [{"source_id": "[1]", "sentences": ["1.1", "1.2"]}]}],
        "corrections": [{"source_id": "[1]", "sentences": ["1.1", "1.2"]}],
    }
    restored = restore_sentences(deepcopy(data), records)
    assert [e["quote"] for e in restored["claims"][0]["evidence"]] == [
        records[sid][1] for sid in ("1.1", "1.2")
    ]
    assert restored["corrections"][0]["quote"] == " … ".join(
        records[sid][1] for sid in ("1.1", "1.2")
    )
    assert (
        restore_sentences({"corrections": [{"sentences": []}]}, records)["corrections"][
            0
        ]["quote"]
        == ""
    )


@pytest.mark.parametrize(
    "line,source,defect",
    [
        ("Не менее 500 м [3].", "Не менее 500 м.", False),
        ("Не менее 500м [3].", "Не менее 500 м.", False),
        ("Не менее 1 500,50 м [3].", "Не менее 1500.5м.", False),
        ("Не менее 500 м [3].", "Доступ обеспечивается.", True),
        ("Не менее 500 м [3].", "Не более 500 м.", True),
        ("Не более 500 м [3].", "Не менее 500 м.", True),
        ("Не ниже 500 м [3].", "Не более 500 м.", True),
        ("Не выше 500 м [3].", "Не более 500 м.", False),
        ("Не менее 500 м [3].", "Не более 200 м, не менее 500 м.", False),
        ("Не допускается расстояние 500 м [3].", "Допускается расстояние 500 м.", True),
        ("Запрещено расстояние 500 м [3].", "Не допускается расстояние 500 м.", False),
        ("Запрещено расстояние 500 м [3].", "Разрешается расстояние 500 м.", True),
        (
            "СП 2.4.3648-20, п. 2.1.2, ред. от 06.12.2024 [3].",
            "Доступ обеспечивается.",
            False,
        ),
    ],
)
def test_number_and_polarity_check(line, source, defect):
    assert bool(match_defect(line, [source])) is defect


@pytest.mark.parametrize("sid", ["1.2", "1.1"])
async def test_code_rejection_downgrades_and_supplies_reviser_evidence(sid):
    def cite(data):
        data["claims"][0]["evidence"][0]["sentences"] = [sid]

    llm = SentenceLlm(cite)
    metrics = DvdRunMetrics("match")
    context = CONTEXT.replace("не менее 500", "не более 500")
    verdict = await AnswerCritic(llm).review(
        "qwen", "Вопрос", context, LINE, metrics=metrics
    )
    assert not verdict.satisfied and not verdict.needs_evidence
    assert verdict.claims[0].status == "insufficient"
    [correction] = verdict.corrections
    assert correction.target == LINE and correction.source_id == "[1]"
    assert correction.quote == verdict.claims[0].evidence[0].quote
    assert "500" in correction.instruction
    assert metrics.counters["critic_code_rejections"] == 1


@pytest.mark.parametrize(
    "reasons,gpt,expected",
    [
        (("numbers", "obligation", "many_sources"), False, (False, "auto")),
        (("absence_claim",), False, (False, "full")),
        (("previous_rejection",), False, (True, "auto")),
        (("partial_context",), False, (True, "auto")),
        (("document_list",), False, (True, "auto")),
        (("no_citations",), False, (True, "auto")),
        (("long_answer",), False, (True, "auto")),
        (("unclassified",), False, (True, "auto")),
        (
            ("multiple_editions", "conditions", "multi_source_claim"),
            False,
            (False, "auto"),
        ),
        (("numbers",), True, (True, "auto")),
    ],
)
def test_pure_critic_plan(reasons, gpt, expected):
    assert critic_plan(AnswerRisk("high", reasons), gpt) == expected


@pytest.mark.parametrize(
    "reasons,adaptive,calls,tier",
    [
        (("numbers", "obligation", "many_sources"), True, 2, "light"),
        (
            (
                "numbers",
                "obligation",
                "absence_claim",
                "many_sources",
                "multi_source_claim",
                "multiple_documents",
            ),
            True,
            2,
            "light",
        ),
        (("absence_claim",), True, 2, "light"),
        (("previous_rejection",), True, 3, "full"),
        (("numbers",), False, 3, "full"),
    ],
)
async def test_tiers_count_actual_group_and_omissions_requests(
    monkeypatch, reasons, adaptive, calls, tier
):
    monkeypatch.setenv("DVD_ADAPTIVE_CRITIC", str(adaptive))
    llm = SentenceLlm()
    metrics = DvdRunMetrics("tier")
    draft = LINE + "\nРасстояние составляет 500 м [1]."
    verdict = await AnswerCritic(llm).review(
        "qwen",
        "Вопрос",
        CONTEXT,
        draft,
        risk=AnswerRisk("high", reasons),
        metrics=metrics,
    )
    assert verdict.satisfied
    assert len(llm.calls) == metrics.counters["critic_calls"] == calls
    assert metrics.decisions["critic_tier"] == tier
    assert metrics.decisions["critic_omissions"] is (tier == "full")
    payloads = [
        call["messages"][1]["content"].split("allowed_claim_lines", 1)[0]
        for call in llm.calls
    ]
    assert len(set(payloads)) == 1
    if tier == "light":
        assert all(
            call["format"]["properties"]["missing_requirements"]["maxItems"] == 0
            for call in llm.calls
        )


@pytest.mark.parametrize("parallel", ["true", "false"])
async def test_light_single_request_cannot_raise_omissions(monkeypatch, parallel):
    monkeypatch.setenv("DVD_PARALLEL_CRITIC", parallel)
    llm = SentenceLlm()
    await AnswerCritic(llm).review(
        "qwen", "Вопрос", CONTEXT, LINE, risk=AnswerRisk("low")
    )
    assert len(llm.calls) == 1
    assert llm.calls[0]["format"]["properties"]["missing_requirements"]["maxItems"] == 0


async def test_light_rejects_a_model_that_returns_omissions_anyway():
    def omit(data):
        data["missing_requirements"] = ["Другое требование"]

    llm = SentenceLlm(omit)
    with pytest.raises(CriticResponseError):
        await AnswerCritic(llm).review(
            "qwen", "Вопрос", CONTEXT, LINE, risk=AnswerRisk("low")
        )
    assert len(llm.calls) == 3


@pytest.mark.parametrize("policy", [None, "", "auto", "full", "cited"])
@pytest.mark.parametrize("reason", ["numbers", "absence_claim", "previous_rejection"])
@pytest.mark.parametrize("adaptive", [True, False])
@pytest.mark.parametrize("fits", [True, False])
async def test_explicit_context_policy_wins_over_tier(
    service, monkeypatch, policy, reason, adaptive, fits
):
    if policy is None:
        monkeypatch.delenv("DVD_CRITIC_CONTEXT", raising=False)
    else:
        monkeypatch.setenv("DVD_CRITIC_CONTEXT", policy)
    monkeypatch.setenv("DVD_ADAPTIVE_CRITIC", str(adaptive))
    service.context_reducer.fits = AsyncMock(return_value=fits)
    reduced = PreparedContext(
        "partial",
        processed_parts=1,
        failed_parts=["round-1/part-2: [2] (coverage_mismatch)"],
        reduction_rounds=2,
    )
    service.context_reducer.prepare = AsyncMock(return_value=reduced)
    metrics = DvdRunMetrics("context")
    prepared = await service._review_context(
        "qwen", "Вопрос", LINE, CONTEXT, metrics, risk=AnswerRisk("high", (reason,))
    )
    full = policy == "full" or (
        policy not in {"full", "cited"} and adaptive and reason == "absence_claim"
    )
    if not full and policy != "cited" and fits:
        assert prepared.text == CONTEXT
        service.context_reducer.prepare.assert_not_awaited()
        assert metrics.decisions["critic_context"] == "full"
    else:
        assert prepared is reduced
        service.context_reducer.prepare.assert_awaited_once()
        supplied = service.context_reducer.prepare.await_args.args[2]
        assert set(source_records(supplied)) == ({"[1]", "[2]"} if full else {"[1]"})
        assert metrics.decisions["critic_context"] == ("full" if full else "cited")
    if full or policy == "cited":
        service.context_reducer.fits.assert_not_awaited()
    else:
        service.context_reducer.fits.assert_awaited_once()


async def test_configured_reason_forces_full_audit(monkeypatch):
    monkeypatch.setenv("DVD_CRITIC_FULL_AUDIT_REASONS", "numbers")
    llm = SentenceLlm()
    metrics = DvdRunMetrics("configured")
    await AnswerCritic(llm).review(
        "qwen",
        "Вопрос",
        CONTEXT,
        LINE + "\nРасстояние 500 м [1].",
        risk=AnswerRisk("high", ("numbers",)),
        metrics=metrics,
    )
    assert len(llm.calls) == 3 and metrics.decisions["critic_tier"] == "full"
