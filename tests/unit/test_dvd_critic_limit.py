"""A critic audit stops at a fixed output cap instead of reasoning without end."""

import pytest

from src.agents.services.dvd import dvd_reasoning, literal_audit
from src.agents.services.dvd.dvd_reasoning import AnswerCritic, critic_output_cap
from src.agents.services.dvd.retry_policy import CriticBudgetError
from tests.helpers import FakeDvdMcpClient, FakeLlmClient, answer_text, plan_json
from tests.unit.test_dvd_answer_revision import ACCESS, SOURCE, audit
from tests.unit.test_dvd_rag_service import _run

LENGTH = object()
INVENTED = "Ширина пандуса не менее 1,5 м [1]."


class CappedLlm(FakeLlmClient):
    """A reply queued as ``LENGTH`` stops at the output limit."""

    def __init__(self):
        super().__init__()
        self.efforts = []

    async def chat(self, model=None, messages=None, options=None, stream=False, **kw):
        if not stream:
            self.efforts.append(kw.get("reasoning_effort"))
            if self.json_responses and self.json_responses[0] is LENGTH:
                self.json_responses[0] = "{}"
                await super().chat(model, messages, options, stream, **kw)
                return {"done_reason": "length", "message": {"content": ""}}
        return await super().chat(model, messages, options, stream, **kw)


@pytest.fixture
def fake_llm():
    return CappedLlm()


@pytest.fixture
def medium(monkeypatch):
    monkeypatch.setattr(
        dvd_reasoning, "critic_reasoning_effort", lambda *a, **k: "medium"
    )


@pytest.mark.parametrize("lines, cap", [(1, 4096), (5, 4096), (20, 7000), (100, 12000)])
def test_cap_follows_the_answer_not_the_evidence(lines, cap):
    assert critic_output_cap(lines) == cap


def test_cap_ceiling_is_configurable(monkeypatch):
    monkeypatch.setenv("DVD_CRITIC_MAX_OUTPUT_TOKENS", "6000")
    assert critic_output_cap(100) == 6000


async def test_capped_audit_is_repeated_once_at_low_effort(medium):
    llm = CappedLlm()
    llm.json_responses = [
        LENGTH,
        audit([(ACCESS, "supported", SOURCE[-50:])], satisfied=True),
    ]
    verdict = await AnswerCritic(llm).review(
        "m", "вопрос", f"[1] СП 257\n{SOURCE}", ACCESS
    )
    assert verdict.satisfied
    assert llm.efforts == ["medium", "low"]
    # Never a wider limit: both audits share one cap.
    assert {c.options["num_predict"] for c in llm.chat_calls} == {critic_output_cap(1)}


async def test_audit_that_never_finishes_is_a_budget_error(medium):
    llm = CappedLlm()
    llm.json_responses = [LENGTH, LENGTH, LENGTH]
    with pytest.raises(CriticBudgetError):
        await AnswerCritic(llm).review("m", "вопрос", f"[1] СП 257\n{SOURCE}", ACCESS)
    assert llm.efforts == ["medium", "low"]


@pytest.mark.parametrize(
    "line, status",
    [
        (ACCESS, "supported"),
        # A figure the fragment does not contain.
        (INVENTED, "insufficient"),
        # Wording the fragment does not contain.
        ("Парковки размещают у главного входа [1].", "insufficient"),
        ("В гостиницах обеспечивается доступ для МГН [7].", "insufficient"),
        ("В гостиницах обеспечивается доступ для МГН.", "insufficient"),
    ],
)
def test_literal_audit_errs_towards_rejection(line, status):
    [claim] = literal_audit.audit([line], f"[1] СП 257\n{SOURCE}")
    assert claim.status == status


async def test_answer_survives_a_critic_that_never_finishes(service, fake_llm, medium):
    llm = fake_llm
    client = FakeDvdMcpClient(hits_per_call=[[{"name": "СП 257", "text": SOURCE}]])
    llm.json_responses = [plan_json(), LENGTH, LENGTH]
    llm.answer_texts = [f"- {ACCESS}\n- {INVENTED}"]
    events = await _run(service, client)
    assert answer_text(events) == f"- {ACCESS}"
    assert llm.efforts[-2:] == ["medium", "low"]


async def test_strict_mode_still_fails_on_a_critic_that_never_finishes(
    service, fake_llm, medium, monkeypatch
):
    monkeypatch.setenv("DVD_KNOWLEDGE_FALLBACK", "false")
    llm = fake_llm
    client = FakeDvdMcpClient(hits_per_call=[[{"name": "СП 257", "text": SOURCE}]])
    llm.json_responses = [plan_json(), LENGTH, LENGTH]
    llm.answer_texts = [ACCESS]
    # The run fails as before; the producer reports it as an error event.
    with pytest.raises(CriticBudgetError):
        await _run(service, client)
