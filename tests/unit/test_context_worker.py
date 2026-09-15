from types import MethodType

from src.agents.workers.context_worker import ContextContent, ContextWorker


class FakeApi:
    def __init__(self):
        self.after_values = []

    async def get(self, endpoint, **kwargs):
        after = (kwargs.get("params") or {}).get("after_seq")
        self.after_values.append(after)
        if after is None:
            return {
                "content": {"summary": "base", "structured": {}},
                "tail": [{"seq": 1, "role": "user", "parts": []}],
                "tail_has_more": True,
                "tail_next_after_seq": 1,
            }
        return {
            "content": {"summary": "base", "structured": {}},
            "tail": [{"seq": 2, "role": "assistant", "parts": []}],
            "tail_has_more": False,
            "tail_next_after_seq": None,
        }


async def test_context_worker_folds_all_tail_pages():
    worker = ContextWorker.__new__(ContextWorker)
    worker.worker_id = "worker"
    worker.api = FakeApi()
    worker.headers = {}

    async def summarize(self, job, previous, messages):
        return ContextContent(
            summary=previous.summary + f"/{messages[0]['seq']}", structured={}
        )

    worker._summarize = MethodType(summarize, worker)
    content = await worker._summarize_job(
        {"job_id": "job", "target_seq": 2, "model": "gpt-oss-20b"}
    )

    assert content.summary == "base/1/2"
    assert worker.api.after_values == [None, 1]


async def test_document_summary_uses_remaining_window_and_retains_document_intent(
    fake_llm,
):
    import json

    worker = ContextWorker.__new__(ContextWorker)
    worker.llm = fake_llm
    fake_llm.json_responses = [
        json.dumps({"summary": "СП 55, выбран раздел 3", "structured": {}})
    ]
    result = await worker._summarize(
        {"model": "gpt-oss-20b", "target_seq": 2, "prompt_version": "documents-v1"},
        ContextContent(summary="", structured={}),
        [
            {
                "seq": 1,
                "role": "user",
                "parts": [{"kind": "text", "payload": {"text": "СП 55 пункт 3"}}],
            }
        ],
    )
    call = fake_llm.chat_calls[-1]
    assert call.options["num_ctx"] == 32000
    assert 6000 < call.options["num_predict"] < 32000
    assert "исходный вопрос" in call.messages[0]["content"]
    assert "СП 55" in result.summary
