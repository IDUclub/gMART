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
