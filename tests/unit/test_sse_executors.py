from src.agents.common.executors.sse_executors import stream_with_error_handling


class FakeRequest:
    method = "GET"
    url = "http://test/scenario-data/qa/stream"
    client = None
    query_params = {}
    headers = {}

    @staticmethod
    async def is_disconnected():
        return False


class ForbiddenErrorExplainer:
    async def execute_request(self, *args, **kwargs):
        raise AssertionError("the fallback must not ask an LLM to explain an error")


async def test_stream_error_is_deterministic_and_does_not_expose_traceback():
    async def failing_pipeline(**kwargs):
        del kwargs
        if False:
            yield {}
        raise ValueError("private traceback details")

    events = [
        event
        async for event in stream_with_error_handling(
            failing_pipeline,
            FakeRequest(),
            ForbiddenErrorExplainer(),
            "model",
            rerun=False,
        )
    ]

    assert events == [
        {
            "type": "chunk",
            "content": {
                "text": (
                    "Не удалось выполнить запрос из-за внутренней ошибки сервера. "
                    "Повторите попытку позже."
                ),
                "done": False,
            },
        },
        {
            "type": "error",
            "content": {
                "message": "Internal stream exception",
                "traceback": "",
            },
        },
        {"type": "chunk", "content": {"text": "", "done": True}},
    ]
    assert "private traceback details" not in repr(events)


async def test_model_failure_is_reported_as_a_model_error():
    from src.agents.model_clients.llm_base import LlmResponseError

    async def failing_pipeline(**kwargs):
        del kwargs
        if False:
            yield {}
        raise LlmResponseError(
            "Model did not produce a complete structured answer after a bounded retry"
        )

    events = [
        event
        async for event in stream_with_error_handling(
            failing_pipeline,
            FakeRequest(),
            ForbiddenErrorExplainer(),
            "model",
            rerun=False,
        )
    ]
    assert events[0]["content"]["text"].startswith("Модель не смогла")
    assert events[1]["content"] == {"message": "Model response error", "traceback": ""}
    assert "structured answer" not in repr(events)
