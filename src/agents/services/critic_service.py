from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Literal

from loguru import logger

from src.agents.model_clients.base_client import BaseLlmClient

_CRITIC_PROMPT = """
Ты — критик качества ответов геопространственного агента.

Агент: {agent_name}
Задача пользователя: {user_query}

Ответ агента:
{response_text}

Оцени, насколько ответ агента решает задачу пользователя.
Ответь строго в формате JSON без каких-либо пояснений:
{{
  "quality": "good" | "poor",
  "feedback": "краткое объяснение оценки на русском языке",
  "refined_query": null | "уточнённый запрос для повторного выполнения — только если quality=poor"
}}

Критерии оценки «good»:
  - Ответ прямо отвечает на поставленную задачу
  - Упоминаются конкретные объекты, параметры и расстояния из запроса
  - Нет очевидных несоответствий или пропущенных требований

Оценка «poor» только если:
  - Ответ не отвечает на поставленный вопрос
  - Ключевые параметры (объекты, расстояния, районы) проигнорированы
  - Содержится очевидная ошибка в интерпретации задачи

Если quality="poor", в refined_query сформулируй уточнённый запрос, устраняющий выявленные проблемы.
Если quality="good", refined_query должен быть null.
""".strip()


@dataclass
class CriticVerdict:
    """
    Result of a critic agent evaluation.

    Attributes:
        quality: ``"good"`` if the sub-pipeline response adequately answers the
            user's query, ``"poor"`` if a significant issue was detected.
        feedback: Human-readable explanation of the verdict (Russian).
        refined_query: A corrected / more specific query to pass to the sub-pipeline
            on retry. Only set when ``quality == "poor"``; ``None`` otherwise.
    """

    quality: Literal["good", "poor"]
    feedback: str
    refined_query: str | None = None

    @property
    def needs_retry(self) -> bool:
        """True when the sub-pipeline should be re-run with ``refined_query``."""
        return self.quality == "poor" and bool(self.refined_query)


class CriticService(BaseLlmClient):
    """
    LLM-based critic that evaluates the text output of a sub-pipeline.

    Called once per sub-pipeline after its final ``chunk done:true`` event.
    If the verdict is ``"poor"`` and a ``refined_query`` is produced, the
    caller (OrchestratorPipelineService) re-runs the sub-pipeline with the
    refined query.  At most ``MAX_RETRIES`` automatic retries are allowed per
    sub-pipeline to prevent infinite loops.

    Falls back to ``quality="good"`` on any LLM or parsing error so the
    pipeline always continues.

    Attributes:
        MAX_RETRIES (int): Maximum number of automatic retries per sub-pipeline.
        host (str): Ollama host URL (from BaseLlmClient).
        llm_client: Async Ollama client (from BaseLlmClient).
    """

    MAX_RETRIES: int = 1

    async def evaluate(
        self,
        *,
        user_query: str,
        response_text: str,
        agent_name: str,
        model: str,
    ) -> CriticVerdict:
        """
        Evaluate the text output of a sub-pipeline against the user's query.

        Args:
            user_query (str): The (possibly decomposed) query that was sent to
                the sub-pipeline.
            response_text (str): Concatenated text from all ``chunk`` events
                produced by the sub-pipeline.
            agent_name (str): Human-readable sub-pipeline name for the prompt
                (e.g. ``"restriction-creation-agent"``).
            model (str): Ollama model name.
        Returns:
            CriticVerdict: Verdict with quality, feedback, and optional
                ``refined_query``.  Always returns a valid verdict — falls back
                to ``quality="good"`` on any error.
        """
        if not response_text.strip():
            logger.warning(
                f"CriticService: empty response from {agent_name}, marking as poor."
            )
            return CriticVerdict(
                quality="poor",
                feedback="Агент не вернул текстового ответа.",
                refined_query=user_query,
            )

        prompt = _CRITIC_PROMPT.format(
            agent_name=agent_name,
            user_query=user_query,
            response_text=response_text,
        )
        try:
            response = await self.llm_client.generate(
                model=model, prompt=prompt, stream=False
            )
            raw = response.response.strip()
            raw = re.sub(r"```(?:json)?", "", raw).strip().rstrip("`")
            data = json.loads(raw)

            quality = data.get("quality", "good")
            if quality not in ("good", "poor"):
                quality = "good"

            refined_query = data.get("refined_query") or None
            # Defensively ignore refined_query when quality is reported as good
            if quality == "good":
                refined_query = None

            verdict = CriticVerdict(
                quality=quality,
                feedback=str(data.get("feedback", "")),
                refined_query=refined_query,
            )
            logger.info(
                f"CriticService: {agent_name} → quality={verdict.quality}, "
                f"needs_retry={verdict.needs_retry}"
            )
            return verdict

        except Exception as exc:
            logger.warning(
                f"CriticService: evaluation failed for {agent_name} ({exc}). "
                "Assuming quality=good."
            )
            return CriticVerdict(
                quality="good",
                feedback="Автоматическая оценка недоступна.",
                refined_query=None,
            )
