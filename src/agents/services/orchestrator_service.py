from __future__ import annotations

import json
import re
from dataclasses import dataclass, field

from loguru import logger

from src.agents.model_clients.base_client import BaseLlmClient

_CLASSIFICATION_PROMPT = """
Ты — классификатор запросов геопространственной системы городского планирования.
Определи, какие агенты нужны для обработки запроса пользователя.

Доступные агенты:
- restriction: строит зоны строительных ограничений (буферы, радиусы вокруг объектов)
- provision: рассчитывает эффекты обеспеченности населения инфраструктурными сервисами

Ответь строго в формате JSON без каких-либо пояснений:
{"restriction": true/false, "provision": true/false}

Примеры:
Запрос: "Зона 200 метров вокруг школ"
Ответ: {"restriction": true, "provision": false}

Запрос: "Эффекты обеспеченности детскими садами"
Ответ: {"restriction": false, "provision": true}

Запрос: "Ограничения вокруг больниц и их влияние на обеспеченность"
Ответ: {"restriction": true, "provision": true}

Запрос: "Привет, как дела?"
Ответ: {"restriction": false, "provision": false}
""".strip()

_DECOMPOSITION_PROMPT = """
Ты — декомпозитор составных запросов геопространственной системы городского планирования.

Пользователь задал запрос, требующий работы двух агентов одновременно:
  • restriction-creation-agent — строит зоны строительных ограничений
    (буферы и радиусы вокруг объектов инфраструктуры)
  • provision-effects-agent   — рассчитывает эффекты обеспеченности населения
    инфраструктурными сервисами

Твоя задача — разбить исходный запрос на два самодостаточных подзапроса.
Каждый подзапрос должен:
  – содержать все необходимые параметры и объекты именно для своего агента;
  – не содержать информации, относящейся к другому агенту;
  – быть сформулирован как самостоятельное законченное задание.

Ответь строго в формате JSON без каких-либо пояснений:
{"restriction_query": "...", "provision_query": "..."}

Примеры:
Запрос: "Построй зону 300 м вокруг школ и рассчитай эффекты обеспеченности детскими садами"
Ответ: {"restriction_query": "Построй зону ограничений 300 метров вокруг школ", "provision_query": "Рассчитай эффекты обеспеченности населения детскими садами"}

Запрос: "Ограничения вокруг больниц 500 м и влияние на обеспеченность поликлиниками в Северном районе"
Ответ: {"restriction_query": "Построй зону ограничений 500 метров вокруг больниц", "provision_query": "Рассчитай эффекты обеспеченности поликлиниками в Северном районе"}
""".strip()


@dataclass
class OrchestratorIntent:
    """
    Result of orchestrator intent classification and optional query decomposition.

    Attributes:
        needs_restriction (bool): Whether to invoke restriction-creation-agent.
        needs_provision (bool): Whether to invoke provision-effects-agent.
        restriction_query (str | None): Focused sub-query for the restriction agent.
            Set only when the original request is compound (both agents needed).
            ``None`` means use the original user query unchanged.
        provision_query (str | None): Focused sub-query for the provision agent.
            Set only when the original request is compound (both agents needed).
            ``None`` means use the original user query unchanged.
    """

    needs_restriction: bool
    needs_provision: bool
    restriction_query: str | None = field(default=None)
    provision_query: str | None = field(default=None)

    @property
    def is_empty(self) -> bool:
        """True if no sub-agent should be invoked."""
        return not self.needs_restriction and not self.needs_provision

    @property
    def is_compound(self) -> bool:
        """True if both sub-agents are needed — decomposition is required."""
        return self.needs_restriction and self.needs_provision


class OrchestratorService(BaseLlmClient):
    """
    LLM-based intent classifier and query decomposer for the orchestrator agent.

    Two distinct LLM calls are used:
    1. ``classify_intent`` — fast binary routing decision (always called).
    2. ``decompose_query`` — focused sub-query extraction (called only for compound
       requests where both agents are needed, to avoid passing irrelevant context
       to each sub-pipeline).

    Falls back gracefully on any LLM or parsing error.

    Attributes:
        host (str): Ollama host URL.
        llm_client (AsyncOllamaClient): Async Ollama client instance.
    """

    async def classify_intent(
        self,
        user_query: str,
        model: str,
    ) -> OrchestratorIntent:
        """
        Classify user query intent via LLM.

        Args:
            user_query (str): Cleaned user query text.
            model (str): Ollama model name.
        Returns:
            OrchestratorIntent: Intent flags for restriction and provision agents.
                ``restriction_query`` and ``provision_query`` are always ``None``
                here; call ``decompose_query`` separately if ``is_compound`` is True.
        """
        prompt = f"{_CLASSIFICATION_PROMPT}\n\nЗапрос: {user_query}\nОтвет:"
        try:
            response = await self.llm_client.generate(
                model=model, prompt=prompt, stream=False
            )
            raw = response.response.strip()
            # Strip markdown code fences if the model wraps its output
            raw = re.sub(r"```(?:json)?", "", raw).strip().rstrip("`")
            data = json.loads(raw)
            return OrchestratorIntent(
                needs_restriction=bool(data.get("restriction", False)),
                needs_provision=bool(data.get("provision", False)),
            )
        except Exception as exc:
            logger.warning(
                f"OrchestratorService: intent classification failed ({exc}). "
                "Defaulting to restriction-only."
            )
            return OrchestratorIntent(needs_restriction=True, needs_provision=False)

    async def decompose_query(
        self,
        user_query: str,
        model: str,
    ) -> tuple[str, str]:
        """
        Decompose a compound query into focused sub-queries for each sub-agent.

        Called only when ``classify_intent`` returns ``is_compound == True``.
        Each sub-query contains only the parameters relevant to its agent so that
        neither sub-pipeline wastes tokens processing context meant for the other.

        Args:
            user_query (str): Original user query that addresses both agents.
            model (str): Ollama model name.
        Returns:
            tuple[str, str]: ``(restriction_query, provision_query)``.
                Falls back to ``(user_query, user_query)`` on any error so the
                pipeline can always continue.
        """
        prompt = f"{_DECOMPOSITION_PROMPT}\n\nЗапрос: {user_query}\nОтвет:"
        try:
            response = await self.llm_client.generate(
                model=model, prompt=prompt, stream=False
            )
            raw = response.response.strip()
            raw = re.sub(r"```(?:json)?", "", raw).strip().rstrip("`")
            data = json.loads(raw)
            restriction_query = str(data.get("restriction_query") or user_query)
            provision_query = str(data.get("provision_query") or user_query)
            logger.info(
                f"OrchestratorService: decomposed query — "
                f"restriction={restriction_query!r}, provision={provision_query!r}"
            )
            return restriction_query, provision_query
        except Exception as exc:
            logger.warning(
                f"OrchestratorService: query decomposition failed ({exc}). "
                "Falling back to original query for both agents."
            )
            return user_query, user_query
