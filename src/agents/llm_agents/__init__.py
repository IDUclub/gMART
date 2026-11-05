"""Agent-framework layer of gMART.

Two frameworks share the work, each where it is strongest:

* **pydantic-ai** runs every typed LLM step inside the pipelines (plans, mappings,
  answers, critics): :class:`~src.agents.llm_agents.stage.LlmStage` wraps a
  ``pydantic_ai.Agent`` with a typed ``output_type``, output validation and retries.
* **Google ADK** runs the autonomous orchestrator, which picks the gMART sub-agents
  as tools (:mod:`src.agents.llm_agents.adk`).

Both talk to the model through the same :class:`BaseLlmAdapter` the services were
built with, so the tuned vLLM/Ollama transport (request priority, pacing, the
``think`` spelling, bounded retries of truncated JSON) is shared, and the tests and
benchmarks that stand in for that adapter keep working unchanged.
"""

import os

# pydantic-ai prints an observability banner on first run; it has no place in a
# service log.
os.environ.setdefault("PYDANTIC_AI_NO_BANNER", "1")
