"""Create a project and its scenario in Urban API from the user's files.

The dialogue settles, in order: which uploads to use (the chat's earlier files are
offered, never taken silently), the Urban API type of every group of objects, the
region, the scenario profile and what to do with the objects already inside the
boundary. Each answer is kept with the dialogue (by chat), so the next message
continues where the question was asked. Then ``ProjectCreator`` makes the calls;
objects that failed are reported and a «да» to the retry question saves only them.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import AsyncGenerator
from datetime import datetime, timezone
from typing import Any

from loguru import logger
from shapely.geometry import shape

from src.agents.api_clients.chat_storage_client.entities import RoleEnum
from src.agents.common.auth.synapse_auth import SynapseCallerVerifier
from src.agents.common.exceptions.base_exceptions import (
    AgentsBaseException,
    AgentsUnauthorizedException,
)
from src.agents.common.exceptions.token_exceptions import TokenExpiredError
from src.agents.dto.project_request_dto import ProjectInputs
from src.agents.services.base_llm_service import BaseLlmService
from src.agents.services.compilance.compliance_entity_mapping import (
    EntityMapper,
    EntityMapping,
    EntityTarget,
    apply_reply,
    mapping_options,
)
from src.agents.services.compilance.compliance_uploads import (
    ComplianceUploadService,
    UserFiles,
)
from src.agents.services.compilance.compliance_user_files import boundary_geometry
from src.agents.services.pipeline_state import PipelineStateStore, PipelineStatus
from src.agents.services.project_files.project_creation import (
    CreationItem,
    ProjectCreator,
    build_items,
    creation_summary,
    describe_error,
)
from src.agents.services.project_files.project_params import (
    EXISTING_POLICIES,
    EXISTING_TITLES,
    ProjectIntent,
    all_regions,
    extract_intent,
    guess_profile,
    match_option,
    project_name,
    regions_for,
    scenario_name,
    yes_no,
)
from src.agents.services.project_files.project_types import (
    PROJECT_CLASSIFY_PROMPT,
    UrbanTypes,
    creation_specs,
    render_type_question,
)
from src.common.service_auth import is_internal_user_context_jwt, user_id_from_jwt

AGENT_ID = "project_files"
_POLYGONS = {"Polygon", "MultiPolygon"}


class ProjectFilesService(BaseLlmService):
    """Dialogue and creation of a project from uploaded files."""

    def __init__(
        self,
        ollama_host: str,
        chat_storage_client,
        urban_api_client,
        state_store: PipelineStateStore,
        uploads: ComplianceUploadService,
        user_token_verifier: SynapseCallerVerifier,
    ) -> None:
        super().__init__(ollama_host, chat_storage_client, urban_api_client)
        self.state_store = state_store
        self.uploads = uploads
        self.user_token_verifier = user_token_verifier
        self.urban = urban_api_client
        self.entity_mapper = EntityMapper(self.llm_client, PROJECT_CLASSIFY_PROMPT)
        self.creator = ProjectCreator(self.urban)

    # ------------------------------------------------------------------
    # Entry point
    # ------------------------------------------------------------------

    async def run_project_pipeline(
        self,
        token: str,
        model: str | None,
        temperature: float,
        user_query: str,
        chat_id: str | None = None,
        request_id: str | None = None,
        persist_history: bool = True,
        conversation_key: str | None = None,
        upload_ids: list[str] | None = None,
        inputs: ProjectInputs | dict | None = None,
    ) -> AsyncGenerator[dict[str, Any], None]:
        request_id = str(request_id or self.state_store.new_request_id())
        async with self.state_store.execution_lock(request_id):
            async for event in self._run_pipeline(
                token,
                model,
                temperature,
                user_query,
                chat_id,
                request_id,
                persist_history,
                conversation_key,
                upload_ids,
                inputs,
            ):
                yield event

    async def _run_pipeline(
        self,
        token,
        model,
        temperature,
        user_query,
        chat_id,
        request_id,
        persist_history,
        conversation_key,
        upload_ids,
        inputs,
    ):
        # Verify before reading/replaying state, fetching history or calling Urban
        # API. Internal contexts are created only after resolve_a2a_caller verifies
        # Synapse; HTTP dependencies reject them at the edge.
        try:
            owner = (
                user_id_from_jwt(token)
                if is_internal_user_context_jwt(token)
                else await self.user_token_verifier.verify_user(token)
            )
        except (AgentsUnauthorizedException, ValueError):
            yield self._text(
                "Не удалось подтвердить токен пользователя, поэтому проект не создан. "
                "Войдите заново и повторите запрос."
            )
            return
        if await self.state_store.exists(request_id):
            # A reconnect replays the stream; the dialogue goes on by chat.
            for event in await self.state_store.get_buffered_events(request_id):
                yield event
            return
        inputs = ProjectInputs.model_validate(inputs or {})
        model = await self.resolve_model(model)
        await self.state_store.create(
            request_id,
            chat_id=chat_id,
            user_query=user_query,
            scenario_id=None,
            model=model,
            temperature=temperature,
        )
        yield await self._buf(
            request_id, "pipeline_started", {"request_id": request_id}
        )

        history: list[dict] = []
        if chat_id:
            try:
                chat = await self.get_chat_messages(token, chat_id)
                history = self.build_llm_history(
                    chat.messages, current_user_query=user_query
                )
                if persist_history:
                    await self.add_single_message(
                        token, chat_id, RoleEnum.USER, user_query
                    )
            except Exception:
                logger.warning("Project chat history is unavailable")
        elif persist_history:
            try:
                chat_id, title = await self.create_chat(
                    token,
                    model,
                    user_query,
                    additional_instructions=(
                        "Создание проекта и сценария в Urban API из файлов "
                        "пользователя."
                    ),
                    resolve_project_id=False,
                    agent_id=AGENT_ID,
                )
                await self.state_store.set_chat_id(request_id, chat_id)
                yield await self._buf(
                    request_id,
                    "service_event",
                    {
                        "event_type": "storage_event",
                        "event": {
                            "storage_event_type": "chat_created",
                            "chat_id": chat_id,
                            "chat_title": title,
                        },
                    },
                )
            except Exception:
                logger.warning("Project could not create chat history")

        key = conversation_key or chat_id
        answer: list[str] = []
        try:
            async for event in self._dialogue(
                token, model, user_query, key, upload_ids, inputs, history, owner
            ):
                if event["type"] == "chunk":
                    answer.append(event["content"].get("text", ""))
                elif event["type"] == "clarification":
                    answer.append(event["content"].get("question") or "")
                yield await self._buf(request_id, event["type"], event["content"])
            await self.state_store.set_status(request_id, PipelineStatus.DONE)
        except Exception as exc:
            logger.opt(exception=exc).error("Project creation from files failed")
            await self.state_store.set_status(request_id, PipelineStatus.FAILED)
            dialogue = await self.state_store.get_project_dialogue(key) if key else None
            resumable = bool(dialogue and dialogue.get("stage") == "retry")
            yield await self._buf(
                request_id,
                "error",
                {"message": self._failure_text(exc, resumable), "traceback": ""},
            )
            return
        if persist_history and chat_id and answer:
            try:
                await self.add_single_message(
                    token, chat_id, RoleEnum.ASSISTANT, "\n\n".join(answer)
                )
            except Exception:
                yield await self._buf(
                    request_id,
                    "warning",
                    {
                        "code": "history_unavailable",
                        "message": "Ответ не удалось сохранить в истории.",
                    },
                )

    # ------------------------------------------------------------------
    # Dialogue
    # ------------------------------------------------------------------

    async def _dialogue(
        self,
        token: str,
        model: str,
        user_query: str,
        key: str | None,
        upload_ids: list[str] | None,
        inputs: ProjectInputs,
        history: list[dict],
        owner: str,
    ) -> AsyncGenerator[dict[str, Any], None]:
        pending = await self.state_store.get_project_dialogue(key) if key else None
        context: dict[str, Any] | None = None
        if pending:
            context = pending["context"]
            saved_owners = (
                context.get("owner"),
                (context.get("run") or {}).get("owner"),
            )
            if any(
                saved_owner and saved_owner != owner for saved_owner in saved_owners
            ):
                yield self._text(
                    "Этот запуск создания проекта принадлежит другому пользователю, "
                    "продолжить его нельзя."
                )
                return
            context["owner"] = owner
            # Parameters sent with the answer override the ones of the request.
            context["inputs"] = {
                **(context.get("inputs") or {}),
                **inputs.model_dump(mode="json", exclude_none=True),
            }
            outcome = await self._answer(model, user_query, pending, upload_ids, owner)
            if outcome == "unresolved":
                yield self._clarification(
                    "Не удалось разобрать ответ. " + pending["question"],
                    pending["stage"],
                    pending.get("options") or [],
                )
                return
            if outcome == "declined":
                await self.state_store.set_project_dialogue(key, None)
                yield self._text(self._declined_text(pending["stage"], context))
                return
            if outcome == "new_request":
                await self.state_store.set_project_dialogue(key, None)
                context = None
        if context is None:
            context = {
                "owner": owner,
                "user_query": user_query,
                "requested_uploads": list(dict.fromkeys(upload_ids or [])),
                "inputs": inputs.model_dump(mode="json"),
            }
        async for event in self._advance(token, model, owner, key, context, history):
            yield event

    async def _answer(
        self,
        model: str,
        reply: str,
        pending: dict[str, Any],
        upload_ids: list[str] | None,
        owner: str,
    ) -> str:
        """Fold the reply into the context: ``answered``, ``declined``,
        ``unresolved`` or ``new_request``."""
        stage = pending["stage"]
        context = pending["context"]
        options = pending.get("options") or []
        if stage == "reuse_files":
            requested = list(dict.fromkeys(upload_ids or []))
            if requested:
                context["requested_uploads"] = requested
                context["upload_ids"] = requested
                return "answered"
            choice = self._choice(reply, options)
            if choice is None:
                return "unresolved"
            if choice == "none":
                return "declined"
            chosen = list(context.get("requested_uploads") or [])
            if choice == "all":
                chosen += pending.get("offered") or []
            context["upload_ids"] = list(dict.fromkeys(chosen))
            return "answered"
        if stage == "entity_mapping":
            mapping = EntityMapping.from_dict(pending["mapping"])
            candidates = [
                target
                for target in (
                    EntityTarget.from_dict(item) for item in pending["candidates"]
                )
                if target is not None
            ]
            answer = await self.entity_mapper.resolve_reply(
                model, reply, mapping.pending, candidates
            )
            if answer.kind in {"unresolved", "new_request"}:
                return answer.kind
            try:
                await self.uploads.confirm_mapping(
                    context["upload_ids"],
                    owner,
                    apply_reply(mapping, answer),
                    namespace="project",
                )
            except AgentsBaseException:
                return "new_request"  # the uploads expired
            return "answered"
        if stage == "region":
            option = match_option(reply, options)
            if option is None:
                return "unresolved"
            context["region"] = {
                "territory_id": int(option["value"]),
                "name": option["label"],
            }
            return "answered"
        if stage == "profile":
            option = match_option(reply, options)
            if option is None:
                return "unresolved"
            context["profile"] = {"name": option["value"], "label": option["label"]}
            return "answered"
        if stage == "existing_objects":
            option = match_option(reply, options)
            if option is None:
                return "unresolved"
            context["existing_policy"] = option["value"]
            return "answered"
        if stage == "retry":
            decision = yes_no(reply)
            if decision is None:
                return "new_request"
            return "answered" if decision else "declined"
        return "new_request"

    async def answers_dialogue(self, model: str, key: str, reply: str) -> bool:
        """Whether a message answers the question pending in the dialogue."""
        pending = await self.state_store.get_project_dialogue(key)
        if not pending:
            return False
        options = pending.get("options") or []
        stage = pending["stage"]
        if stage == "reuse_files":
            return self._choice(reply, options) is not None
        if stage == "entity_mapping":
            mapping = EntityMapping.from_dict(pending["mapping"])
            answer = await self.entity_mapper.resolve_reply(
                model,
                reply,
                mapping.pending,
                [
                    target
                    for target in (
                        EntityTarget.from_dict(item) for item in pending["candidates"]
                    )
                    if target is not None
                ],
            )
            return answer.kind in {"accept", "reject", "edit"}
        if stage == "retry":
            return yes_no(reply) is not None
        return match_option(reply, options) is not None

    @staticmethod
    def _choice(reply: str, options: list[dict[str, Any]]) -> str | None:
        decision = yes_no(reply)
        if decision is not None:
            return options[0]["value"] if decision else options[-1]["value"]
        option = match_option(reply, options)
        return option["value"] if option else None

    async def _advance(
        self,
        token: str,
        model: str,
        owner: str,
        key: str | None,
        context: dict[str, Any],
        history: list[dict],
    ) -> AsyncGenerator[dict[str, Any], None]:
        inputs = ProjectInputs.model_validate(context.get("inputs") or {})

        # 1. The uploads.
        if context.get("upload_ids") is None:
            requested = context.get("requested_uploads") or []
            earlier = [
                upload_id
                for upload_id in (await self._alive_uploads(key, owner) if key else [])
                if upload_id not in requested
            ]
            if earlier:
                described = await self._describe_uploads(earlier, owner)
                if requested:
                    question = (
                        "Кроме новых файлов, в этом чате уже загружены: "
                        f"{described}. Добавить их в проект?"
                    )
                    options = [
                        {"value": "all", "label": "Да, новые и прежние файлы"},
                        {"value": "new", "label": "Нет, только новые файлы"},
                    ]
                else:
                    question = (
                        f"В этом чате уже загружены файлы: {described}. Создать "
                        "проект по ним? Ответьте «да» или загрузите новые файлы."
                    )
                    options = [
                        {"value": "all", "label": "Да, использовать их"},
                        {"value": "none", "label": "Нет, загружу новые"},
                    ]
                async for event in self._ask(
                    key, context, "reuse_files", question, options, offered=earlier
                ):
                    yield event
                return
            if not requested:
                yield self._text(
                    "Для создания проекта загрузите файлы проекта (GeoJSON, ZIP с "
                    "GeoJSON или Shapefile, GeoPackage) через POST /compliance/uploads "
                    "и передайте их upload_ids."
                )
                return
            context["upload_ids"] = requested
        if key:
            await self.state_store.add_chat_uploads(key, context["upload_ids"])

        # 2. The files: only the design and its boundary become the project.
        files = await self.uploads.load(
            context["upload_ids"], owner, namespace="project"
        )
        project_files = UserFiles(
            upload_ids=files.upload_ids,
            layers=[layer for layer in files.layers if layer.role != "context"],
            confirmed_mapping=files.confirmed_mapping,
            classified=files.classified,
        )
        if not project_files.has_project:
            yield self._text(
                "В файлах нет объектов проекта: все слои определены как окружение или "
                "граница. Укажите роли слоёв (PATCH /compliance/uploads/{upload_id}) "
                "и повторите запрос."
            )
            return
        boundary = boundary_geometry(project_files)
        if boundary is None or boundary.get("type") not in _POLYGONS:
            yield self._text(
                "Не удалось определить границу проекта: загрузите слой границы "
                "(полигон) или объекты, охватывающие площадь."
            )
            return

        # 3. The Urban API type of every group of objects.
        yield self._status("mapping", "Сопоставляю объекты файлов с типами Urban API…")
        types = await self._types(token)
        candidates = types.targets()
        mapping = await self.entity_mapper.propose(
            model,
            project_files,
            candidates,
            sources=[],
            llm_types=("physical_object",),
        )
        await self.uploads.cache_classifications(
            project_files.upload_ids, owner, mapping.classified, namespace="project"
        )
        yield {
            "type": "user_files",
            "content": {
                "upload_ids": list(project_files.upload_ids),
                "layers": files.summary(),
                "mapping": [item.to_dict() for item in mapping.proposals],
                "pending": [item.signature for item in mapping.pending],
            },
        }
        if mapping.pending:
            async for event in self._ask(
                key,
                context,
                "entity_mapping",
                render_type_question(mapping.pending),
                mapping_options(mapping.pending),
                mapping=mapping.to_dict(),
                candidates=[target.to_dict() for target in candidates],
            ):
                yield event
            return
        specs, problems = creation_specs(mapping, types)
        zones, objects, rejected = build_items(project_files, specs)
        if not zones and not objects:
            yield self._text(
                "Ни один объект файлов не сопоставлен с типами Urban API — проект не "
                "создан. Сопоставление показано в событии user_files."
            )
            return

        # 4. The names and the explicit choices of the request.
        if "intent" not in context:
            context["intent"] = (
                await extract_intent(
                    self.llm_client,
                    model,
                    context["user_query"],
                    types.profiles(),
                    history,
                )
            ).model_dump(mode="json")
        intent = ProjectIntent.model_validate(context["intent"])

        # 5. The region.
        if context.get("region") is None:
            region = await self._region(token, inputs, boundary)
            if isinstance(region, dict):
                context["region"] = region
            else:
                question, options = region
                async for event in self._ask(key, context, "region", question, options):
                    yield event
                return

        # 6. The profile of the scenario.
        if context.get("profile") is None:
            profile = self._profile(inputs, intent, project_files, specs, types)
            if profile is not None:
                context["profile"] = profile
            else:
                _, reason = guess_profile(project_files, specs, types)
                options = [
                    {"value": zone.name, "label": zone.label}
                    for zone in types.profiles()
                ]
                async for event in self._ask(
                    key,
                    context,
                    "profile",
                    "Не удалось определить профиль сценария "
                    f"({reason}). Выберите профиль: "
                    + "; ".join(
                        f"{number}. {option['label']}"
                        for number, option in enumerate(options, start=1)
                    ),
                    options,
                ):
                    yield event
                return

        # 7. The objects already inside the boundary.
        if context.get("existing_policy") is None:
            policy = inputs.existing_objects or intent.existing_objects
            if policy is None:
                existing = await self._existing_objects(token, boundary)
                if existing or zones:
                    question = self._existing_question(existing, bool(zones))
                    options = [
                        {"value": value, "label": EXISTING_TITLES[value]}
                        for value in EXISTING_POLICIES
                    ]
                    async for event in self._ask(
                        key, context, "existing_objects", question, options
                    ):
                        yield event
                    return
                policy = "keep"
            context["existing_policy"] = policy

        # 8. The creation.
        files_names = sorted({layer.name for layer in project_files.layers})
        run = context.setdefault(
            "run",
            {
                "owner": owner,
                "project_name": project_name(
                    project_files, inputs.project_name or intent.project_name
                ),
                "scenario_name": scenario_name(
                    inputs.scenario_name or intent.scenario_name
                ),
                "region_id": context["region"]["territory_id"],
                "region_name": context["region"].get("name"),
                "profile_id": types.zone_type(context["profile"]["name"]).id,
                "profile_name": context["profile"].get("label")
                or context["profile"]["name"],
                "existing_policy": context["existing_policy"],
                "boundary": boundary,
                "files": files_names,
                "year": datetime.now(timezone.utc).year,
            },
        )

        async def save() -> None:
            if key:
                await self.state_store.set_project_dialogue(
                    key,
                    {
                        "stage": "retry",
                        "question": "Повторить загрузку несохранённых объектов?",
                        "options": [],
                        "context": context,
                    },
                )

        await save()
        async for event in self.creator.create(
            token, run, zones, objects, save, owner=owner
        ):
            yield event
        summary = creation_summary(run, zones, objects)
        summary["not_created"] = self._not_created(mapping, specs, problems, rejected)
        yield {"type": "project_summary", "content": summary}
        yield self._text(self._summary_text(summary))
        if summary["failed_total"] or summary["zones"]["error"] or not summary["ready"]:
            async for event in self._ask(
                key,
                context,
                "retry",
                "Повторить загрузку несохранённых объектов и проверку доступности проекта? "
                "Ответьте «да» или «нет».",
                [
                    {"value": "yes", "label": "Да, повторить"},
                    {"value": "no", "label": "Нет"},
                ],
            ):
                yield event
            return
        if key:
            await self.state_store.set_project_dialogue(key, None)

    # ------------------------------------------------------------------
    # Steps
    # ------------------------------------------------------------------

    async def _types(self, token: str) -> UrbanTypes:
        return UrbanTypes.from_dictionaries(
            await self.urban.get_physical_object_types(token),
            await self.urban.get_service_types(token),
            await self.urban.get_functional_zone_types(token),
        )

    async def _alive_uploads(self, key: str, owner: str) -> list[str]:
        alive = []
        for upload_id in await self.state_store.get_chat_uploads(key):
            try:
                await self.uploads.get(upload_id, owner)
            except AgentsBaseException:
                continue
            alive.append(upload_id)
        return alive

    async def _describe_uploads(self, upload_ids: list[str], owner: str) -> str:
        names: list[str] = []
        for upload_id in upload_ids:
            record = await self.uploads.get(upload_id, owner)
            names.extend(record.get("files") or [upload_id])
        return ", ".join(f"«{name}»" for name in names)

    async def _region(
        self, token: str, inputs: ProjectInputs, boundary: dict
    ) -> dict[str, Any] | tuple[str, list[dict[str, Any]]]:
        if inputs.territory_id:
            return {"territory_id": inputs.territory_id, "name": None}
        regions = await regions_for(self.urban, token, boundary)
        if len(regions) == 1:
            return regions[0]
        if regions:
            options = [
                {"value": item["territory_id"], "label": item["name"]}
                for item in regions
            ]
            question = "Граница проекта лежит в нескольких регионах. Выберите регион: "
        else:
            options = [
                {"value": item["territory_id"], "label": item["name"]}
                for item in sorted(
                    await all_regions(self.urban, token),
                    key=lambda item: item["name"],
                )
            ]
            question = (
                "Не удалось определить регион по границе проекта. Напишите название "
                "региона или его номер в списке: "
            )
        return (
            question
            + "; ".join(
                f"{number}. {option['label']}"
                for number, option in enumerate(options, start=1)
            ),
            options,
        )

    @staticmethod
    def _profile(
        inputs: ProjectInputs,
        intent: ProjectIntent,
        files: UserFiles,
        specs: dict,
        types: UrbanTypes,
    ) -> dict[str, Any] | None:
        if inputs.functional_zone_type_id:
            zone = next(
                (
                    zone
                    for zone in types.zones.values()
                    if zone.id == inputs.functional_zone_type_id
                ),
                None,
            )
            if zone is not None:
                return {"name": zone.name, "label": zone.label}
        zone = types.zone_type(intent.profile) if intent.profile else None
        if zone is None:
            zone, _ = guess_profile(files, specs, types)
        return {"name": zone.name, "label": zone.label} if zone else None

    async def _existing_objects(self, token: str, boundary: dict) -> Counter[str]:
        border = shape(boundary)
        counts: Counter[str] = Counter()
        try:
            objects = await self.urban.get_physical_objects_around(token, boundary)
        except Exception as exc:
            logger.warning(f"Objects inside the project boundary are unknown: {exc!r}")
            return Counter({"": 1})
        seen = set()
        for item in objects or []:
            if item.get("physical_object_id") in seen:
                continue
            try:
                geometry = shape(item["geometry"])
            except (ValueError, TypeError, KeyError, AttributeError):
                continue
            if geometry.is_empty or not border.contains(
                geometry.representative_point()
            ):
                continue
            seen.add(item.get("physical_object_id"))
            counts[(item.get("physical_object_type") or {}).get("name") or ""] += 1
        return counts

    @staticmethod
    def _existing_question(existing: Counter[str], has_zones: bool) -> str:
        known = {name: count for name, count in existing.items() if name}
        if known:
            found = "; ".join(
                f"{name} — {count}"
                for name, count in sorted(known.items(), key=lambda item: -item[1])
            )
            head = f"Внутри границы проекта уже есть объекты: {found}."
        elif existing:
            head = "Не удалось проверить, есть ли объекты внутри границы проекта."
        else:
            head = "Внутри границы проекта объектов не найдено."
        zones = (
            " Функциональные зоны сценария при замене или удалении заменяются зонами "
            "файлов."
            if has_zones
            else ""
        )
        return (
            f"{head}{zones} Что с ними сделать в новом сценарии? "
            "1. Заменить объекты тех же типов, что в файлах; 2. удалить все "
            "существующие объекты внутри границы; 3. оставить, а объекты файлов "
            "добавить поверх."
        )

    async def _ask(
        self,
        key: str | None,
        context: dict[str, Any],
        stage: str,
        question: str,
        options: list[dict[str, Any]],
        **extra: Any,
    ) -> AsyncGenerator[dict[str, Any], None]:
        if key:
            await self.state_store.set_project_dialogue(
                key,
                {
                    "stage": stage,
                    "question": question,
                    "options": options,
                    "context": context,
                    **extra,
                },
            )
        else:
            question += (
                "\n\nОтвет не сохранится: запрос выполнен без чата. Повторите его с "
                "chat_id или задайте параметр в inputs."
            )
        yield self._clarification(question, stage, options)

    # ------------------------------------------------------------------
    # Texts
    # ------------------------------------------------------------------

    @staticmethod
    def _not_created(
        mapping: EntityMapping,
        specs: dict,
        problems: dict[str, str],
        rejected: dict[str, int],
    ) -> list[dict[str, Any]]:
        rows = []
        for item in mapping.proposals:
            if not item.roles.get("project") or item.signature in specs:
                continue
            reason = problems.get(item.signature) or (
                "не сопоставлено с типом Urban API"
                if item.source != "table"
                else "служебный объект генератора"
            )
            if item.source == "user" and item.target is None:
                reason = "исключено пользователем"
            rows.append({"label": item.label, "count": item.count, "reason": reason})
        rows += [
            {"label": reason, "count": count, "reason": reason}
            for reason, count in rejected.items()
        ]
        return rows

    @staticmethod
    def _summary_text(summary: dict[str, Any]) -> str:
        lines = []
        region = summary["region"].get("name")
        lines.append(
            f"Проект «{summary['project_name']}» создан в Urban API "
            f"(project_id {summary['project_id']}"
            + (f", регион — {region}" if region else "")
            + ")."
        )
        lines.append(
            f"Сценарий «{summary['scenario_name']}» (scenario_id "
            f"{summary['scenario_id']}, профиль — {summary['profile']}) создан из "
            f"базового сценария {summary['base_scenario_id']}."
        )
        if summary.get("ready"):
            lines.append(
                "Проект готов: доступность территории, сценария и сохранённых объектов подтверждена."
            )
        else:
            lines.append(
                "Доступность созданного проекта не подтверждена за время ожидания. "
                f"Причина: {summary.get('ready_error') or 'проверка не завершена'}."
            )
        if summary["deleted_existing"]:
            lines.append(
                "Удалено существующих объектов внутри границы: "
                f"{summary['deleted_existing']}."
            )
        if summary.get("locked_existing"):
            lines.append(
                "Не удалены закреплённые в Urban API объекты (is_locked): "
                f"{summary['locked_existing']} — они остаются в сценарии."
            )
        zones = summary["zones"]
        if zones["total"]:
            lines.append(
                f"Функциональные зоны: {zones['saved']} из {zones['total']}"
                + (f" (ошибка: {zones['error']})" if zones["error"] else "")
                + "."
            )
        objects = summary["objects"]
        rows = "; ".join(
            f"{name} — {row['saved']}"
            + (f" из {row['total']}" if row["saved"] != row["total"] else "")
            for name, row in sorted(
                objects["by_type"].items(), key=lambda item: -item[1]["total"]
            )
        )
        lines.append(
            f"Объекты: сохранено {objects['saved']} из {objects['total']}"
            + (f" ({rows})" if rows else "")
            + "."
        )
        if objects["services"]:
            lines.append(
                "Сервисы: "
                + "; ".join(
                    f"{name} — {count}" for name, count in objects["services"].items()
                )
                + "."
            )
        not_created = summary.get("not_created") or []
        if not_created:
            lines.append(
                "Не загружено: "
                + "; ".join(
                    f"{row['label']} — {row['count']} ({row['reason']})"
                    for row in not_created
                )
                + "."
            )
        if summary["failed_total"]:
            reasons = Counter(item["error"] for item in summary["failed"])
            lines.append(
                f"Не сохранено объектов: {summary['failed_total']}. Причины: "
                + "; ".join(f"{error} — {count}" for error, count in reasons.items())
                + "."
            )
        if summary["cleanup_errors"]:
            lines.append(
                "Не удалось удалить часть существующих объектов: "
                + "; ".join(summary["cleanup_errors"][:3])
                + "."
            )
        return "\n\n".join(lines)

    @staticmethod
    def _declined_text(stage: str, context: dict[str, Any]) -> str:
        if stage == "reuse_files":
            return (
                "Хорошо, прежние файлы не используются. Загрузите новые файлы через "
                "POST /compliance/uploads и повторите запрос с их upload_ids."
            )
        if stage == "retry":
            run = context.get("run") or {}
            return (
                "Хорошо, повторной загрузки не будет. Проект "
                f"{run.get('project_id')} и сценарий {run.get('scenario_id')} "
                "остаются в Urban API с уже сохранёнными объектами."
            )
        return "Хорошо, создание проекта отменено."

    @staticmethod
    def _failure_text(exc: Exception, resumable: bool) -> str:
        if isinstance(exc, TokenExpiredError):
            reason = "истёк срок действия токена"
        elif isinstance(exc, AgentsBaseException):
            reason = describe_error(exc)
        else:
            reason = "внутренняя ошибка"
        text = f"Не удалось создать проект из файлов: {reason}."
        if resumable:
            text += (
                " Уже созданное сохранено; ответьте «да» в этом чате, чтобы "
                "продолжить с места остановки."
            )
        return text

    # ------------------------------------------------------------------
    # Events
    # ------------------------------------------------------------------

    async def _buf(self, request_id: str, kind: str, content: dict) -> dict:
        event = {"type": kind, "content": content}
        await self.state_store.buffer_event(request_id, event)
        return event

    @staticmethod
    def _status(stage: str, text: str) -> dict[str, Any]:
        return {
            "type": "status",
            "content": {"status": f"project_{stage}", "text": text},
        }

    @staticmethod
    def _text(text: str) -> dict[str, Any]:
        return {"type": "chunk", "content": {"text": text, "done": True}}

    @staticmethod
    def _clarification(
        question: str, kind: str, options: list[dict[str, Any]]
    ) -> dict[str, Any]:
        return {
            "type": "clarification",
            "content": {"question": question, "kind": kind, "options": options},
        }


__all__ = ["ProjectFilesService", "CreationItem"]
