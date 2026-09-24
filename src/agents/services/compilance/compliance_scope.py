"""Which norms a compliance check covers: topic entities and documents.

A request such as «проверь нормы по школам из СП 42» names a topic and a document.
The LLM only reads the request: it lists the topics and the documents as the user
named them, and picks the NormGraph entities that mean a topic out of the candidates
NormGraph returned.  Documents are matched in code: a designation written in the
request («СП 42.13330») selects a document on its own only when it matches exactly
one NormGraph document; anything else becomes a numbered choice that the user
answers in the next message of the same conversation.
"""

from __future__ import annotations

import json
import re
import unicodedata
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Literal

from loguru import logger
from pydantic import BaseModel, Field

from src.agents.services.dvd.document_reference import parse_reference
from src.agents.services.normgraph.normgraph_reasoning import _request_json

if TYPE_CHECKING:
    from src.agents.mcp_clients.normgraph_mcp_client import NormGraphMcpClient

MAX_TOPICS = 5
MAX_CHOICE_OPTIONS = 10
# Documents per listing; NormGraph caps a response at 500 rows.
DOCUMENT_POOL_LIMIT = 500
ENTITY_CANDIDATES_PER_TOPIC = 10
_SHORT_REPLY_WORDS = 8

_ORDINALS = {
    "первый": 1,
    "первая": 1,
    "первое": 1,
    "второй": 2,
    "вторая": 2,
    "второе": 2,
    "третий": 3,
    "третья": 3,
    "третье": 3,
    "четвертый": 4,
    "четвертая": 4,
    "пятый": 5,
    "пятая": 5,
    "шестой": 6,
    "седьмой": 7,
    "восьмой": 8,
    "девятый": 9,
    "десятый": 10,
}
_ALL_OPTIONS = re.compile(
    r"^(?:все|всё|все документы|все варианты|все перечисленные|по всем|оба|обе)$"
)
_NUMBERS_REPLY = re.compile(
    r"^(?:(?:вариант|варианты|номер|номера|документ|документы|№)\s*)?"
    r"\d+(?:\s*(?:,|;|и|-|–|—)\s*\d+)*(?:\s*(?:вариант\w*|документ\w*))?$"
)
_FLEXION = re.compile(r"[аеиоуыэюяйь]+$")
_TOKEN = re.compile(r"[^\W_]+")


def normalized(text: str) -> str:
    return " ".join(unicodedata.normalize("NFKC", text).casefold().split()).replace(
        "ё", "е"
    )


def designates(reference: str, name: str) -> bool:
    """Whether a written designation («СП 42.13330») names the document ``name``.

    Spacing is ignored and a trailing number must not continue («СП 42» names
    «СП 42.13330.2016», never «СП 421»).
    """

    compact = re.sub(r"\s+", "", normalized(reference))
    if not compact:
        return False
    pattern = r"(?<!\w)" + r"\s*".join(map(re.escape, compact)) + r"(?!\d)"
    return re.search(pattern, normalized(name)) is not None


def _stems(text: str) -> set[str]:
    stems = set()
    for token in _TOKEN.findall(normalized(text)):
        if token.isdigit():
            stems.add(token)
        elif len(token) >= 3:
            stem = _FLEXION.sub("", token) if len(token) > 3 else token
            stems.add(stem if len(stem) >= 3 else token)
    return stems


def _overlap(reference: str, name: str) -> int:
    name_stems = _stems(name)
    return sum(
        any(a.startswith(b) or b.startswith(a) for b in name_stems)
        for a in _stems(reference)
    )


@dataclass(frozen=True)
class ComplianceScope:
    """Resolved filters of one compliance run; empty means the full corpus."""

    topics: tuple[str, ...] = ()
    entities: tuple[str, ...] = ()
    documents: tuple[str, ...] = ()

    @property
    def is_filtered(self) -> bool:
        return bool(self.entities or self.documents)

    def filters(self) -> dict[str, Any]:
        filters: dict[str, Any] = {}
        if self.entities:
            filters["entities"] = list(self.entities)
        if self.documents:
            filters["document_names"] = list(self.documents)
        return filters

    def label(self) -> str:
        parts = []
        if self.topics:
            parts.append("темы: " + ", ".join(f"«{topic}»" for topic in self.topics))
        if self.documents:
            parts.append("документы: " + ", ".join(self.documents))
        return "; ".join(parts)

    def to_dict(self) -> dict[str, Any]:
        return {
            "topics": list(self.topics),
            "entities": list(self.entities),
            "documents": list(self.documents),
            "label": self.label(),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> "ComplianceScope":
        data = data or {}
        return cls(
            topics=tuple(data.get("topics") or ()),
            entities=tuple(data.get("entities") or ()),
            documents=tuple(data.get("documents") or ()),
        )


@dataclass(frozen=True)
class ScopeOutcome:
    """``scoped`` runs the check; ``choice`` waits for the user; ``empty`` stops."""

    kind: Literal["scoped", "choice", "empty"]
    scope: ComplianceScope = field(default_factory=ComplianceScope)
    message: str | None = None
    # The pending document choice, persisted until the user's next message.
    choice: dict[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "kind": self.kind,
            "scope": self.scope.to_dict(),
            "message": self.message,
            "choice": self.choice,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ScopeOutcome":
        return cls(
            kind=data["kind"],
            scope=ComplianceScope.from_dict(data.get("scope")),
            message=data.get("message"),
            choice=data.get("choice"),
        )


@dataclass(frozen=True)
class ChoiceReply:
    """How the user's message answers a pending document choice."""

    kind: Literal["selected", "unresolved", "not_choice"]
    documents: tuple[str, ...] = ()


class ScopeRequest(BaseModel):
    topics: list[str] = Field(default_factory=list)
    documents: list[str] = Field(default_factory=list)


class TopicEntities(BaseModel):
    topic: str
    entities: list[str] = Field(default_factory=list)


class EntitySelection(BaseModel):
    selections: list[TopicEntities] = Field(default_factory=list)


class ChoiceAnswer(BaseModel):
    is_choice: bool = False
    numbers: list[int] = Field(default_factory=list)


def clarification_options(choice: dict[str, Any]) -> list[dict[str, Any]]:
    return [
        {
            "number": number,
            "label": _option_label(candidate),
            "value": candidate["name"],
        }
        for number, candidate in enumerate(choice.get("candidates") or [], start=1)
    ]


def _option_label(candidate: dict[str, Any]) -> str:
    return f"{candidate['name']} — исполнимых норм: {candidate['executable_count']}"


def render_choice(choice: dict[str, Any]) -> str:
    topics = choice.get("topics") or []
    about = " по теме " + ", ".join(f"«{topic}»" for topic in topics) if topics else ""
    references = ", ".join(f"«{ref}»" for ref in choice.get("references") or [])
    if choice.get("matched"):
        header = (
            f"Не удалось однозначно определить документ {references}. "
            f"Выберите документ для проверки норм{about}:"
        )
    else:
        header = (
            f"Документ {references} не найден среди документов с исполнимыми "
            f"нормами{about}. Выберите документ из доступных:"
        )
    lines = [header, ""]
    lines += [
        f"{option['number']}. {option['label']}"
        for option in clarification_options(choice)
    ]
    lines += [
        "",
        "Ответьте номером (например, «1» или «1, 3»), словом «все» "
        "или названием документа.",
    ]
    return "\n".join(lines)


class ComplianceScopeResolver:
    """Resolve a compliance request into NormGraph filters or a document choice."""

    def __init__(self, llm_client) -> None:
        self.llm_client = llm_client

    async def resolve(
        self,
        client: "NormGraphMcpClient",
        model: str,
        user_query: str,
        history: list[dict] | None = None,
    ) -> ScopeOutcome:
        request = await self._extract(model, user_query, history)
        topics = _unique(request.topics)[:MAX_TOPICS]
        exact_refs = parse_reference(user_query).document_names
        described = [
            ref
            for ref in _unique(request.documents)
            if not any(
                designates(ref, exact) or designates(exact, ref) for exact in exact_refs
            )
        ]
        if not topics and not exact_refs and not described:
            return ScopeOutcome(kind="scoped")

        entities: list[str] = []
        if topics:
            by_topic = await self._topic_entities(client, model, user_query, topics)
            missing = [topic for topic in topics if not by_topic.get(topic)]
            if missing:
                return ScopeOutcome(
                    kind="empty",
                    message=(
                        "Проверка не выполнена: в графе норм нет объектов, "
                        "соответствующих теме "
                        + ", ".join(f"«{topic}»" for topic in missing)
                        + ". Уточните тему — например, назовите вид объектов "
                        "так, как он называется в нормах."
                    ),
                )
            entities = sorted({key for keys in by_topic.values() for key in keys})
        scope = ComplianceScope(topics=tuple(topics), entities=tuple(entities))
        if not exact_refs and not described:
            return ScopeOutcome(kind="scoped", scope=scope)
        return await self._resolve_documents(
            client, user_query, scope, exact_refs, described
        )

    async def resolve_choice(
        self, model: str, reply: str, choice: dict[str, Any]
    ) -> ChoiceReply:
        candidates = [item["name"] for item in choice.get("candidates") or []]
        text = normalized(reply).strip(" .!")
        if not candidates or not text:
            return ChoiceReply(kind="not_choice")
        if _ALL_OPTIONS.match(text):
            return ChoiceReply(kind="selected", documents=tuple(candidates))
        numbers = _reply_numbers(text)
        if numbers is not None:
            return _by_numbers(numbers, candidates)
        # A long message naming a document is more likely a new request about it;
        # only a short reply is taken as a pick by name without asking the model.
        if len(text.split()) <= _SHORT_REPLY_WORDS:
            references = parse_reference(reply).document_names
            # Repeating a designation that fits several options («СП 42» for two
            # editions) does not choose between them.
            if any(
                sum(designates(ref, name) for name in candidates) > 1
                for ref in references
            ):
                return ChoiceReply(kind="unresolved")
            named = [
                name
                for name in candidates
                if normalized(name) in text
                or any(designates(ref, name) for ref in references)
            ]
            if named:
                return ChoiceReply(kind="selected", documents=tuple(named))
        answer = await _request_json(
            self.llm_client,
            model,
            [
                {"role": "system", "content": _CHOICE_PROMPT},
                {
                    "role": "user",
                    "content": json.dumps(
                        {
                            "options": {
                                str(number): name
                                for number, name in enumerate(candidates, start=1)
                            },
                            "reply": reply,
                        },
                        ensure_ascii=False,
                    ),
                },
            ],
            ChoiceAnswer,
        )
        if not answer.is_choice:
            return ChoiceReply(kind="not_choice")
        return _by_numbers(answer.numbers, candidates)

    async def empty_scope_message(
        self, client: "NormGraphMcpClient", scope: ComplianceScope, found: int
    ) -> str:
        """Explain a scoped check with no executable norm and name where they exist."""

        lines = [
            f"Проверка не выполнена: по условиям ({scope.label()}) нет норм "
            "с исполнимым планом проверки. "
            f"Найдено норм по условиям: {found}, исполнимых из них: 0."
        ]
        documents = await client.list_restriction_documents(
            executable_only=True,
            limit=MAX_CHOICE_OPTIONS,
            **({"entities": list(scope.entities)} if scope.entities else {}),
        )
        about = (
            " по теме " + ", ".join(f"«{topic}»" for topic in scope.topics)
            if scope.topics
            else ""
        )
        if documents:
            lines += ["", f"Исполнимые нормы{about} есть в документах:"]
            lines += [
                f"- {item['name']} — {item['executable_count']}"
                for item in documents
                if item.get("name")
            ]
        elif scope.topics:
            lines += [
                "",
                f"Исполнимых норм{about} нет ни в одном документе графа норм.",
            ]
        return "\n".join(lines)

    async def _extract(
        self, model: str, user_query: str, history: list[dict] | None
    ) -> ScopeRequest:
        request = await _request_json(
            self.llm_client,
            model,
            [
                {"role": "system", "content": _SCOPE_PROMPT},
                *(history or [])[-6:],
                {"role": "user", "content": user_query},
            ],
            ScopeRequest,
        )
        logger.info(f"Compliance scope request: {request.model_dump_json()}")
        return request

    async def _topic_entities(
        self,
        client: "NormGraphMcpClient",
        model: str,
        user_query: str,
        topics: list[str],
    ) -> dict[str, list[str]]:
        resolutions = await client.resolve_entities(
            topics, limit=ENTITY_CANDIDATES_PER_TOPIC
        )
        candidates = {
            str(item.get("term")): [
                candidate
                for candidate in item.get("candidates") or []
                if candidate.get("normalized")
            ]
            for item in resolutions
        }
        offered = {
            topic: {c["normalized"] for c in candidates.get(topic, [])}
            for topic in topics
        }
        if not any(offered.values()):
            return {}
        selection = await _request_json(
            self.llm_client,
            model,
            [
                {"role": "system", "content": _ENTITY_PROMPT},
                {
                    "role": "user",
                    "content": json.dumps(
                        {
                            "request": user_query,
                            "topics": [
                                {
                                    "topic": topic,
                                    "candidates": [
                                        {
                                            "entity": c["normalized"],
                                            "aliases": (c.get("aliases") or [])[:5],
                                            "norms": c.get("restriction_count", 0),
                                            "executable_norms": c.get(
                                                "executable_count", 0
                                            ),
                                        }
                                        for c in candidates.get(topic, [])
                                    ],
                                }
                                for topic in topics
                            ],
                        },
                        ensure_ascii=False,
                    ),
                },
            ],
            EntitySelection,
        )
        chosen = {item.topic: item.entities for item in selection.selections}
        by_topic: dict[str, list[str]] = {}
        for topic in topics:
            # The model may only narrow NormGraph's candidates, never invent names.
            picked = [key for key in chosen.get(topic, []) if key in offered[topic]]
            if not picked:
                # Its own name (or alias) always means the topic.
                picked = [
                    c["normalized"]
                    for c in candidates.get(topic, [])
                    if c.get("match") in {"exact", "alias"}
                ]
            by_topic[topic] = sorted(set(picked))
        logger.info(f"Compliance topic entities: {by_topic}")
        return by_topic

    async def _resolve_documents(
        self,
        client: "NormGraphMcpClient",
        user_query: str,
        scope: ComplianceScope,
        exact_refs: list[str],
        described: list[str],
    ) -> ScopeOutcome:
        topic_filter = {"entities": list(scope.entities)} if scope.entities else {}
        pool = await client.list_restriction_documents(
            executable_only=True, limit=DOCUMENT_POOL_LIMIT, **topic_filter
        )
        executable = {
            item["name"]: int(item.get("executable_count") or 0)
            for item in pool
            if item.get("name")
        }
        catalogue = [
            item["name"]
            for item in await client.list_restriction_documents(
                limit=DOCUMENT_POOL_LIMIT
            )
            if item.get("name")
        ]

        selected: list[str] = []
        ambiguous: list[str] = []
        unresolved: list[str] = list(described)
        for ref in exact_refs:
            matches = [name for name in catalogue if designates(ref, name)]
            if len(matches) == 1:
                selected.append(matches[0])
            elif matches:
                ambiguous.extend(matches)
                unresolved.append(ref)
            else:
                unresolved.append(ref)
        if not unresolved:
            return ScopeOutcome(
                kind="scoped",
                scope=ComplianceScope(
                    topics=scope.topics,
                    entities=scope.entities,
                    documents=tuple(_unique(selected)),
                ),
            )

        ranked = list(dict.fromkeys(ambiguous))
        scored = sorted(
            (
                (max(_overlap(ref, name) for ref in unresolved), count, name)
                for name, count in executable.items()
                if name not in ranked and name not in selected
            ),
            key=lambda row: (-row[0], -row[1], row[2]),
        )
        matched = bool(ranked) or any(score > 0 for score, _, _ in scored)
        ranked += [name for score, _, name in scored if score > 0 or not matched]
        candidates = [
            {"name": name, "executable_count": executable.get(name, 0)}
            for name in ranked[:MAX_CHOICE_OPTIONS]
        ]
        if not candidates:
            about = (
                " по теме " + ", ".join(f"«{t}»" for t in scope.topics)
                if scope.topics
                else ""
            )
            return ScopeOutcome(
                kind="empty",
                scope=scope,
                message=(
                    f"Проверка не выполнена: документов с исполнимыми нормами{about} "
                    "в графе норм нет."
                ),
            )
        choice = {
            "query": user_query,
            "topics": list(scope.topics),
            "entities": list(scope.entities),
            "documents": _unique(selected),
            "references": unresolved,
            "matched": matched,
            "candidates": candidates,
        }
        return ScopeOutcome(
            kind="choice", scope=scope, message=render_choice(choice), choice=choice
        )


def scope_for_choice(
    choice: dict[str, Any], documents: tuple[str, ...]
) -> ComplianceScope:
    """The scope of the original request once the user picked its documents."""

    return ComplianceScope(
        topics=tuple(choice.get("topics") or ()),
        entities=tuple(choice.get("entities") or ()),
        documents=tuple(_unique([*(choice.get("documents") or []), *documents])),
    )


def _unique(values: list[str]) -> list[str]:
    seen: dict[str, str] = {}
    for value in values:
        text = " ".join(str(value).split())
        if text and normalized(text) not in seen:
            seen[normalized(text)] = text
    return list(seen.values())


def _reply_numbers(text: str) -> list[int] | None:
    if _NUMBERS_REPLY.match(text):
        numbers: list[int] = []
        for start, end in re.findall(r"(\d+)(?:\s*[-–—]\s*(\d+))?", text):
            first = int(start)
            last = int(end) if end else first
            numbers.extend(range(first, max(first, last) + 1))
        return numbers
    words = [word for word in re.split(r"[\s,;]+|\bи\b", text) if word]
    words = [word for word in words if word not in {"вариант", "документ", "и"}]
    if words and all(word in _ORDINALS for word in words):
        return [_ORDINALS[word] for word in words]
    return None


def _by_numbers(numbers: list[int], candidates: list[str]) -> ChoiceReply:
    if not numbers or any(not 0 < number <= len(candidates) for number in numbers):
        return ChoiceReply(kind="unresolved")
    return ChoiceReply(
        kind="selected",
        documents=tuple(dict.fromkeys(candidates[number - 1] for number in numbers)),
    )


_SCOPE_PROMPT = f"""Ты разбираешь запрос на проверку объектов сценария на соответствие \
градостроительным нормам. Определи, чем пользователь ограничивает проверку. Верни только \
валидный JSON без markdown и пояснений:
{json.dumps({"topics": ["тема"], "documents": ["документ"]}, ensure_ascii=False)}

Правила:
- topics — виды объектов или застройки, нормы о которых нужно проверить: «школа», \
«жилой дом», «детский сад», «жилая застройка», «автозаправочная станция». Пиши каждую \
тему в именительном падеже единственного числа, без слов «нормы», «ограничения», \
«требования». Не включай в topics сценарий, территорию, проект, сами слова «нормы» \
или «соответствие». Если пользователь проверяет все нормы или тему не называет — [].
- documents — нормативные документы, которыми пользователь ограничивает проверку, так, \
как он их назвал: обозначение («СП 42.13330», «СанПиН 2.2.1/2.1.1.1200-03») или \
описание («свод правил по планировке городов»). Не придумывай обозначения, которых нет \
в запросе, и не добавляй документы от себя. Если документ не назван — [].
- Если текущее сообщение уточняет предыдущий запрос из диалога («а теперь только по \
школам»), учитывай предыдущий запрос.
"""

_ENTITY_PROMPT = """Ты сопоставляешь темы проверки нормативного соответствия с сущностями \
графа норм. Для каждой темы из "topics" выбери из её "candidates" те сущности, которые \
обозначают тот же вид объектов, что и тема: синонимы, другие формы названия и более \
узкие разновидности (для «школа» — «общеобразовательная школа», «общеобразовательная \
организация»). Не выбирай смежные и другие объекты: для «школа» не подходят «детский \
сад», «жилой дом», «территория школы», если тема не про неё. Выбирай только значения \
"entity" из кандидатов этой темы, без изменений. Если подходящих нет — пустой список.
Верни только валидный JSON без markdown и пояснений:
{"selections": [{"topic": "тема", "entities": ["сущность"]}]}
"""

_CHOICE_PROMPT = """Пользователю предложили выбрать документы из нумерованного списка \
"options". Определи, отвечает ли его сообщение "reply" на этот выбор. is_choice = true, \
если сообщение выбирает варианты (по номеру, названию, описанию или «последний»), и \
numbers — номера выбранных вариантов. is_choice = false, если это новый вопрос или \
просьба, не связанная с выбором. Верни только валидный JSON без markdown и пояснений:
{"is_choice": true, "numbers": [1]}
"""
