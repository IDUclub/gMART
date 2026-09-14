"""Opt-in GPT-OSS completion transport with local, strict Harmony parsing.

Only an output carrier is advertised. Domain tools are never dispatched here.
"""

from functools import lru_cache

from openai.types.chat import ChatCompletion
from openai_harmony import (
    Conversation,
    DeveloperContent,
    HarmonyEncodingName,
    Message,
    ReasoningEffort,
    Role,
    SystemContent,
    ToolDescription,
    load_harmony_encoding,
)

from .llm_base import LlmResponseError


@lru_cache(maxsize=1)
def encoding():
    return load_harmony_encoding(HarmonyEncodingName.HARMONY_GPT_OSS)


async def create_harmony_completion(client, call, schema):
    if "gpt-oss" not in call["model"].lower():
        raise LlmResponseError("harmony_completion requires a GPT-OSS model", 400)
    enc = encoding()
    instructions = (
        "Return the complete requested JSON object by calling functions.emit_structured_response. "
        "This is an output-format function. Operation names in the input are data values, "
        "not callable functions. Do not call any other native function.\n"
    )
    messages = []
    for item in call["messages"]:
        if not isinstance(item.get("content"), str):
            raise LlmResponseError("Harmony transport accepts text messages only", 400)
        if item["role"] in {"system", "developer"}:
            instructions += item["content"] + "\n"
        elif item["role"] in {"user", "assistant"}:
            messages.append(
                Message.from_role_and_content(Role(item["role"]), item["content"])
            )
        else:
            raise LlmResponseError(
                "Native tool messages are not supported by this transport", 400
            )
    effort = call.get("reasoning_effort", "low")
    if effort not in {"low", "medium", "high"}:
        effort = "low"
    context = [
        Message.from_role_and_content(
            Role.SYSTEM,
            SystemContent.new().with_reasoning_effort(ReasoningEffort[effort.upper()]),
        ),
        Message.from_role_and_content(
            Role.DEVELOPER,
            DeveloperContent(instructions=instructions).with_function_tools(
                [
                    ToolDescription(
                        name="emit_structured_response",
                        description="Return the structured decision requested by the input.",
                        parameters=schema["schema"],
                    )
                ]
            ),
        ),
        *messages,
    ]
    payload = {
        "model": call["model"],
        "prompt": enc.render_conversation_for_completion(
            Conversation.from_messages(context), Role.ASSISTANT
        ),
        "max_tokens": call.get("max_tokens", 2048),
        "extra_body": {
            "stop_token_ids": enc.stop_tokens_for_assistant_actions(),
            "return_token_ids": True,
            "skip_special_tokens": False,
        },
    }
    for key in ("temperature", "top_p", "seed"):
        if key in call:
            payload[key] = call[key]
    result = await client.completions.create(**payload)
    data = result.model_dump()
    choice = data["choices"][0]
    content = ""
    if choice["finish_reason"] != "length":
        tokens = choice.get("token_ids")
        if not tokens:
            raise LlmResponseError(
                "Completion server must return Harmony token IDs", 502
            )
        try:
            parsed = enc.parse_messages_from_completion_tokens(
                tokens, role=Role.ASSISTANT
            )
        except Exception as exc:
            raise LlmResponseError("Invalid Harmony completion envelope", 502) from exc
        outputs = [m for m in parsed if m.channel != "analysis" or m.recipient]
        if len(outputs) != 1 or not (
            outputs[0].author.role == Role.ASSISTANT
            and (
                outputs[0].recipient == "functions.emit_structured_response"
                or (outputs[0].recipient is None and outputs[0].channel == "final")
            )
        ):
            raise LlmResponseError(
                "Unexpected Harmony action; no operation was executed", 502
            )
        content = "".join(getattr(part, "text", "") for part in outputs[0].content)
    return ChatCompletion.model_validate(
        {
            "id": data["id"],
            "object": "chat.completion",
            "model": data["model"],
            "created": data["created"],
            "usage": data.get("usage"),
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": content},
                    "finish_reason": choice["finish_reason"],
                }
            ],
        }
    )
