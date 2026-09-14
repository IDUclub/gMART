"""Check the actual calculation scope, not merely any missing-normative message."""

import re


def verify_provision_scope(events, *, service_type_id=22, request_territory_id=58):
    messages = [
        event["content"]["event"]["content"].get("message", "")
        for event in events
        if event.get("type") == "step_event"
        and event["content"].get("agent") == "provision"
        and event["content"].get("event", {}).get("type") == "error"
    ]
    messages = [
        message for message in messages if "missing_service_normative" in message
    ]
    assert messages, "The calculation did not report a structured missing normative"
    text = "\n".join(messages)
    service_ids = set(
        map(int, re.findall(r"[\"']service_type_id[\"']\s*:\s*(\d+)", text))
    )
    territory_ids = set(
        map(int, re.findall(r"[\"']request_ter_id[\"']\s*:\s*(\d+)", text))
    )
    assert service_ids == {
        service_type_id
    }, "Calculation used the wrong or additional service types"
    assert territory_ids == {
        request_territory_id
    }, "Calculation used the wrong request territory"
