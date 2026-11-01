"""Deterministic checks over a stage's output.

A case lists its expectations as checks, each one a ``path`` into the output plus
exactly one operator::

    expect:
      - {path: mode, eq: execute}
      - {path: "steps[*].agent", set_eq: [norms, documents]}
      - {path: target_population, eq: 30000}
      - {path: clarification_question, null: true}
      - {path: search_query, contains: ["школ"]}

Paths are dotted, with list indices (``steps.0.agent``) and ``[*]`` to collect a
field across a list. String comparisons ignore case, ``ё``/``е`` and repeated
whitespace, so a check states meaning, not formatting. ``weight`` (default 1)
scales a check in the case score; ``name`` labels it in the report.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from typing import Any

_MISSING = object()

OPERATORS = (
    "eq",
    "ne",
    "in",
    "not_in",
    "set_eq",
    "contains",
    "not_contains",
    "contains_any",
    "subset",
    "len",
    "min_len",
    "max_len",
    "null",
    "regex",
    "not_regex",
    "approx",
    "gte",
    "lte",
)
_META = {"path", "weight", "name", "tol"}


def normalize(value: Any) -> Any:
    if isinstance(value, str):
        return " ".join(value.casefold().replace("ё", "е").split())
    if isinstance(value, bool) or value is None:
        return value
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, list | tuple):
        return [normalize(item) for item in value]
    if isinstance(value, dict):
        return {key: normalize(item) for key, item in value.items()}
    return value


def _hashable(value: Any) -> Any:
    if isinstance(value, list):
        return tuple(_hashable(item) for item in value)
    if isinstance(value, dict):
        return tuple(sorted((key, _hashable(item)) for key, item in value.items()))
    return value


def _tokens(path: str) -> list[str]:
    tokens: list[str] = []
    for part in path.split("."):
        while part.endswith("[*]"):
            base = part[:-3]
            if base:
                tokens.append(base)
            tokens.append("*")
            part = ""
        if part:
            tokens.append(part)
    return tokens


def resolve(data: Any, path: str) -> Any:
    """Value at ``path``; ``[*]`` maps the rest of the path over a list."""
    if path in ("", "."):
        return data

    def walk(node: Any, tokens: list[str]) -> Any:
        if not tokens:
            return node
        head, rest = tokens[0], tokens[1:]
        if head == "*":
            if not isinstance(node, list):
                return _MISSING
            values = [walk(item, rest) for item in node]
            return [value for value in values if value is not _MISSING]
        if isinstance(node, dict):
            return walk(node[head], rest) if head in node else _MISSING
        if isinstance(node, list) and re.fullmatch(r"-?\d+", head):
            index = int(head)
            return (
                walk(node[index], rest) if -len(node) <= index < len(node) else _MISSING
            )
        return _MISSING

    return walk(data, _tokens(path))


@dataclass
class CheckResult:
    name: str
    passed: bool
    weight: float
    actual: Any
    expected: Any

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "passed": self.passed,
            "weight": self.weight,
            "actual": self.actual,
            "expected": self.expected,
        }


def _operator(check: dict[str, Any]) -> str:
    ops = [key for key in check if key not in _META]
    if len(ops) != 1 or ops[0] not in OPERATORS:
        raise ValueError(
            f"check {check!r} must have exactly one operator out of {OPERATORS}"
        )
    return ops[0]


def _as_list(value: Any) -> list:
    if value is _MISSING or value is None:
        return []
    return value if isinstance(value, list) else [value]


def _contains(actual: Any, item: Any) -> bool:
    if isinstance(actual, str):
        return normalize(item) in normalize(actual)
    return normalize(item) in normalize(_as_list(actual))


def evaluate_check(output: Any, check: dict[str, Any]) -> CheckResult:
    op = _operator(check)
    expected = check[op]
    path = str(check.get("path", ""))
    actual = resolve(output, path)
    found = actual is not _MISSING
    norm = normalize(actual) if found else None
    passed: bool
    if op == "eq":
        passed = found and norm == normalize(expected)
    elif op == "ne":
        passed = not found or norm != normalize(expected)
    elif op == "in":
        passed = found and norm in normalize(list(expected))
    elif op == "not_in":
        passed = not found or norm not in normalize(list(expected))
    elif op == "set_eq":
        passed = found and {_hashable(v) for v in normalize(_as_list(actual))} == {
            _hashable(v) for v in normalize(list(expected))
        }
    elif op == "contains":
        passed = found and all(_contains(actual, item) for item in _as_list(expected))
    elif op == "not_contains":
        passed = not found or not any(
            _contains(actual, item) for item in _as_list(expected)
        )
    elif op == "contains_any":
        passed = found and any(_contains(actual, item) for item in _as_list(expected))
    elif op == "subset":
        allowed = {_hashable(v) for v in normalize(list(expected))}
        passed = found and all(
            _hashable(v) in allowed for v in normalize(_as_list(actual))
        )
    elif op in ("len", "min_len", "max_len"):
        size = len(actual) if found and hasattr(actual, "__len__") else None
        passed = size is not None and (
            size == expected
            if op == "len"
            else size >= expected if op == "min_len" else size <= expected
        )
    elif op == "null":
        is_null = (not found) or actual is None or actual == [] or actual == ""
        passed = is_null if expected else not is_null
    elif op in ("regex", "not_regex"):
        text = actual if isinstance(actual, str) else ""
        hit = bool(re.search(str(expected), text, re.IGNORECASE | re.MULTILINE))
        passed = found and (hit if op == "regex" else not hit)
    elif op in ("approx", "gte", "lte"):
        number = (
            float(actual)
            if found
            and isinstance(actual, (int, float))
            and not isinstance(actual, bool)
            else None
        )
        if number is None or math.isnan(number):
            passed = False
        elif op == "approx":
            passed = abs(number - float(expected)) <= float(check.get("tol", 1e-6))
        elif op == "gte":
            passed = number >= float(expected)
        else:
            passed = number <= float(expected)
    else:  # pragma: no cover - _operator guards the set
        raise AssertionError(op)
    return CheckResult(
        name=str(check.get("name") or f"{path or '.'} {op}"),
        passed=bool(passed),
        weight=float(check.get("weight", 1)),
        actual=actual if found else None,
        expected=expected,
    )


def validate_checks(checks: list[dict[str, Any]]) -> None:
    for check in checks:
        _operator(check)


def score(results: list[CheckResult]) -> float:
    total = sum(result.weight for result in results)
    if not total:
        return 1.0
    return sum(result.weight for result in results if result.passed) / total
