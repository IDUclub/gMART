"""Shared GIS identifiers keep role detection and entity grouping consistent."""

from pathlib import PurePosixPath
from typing import Any

GISOGD_LAYERS = (
    "redline",
    "gr_dpt",
    "formedland",
    "constructionzoneborders",
    "elementplanningstructure",
    "publicterritoryborders",
    "indentline",
    "easement",
)


def class_id(properties: dict[str, Any]) -> str:
    """Normalize numeric DBF identifiers without changing classifier codes."""
    for key, value in properties.items():
        if str(key).casefold() not in {"classid", "class_id"}:
            continue
        if value is None or isinstance(value, (bool, dict, list)):
            continue
        text = str(value).strip()
        if text.endswith(".0") and text[:-2].isdigit():
            text = text[:-2]
        if text:
            return text
    return ""


def gisogd_layer_kind(name: str) -> str:
    """Recognize schema names even inside archive folders and geometry variants."""
    stem = PurePosixPath(name.replace("\\", "/")).stem.casefold()
    for suffix in ("_line", "_poly"):
        if stem.endswith(suffix):
            stem = stem[: -len(suffix)]
            break
    return stem if stem in GISOGD_LAYERS else ""
