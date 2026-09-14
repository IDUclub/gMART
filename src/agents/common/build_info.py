"""Content identity for reproducible live acceptance, independent of image tags."""

import hashlib
import json
import os
from pathlib import Path


def application_digest(root=None):
    root = Path(root) if root else Path(__file__).resolve().parents[3]
    paths = sorted(
        [*(root / "src/agents").rglob("*.py"), *(root / "src/common").rglob("*.py")]
    )
    records = {
        p.relative_to(root)
        .as_posix(): hashlib.sha256(p.read_bytes().replace(b"\r\n", b"\n"))
        .hexdigest()
        for p in paths
    }
    return hashlib.sha256(json.dumps(records, sort_keys=True).encode()).hexdigest()


def build_info():
    # Publish only hashes. Endpoint settings and credential values stay private.
    settings = {
        k: v
        for k, v in os.environ.items()
        if k.startswith(
            (
                "ORCHESTRATOR_",
                "OPENAI_",
                "LLM_",
                "GENPLANNER_",
                "GENBUILDER_",
                "PZZ_",
                "DVD_",
                "NORM_GRAPH_",
                "URBAN_",
                "OBJECTS_EFFECTS_",
                "IDU_MCP_",
            )
        )
    }
    return {
        "application_sha256": application_digest(),
        "configuration_sha256": hashlib.sha256(
            json.dumps(settings, sort_keys=True).encode()
        ).hexdigest(),
    }
