"""Benchmark configuration: endpoints, named models and the profiles to compare.

A *profile* assigns a model to every stage: ``default`` covers all stages, ``stages``
overrides some of them by stage id or glob (``scenario.*``). Comparing profiles is how
a stage x model matrix is measured; ``--models a,b`` on the command line builds one
single-model profile per name without editing the file.
"""

from __future__ import annotations

import fnmatch
import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

_ENV_REF = re.compile(r"\$\{([A-Z0-9_]+)(?::-([^}]*))?\}")


def _expand_env(value: Any) -> Any:
    """Substitute ``${VAR}`` / ``${VAR:-default}`` so secrets stay out of the file."""
    if isinstance(value, str):
        return _ENV_REF.sub(lambda m: os.getenv(m.group(1), m.group(2) or ""), value)
    if isinstance(value, dict):
        return {key: _expand_env(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_expand_env(item) for item in value]
    return value


@dataclass(frozen=True)
class EndpointConfig:
    name: str
    backend: str = "openai"
    base_url: str | None = None
    host: str | None = None
    api_key: str | None = None
    timeout: float | None = 300.0
    think_mode: str | None = None
    think_effort: str | None = None
    think_chat_template_kwarg: str | None = None


@dataclass(frozen=True)
class ModelConfig:
    name: str
    endpoint: str
    model: str
    # Merged into every call's ``options`` (e.g. num_ctx for Ollama).
    options: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ProfileConfig:
    name: str
    default: str | None
    stages: dict[str, str] = field(default_factory=dict)

    def model_for(self, stage: str) -> str | None:
        if stage in self.stages:
            return self.stages[stage]
        for pattern, model in self.stages.items():
            if fnmatch.fnmatchcase(stage, pattern):
                return model
        return self.default


@dataclass
class BenchConfig:
    endpoints: dict[str, EndpointConfig]
    models: dict[str, ModelConfig]
    profiles: dict[str, ProfileConfig]
    repeat: int = 1
    concurrency: int = 2
    stages: list[str] = field(default_factory=lambda: ["*"])
    tags: list[str] = field(default_factory=list)
    temperature: float = 0.0

    def select_profiles(self, names: list[str] | None) -> list[ProfileConfig]:
        if not names:
            return list(self.profiles.values())
        missing = [name for name in names if name not in self.profiles]
        if missing:
            raise ValueError(f"unknown profiles: {missing}")
        return [self.profiles[name] for name in names]

    def add_single_model_profiles(self, model_names: list[str]) -> list[str]:
        created = []
        for name in model_names:
            if name not in self.models:
                raise ValueError(f"unknown model: {name}")
            profile = f"only:{name}"
            self.profiles[profile] = ProfileConfig(name=profile, default=name)
            created.append(profile)
        return created

    def validate(self) -> None:
        for model in self.models.values():
            if model.endpoint not in self.endpoints:
                raise ValueError(
                    f"model {model.name!r} uses unknown endpoint {model.endpoint!r}"
                )
        for profile in self.profiles.values():
            for used in [profile.default, *profile.stages.values()]:
                if used is not None and used not in self.models:
                    raise ValueError(
                        f"profile {profile.name!r} uses unknown model {used!r}"
                    )


def load_config(path: str | Path) -> BenchConfig:
    raw = _expand_env(yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {})
    endpoints = {
        name: EndpointConfig(name=name, **(body or {}))
        for name, body in (raw.get("endpoints") or {}).items()
    }
    models = {
        name: ModelConfig(name=name, **(body or {}))
        for name, body in (raw.get("models") or {}).items()
    }
    profiles = {
        name: ProfileConfig(
            name=name,
            default=(body or {}).get("default"),
            stages=dict((body or {}).get("stages") or {}),
        )
        for name, body in (raw.get("profiles") or {}).items()
    }
    run = raw.get("run") or {}
    config = BenchConfig(
        endpoints=endpoints,
        models=models,
        profiles=profiles,
        repeat=int(run.get("repeat", 1)),
        concurrency=int(run.get("concurrency", 2)),
        stages=list(run.get("stages") or ["*"]),
        tags=list(run.get("tags") or []),
        temperature=float(run.get("temperature", 0.0)),
    )
    config.validate()
    return config
