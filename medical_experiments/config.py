"""Configuration helpers shared by the medical diagnostic experiments."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Mapping, MutableMapping, Union


PathLike = Union[str, Path]


def load_config(path: PathLike) -> Dict[str, Any]:
    """Load JSON or YAML and record the config directory for relative paths."""

    config_path = Path(path).expanduser().resolve()
    suffix = config_path.suffix.lower()
    with config_path.open("r", encoding="utf-8") as handle:
        if suffix == ".json":
            data = json.load(handle)
        elif suffix in {".yaml", ".yml"}:
            try:
                import yaml
            except ImportError as exc:  # pragma: no cover - environment specific
                raise RuntimeError("PyYAML is required to read YAML configs") from exc
            data = yaml.safe_load(handle)
        else:
            raise ValueError("Config must be .json, .yaml, or .yml: %s" % config_path)

    if not isinstance(data, dict):
        raise ValueError("Top-level config value must be a mapping")
    data["__config_path__"] = str(config_path)
    data["__config_dir__"] = str(config_path.parent)
    return data


def resolve_path(value: PathLike, config: Mapping[str, Any]) -> Path:
    """Resolve a user path relative to its configuration file."""

    path = Path(value).expanduser()
    if not path.is_absolute():
        path = Path(config.get("__config_dir__", ".")) / path
    return path.resolve()


def json_ready(value: Any) -> Any:
    """Recursively convert common numerical/path objects to JSON-safe values."""

    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return {str(key): json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_ready(item) for item in value]
    if hasattr(value, "item"):
        try:
            return value.item()
        except (TypeError, ValueError):
            pass
    return value


def write_json(path: PathLike, payload: Any) -> None:
    """Write deterministic UTF-8 JSON and create the parent directory."""

    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        json.dump(json_ready(payload), handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")


def without_private_keys(config: Mapping[str, Any]) -> Dict[str, Any]:
    """Return a copy suitable for recording in result artifacts."""

    return {key: value for key, value in config.items() if not str(key).startswith("__")}


def resolve_known_paths(config: Mapping[str, Any]) -> Dict[str, Any]:
    """Resolve conventional experiment path fields without touching model text."""

    result = dict(config)
    scalar_keys = {
        "manifest",
        "output_dir",
        "output_manifest",
        "input_path",
        "lafite_config",
        "checkpoint",
        "summary",
    }
    sequence_keys = {"cache_paths", "npz_paths", "input_paths"}
    for key in scalar_keys:
        if result.get(key):
            result[key] = str(resolve_path(result[key], config))
    for key in sequence_keys:
        value = result.get(key)
        if value:
            values = [value] if isinstance(value, (str, Path)) else list(value)
            result[key] = [str(resolve_path(item, config)) for item in values]
    if isinstance(result.get("sources"), list):
        sources = []
        for raw in result["sources"]:
            source = dict(raw)
            if source.get("cache_dir"):
                source["cache_dir"] = str(resolve_path(source["cache_dir"], config))
            sources.append(source)
        result["sources"] = sources
    if isinstance(result.get("factory_kwargs"), Mapping):
        kwargs = dict(result["factory_kwargs"])
        for key in ("lafite_config", "checkpoint", "knowledge_file", "model_path"):
            if kwargs.get(key):
                kwargs[key] = str(resolve_path(kwargs[key], config))
        result["factory_kwargs"] = kwargs
    return result
