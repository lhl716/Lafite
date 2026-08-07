"""Generic PyTorch feature taps for otherwise model-specific MLLMs."""

from __future__ import annotations

import importlib
from contextlib import AbstractContextManager
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Mapping, MutableMapping, Optional, Sequence


def import_object(path: str) -> Any:
    """Import ``package.module:object`` or ``package.module.object``."""

    if ":" in path:
        module_name, object_name = path.split(":", 1)
    else:
        module_name, object_name = path.rsplit(".", 1)
    module = importlib.import_module(module_name)
    return getattr(module, object_name)


def resolve_module(model: Any, dotted_name: str) -> Any:
    modules = dict(model.named_modules())
    if dotted_name not in modules:
        candidates = sorted(name for name in modules if dotted_name in name)
        hint = ", ".join(candidates[:8]) or "no similar module names"
        raise KeyError("module %r not found; candidates: %s" % (dotted_name, hint))
    return modules[dotted_name]


def select_value(value: Any, selector: Optional[str] = None) -> Any:
    """Select a tensor-like value from nested model outputs.

    Selector components are separated by dots. Integer components index tuples;
    names index mappings, attributes, or Hugging Face ModelOutput objects.
    """

    current = value
    if selector:
        for component in selector.split("."):
            if component == "":
                continue
            if component.lstrip("-").isdigit():
                current = current[int(component)]
            elif isinstance(current, Mapping):
                current = current[component]
            else:
                current = getattr(current, component)
    if isinstance(current, Mapping):
        tensors = [item for item in current.values() if hasattr(item, "detach")]
        if len(tensors) == 1:
            current = tensors[0]
    if isinstance(current, (tuple, list)):
        tensors = [item for item in current if hasattr(item, "detach")]
        if len(tensors) == 1:
            current = tensors[0]
    if not hasattr(current, "detach"):
        raise TypeError("selected hook output is not tensor-like; provide a selector")
    return current


@dataclass(frozen=True)
class TapSpec:
    name: str
    module: str
    selector: Optional[str] = None

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "TapSpec":
        return cls(name=str(value["name"]), module=str(value["module"]), selector=value.get("selector"))


class FeatureCollector(AbstractContextManager):
    """Register deterministic forward hooks and detach their latest outputs."""

    def __init__(self, model: Any, taps: Sequence[TapSpec]):
        self.model = model
        self.taps = list(taps)
        self.handles: List[Any] = []
        self.values: Dict[str, Any] = {}

    def __enter__(self) -> "FeatureCollector":
        names = [tap.name for tap in self.taps]
        if len(set(names)) != len(names):
            raise ValueError("tap names must be unique")
        for tap in self.taps:
            module = resolve_module(self.model, tap.module)

            def hook(_module: Any, _inputs: Any, output: Any, spec: TapSpec = tap) -> None:
                tensor = select_value(output, spec.selector)
                self.values[spec.name] = tensor.detach().to("cpu")

            self.handles.append(module.register_forward_hook(hook))
        return self

    def clear(self) -> None:
        self.values.clear()

    def snapshot(self) -> Dict[str, Any]:
        missing = [tap.name for tap in self.taps if tap.name not in self.values]
        if missing:
            raise RuntimeError("forward pass did not execute taps: %s" % ", ".join(missing))
        return dict(self.values)

    def __exit__(self, exc_type: Any, exc_value: Any, traceback: Any) -> None:
        for handle in self.handles:
            handle.remove()
        self.handles = []


def move_to_device(value: Any, device: str) -> Any:
    if hasattr(value, "to"):
        return value.to(device)
    if isinstance(value, dict):
        return {key: move_to_device(item, device) for key, item in value.items()}
    if isinstance(value, tuple):
        return tuple(move_to_device(item, device) for item in value)
    if isinstance(value, list):
        return [move_to_device(item, device) for item in value]
    return value
