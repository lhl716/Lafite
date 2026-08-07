"""Model-agnostic MLLM feature export through a small factory protocol.

The configured factory receives ``factory_kwargs`` and returns a mapping with:

``model``
    A PyTorch module.
``preprocess(image_hwc, record)``
    Converts a NumPy medical image and manifest record into model inputs.
``forward(model, prepared)`` (optional)
    Defaults to ``model(**prepared)`` for mappings or ``model(prepared)``.
``extract_output(output)`` (optional)
    Returns additional named tensors, e.g. projector or selected LLM states.

Forward hooks cover intermediate vision/projector modules without coupling this
repository to one LLaVA/Qwen/InternVL implementation.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Mapping, Optional

from .config import resolve_path
from .feature_cache import safe_sample_name, save_feature_record
from .hooks import FeatureCollector, TapSpec, import_object, move_to_device
from .manifest import (
    assert_valid_manifest,
    manifest_fingerprint,
    normalize_split,
    read_jsonl,
)
from .medical_io import load_array, prepare_model_image


def _default_forward(model: Any, prepared: Any) -> Any:
    if isinstance(prepared, Mapping):
        return model(**prepared)
    if isinstance(prepared, tuple):
        return model(*prepared)
    return model(prepared)


def _to_numpy(value: Any):
    if hasattr(value, "detach"):
        return value.detach().to("cpu").float().numpy()
    try:
        import numpy as np
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("NumPy is required for MLLM feature export") from exc
    return np.asarray(value)


def export_mllm_features(config: Mapping[str, Any]) -> Dict[str, Any]:
    """Run a configured frozen MLLM once and write one feature cache per sample."""

    try:
        import torch
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("PyTorch is required for MLLM feature export") from exc

    manifest_path = resolve_path(config["manifest"], config)
    output_dir = resolve_path(config["output_dir"], config)
    records = read_jsonl(manifest_path)
    assert_valid_manifest(records, manifest_path=manifest_path, check_paths=True)

    factory = import_object(str(config["factory"]))
    bundle = factory(dict(config.get("factory_kwargs", {})))
    if not isinstance(bundle, Mapping) or "model" not in bundle or "preprocess" not in bundle:
        raise TypeError("MLLM factory must return a mapping with model and preprocess")
    model = bundle["model"]
    preprocess = bundle["preprocess"]
    forward = bundle.get("forward", _default_forward)
    extract_output = bundle.get("extract_output")
    device = str(config.get("device", "cuda" if torch.cuda.is_available() else "cpu"))
    model.to(device)
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)

    taps = [TapSpec.from_mapping(item) for item in config.get("taps", [])]
    preprocessing = dict(config.get("preprocessing", {}))
    exported = 0
    skipped = 0
    with FeatureCollector(model, taps) as collector, torch.inference_mode():
        for record in records:
            if config.get("split") and normalize_split(record.get("split")) != normalize_split(
                config["split"]
            ):
                continue
            target = output_dir / (safe_sample_name(str(record["sample_id"])) + ".npz")
            if target.exists() and not bool(config.get("overwrite", False)):
                skipped += 1
                continue
            image_path = Path(str(record["image"]))
            if not image_path.is_absolute():
                image_path = manifest_path.parent / image_path
            raw = load_array(image_path, key=record.get("image_key"))
            local_preprocessing = dict(preprocessing)
            if record.get("slice_index") is not None:
                local_preprocessing["slice_index"] = int(record["slice_index"])
            image = prepare_model_image(raw, str(record.get("modality", "unknown")), local_preprocessing)
            target_fields = set(
                config.get(
                    "preprocess_target_fields",
                    ["finding", "report", "label", "target", "ground_truth", "mask"],
                )
            )
            visible_record = {
                key: value for key, value in record.items() if key not in target_fields
            }
            prepared = move_to_device(preprocess(image, visible_record), device)
            collector.clear()
            output = forward(model, prepared)
            arrays = {name: _to_numpy(value) for name, value in collector.snapshot().items()}
            if extract_output is not None:
                extra = extract_output(output)
                if not isinstance(extra, Mapping):
                    raise TypeError("extract_output must return a mapping")
                arrays.update({str(name): _to_numpy(value) for name, value in extra.items()})
            if record.get("mask"):
                mask_path = Path(str(record["mask"]))
                if not mask_path.is_absolute():
                    mask_path = manifest_path.parent / mask_path
                mask = load_array(mask_path, key=record.get("mask_key"))
                if record.get("slice_index") is not None and getattr(mask, "ndim", 0) == 3:
                    import numpy as np

                    mask = np.moveaxis(mask, int(record.get("mask_depth_axis", record.get("depth_axis", 0))), 0)[int(record["slice_index"])]
                arrays["mask"] = mask
            metadata = {key: value for key, value in record.items() if key not in {"report"}}
            metadata["source"] = "mllm"
            save_feature_record(output_dir, metadata, arrays, overwrite=bool(config.get("overwrite", False)))
            exported += 1
    return {
        "output_dir": str(output_dir),
        "exported": exported,
        "skipped": skipped,
        "num_taps": len(taps),
        "manifest_sha256": manifest_fingerprint(records),
    }
