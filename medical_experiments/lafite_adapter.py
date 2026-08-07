"""LafitE-specific feature and reconstruction cache exporter."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Mapping, Tuple

from .config import resolve_path
from .feature_cache import save_feature_record
from .manifest import (
    assert_valid_manifest,
    manifest_fingerprint,
    normalize_split,
    read_jsonl,
)
from .medical_io import load_array, prepare_model_image


def _load_yaml(path: Path):
    try:
        import yaml
        from easydict import EasyDict
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("PyYAML and easydict are required to build LafitE") from exc
    with path.open("r", encoding="utf-8") as handle:
        return EasyDict(yaml.safe_load(handle))


def load_lafite(config_path: Path, checkpoint_path: Path, device: str = "cpu") -> Tuple[Any, Dict[str, Any]]:
    """Build LafitE and load either DDP or plain checkpoints without dist init."""

    try:
        import torch
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("PyTorch is required to load LafitE") from exc
    from lib.models.model_helper import ModelHelper
    from lib.utils.misc_helper import update_config

    cfg = update_config(_load_yaml(config_path))
    # A full LafitE checkpoint is loaded immediately below.  Disable the
    # backbone's constructor-time download so feature export is reproducible
    # and works on offline clinical systems.
    if cfg.net and hasattr(cfg.net[0], "kwargs"):
        cfg.net[0].kwargs["pretrained"] = False
    model = ModelHelper(cfg.net)
    checkpoint = torch.load(str(checkpoint_path), map_location="cpu")
    state = checkpoint.get("state_dict", checkpoint)
    cleaned = {}
    for key, value in state.items():
        cleaned[key[7:] if key.startswith("module.") else key] = value
    incompatible = model.load_state_dict(cleaned, strict=False)
    if device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA device requested but torch.cuda.is_available() is false")
    if device.startswith("cuda"):
        model.device = torch.device(device)
        model.to(device)
    else:
        model.cpu()
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    load_info = {
        "missing_keys": list(incompatible.missing_keys),
        "unexpected_keys": list(incompatible.unexpected_keys),
        "input_size": list(cfg.dataset.input_size),
        "pixel_mean": list(cfg.dataset.pixel_mean),
        "pixel_std": list(cfg.dataset.pixel_std),
    }
    return model, load_info


def _image_tensor(image, size, mean, std, device):
    import torch
    import torch.nn.functional as functional

    tensor = torch.as_tensor(image).permute(2, 0, 1).unsqueeze(0).float()
    tensor = functional.interpolate(tensor, size=tuple(size), mode="bilinear", align_corners=False)
    mean_tensor = torch.as_tensor(mean, dtype=tensor.dtype).view(1, -1, 1, 1)
    std_tensor = torch.as_tensor(std, dtype=tensor.dtype).view(1, -1, 1, 1)
    tensor = (tensor - mean_tensor) / std_tensor
    return tensor.to(device)


def _mask_array(mask, record):
    import numpy as np

    value = np.asarray(mask)
    if value.ndim == 3 and record.get("slice_index") is not None:
        depth_axis = int(record.get("mask_depth_axis", record.get("depth_axis", 0)))
        value = np.moveaxis(value, depth_axis, 0)[int(record["slice_index"])]
    if value.ndim == 3 and value.shape[-1] == 1:
        value = value[..., 0]
    return (value > 0).astype("uint8")


def export_lafite_features(config: Mapping[str, Any]) -> Dict[str, Any]:
    """Export aligned/reconstructed features and residual maps for a manifest."""

    try:
        import torch
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("PyTorch is required for LafitE feature export") from exc

    manifest_path = resolve_path(config["manifest"], config)
    model_config = resolve_path(config["lafite_config"], config)
    checkpoint = resolve_path(config["checkpoint"], config)
    output_dir = resolve_path(config["output_dir"], config)
    records = read_jsonl(manifest_path)
    assert_valid_manifest(records, manifest_path=manifest_path, check_paths=True, require_mask=bool(config.get("require_mask", True)))

    device = str(config.get("device", "cuda" if torch.cuda.is_available() else "cpu"))
    model, load_info = load_lafite(model_config, checkpoint, device=device)
    preprocessing = dict(config.get("preprocessing", {}))
    mode = str(config.get("mode", "reconstruction"))
    if mode not in {"neck", "reconstruction"}:
        raise ValueError("mode must be neck or reconstruction")
    exported = 0
    skipped = 0
    with torch.inference_mode():
        for record in records:
            if config.get("split") and normalize_split(record.get("split")) != normalize_split(
                config["split"]
            ):
                continue
            image_path = Path(str(record["image"]))
            if not image_path.is_absolute():
                image_path = manifest_path.parent / image_path
            raw = load_array(image_path, key=record.get("image_key"))
            local_preprocessing = dict(preprocessing)
            if record.get("slice_index") is not None:
                local_preprocessing["slice_index"] = int(record["slice_index"])
                local_preprocessing["depth_axis"] = int(record.get("depth_axis", 0))
            image = prepare_model_image(raw, str(record.get("modality", "unknown")), local_preprocessing)
            tensor = _image_tensor(image, load_info["input_size"], load_info["pixel_mean"], load_info["pixel_std"], device)
            outputs = model({"image": tensor}, neck_stop=(mode == "neck"))
            arrays = {"feature_align": outputs["feature_align"].detach().cpu().float().numpy()[0]}
            if mode == "reconstruction":
                feature_rec = outputs["feature_rec"].detach().cpu().float()
                feature_align = outputs["feature_align"].detach().cpu().float()
                arrays["feature_rec"] = feature_rec.numpy()[0]
                arrays["lafite_residual"] = (feature_align - feature_rec).abs().numpy()[0]
                arrays["lafite_pred"] = outputs["pred"].detach().cpu().float().numpy()[0]
            if record.get("mask"):
                mask_path = Path(str(record["mask"]))
                if not mask_path.is_absolute():
                    mask_path = manifest_path.parent / mask_path
                arrays["mask"] = _mask_array(load_array(mask_path, key=record.get("mask_key")), record)
            metadata = {key: value for key, value in record.items() if key != "report"}
            metadata.update({"source": "lafite", "mode": mode, "input_size": load_info["input_size"]})
            try:
                save_feature_record(output_dir, metadata, arrays, overwrite=bool(config.get("overwrite", False)))
                exported += 1
            except FileExistsError:
                skipped += 1
    return {
        "output_dir": str(output_dir),
        "exported": exported,
        "skipped": skipped,
        "checkpoint_load": load_info,
        "manifest_sha256": manifest_fingerprint(records),
    }
