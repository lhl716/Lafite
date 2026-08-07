"""Reference intervention predictor backed by an existing LafitE checkpoint."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Mapping

from ..lafite_adapter import _image_tensor, load_lafite
from ..medical_io import prepare_model_image


def make_lafite_predictor(kwargs: Mapping[str, Any]):
    """Factory compatible with ``run_intervention_pipeline``.

    Required kwargs are ``lafite_config`` and ``checkpoint``.  ``mask_threshold``
    should be a calibration-set threshold; when omitted the factory returns only
    an anomaly score, preventing per-image normalization from masquerading as
    mask stability.
    """

    try:
        import numpy as np
        import torch
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("NumPy and PyTorch are required for the LafitE predictor") from exc

    device = str(kwargs.get("device", "cuda" if torch.cuda.is_available() else "cpu"))
    model, load_info = load_lafite(
        Path(str(kwargs["lafite_config"])).expanduser().resolve(),
        Path(str(kwargs["checkpoint"])).expanduser().resolve(),
        device=device,
    )
    preprocessing = dict(kwargs.get("preprocessing", {}))
    reduction = str(kwargs.get("score_reduction", "topk_mean")).lower()
    topk_fraction = float(kwargs.get("topk_fraction", 0.01))
    if not 0 < topk_fraction <= 1:
        raise ValueError("topk_fraction must lie in (0, 1]")
    mask_threshold = kwargs.get("mask_threshold")

    def predict(image: Any, record: Mapping[str, Any]) -> Dict[str, Any]:
        model_image = prepare_model_image(image, str(record.get("modality", "unknown")), preprocessing)
        tensor = _image_tensor(
            model_image,
            load_info["input_size"],
            load_info["pixel_mean"],
            load_info["pixel_std"],
            device,
        )
        with torch.inference_mode():
            output = model({"image": tensor})
        anomaly_map = output["pred"].detach().cpu().float().numpy()[0, 0]
        flat = anomaly_map.reshape(-1)
        if reduction == "max":
            score = float(flat.max())
        elif reduction == "mean":
            score = float(flat.mean())
        elif reduction == "topk_mean":
            count = max(1, int(round(flat.size * topk_fraction)))
            score = float(np.partition(flat, flat.size - count)[-count:].mean())
        else:
            raise ValueError("score_reduction must be max, mean, or topk_mean")
        result: Dict[str, Any] = {"score": score}
        if mask_threshold is not None:
            result["mask"] = (anomaly_map >= float(mask_threshold)).astype("float32")
        return result

    return {"predict": predict}
