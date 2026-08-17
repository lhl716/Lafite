"""Validation-only threshold calibration for modality-specific LafitE models."""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence, Tuple

from .config import resolve_path
from .lafite_adapter import _image_tensor, load_lafite
from .manifest import manifest_fingerprint, normalize_split, read_jsonl
from .medical_io import load_array, prepare_model_image


def _numpy():
    try:
        import numpy as np
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("NumPy is required for LafitE calibration") from exc
    return np


def _balanced_accuracy(labels: Sequence[int], predictions: Sequence[int]) -> float:
    np = _numpy()
    labels_array = np.asarray(labels, dtype=np.int64)
    predictions_array = np.asarray(predictions, dtype=np.int64)
    positive = labels_array == 1
    negative = labels_array == 0
    if not positive.any() or not negative.any():
        raise ValueError("score calibration requires positive and negative patients")
    sensitivity = float((predictions_array[positive] == 1).mean())
    specificity = float((predictions_array[negative] == 0).mean())
    return 0.5 * (sensitivity + specificity)


def select_score_threshold(
    patient_scores: Mapping[str, float], patient_labels: Mapping[str, int]
) -> Tuple[float, Dict[str, float]]:
    """Choose the patient-level threshold maximizing balanced accuracy."""

    np = _numpy()
    patients = sorted(patient_scores)
    if set(patients) != set(patient_labels):
        raise ValueError("patient score/label sets differ")
    scores = np.asarray([float(patient_scores[item]) for item in patients], dtype=np.float64)
    labels = np.asarray([int(patient_labels[item]) for item in patients], dtype=np.int64)
    unique = np.unique(scores)
    if unique.size == 1:
        candidates = unique
    else:
        candidates = np.concatenate(
            ([np.nextafter(unique[0], -np.inf)], (unique[:-1] + unique[1:]) / 2.0, [unique[-1]])
        )
    ranked = []
    for value in candidates:
        predictions = (scores >= value).astype(np.int64)
        balanced = _balanced_accuracy(labels, predictions)
        sensitivity = float((predictions[labels == 1] == 1).mean())
        specificity = float((predictions[labels == 0] == 0).mean())
        ranked.append((balanced, min(sensitivity, specificity), -abs(float(value) - float(np.median(scores))), float(value), sensitivity, specificity))
    ranked.sort(reverse=True)
    balanced, _minimum_class_recall, _distance, threshold, sensitivity, specificity = ranked[0]
    return threshold, {
        "balanced_accuracy": balanced,
        "sensitivity": sensitivity,
        "specificity": specificity,
    }


def select_mask_threshold(
    predictions: Sequence[Any], masks: Sequence[Any], *, candidate_count: int = 101
) -> Tuple[float, Dict[str, float]]:
    """Choose a global threshold maximizing mean Dice over positive slices."""

    np = _numpy()
    positive_pairs = []
    samples = []
    for prediction, mask in zip(predictions, masks):
        pred = np.asarray(prediction, dtype=np.float32).reshape(-1)
        target = np.asarray(mask).astype(bool).reshape(-1)
        if target.any():
            positive_pairs.append((pred, target))
        if pred.size > 20000:
            indices = np.linspace(0, pred.size - 1, 20000, dtype=np.int64)
            samples.append(pred[indices])
        else:
            samples.append(pred)
    if not positive_pairs:
        raise ValueError("mask calibration requires at least one positive slice")
    sampled = np.concatenate(samples)
    quantiles = np.linspace(0.0, 1.0, max(3, int(candidate_count)))
    candidates = np.unique(np.quantile(sampled, quantiles))
    ranked = []
    for threshold in candidates:
        dice_values = []
        global_tp = 0
        global_pred = 0
        global_target = 0
        for pred, target in positive_pairs:
            binary = pred >= threshold
            true_positive = int(np.logical_and(binary, target).sum())
            predicted_positive = int(binary.sum())
            target_positive = int(target.sum())
            dice_values.append(
                2.0 * true_positive / max(1, predicted_positive + target_positive)
            )
            global_tp += true_positive
            global_pred += predicted_positive
            global_target += target_positive
        mean_dice = float(np.mean(dice_values))
        global_dice = 2.0 * global_tp / max(1, global_pred + global_target)
        ranked.append((mean_dice, global_dice, -float(threshold), float(threshold)))
    ranked.sort(reverse=True)
    mean_dice, global_dice, _tie_break, threshold = ranked[0]
    return threshold, {
        "mean_positive_slice_dice": mean_dice,
        "global_positive_pixel_dice": global_dice,
        "positive_slices": len(positive_pairs),
        "candidate_count": len(candidates),
    }


def calibrate_lafite(config: Mapping[str, Any]) -> Dict[str, Any]:
    """Run LafitE on a validation manifest and lock global mask/score thresholds."""

    try:
        import torch
        import torch.nn.functional as functional
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("PyTorch is required for LafitE calibration") from exc
    np = _numpy()
    manifest_path = resolve_path(config["manifest"], config)
    records = [
        record
        for record in read_jsonl(manifest_path)
        if normalize_split(record.get("split")) == normalize_split(config.get("split", "val"))
    ]
    if not records:
        raise ValueError("calibration manifest has no matching validation records")
    device = str(config.get("device", "cuda:0"))
    model, load_info = load_lafite(
        resolve_path(config["lafite_config"], config),
        resolve_path(config["checkpoint"], config),
        device=device,
    )
    preprocessing = dict(config.get("preprocessing", {}))
    reduction = str(config.get("score_reduction", "topk_mean"))
    topk_fraction = float(config.get("topk_fraction", 0.01))
    predictions: List[Any] = []
    masks: List[Any] = []
    patient_slice_scores: Dict[str, List[float]] = defaultdict(list)
    patient_labels: Dict[str, int] = defaultdict(int)
    with torch.inference_mode():
        for record in records:
            image_path = Path(str(record["image"]))
            if not image_path.is_absolute():
                image_path = manifest_path.parent / image_path
            raw = load_array(image_path, key=record.get("image_key"))
            image = prepare_model_image(raw, str(record.get("modality", "unknown")), preprocessing)
            tensor = _image_tensor(
                image,
                load_info["input_size"],
                load_info["pixel_mean"],
                load_info["pixel_std"],
                device,
            )
            output = model({"image": tensor})
            anomaly_map = output["pred"].detach().cpu().float()[0, 0]
            mask_path = Path(str(record["mask"]))
            if not mask_path.is_absolute():
                mask_path = manifest_path.parent / mask_path
            raw_mask = np.asarray(load_array(mask_path, key=record.get("mask_key")))
            if raw_mask.ndim == 3:
                raw_mask = raw_mask.any(axis=-1)
            mask_tensor = torch.as_tensor(raw_mask > 0).float()[None, None]
            resized_mask = functional.interpolate(
                mask_tensor,
                size=tuple(anomaly_map.shape),
                mode="nearest",
            )[0, 0].numpy().astype(bool)
            anomaly_array = anomaly_map.numpy()
            flat = anomaly_array.reshape(-1)
            if reduction == "mean":
                score = float(flat.mean())
            elif reduction == "max":
                score = float(flat.max())
            elif reduction == "topk_mean":
                count = max(1, int(round(flat.size * topk_fraction)))
                score = float(np.partition(flat, flat.size - count)[-count:].mean())
            else:
                raise ValueError("score_reduction must be mean, max, or topk_mean")
            patient = str(record["patient_id"])
            patient_slice_scores[patient].append(score)
            patient_labels[patient] = max(patient_labels[patient], int(record.get("label", 0)))
            predictions.append(anomaly_array)
            masks.append(resized_mask)

    aggregation = str(config.get("patient_aggregation", "mean"))
    if aggregation != "mean":
        raise ValueError("patient_aggregation currently supports only mean")
    patient_scores = {
        patient: float(np.mean(values)) for patient, values in patient_slice_scores.items()
    }
    mask_threshold, mask_metrics = select_mask_threshold(
        predictions,
        masks,
        candidate_count=int(config.get("mask_candidate_count", 101)),
    )
    score_threshold, score_metrics = select_score_threshold(patient_scores, patient_labels)
    return {
        "manifest": str(manifest_path),
        "manifest_sha256": manifest_fingerprint(records),
        "split": normalize_split(config.get("split", "val")),
        "checkpoint": str(resolve_path(config["checkpoint"], config)),
        "mask_threshold": mask_threshold,
        "score_threshold": score_threshold,
        "mask_metrics": mask_metrics,
        "score_metrics": score_metrics,
        "samples": len(records),
        "patients": len(patient_scores),
        "positive_patients": sum(patient_labels.values()),
        "negative_patients": len(patient_labels) - sum(patient_labels.values()),
        "score_reduction": reduction,
        "topk_fraction": topk_fraction,
        "patient_aggregation": aggregation,
        "preprocessing": preprocessing,
    }
