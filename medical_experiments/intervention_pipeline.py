"""Manifest-driven generation and optional model scoring of paired interventions.

A configured model factory returns either a callable or a mapping containing
``predict(image, record)``.  The predictor returns a scalar anomaly score or a
mapping with ``score`` plus optional ``mask``/``report_entities`` fields.  This small
protocol keeps counterfactual generation identical across LafitE, a baseline
MLLM, and a future fused model.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Sequence

from .config import resolve_path
from .feature_cache import load_feature_record, safe_sample_name, save_feature_record
from .hooks import import_object
from .interventions import apply_nuisance, generate_intervention_variants, intervention_quality_metrics
from .manifest import (
    assert_valid_manifest,
    manifest_fingerprint,
    normalize_split,
    read_jsonl,
    write_jsonl,
)
from .medical_io import load_array


VARIANTS = ("original", "lesion_removed", "nuisance_changed", "placebo")


def _numpy():
    try:
        import numpy as np
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("NumPy is required for intervention generation") from exc
    return np


def _resolve_record_path(record: Mapping[str, Any], field: str, manifest_path: Path) -> Optional[Path]:
    value = record.get(field)
    if not value:
        return None
    path = Path(str(value)).expanduser()
    if not path.is_absolute():
        path = manifest_path.parent / path
    return path.resolve()


def _select_sample(array: Any, record: Mapping[str, Any], *, mask: bool = False):
    np = _numpy()
    value = np.asarray(array)
    if value.ndim == 3 and record.get("slice_index") is not None:
        field = "mask_depth_axis" if mask else "depth_axis"
        depth_axis = int(record.get(field, record.get("depth_axis", 0)))
        value = np.moveaxis(value, depth_axis, 0)[int(record["slice_index"])]
    return value


def _load_optional(record: Mapping[str, Any], field: str, manifest_path: Path, *, mask: bool = False):
    path = _resolve_record_path(record, field, manifest_path)
    if path is None:
        return None
    key = record.get(field + "_key")
    return _select_sample(load_array(path, key=key), record, mask=mask)


def _predictor(config: Mapping[str, Any]):
    factory_path = config.get("model_factory")
    if not factory_path:
        return None
    factory = import_object(str(factory_path))
    bundle = factory(dict(config.get("factory_kwargs", {})))
    if callable(bundle):
        return bundle
    if isinstance(bundle, Mapping) and callable(bundle.get("predict")):
        return bundle["predict"]
    raise TypeError("model_factory must return a callable or {'predict': callable}")


def _prediction_payload(raw: Any) -> Dict[str, Any]:
    np = _numpy()
    if isinstance(raw, Mapping):
        if "score" not in raw and "anomaly_score" not in raw:
            raise ValueError("predictor mapping must contain score or anomaly_score")
        payload = dict(raw)
        payload["score"] = float(payload.pop("anomaly_score", payload.get("score")))
        report = payload.get("report")
        if "report_entities" not in payload and "entities" not in payload:
            if isinstance(report, Mapping):
                entities = report.get("report_entities", report.get("entities"))
                if entities is not None:
                    payload["report_entities"] = entities
            elif isinstance(report, (list, tuple, set)):
                payload["report_entities"] = list(report)
        return payload
    if np.asarray(raw).ndim != 0:
        raise TypeError("predictor must return a scalar or mapping")
    return {"score": float(raw)}


def run_intervention_pipeline(config: Mapping[str, Any]) -> Dict[str, Any]:
    """Generate four variants and, when configured, score them immediately."""

    np = _numpy()
    manifest_path = resolve_path(config["manifest"], config)
    output_dir = resolve_path(config["output_dir"], config)
    cache_dir = output_dir / "interventions_cache"
    prediction_path = output_dir / "model_outputs.jsonl"
    records = read_jsonl(manifest_path)
    assert_valid_manifest(records, manifest_path=manifest_path, check_paths=True, require_mask=True)
    selected_split = config.get("split", "test")
    selected = [
        record
        for record in records
        if selected_split is None
        or normalize_split(record.get("split")) == normalize_split(selected_split)
    ]
    if not selected:
        raise ValueError("no records match split %r" % selected_split)
    predict = _predictor(config)
    intervention_config = dict(config.get("interventions", {}))
    cohort = str(config.get("cohort", "positive")).lower()
    if cohort not in {"positive", "normal", "all"}:
        raise ValueError("cohort must be positive, normal, or all")
    overwrite = bool(config.get("overwrite", False))
    output_records = []
    generated = 0
    reused = 0
    processed = 0

    for index, record in enumerate(selected):
        sample_id = str(record["sample_id"])
        cache_path = cache_dir / (safe_sample_name(sample_id) + ".npz")
        if cache_path.exists() and not overwrite:
            metadata, arrays = load_feature_record(cache_path)
            quality = metadata.get("quality", {})
            available_variants = [variant for variant in VARIANTS if variant in arrays]
            label = int(metadata.get("label", record.get("label", 1)))
            if (cohort == "positive" and label == 0) or (cohort == "normal" and label == 1):
                continue
            reused += 1
        else:
            image_path = _resolve_record_path(record, "image", manifest_path)
            assert image_path is not None
            image = _select_sample(load_array(image_path, key=record.get("image_key")), record)
            lesion = _load_optional(record, "mask", manifest_path, mask=True)
            label = int(record.get("label", bool(np.asarray(lesion).sum() > 0)))
            if (cohort == "positive" and label == 0) or (cohort == "normal" and label == 1):
                continue
            normal_reference = _load_optional(record, "normal_reference", manifest_path)
            nuisance_mask = _load_optional(record, "nuisance_mask", manifest_path, mask=True)
            if bool(config.get("require_nuisance_mask", False)) and nuisance_mask is None:
                raise ValueError("sample %s requires nuisance_mask" % sample_id)
            placebo_mask = _load_optional(record, "placebo_mask", manifest_path, mask=True)
            placebo_allowed_mask = _load_optional(
                record, "placebo_allowed_mask", manifest_path, mask=True
            )
            style_reference = _load_optional(record, "style_reference", manifest_path)
            seed = int(intervention_config.get("seed", config.get("seed", 42))) + index
            if label == 1:
                variants = generate_intervention_variants(
                    image,
                    lesion,
                    normal_reference=normal_reference,
                    nuisance_mask=nuisance_mask,
                    placebo_mask=placebo_mask,
                    removal_method=intervention_config.get("removal_method"),
                    nuisance_mode=str(intervention_config.get("nuisance_mode", "gaussian_noise")),
                    nuisance_strength=float(intervention_config.get("nuisance_strength", 0.15)),
                    dilation_radius=int(intervention_config.get("dilation_radius", 3)),
                    gaussian_sigma=float(intervention_config.get("gaussian_sigma", 2.0)),
                    blend_sigma=float(intervention_config.get("blend_sigma", 0.75)),
                    style_reference=style_reference,
                    seed=seed,
                    protection_threshold=float(
                        intervention_config.get("protection_threshold", 0.05)
                    ),
                    placebo_mass_tolerance=float(
                        intervention_config.get("placebo_mass_tolerance", 0.05)
                    ),
                    placebo_allowed_mask=placebo_allowed_mask,
                    placebo_max_outside=float(
                        intervention_config.get("placebo_max_outside", 0.05)
                    ),
                )
                available_variants = list(VARIANTS)
                arrays = {name: np.asarray(variants[name]) for name in available_variants}
                arrays.update({"edit_mask_" + name: np.asarray(value) for name, value in variants["masks"].items()})
                quality = variants["quality"]
                variant_metadata = variants["metadata"]
            else:
                zero_lesion = np.zeros_like(image, dtype="float64")
                nuisance_changed = apply_nuisance(
                    image,
                    zero_lesion,
                    nuisance_mask=nuisance_mask,
                    mode=str(intervention_config.get("nuisance_mode", "gaussian_noise")),
                    strength=float(intervention_config.get("nuisance_strength", 0.15)),
                    dilation_radius=0,
                    gaussian_sigma=float(intervention_config.get("gaussian_sigma", 2.0)),
                    style_reference=style_reference,
                    seed=seed,
                    protection_threshold=float(
                        intervention_config.get("protection_threshold", 0.05)
                    ),
                )
                available_variants = ["original", "nuisance_changed"]
                arrays = {"original": np.asarray(image), "nuisance_changed": np.asarray(nuisance_changed)}
                edit_mask = np.ones_like(image) if nuisance_mask is None else np.asarray(nuisance_mask)
                quality = {
                    "original": intervention_quality_metrics(image, image, np.zeros_like(image)),
                    "nuisance_changed": intervention_quality_metrics(image, nuisance_changed, edit_mask),
                }
                variant_metadata = {
                    "normal_control": True,
                    "nuisance_mode": str(intervention_config.get("nuisance_mode", "gaussian_noise")),
                    "seed": seed,
                }
            metadata = {key: value for key, value in record.items() if key != "report"}
            metadata.update({"source": "paired_interventions", "quality": quality, "available_variants": available_variants, **variant_metadata})
            save_feature_record(cache_dir, metadata, arrays, overwrite=overwrite)
            generated += 1

        processed += 1
        for variant in available_variants:
            row: Dict[str, Any] = {
                "sample_id": sample_id,
                "patient_id": str(record["patient_id"]),
                "variant": variant,
                "label": label,
                "image": {"path": str(cache_path.resolve()), "key": variant},
                "artifact_score": quality.get(variant, {}).get("artifact_score"),
                "continuity_score": quality.get(variant, {}).get("continuity_score"),
            }
            target_entity = record.get("target_entity", record.get("finding"))
            if target_entity:
                row["target_entity"] = str(target_entity)
            if predict is not None:
                target_fields = set(
                    config.get(
                        "predictor_target_fields",
                        ["finding", "report", "label", "target_entity", "ground_truth"],
                    )
                )
                visible_record = {
                    key: value for key, value in record.items() if key not in target_fields
                }
                visible_record["variant"] = variant
                prediction = _prediction_payload(predict(arrays[variant], visible_record))
                row["score"] = prediction.pop("score")
                predicted_mask = prediction.pop("mask", prediction.pop("pred_mask", None))
                if predicted_mask is not None:
                    # Store masks separately to avoid rewriting the immutable
                    # intervention cache during each model comparison.
                    mask_dir = output_dir / "predicted_masks"
                    mask_dir.mkdir(parents=True, exist_ok=True)
                    mask_path = mask_dir / (safe_sample_name(sample_id + "-" + variant) + ".npy")
                    if mask_path.exists() and not overwrite:
                        raise FileExistsError("predicted mask exists: %s" % mask_path)
                    np.save(str(mask_path), np.asarray(predicted_mask), allow_pickle=False)
                    row["mask"] = str(mask_path.resolve())
                row.update(prediction)
            output_records.append(row)

    if processed == 0:
        raise ValueError("no samples match split %r and cohort %r" % (selected_split, cohort))
    write_jsonl(prediction_path, output_records)
    return {
        "output_dir": str(output_dir),
        "intervention_cache": str(cache_dir),
        "model_outputs": str(prediction_path),
        "samples": processed,
        "records": len(output_records),
        "generated": generated,
        "reused": reused,
        "scored": predict is not None,
        "cohort": cohort,
        "manifest_sha256": manifest_fingerprint(records),
    }
