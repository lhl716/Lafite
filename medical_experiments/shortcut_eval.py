"""Patient-level paired evaluation for lesion/background shortcut experiments.

Input may be JSON Lines or NPZ.  The preferred long-form schema has one row per
patient/sample/variant with ``patient_id``, ``variant``, ``score`` and an
optional predicted ``mask``.  Wide JSONL/NPZ records with keys such as
``original_score`` and ``nuisance_changed_score`` are also accepted.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np


CANONICAL_VARIANTS = ("original", "lesion_removed", "nuisance_changed", "placebo")

DEFAULT_RED_FLAG_RULES = {
    "min_lesion_score_drop": 0.10,
    "max_nuisance_mean_abs_delta": 0.05,
    "max_placebo_mean_abs_delta": 0.05,
    "min_nuisance_mask_dice": 0.80,
    "min_placebo_mask_dice": 0.80,
    "max_nuisance_to_lesion_ratio": 0.50,
    "min_mask_patient_coverage": 0.80,
    "max_normal_nuisance_fpr_increase_pp": 5.0,
    "max_normal_placebo_fpr_increase_pp": 5.0,
    "min_paired_patients": 20,
}


def _json_safe(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, np.ndarray):
        return [_json_safe(item) for item in value.tolist()]
    if isinstance(value, (np.floating, float)):
        number = float(value)
        return number if np.isfinite(number) else None
    if isinstance(value, (np.integer, int)) and not isinstance(value, bool):
        return int(value)
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    if value is None or isinstance(value, str):
        return value
    return str(value)


def bootstrap_ci(
    values: Sequence[float],
    iterations: int = 2000,
    confidence: float = 0.95,
    seed: int = 0,
    statistic: Optional[Callable[[np.ndarray], float]] = None,
) -> Dict[str, Optional[float]]:
    """Percentile bootstrap confidence interval over patient-level values."""
    array = np.asarray(values, dtype=np.float64)
    array = array[np.isfinite(array)]
    if statistic is None:
        statistic = lambda sample: float(np.mean(sample))
    if array.size == 0:
        return {"estimate": None, "lower": None, "upper": None, "confidence": float(confidence)}
    if iterations <= 0:
        raise ValueError("bootstrap iterations must be positive")
    if not 0.0 < confidence < 1.0:
        raise ValueError("bootstrap confidence must lie in (0, 1)")

    estimate = float(statistic(array))
    if array.size == 1:
        return {
            "estimate": estimate,
            "lower": estimate,
            "upper": estimate,
            "confidence": float(confidence),
        }
    rng = np.random.RandomState(int(seed))
    estimates = np.empty(int(iterations), dtype=np.float64)
    for index in range(int(iterations)):
        sample = array[rng.randint(0, array.size, size=array.size)]
        estimates[index] = float(statistic(sample))
    tail = (1.0 - float(confidence)) / 2.0
    return {
        "estimate": estimate,
        "lower": float(np.quantile(estimates, tail)),
        "upper": float(np.quantile(estimates, 1.0 - tail)),
        "confidence": float(confidence),
    }


def soft_dice(mask_a: Any, mask_b: Any, epsilon: float = 1e-8) -> float:
    """Soft Sørensen-Dice similarity; both-empty masks have stability 1."""
    first = np.clip(np.asarray(mask_a, dtype=np.float64), 0.0, 1.0)
    second = np.clip(np.asarray(mask_b, dtype=np.float64), 0.0, 1.0)
    if first.shape != second.shape:
        raise ValueError("mask shapes differ: %r vs %r" % (first.shape, second.shape))
    if first.ndim not in (2, 3):
        raise ValueError("predicted masks must be 2-D or 3-D")
    denominator = float(np.sum(first * first) + np.sum(second * second))
    if denominator <= epsilon:
        return 1.0
    return float((2.0 * np.sum(first * second)) / denominator)


def _canonical_variant(raw: Any, variant_map: Mapping[str, str]) -> str:
    if isinstance(raw, (bytes, np.bytes_)):
        raw = bytes(raw).decode("utf-8")
    name = str(raw)
    if name in variant_map:
        name = str(variant_map[name])
    elif name not in CANONICAL_VARIANTS:
        reverse = {str(value): str(key) for key, value in variant_map.items()}
        if name in reverse and reverse[name] in CANONICAL_VARIANTS:
            name = reverse[name]
    aliases = {
        "baseline": "original",
        "orig": "original",
        "removed": "lesion_removed",
        "lesionremoved": "lesion_removed",
        "background_changed": "nuisance_changed",
        "nuisance": "nuisance_changed",
        "control": "placebo",
    }
    name = aliases.get(name.lower(), name.lower())
    if name not in CANONICAL_VARIANTS:
        raise ValueError("unknown variant %r; expected one of %r" % (raw, CANONICAL_VARIANTS))
    return name


def _field(
    record: Mapping[str, Any],
    fields: Mapping[str, str],
    canonical: str,
    candidates: Sequence[str],
) -> Any:
    configured = fields.get(canonical)
    if configured is not None and configured in record:
        return record[configured]
    for candidate in candidates:
        if candidate in record:
            return record[candidate]
    return None


def _load_mask(value: Any, base_directory: Path, allow_pickle: bool = False) -> Optional[np.ndarray]:
    if value is None:
        return None
    mask_key = None
    if isinstance(value, Mapping):
        mask_key = value.get("key")
        value = value.get("path", value.get("mask_path"))
    if isinstance(value, str):
        path = Path(value)
        if not path.is_absolute():
            path = base_directory / path
        if path.suffix.lower() == ".npy":
            array = np.load(str(path), allow_pickle=allow_pickle)
        elif path.suffix.lower() == ".npz":
            with np.load(str(path), allow_pickle=allow_pickle) as archive:
                key = str(mask_key) if mask_key is not None else None
                if key is None:
                    if len(archive.files) != 1:
                        raise ValueError("mask NPZ %s needs an explicit key" % path)
                    key = archive.files[0]
                array = np.asarray(archive[key])
        else:
            raise ValueError("mask_path must point to .npy or .npz: %s" % path)
    else:
        array = np.asarray(value)
    if array.ndim not in (2, 3):
        raise ValueError("predicted mask must be 2-D or 3-D; got shape %r" % (array.shape,))
    array = np.asarray(array, dtype=np.float64)
    if not np.all(np.isfinite(array)):
        raise ValueError("predicted mask contains NaN or infinite values")
    return np.clip(array, 0.0, 1.0)


def _wide_record_rows(record: Mapping[str, Any], fields: Mapping[str, str]) -> List[Dict[str, Any]]:
    """Expand a wide record into canonical long rows when no variant is present."""
    patient_key = fields.get("patient_id", "patient_id")
    sample_key = fields.get("sample_id", "sample_id")
    patient = record.get(patient_key, record.get("patient_id"))
    sample = record.get(sample_key, record.get("sample_id"))
    label = record.get(fields.get("label", "label"), record.get("label"))
    rows: List[Dict[str, Any]] = []
    for variant in CANONICAL_VARIANTS:
        payload = record.get(variant)
        score = None
        mask = None
        entities = None
        target_entity = record.get("target_entity")
        payload_artifact = None
        payload_continuity = None
        if isinstance(payload, Mapping):
            score = payload.get("score", payload.get("anomaly_score"))
            mask = payload.get("mask", payload.get("pred_mask", payload.get("mask_path")))
            entities = payload.get("entities", payload.get("report_entities"))
            target_entity = payload.get("target_entity", target_entity)
            payload_quality = payload.get("quality", {})
            if not isinstance(payload_quality, Mapping):
                payload_quality = {}
            payload_artifact = payload.get(
                "artifact_score", payload_quality.get("artifact_score")
            )
            payload_continuity = payload.get(
                "continuity_score", payload_quality.get("continuity_score")
            )
        elif payload is not None and np.isscalar(payload):
            score = payload
        if score is None:
            score = record.get("%s_score" % variant, record.get("score_%s" % variant))
        if mask is None:
            mask = record.get("%s_mask" % variant, record.get("mask_%s" % variant))
        if entities is None:
            entities = record.get(
                "%s_entities" % variant,
                record.get("entities_%s" % variant),
            )
        if score is not None:
            artifact_value = record.get("%s_artifact_score" % variant)
            continuity_value = record.get("%s_continuity_score" % variant)
            if artifact_value is None:
                artifact_value = payload_artifact
            if continuity_value is None:
                continuity_value = payload_continuity
            rows.append(
                {
                    "patient_id": patient,
                    "sample_id": sample,
                    "label": label,
                    "variant": variant,
                    "score": score,
                    "mask": mask,
                    "entities": entities,
                    "target_entity": target_entity,
                    "artifact_score": artifact_value,
                    "continuity_score": continuity_value,
                    "_base_directory": record.get("_base_directory", ".") or ".",
                }
            )
    return rows


def _read_jsonl(path: Path) -> List[Dict[str, Any]]:
    records: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError("invalid JSON at %s:%d: %s" % (path, line_number, exc))
            if not isinstance(record, dict):
                raise ValueError("JSONL row %d must be an object" % line_number)
            record["_base_directory"] = str(path.parent)
            records.append(record)
    return records


def _npz_value(array: np.ndarray, index: int) -> Any:
    value = array[index]
    if isinstance(value, np.ndarray):
        return np.array(value, copy=True)
    if isinstance(value, np.generic):
        return value.item()
    return value


def _read_npz(path: Path, config: Mapping[str, Any]) -> List[Dict[str, Any]]:
    fields = config.get("fields", {})
    allow_pickle = bool(config.get("allow_pickle", False))
    with np.load(str(path), allow_pickle=allow_pickle) as archive:
        arrays = {key: np.array(archive[key], copy=True) for key in archive.files}

    patient_key = fields.get("patient_id", "patient_id")
    variant_key = fields.get("variant", "variant")
    score_key = fields.get("score", "score")
    mask_key = fields.get("mask", "mask")
    sample_key = fields.get("sample_id", "sample_id")
    label_key = fields.get("label", "label")
    entities_key = fields.get("entities", "entities")
    target_entity_key = fields.get("target_entity", "target_entity")
    if patient_key not in arrays:
        for alternative in ("patient_ids", "patient", "case_id"):
            if alternative in arrays:
                patient_key = alternative
                break
    if patient_key not in arrays:
        raise ValueError("NPZ %s has no patient_id array" % path)
    patient_array = arrays[patient_key]
    try:
        count = int(len(patient_array))
    except TypeError:
        raise ValueError("NPZ patient_id must have a leading record dimension")

    def require_aligned(keys: Sequence[str]) -> None:
        for key in keys:
            if key not in arrays:
                continue
            try:
                key_count = len(arrays[key])
            except TypeError:
                raise ValueError("NPZ array %s needs a leading record dimension" % key)
            if key_count != count:
                raise ValueError("NPZ array %s does not align with patient_id" % key)

    if mask_key not in arrays:
        for alternative in ("pred_mask", "prediction_mask", "segmentation"):
            if alternative in arrays:
                mask_key = alternative
                break
    if entities_key not in arrays and "report_entities" in arrays:
        entities_key = "report_entities"

    records: List[Dict[str, Any]] = []
    if variant_key in arrays or "variant" in arrays or "condition" in arrays:
        if variant_key not in arrays:
            variant_key = "variant" if "variant" in arrays else "condition"
        if score_key not in arrays:
            for alternative in ("anomaly_score", "prediction_score", "probability"):
                if alternative in arrays:
                    score_key = alternative
                    break
        if score_key not in arrays:
            raise ValueError("NPZ %s has no score array" % path)
        optional_long_keys = [
            sample_key,
            label_key,
            mask_key,
            entities_key,
            target_entity_key,
            fields.get("artifact_score", "artifact_score"),
            fields.get("continuity_score", "continuity_score"),
        ]
        require_aligned([variant_key, score_key] + optional_long_keys)
        for index in range(count):
            row = {
                "patient_id": _npz_value(patient_array, index),
                "variant": _npz_value(arrays[variant_key], index),
                "score": _npz_value(arrays[score_key], index),
                "_base_directory": str(path.parent),
            }
            if sample_key in arrays:
                row["sample_id"] = _npz_value(arrays[sample_key], index)
            if label_key in arrays:
                row["label"] = _npz_value(arrays[label_key], index)
            if mask_key in arrays:
                row["mask"] = _npz_value(arrays[mask_key], index)
            if entities_key in arrays:
                row["entities"] = _npz_value(arrays[entities_key], index)
            if target_entity_key in arrays:
                row["target_entity"] = _npz_value(arrays[target_entity_key], index)
            for quality_key in ("artifact_score", "continuity_score"):
                actual_key = fields.get(quality_key, quality_key)
                if actual_key in arrays:
                    row[quality_key] = _npz_value(arrays[actual_key], index)
            records.append(row)
        return records

    # Wide NPZ: patient_id plus original_score, lesion_removed_score, etc.
    wide_keys = [sample_key, label_key, target_entity_key]
    for variant in CANONICAL_VARIANTS:
        wide_keys.extend(
            [
                "%s_score" % variant,
                "score_%s" % variant,
                "%s_mask" % variant,
                "mask_%s" % variant,
                "%s_entities" % variant,
                "entities_%s" % variant,
            ]
        )
    require_aligned(wide_keys)
    for index in range(count):
        wide: Dict[str, Any] = {
            "patient_id": _npz_value(patient_array, index),
            "_base_directory": str(path.parent),
        }
        if sample_key in arrays:
            wide["sample_id"] = _npz_value(arrays[sample_key], index)
        if label_key in arrays:
            wide["label"] = _npz_value(arrays[label_key], index)
        if target_entity_key in arrays:
            wide["target_entity"] = _npz_value(arrays[target_entity_key], index)
        for variant in CANONICAL_VARIANTS:
            for candidate in ("%s_score" % variant, "score_%s" % variant):
                if candidate in arrays:
                    wide["%s_score" % variant] = _npz_value(arrays[candidate], index)
                    break
            for candidate in ("%s_mask" % variant, "mask_%s" % variant):
                if candidate in arrays:
                    wide["%s_mask" % variant] = _npz_value(arrays[candidate], index)
                    break
            for candidate in ("%s_entities" % variant, "entities_%s" % variant):
                if candidate in arrays:
                    wide["%s_entities" % variant] = _npz_value(arrays[candidate], index)
                    break
        records.append(wide)
    return records


def load_model_outputs(config: Mapping[str, Any]) -> List[Dict[str, Any]]:
    """Load raw model-output records from config ``records`` or JSONL/NPZ files."""
    if "records" in config:
        raw_records = config["records"]
        if not isinstance(raw_records, list):
            raise ValueError("config['records'] must be a list")
        return [dict(record) for record in raw_records]

    paths = config.get("input_paths", config.get("input_path"))
    if paths is None:
        raise ValueError("config needs input_path, input_paths, or records")
    if isinstance(paths, (str, Path)):
        paths = [paths]
    records: List[Dict[str, Any]] = []
    for raw_path in paths:
        path = Path(raw_path)
        if not path.exists():
            raise FileNotFoundError(str(path))
        format_name = str(config.get("format", path.suffix.lstrip("."))).lower()
        if format_name in ("jsonl", "jsonlines", "ndjson"):
            records.extend(_read_jsonl(path))
        elif format_name == "npz":
            records.extend(_read_npz(path, config))
        else:
            raise ValueError("unsupported model-output format %r" % format_name)
    return records


def _normalize_entities(value: Any) -> Optional[Tuple[str, ...]]:
    if value is None:
        return None
    if isinstance(value, str):
        values = [value]
    elif isinstance(value, (list, tuple, set, np.ndarray)):
        values = list(value)
    else:
        raise ValueError("entities/report_entities must be a list of strings")
    normalized = []
    for entity in values:
        if isinstance(entity, np.generic):
            entity = entity.item()
        if isinstance(entity, bytes):
            entity = entity.decode("utf-8")
        if not isinstance(entity, str):
            raise ValueError("each report entity must be a string")
        entity = entity.strip()
        if entity:
            normalized.append(entity)
    return tuple(sorted(set(normalized)))


def _normalize_label(value: Any) -> Optional[int]:
    if value is None:
        return None
    scalar = np.asarray(value)
    if scalar.size != 1:
        raise ValueError("label must be a scalar 0 or 1")
    raw = scalar.reshape(()).item()
    if isinstance(raw, str):
        raw = raw.strip()
        if raw not in ("0", "1"):
            raise ValueError("label must be 0 or 1")
        return int(raw)
    if isinstance(raw, (bool, np.bool_)):
        return int(raw)
    try:
        numeric = float(raw)
    except (TypeError, ValueError):
        raise ValueError("label must be 0 or 1")
    if not np.isfinite(numeric) or numeric not in (0.0, 1.0):
        raise ValueError("label must be 0 or 1")
    return int(numeric)


def _normalize_records(raw_records: Iterable[Mapping[str, Any]], config: Mapping[str, Any]) -> List[Dict[str, Any]]:
    fields: Mapping[str, str] = config.get("fields", {})
    variant_map: Mapping[str, str] = config.get("variant_map", {})
    allow_pickle = bool(config.get("allow_pickle", False))
    skip_invalid = bool(config.get("skip_invalid", False))
    normalized: List[Dict[str, Any]] = []

    for record_index, original_record in enumerate(raw_records):
        record = dict(original_record)
        raw_variant = _field(record, fields, "variant", ("variant", "condition", "intervention"))
        candidate_rows = [record] if raw_variant is not None else _wide_record_rows(record, fields)
        if not candidate_rows:
            if skip_invalid:
                continue
            raise ValueError("record %d has neither a variant nor wide variant scores" % record_index)
        for row in candidate_rows:
            try:
                patient = _field(row, fields, "patient_id", ("patient_id", "patient", "case_id"))
                raw_variant = _field(row, fields, "variant", ("variant", "condition", "intervention"))
                score = _field(
                    row,
                    fields,
                    "score",
                    ("score", "anomaly_score", "prediction_score", "probability"),
                )
                if patient is None or raw_variant is None or score is None:
                    raise ValueError("patient_id, variant, and score are required")
                score_value = float(np.asarray(score).reshape(()))
                if not np.isfinite(score_value):
                    raise ValueError("score is NaN or infinite")

                base_directory = Path(str(row.get("_base_directory", ".")))
                mask_value = _field(
                    row,
                    fields,
                    "mask",
                    ("mask", "pred_mask", "prediction_mask", "segmentation", "mask_path"),
                )
                mask = _load_mask(mask_value, base_directory, allow_pickle=allow_pickle)
                sample = _field(row, fields, "sample_id", ("sample_id", "study_id", "finding_id"))
                label = _normalize_label(
                    _field(row, fields, "label", ("label", "is_anomaly", "binary_label"))
                )
                entities_value = _field(
                    row,
                    fields,
                    "entities",
                    ("entities", "report_entities"),
                )
                entities = _normalize_entities(entities_value)
                target_entity = _field(
                    row,
                    fields,
                    "target_entity",
                    ("target_entity",),
                )
                if target_entity is not None:
                    if isinstance(target_entity, (bytes, np.bytes_)):
                        target_entity = bytes(target_entity).decode("utf-8")
                    if not isinstance(target_entity, str):
                        raise ValueError("target_entity must be a string")
                    target_entity = target_entity.strip() or None
                quality = row.get("quality", {}) if isinstance(row.get("quality"), Mapping) else {}
                artifact = _field(row, fields, "artifact_score", ("artifact_score",))
                continuity = _field(row, fields, "continuity_score", ("continuity_score",))
                if artifact is None:
                    artifact = quality.get("artifact_score")
                if continuity is None:
                    continuity = quality.get("continuity_score")
                normalized.append(
                    {
                        "patient_id": (
                            bytes(patient).decode("utf-8")
                            if isinstance(patient, (bytes, np.bytes_))
                            else str(patient)
                        ),
                        "sample_id": (
                            None
                            if sample is None
                            else (
                                bytes(sample).decode("utf-8")
                                if isinstance(sample, (bytes, np.bytes_))
                                else str(sample)
                            )
                        ),
                        "label": label,
                        "variant": _canonical_variant(raw_variant, variant_map),
                        "score": score_value,
                        "mask": mask,
                        "entities": entities,
                        "target_entity": target_entity,
                        "artifact_score": None if artifact is None else float(artifact),
                        "continuity_score": None if continuity is None else float(continuity),
                        "_order": len(normalized),
                    }
                )
            except (TypeError, ValueError, OSError) as exc:
                if skip_invalid:
                    continue
                raise ValueError("invalid model-output record %d: %s" % (record_index, exc))
    unique_records: List[Dict[str, Any]] = []
    seen_sample_variants = set()
    for row in normalized:
        if row["sample_id"] is not None:
            key = (row["patient_id"], row["sample_id"], row["variant"])
            if key in seen_sample_variants:
                if skip_invalid:
                    continue
                raise ValueError(
                    "duplicate patient_id/sample_id/variant row: %r" % (key,)
                )
            seen_sample_variants.add(key)
        row["_order"] = len(unique_records)
        unique_records.append(row)
    normalized = unique_records
    if not normalized:
        raise ValueError("no valid model-output records")
    return normalized


def _patient_variant_groups(records: Sequence[Mapping[str, Any]]) -> Dict[str, Dict[str, List[Mapping[str, Any]]]]:
    groups: Dict[str, Dict[str, List[Mapping[str, Any]]]] = {}
    for record in records:
        groups.setdefault(str(record["patient_id"]), {}).setdefault(str(record["variant"]), []).append(record)
    return groups


def _patient_labels(
    groups: Mapping[str, Mapping[str, Sequence[Mapping[str, Any]]]],
) -> Tuple[Dict[str, Optional[int]], bool]:
    labels: Dict[str, Optional[int]] = {}
    any_label = False
    for patient_id, variants in groups.items():
        observed = {
            int(row["label"])
            for rows in variants.values()
            for row in rows
            if row.get("label") is not None
        }
        if len(observed) > 1:
            raise ValueError("patient %s has inconsistent labels" % patient_id)
        label = next(iter(observed)) if observed else None
        labels[patient_id] = label
        any_label = any_label or label is not None
    return labels, any_label


def _paired_patient_scores(
    baseline: Sequence[Mapping[str, Any]],
    changed: Sequence[Mapping[str, Any]],
) -> Optional[Tuple[float, float]]:
    """Return paired baseline/changed means, matching sample IDs when present."""
    baseline_has_ids = any(row.get("sample_id") is not None for row in baseline)
    changed_has_ids = any(row.get("sample_id") is not None for row in changed)
    if baseline_has_ids or changed_has_ids:
        if not all(row.get("sample_id") is not None for row in baseline):
            return None
        if not all(row.get("sample_id") is not None for row in changed):
            return None
        baseline_by_id: Dict[str, List[float]] = {}
        changed_by_id: Dict[str, List[float]] = {}
        for row in baseline:
            if row.get("sample_id") is not None:
                baseline_by_id.setdefault(str(row["sample_id"]), []).append(float(row["score"]))
        for row in changed:
            if row.get("sample_id") is not None:
                changed_by_id.setdefault(str(row["sample_id"]), []).append(float(row["score"]))
        shared = sorted(set(baseline_by_id).intersection(changed_by_id))
        if not shared:
            return None
        baseline_mean = float(
            np.mean([float(np.mean(baseline_by_id[sample_id])) for sample_id in shared])
        )
        changed_mean = float(
            np.mean([float(np.mean(changed_by_id[sample_id])) for sample_id in shared])
        )
        return baseline_mean, changed_mean
    if len(baseline) != len(changed):
        return None
    return (
        float(np.mean([float(row["score"]) for row in baseline])),
        float(np.mean([float(row["score"]) for row in changed])),
    )


def _paired_patient_score_delta(
    baseline: Sequence[Mapping[str, Any]],
    changed: Sequence[Mapping[str, Any]],
) -> Optional[float]:
    scores = _paired_patient_scores(baseline, changed)
    if scores is None:
        return None
    return float(scores[1] - scores[0])


def _paired_score_deltas(
    groups: Mapping[str, Mapping[str, Sequence[Mapping[str, Any]]]],
    bootstrap: Mapping[str, Any],
) -> Tuple[Dict[str, Any], Dict[str, List[float]]]:
    result: Dict[str, Any] = {}
    patient_values: Dict[str, List[float]] = {}
    iterations = int(bootstrap["iterations"])
    confidence = float(bootstrap["confidence"])
    seed = int(bootstrap["seed"])
    for variant_index, variant in enumerate(CANONICAL_VARIANTS[1:]):
        deltas: List[float] = []
        patient_ids: List[str] = []
        unmatched_patients = 0
        for patient_id, variants in groups.items():
            if "original" not in variants or variant not in variants:
                continue
            delta = _paired_patient_score_delta(variants["original"], variants[variant])
            if delta is None:
                unmatched_patients += 1
                continue
            deltas.append(delta)
            patient_ids.append(patient_id)
        patient_values[variant] = deltas
        absolute = [abs(value) for value in deltas]
        result[variant] = {
            "definition": "%s - original" % variant,
            "n_patients": len(deltas),
            "n_unmatched_patients": unmatched_patients,
            "mean_delta": float(np.mean(deltas)) if deltas else None,
            "median_delta": float(np.median(deltas)) if deltas else None,
            "mean_absolute_delta": float(np.mean(absolute)) if absolute else None,
            "mean_delta_ci": bootstrap_ci(
                deltas, iterations, confidence, seed + 11 * variant_index
            ),
            "mean_absolute_delta_ci": bootstrap_ci(
                absolute, iterations, confidence, seed + 11 * variant_index + 1
            ),
        }
        if bootstrap.get("include_patient_values", False):
            result[variant]["patient_deltas"] = dict(zip(patient_ids, deltas))
    return result, patient_values


def _normal_cohort_fpr(
    groups: Mapping[str, Mapping[str, Sequence[Mapping[str, Any]]]],
    patient_labels: Mapping[str, Optional[int]],
    labels_available: bool,
    decision_threshold: Optional[float],
    bootstrap: Mapping[str, Any],
) -> Dict[str, Any]:
    iterations = int(bootstrap["iterations"])
    confidence = float(bootstrap["confidence"])
    seed = int(bootstrap["seed"])
    by_variant: Dict[str, Any] = {}
    paired_changes: Dict[str, Any] = {}

    if not labels_available:
        reason = "label is absent"
    elif decision_threshold is None:
        reason = "decision_threshold is absent"
    else:
        reason = None

    for variant_index, variant in enumerate(CANONICAL_VARIANTS):
        patient_predictions: List[float] = []
        if reason is None:
            for patient_id, variants in groups.items():
                if patient_labels.get(patient_id) != 0 or variant not in variants:
                    continue
                patient_score = float(
                    np.mean([float(row["score"]) for row in variants[variant]])
                )
                patient_predictions.append(
                    float(patient_score >= float(decision_threshold))
                )
        ci = bootstrap_ci(
            patient_predictions,
            iterations,
            confidence,
            seed + 503 + 17 * variant_index,
        )
        by_variant[variant] = {
            "status": "evaluable" if patient_predictions else "not_evaluable",
            "n_normal_patients": len(patient_predictions),
            "fpr": float(np.mean(patient_predictions)) if patient_predictions else None,
            "fpr_percent": (
                100.0 * float(np.mean(patient_predictions))
                if patient_predictions
                else None
            ),
            "fpr_ci": ci,
            "reason": reason if not patient_predictions else None,
        }

    for variant_index, variant in enumerate(("nuisance_changed", "placebo")):
        patient_changes: List[float] = []
        unmatched_patients = 0
        if reason is None:
            for patient_id, variants in groups.items():
                if patient_labels.get(patient_id) != 0:
                    continue
                if "original" not in variants or variant not in variants:
                    continue
                scores = _paired_patient_scores(
                    variants["original"], variants[variant]
                )
                if scores is None:
                    unmatched_patients += 1
                    continue
                original_positive = float(scores[0] >= float(decision_threshold))
                changed_positive = float(scores[1] >= float(decision_threshold))
                patient_changes.append(100.0 * (changed_positive - original_positive))
        ci = bootstrap_ci(
            patient_changes,
            iterations,
            confidence,
            seed + 577 + 19 * variant_index,
        )
        paired_changes[variant] = {
            "status": "evaluable" if patient_changes else "not_evaluable",
            "n_normal_patients": len(patient_changes),
            "n_unmatched_patients": unmatched_patients,
            "mean_change_percentage_points": (
                float(np.mean(patient_changes)) if patient_changes else None
            ),
            "mean_change_percentage_points_ci": ci,
            "definition": "%s FPR minus original FPR, paired by patient" % variant,
            "reason": reason if not patient_changes else None,
        }

    normal_patient_count = sum(1 for label in patient_labels.values() if label == 0)
    global_evaluable = reason is None and normal_patient_count > 0
    if reason is None and normal_patient_count == 0:
        reason = "no label=0 patients"
    return {
        "status": "evaluable" if global_evaluable else "not_evaluable",
        "reason": reason,
        "decision_threshold": decision_threshold,
        "decision_rule": (
            "score >= decision_threshold" if decision_threshold is not None else None
        ),
        "n_normal_patients": normal_patient_count,
        "by_variant": by_variant,
        "paired_change_vs_original": paired_changes,
    }


def _paired_rows(
    baseline: Sequence[Mapping[str, Any]],
    changed: Sequence[Mapping[str, Any]],
) -> List[Tuple[Mapping[str, Any], Mapping[str, Any]]]:
    if not baseline or not changed:
        return []
    baseline_any_ids = any(row.get("sample_id") is not None for row in baseline)
    changed_any_ids = any(row.get("sample_id") is not None for row in changed)
    if baseline_any_ids or changed_any_ids:
        if not all(row.get("sample_id") is not None for row in baseline):
            return []
        if not all(row.get("sample_id") is not None for row in changed):
            return []
    baseline_with_ids = {str(row["sample_id"]): row for row in baseline if row.get("sample_id") is not None}
    changed_with_ids = {str(row["sample_id"]): row for row in changed if row.get("sample_id") is not None}
    pairs: List[Tuple[Mapping[str, Any], Mapping[str, Any]]] = []
    shared = sorted(set(baseline_with_ids).intersection(changed_with_ids))
    if baseline_with_ids or changed_with_ids:
        for sample_id in shared:
            pairs.append((baseline_with_ids[sample_id], changed_with_ids[sample_id]))
        return pairs
    if len(baseline) == 1 and len(changed) == 1:
        return [(baseline[0], changed[0])]
    return pairs


def _mask_pairs(
    baseline: Sequence[Mapping[str, Any]],
    changed: Sequence[Mapping[str, Any]],
) -> List[Tuple[np.ndarray, np.ndarray]]:
    baseline = [row for row in baseline if row.get("mask") is not None]
    changed = [row for row in changed if row.get("mask") is not None]
    return [
        (left["mask"], right["mask"])
        for left, right in _paired_rows(baseline, changed)
    ]


def _mask_stability(
    groups: Mapping[str, Mapping[str, Sequence[Mapping[str, Any]]]],
    bootstrap: Mapping[str, Any],
) -> Dict[str, Any]:
    result: Dict[str, Any] = {}
    iterations = int(bootstrap["iterations"])
    confidence = float(bootstrap["confidence"])
    seed = int(bootstrap["seed"])
    for variant_index, variant in enumerate(CANONICAL_VARIANTS[1:]):
        patient_dice: List[float] = []
        patient_mae: List[float] = []
        shape_mismatches = 0
        both_empty_pairs = 0
        patients_with_only_both_empty = 0
        for variants in groups.values():
            if "original" not in variants or variant not in variants:
                continue
            dice_values: List[float] = []
            mae_values: List[float] = []
            patient_both_empty = 0
            for baseline_mask, changed_mask in _mask_pairs(variants["original"], variants[variant]):
                if baseline_mask.shape != changed_mask.shape:
                    shape_mismatches += 1
                    continue
                if (
                    float(np.sum(np.abs(baseline_mask))) <= 1e-12
                    and float(np.sum(np.abs(changed_mask))) <= 1e-12
                ):
                    both_empty_pairs += 1
                    patient_both_empty += 1
                    continue
                dice_values.append(soft_dice(baseline_mask, changed_mask))
                mae_values.append(float(np.mean(np.abs(baseline_mask - changed_mask))))
            if dice_values:
                patient_dice.append(float(np.mean(dice_values)))
                patient_mae.append(float(np.mean(mae_values)))
            elif patient_both_empty:
                patients_with_only_both_empty += 1
        result[variant] = {
            "status": "evaluable" if patient_dice else "not_evaluable",
            "n_patients": len(patient_dice),
            "mean_soft_dice": float(np.mean(patient_dice)) if patient_dice else None,
            "mean_soft_dice_ci": bootstrap_ci(
                patient_dice, iterations, confidence, seed + 101 + 13 * variant_index
            ),
            "mean_mask_mae": float(np.mean(patient_mae)) if patient_mae else None,
            "mean_mask_mae_ci": bootstrap_ci(
                patient_mae, iterations, confidence, seed + 102 + 13 * variant_index
            ),
            "shape_mismatch_pairs": shape_mismatches,
            "both_empty_pairs_excluded": both_empty_pairs,
            "patients_with_only_both_empty_masks": patients_with_only_both_empty,
        }
    return result


def _entity_jaccard(first: Sequence[str], second: Sequence[str]) -> float:
    left = set(first)
    right = set(second)
    union = left | right
    if not union:
        return 1.0
    return float(len(left & right)) / float(len(union))


def _report_entity_endpoints(
    groups: Mapping[str, Mapping[str, Sequence[Mapping[str, Any]]]],
    bootstrap: Mapping[str, Any],
) -> Dict[str, Any]:
    """Structured report consistency endpoints; no free-text overlap metric."""
    iterations = int(bootstrap["iterations"])
    confidence = float(bootstrap["confidence"])
    seed = int(bootstrap["seed"])
    jaccard_result: Dict[str, Any] = {}

    for variant_index, variant in enumerate(("nuisance_changed", "placebo")):
        patient_values: List[float] = []
        sample_pair_count = 0
        for variants in groups.values():
            if "original" not in variants or variant not in variants:
                continue
            sample_values = []
            for original_row, changed_row in _paired_rows(
                variants["original"], variants[variant]
            ):
                original_entities = original_row.get("entities")
                changed_entities = changed_row.get("entities")
                if original_entities is None or changed_entities is None:
                    continue
                sample_values.append(
                    _entity_jaccard(original_entities, changed_entities)
                )
            if sample_values:
                patient_values.append(float(np.mean(sample_values)))
                sample_pair_count += len(sample_values)
        ci = bootstrap_ci(
            patient_values,
            iterations,
            confidence,
            seed + 401 + 19 * variant_index,
        )
        jaccard_result[variant] = {
            "status": "evaluable" if patient_values else "not_evaluable",
            "n_patients": len(patient_values),
            "n_sample_pairs": sample_pair_count,
            "mean_jaccard": float(np.mean(patient_values)) if patient_values else None,
            "mean_jaccard_ci": ci,
        }

    disappearance_by_patient: List[float] = []
    eligible_sample_pairs = 0
    for variants in groups.values():
        if "original" not in variants or "lesion_removed" not in variants:
            continue
        patient_disappearance = []
        for original_row, removed_row in _paired_rows(
            variants["original"], variants["lesion_removed"]
        ):
            original_entities = original_row.get("entities")
            removed_entities = removed_row.get("entities")
            target = original_row.get("target_entity") or removed_row.get("target_entity")
            if original_entities is None or removed_entities is None or target is None:
                continue
            if target not in set(original_entities):
                # A disappearance test is only meaningful when the target was
                # emitted for the unmodified image.
                continue
            patient_disappearance.append(float(target not in set(removed_entities)))
        if patient_disappearance:
            disappearance_by_patient.append(float(np.mean(patient_disappearance)))
            eligible_sample_pairs += len(patient_disappearance)
    disappearance_ci = bootstrap_ci(
        disappearance_by_patient,
        iterations,
        confidence,
        seed + 449,
    )
    disappearance = {
        "status": "evaluable" if disappearance_by_patient else "not_evaluable",
        "n_patients": len(disappearance_by_patient),
        "n_eligible_sample_pairs": eligible_sample_pairs,
        "disappearance_rate": (
            float(np.mean(disappearance_by_patient))
            if disappearance_by_patient
            else None
        ),
        "disappearance_rate_ci": disappearance_ci,
        "eligibility": "target_entity must be present in original report_entities",
    }
    return {
        "entity_jaccard": jaccard_result,
        "target_entity_disappearance_after_lesion_removal": disappearance,
    }


def _quality_summary(
    groups: Mapping[str, Mapping[str, Sequence[Mapping[str, Any]]]],
    bootstrap: Mapping[str, Any],
) -> Dict[str, Any]:
    result: Dict[str, Any] = {}
    for variant_index, variant in enumerate(CANONICAL_VARIANTS):
        variant_result: Dict[str, Any] = {}
        for metric_index, metric in enumerate(("artifact_score", "continuity_score")):
            patient_values = []
            for variants in groups.values():
                values = [
                    float(row[metric])
                    for row in variants.get(variant, [])
                    if row.get(metric) is not None and np.isfinite(float(row[metric]))
                ]
                if values:
                    patient_values.append(float(np.mean(values)))
            if patient_values:
                variant_result[metric] = {
                    "n_patients": len(patient_values),
                    "mean": float(np.mean(patient_values)),
                    "ci": bootstrap_ci(
                        patient_values,
                        int(bootstrap["iterations"]),
                        float(bootstrap["confidence"]),
                        int(bootstrap["seed"]) + 211 + 17 * variant_index + metric_index,
                    ),
                }
        if variant_result:
            result[variant] = variant_result
    return result


def _rule_result(
    name: str,
    observed: Optional[float],
    operator: str,
    threshold: float,
    explanation: str,
) -> Dict[str, Any]:
    if observed is None or not np.isfinite(float(observed)):
        return {
            "rule": name,
            "observed": None,
            "operator": operator,
            "threshold": float(threshold),
            "triggered": None,
            "status": "not_evaluable",
            "explanation": explanation,
        }
    if operator == "<":
        triggered = float(observed) < float(threshold)
    elif operator == ">":
        triggered = float(observed) > float(threshold)
    else:  # pragma: no cover - only internal fixed operators are used.
        raise ValueError("unsupported rule operator")
    return {
        "rule": name,
        "observed": float(observed),
        "operator": operator,
        "threshold": float(threshold),
        "triggered": bool(triggered),
        "status": "red_flag" if triggered else "pass",
        "explanation": explanation,
    }


def _red_flag_decisions(
    score_deltas: Mapping[str, Any],
    mask_stability: Mapping[str, Any],
    normal_cohort_fpr: Mapping[str, Any],
    rules: Mapping[str, float],
) -> Dict[str, Any]:
    lesion_delta = score_deltas.get("lesion_removed", {}).get("mean_delta")
    lesion_drop = None if lesion_delta is None else -float(lesion_delta)
    nuisance_abs = score_deltas.get("nuisance_changed", {}).get("mean_absolute_delta")
    placebo_abs = score_deltas.get("placebo", {}).get("mean_absolute_delta")
    nuisance_dice = mask_stability.get("nuisance_changed", {}).get("mean_soft_dice")
    placebo_dice = mask_stability.get("placebo", {}).get("mean_soft_dice")
    nuisance_score_n = int(
        score_deltas.get("nuisance_changed", {}).get("n_patients", 0)
    )
    placebo_score_n = int(score_deltas.get("placebo", {}).get("n_patients", 0))
    nuisance_mask_n = int(
        mask_stability.get("nuisance_changed", {}).get("n_patients", 0)
    )
    placebo_mask_n = int(mask_stability.get("placebo", {}).get("n_patients", 0))
    nuisance_mask_coverage = (
        float(nuisance_mask_n) / float(nuisance_score_n)
        if nuisance_score_n > 0
        else None
    )
    placebo_mask_coverage = (
        float(placebo_mask_n) / float(placebo_score_n)
        if placebo_score_n > 0
        else None
    )
    nuisance_fpr_increase = normal_cohort_fpr.get(
        "paired_change_vs_original", {}
    ).get("nuisance_changed", {}).get("mean_change_percentage_points")
    placebo_fpr_increase = normal_cohort_fpr.get(
        "paired_change_vs_original", {}
    ).get("placebo", {}).get("mean_change_percentage_points")
    ratio = None
    if lesion_drop is not None and nuisance_abs is not None and lesion_drop > 1e-12:
        ratio = float(nuisance_abs) / float(lesion_drop)

    decisions = [
        _rule_result(
            "lesion_response_too_small",
            lesion_drop,
            "<",
            rules["min_lesion_score_drop"],
            "Original minus lesion-removed anomaly score should exceed the preregistered minimum.",
        ),
        _rule_result(
            "nuisance_score_sensitivity",
            nuisance_abs,
            ">",
            rules["max_nuisance_mean_abs_delta"],
            "Background-only changes should not materially change the anomaly score.",
        ),
        _rule_result(
            "placebo_score_sensitivity",
            placebo_abs,
            ">",
            rules["max_placebo_mean_abs_delta"],
            "A size-matched edit away from the lesion should have little effect.",
        ),
        _rule_result(
            "nuisance_mask_instability",
            nuisance_dice,
            "<",
            rules["min_nuisance_mask_dice"],
            "Predicted lesion masks should remain stable under protected nuisance changes.",
        ),
        _rule_result(
            "placebo_mask_instability",
            placebo_dice,
            "<",
            rules["min_placebo_mask_dice"],
            "Predicted lesion masks should remain stable under placebo edits.",
        ),
        _rule_result(
            "nuisance_to_lesion_ratio",
            ratio,
            ">",
            rules["max_nuisance_to_lesion_ratio"],
            "Background sensitivity should be small relative to the true lesion-removal response.",
        ),
        _rule_result(
            "insufficient_nuisance_mask_coverage",
            nuisance_mask_coverage,
            "<",
            rules["min_mask_patient_coverage"],
            "Nuisance mask stability must cover enough of the paired score cohort.",
        ),
        _rule_result(
            "insufficient_placebo_mask_coverage",
            placebo_mask_coverage,
            "<",
            rules["min_mask_patient_coverage"],
            "Placebo mask stability must cover enough of the paired score cohort.",
        ),
        _rule_result(
            "normal_nuisance_fpr_increase",
            nuisance_fpr_increase,
            ">",
            rules["max_normal_nuisance_fpr_increase_pp"],
            "Nuisance changes must not materially increase false positives in normal patients.",
        ),
        _rule_result(
            "normal_placebo_fpr_increase",
            placebo_fpr_increase,
            ">",
            rules["max_normal_placebo_fpr_increase_pp"],
            "Placebo edits must not materially increase false positives in normal patients.",
        ),
    ]

    paired_counts = [
        int(score_deltas.get(variant, {}).get("n_patients", 0))
        for variant in CANONICAL_VARIANTS[1:]
    ]
    minimum_count = min(paired_counts) if paired_counts else 0
    decisions.append(
        _rule_result(
            "insufficient_paired_patients",
            float(minimum_count),
            "<",
            rules["min_paired_patients"],
            "Every intervention comparison must meet the preregistered patient count.",
        )
    )
    required_decisions = [
        decision
        for decision in decisions
        if not (
            decision["rule"]
            in ("normal_nuisance_fpr_increase", "normal_placebo_fpr_increase")
            and decision["triggered"] is None
        )
    ]
    evaluated = [
        decision
        for decision in required_decisions
        if decision["triggered"] is not None
    ]
    has_flag = any(bool(decision["triggered"]) for decision in evaluated)
    has_missing = len(evaluated) != len(required_decisions)
    status = "red_flag" if has_flag else ("indeterminate" if has_missing else "pass")
    return {
        "status": status,
        "any_red_flag": bool(has_flag),
        "decisions": decisions,
        "derived_metrics": {
            "mean_lesion_score_drop": lesion_drop,
            "nuisance_to_lesion_ratio": ratio,
            "nuisance_mask_patient_coverage": nuisance_mask_coverage,
            "placebo_mask_patient_coverage": placebo_mask_coverage,
            "normal_nuisance_fpr_increase_percentage_points": nuisance_fpr_increase,
            "normal_placebo_fpr_increase_percentage_points": placebo_fpr_increase,
        },
    }


def run_shortcut_evaluation(config_dict: Mapping[str, Any]) -> Dict[str, Any]:
    """Run patient-level shortcut evaluation and return JSON-serializable data.

    Important config keys
    ---------------------
    input_path / input_paths:
        JSONL or NPZ model output.  The extension is used unless ``format`` is
        supplied.
    fields:
        Optional canonical-to-file field mapping.  Canonical optional fields
        include ``label``, ``entities``/``report_entities`` and
        ``target_entity`` in addition to score/mask identifiers.
    decision_threshold:
        Optional finite score threshold.  Together with ``label=0`` rows it
        enables patient-level normal-cohort FPR endpoints.
    bootstrap:
        ``iterations``, ``confidence`` and ``seed``.  Resampling units are
        patients, after within-patient scores have been averaged.
    red_flag_rules:
        Fixed thresholds recorded verbatim in the result.  Defaults are provided
        for smoke tests, but confirm them prospectively for a real study.
    """
    if not isinstance(config_dict, Mapping):
        raise TypeError("config_dict must be a mapping")
    raw_records = load_model_outputs(config_dict)
    records = _normalize_records(raw_records, config_dict)
    groups = _patient_variant_groups(records)
    patient_labels, labels_available = _patient_labels(groups)
    lesion_groups = (
        {
            patient_id: variants
            for patient_id, variants in groups.items()
            if patient_labels.get(patient_id) == 1
        }
        if labels_available
        else groups
    )

    raw_decision_threshold = config_dict.get("decision_threshold")
    decision_threshold = (
        None if raw_decision_threshold is None else float(raw_decision_threshold)
    )
    if decision_threshold is not None and not np.isfinite(decision_threshold):
        raise ValueError("decision_threshold must be finite")

    bootstrap_config = dict(config_dict.get("bootstrap", {}))
    bootstrap_config.setdefault("iterations", int(config_dict.get("bootstrap_iterations", 2000)))
    bootstrap_config.setdefault("confidence", float(config_dict.get("confidence", 0.95)))
    bootstrap_config.setdefault("seed", int(config_dict.get("seed", 0)))
    bootstrap_config.setdefault("include_patient_values", bool(config_dict.get("include_patient_values", False)))
    bootstrap_config["iterations"] = int(bootstrap_config["iterations"])
    bootstrap_config["confidence"] = float(bootstrap_config["confidence"])
    bootstrap_config["seed"] = int(bootstrap_config["seed"])
    if bootstrap_config["iterations"] <= 0:
        raise ValueError("bootstrap iterations must be positive")
    if not 0.0 < bootstrap_config["confidence"] < 1.0:
        raise ValueError("bootstrap confidence must lie in (0, 1)")

    score_deltas, _ = _paired_score_deltas(lesion_groups, bootstrap_config)
    mask_stability = _mask_stability(lesion_groups, bootstrap_config)
    report_entity_endpoints = _report_entity_endpoints(
        lesion_groups, bootstrap_config
    )
    normal_cohort_fpr = _normal_cohort_fpr(
        groups,
        patient_labels,
        labels_available,
        decision_threshold,
        bootstrap_config,
    )
    quality = _quality_summary(groups, bootstrap_config)

    supplied_rules = config_dict.get(
        "red_flag_rules",
        config_dict.get("preregistered_rules", config_dict.get("red_flags", {})),
    )
    if not isinstance(supplied_rules, Mapping):
        raise ValueError("red_flag_rules must be a mapping")
    unknown_rules = set(supplied_rules).difference(DEFAULT_RED_FLAG_RULES)
    if unknown_rules:
        raise ValueError("unknown red-flag rules: %s" % sorted(unknown_rules))
    rules = dict(DEFAULT_RED_FLAG_RULES)
    numeric_rules = {str(key): float(value) for key, value in dict(supplied_rules).items()}
    if any(not np.isfinite(value) for value in numeric_rules.values()):
        raise ValueError("red-flag thresholds must be finite")
    if any(value < 0 for value in numeric_rules.values()):
        raise ValueError("red-flag thresholds must be non-negative")
    rules.update(numeric_rules)
    missing_rule_names = set(DEFAULT_RED_FLAG_RULES).difference(rules)
    if missing_rule_names:  # Defensive; defaults make this unreachable.
        raise ValueError("missing red-flag rules: %s" % sorted(missing_rule_names))
    decisions = _red_flag_decisions(
        score_deltas, mask_stability, normal_cohort_fpr, rules
    )

    patients_by_variant = {
        variant: sum(1 for variants in groups.values() if variant in variants)
        for variant in CANONICAL_VARIANTS
    }
    result = {
        "schema_version": "1.0",
        "analysis_unit": "patient",
        "n_records": len(records),
        "n_patients": len(groups),
        "patients_by_variant": patients_by_variant,
        "paired_score_deltas": score_deltas,
        "mask_stability": mask_stability,
        "report_entity_endpoints": report_entity_endpoints,
        "normal_cohort_fpr": normal_cohort_fpr,
        "intervention_quality": quality,
        "analysis_cohorts": {
            "labels_available": labels_available,
            "n_label_1_patients": sum(
                1 for label in patient_labels.values() if label == 1
            ),
            "n_label_0_patients": sum(
                1 for label in patient_labels.values() if label == 0
            ),
            "n_unlabeled_patients": sum(
                1 for label in patient_labels.values() if label is None
            ),
            "paired_lesion_metrics_cohort": (
                "label=1" if labels_available else "all patients (label absent)"
            ),
        },
        "bootstrap": {
            "iterations": int(bootstrap_config["iterations"]),
            "confidence": float(bootstrap_config["confidence"]),
            "seed": int(bootstrap_config["seed"]),
            "resampling_unit": "patient",
        },
        "preregistered_red_flag_rules": rules,
        "red_flag_assessment": decisions,
        "notes": [
            "Score delta is variant minus original after within-patient aggregation.",
            "Mask stability is paired within patient/sample and then averaged per patient.",
            "Report endpoints use structured entity sets (Jaccard/disappearance), not BLEU.",
            "When labels exist, paired lesion metrics use only label=1 patients.",
            "Normal-cohort FPR uses score >= decision_threshold at patient level.",
            "Default thresholds are placeholders; a confirmatory study should preregister overrides.",
        ],
    }
    safe_result = _json_safe(result)
    # Fail here rather than returning a structure containing NaN/Infinity.
    json.dumps(safe_result, allow_nan=False)
    return safe_result


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Evaluate lesion/background shortcut interventions")
    parser.add_argument("config", help="Path to a JSON configuration file")
    parser.add_argument("--output", help="Optional output JSON path")
    arguments = parser.parse_args(argv)
    with Path(arguments.config).open("r", encoding="utf-8") as handle:
        config = json.load(handle)
    result = run_shortcut_evaluation(config)
    rendered = json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False)
    if arguments.output:
        Path(arguments.output).write_text(rendered + "\n", encoding="utf-8")
    else:
        print(rendered)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())


__all__ = [
    "bootstrap_ci",
    "load_model_outputs",
    "run_shortcut_evaluation",
    "soft_dice",
]
