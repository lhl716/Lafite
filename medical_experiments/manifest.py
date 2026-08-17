"""JSONL manifest schema and patient-level leakage guards.

Each line minimally contains::

    {"sample_id": "case001-z042", "patient_id": "case001",
     "image": "/data/case001.npy", "mask": "/data/case001_mask.npy",
     "split": "train", "modality": "CT", "label": 1}

Additional experiment-three fields (``organ``, ``region``, ``finding``,
``report``) are preserved verbatim.
"""

from __future__ import annotations

import hashlib
import json
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Mapping, MutableMapping, Optional, Sequence, Tuple, Union


PathLike = Union[str, Path]
ALLOWED_SPLITS = {"train", "val", "test", "calibration", "external"}
REQUIRED_FIELDS = ("sample_id", "patient_id", "image", "split")


def normalize_split(value: Any) -> str:
    """Return the canonical split spelling used by every experiment stage."""

    return str(value).strip().lower()


def read_jsonl(path: PathLike) -> List[Dict[str, Any]]:
    records: List[Dict[str, Any]] = []
    with Path(path).expanduser().open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            text = line.strip()
            if not text:
                continue
            try:
                record = json.loads(text)
            except json.JSONDecodeError as exc:
                raise ValueError("Invalid JSON on line %d of %s" % (line_number, path)) from exc
            if not isinstance(record, dict):
                raise ValueError("Manifest line %d is not an object" % line_number)
            records.append(record)
    return records


def write_jsonl(path: PathLike, records: Iterable[Mapping[str, Any]]) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(dict(record), ensure_ascii=False, sort_keys=True))
            handle.write("\n")


def _resolve_record_path(value: str, manifest_path: Optional[PathLike]) -> Path:
    path = Path(value).expanduser()
    if not path.is_absolute() and manifest_path is not None:
        path = Path(manifest_path).expanduser().resolve().parent / path
    return path.resolve()


def validate_manifest(
    records: Sequence[Mapping[str, Any]],
    *,
    manifest_path: Optional[PathLike] = None,
    check_paths: bool = False,
    require_mask: bool = False,
    require_semantics: bool = False,
    progress: bool = False,
) -> Dict[str, Any]:
    """Validate schema and fail on patient leakage across data splits."""

    errors: List[str] = []
    warnings: List[str] = []
    sample_ids = set()
    patient_splits: MutableMapping[str, set] = defaultdict(set)
    split_counts: Counter = Counter()
    modality_counts: Counter = Counter()

    indexed_records: Any = enumerate(records)
    if progress:
        try:
            from tqdm.auto import tqdm
        except ImportError:  # pragma: no cover - minimal environments
            def plain_progress():
                total = len(records)
                for index, record in enumerate(records):
                    if index == 0 or (index + 1) % 1000 == 0 or index + 1 == total:
                        print(
                            "Validating manifest paths: {}/{}".format(index + 1, total),
                            flush=True,
                        )
                    yield index, record

            indexed_records = plain_progress()
        else:
            indexed_records = tqdm(
                indexed_records,
                total=len(records),
                desc="Validating manifest paths",
                unit="record",
                dynamic_ncols=True,
            )

    for index, record in indexed_records:
        prefix = "record[%d]" % index
        missing = [field for field in REQUIRED_FIELDS if record.get(field) in (None, "")]
        if missing:
            errors.append("%s missing fields: %s" % (prefix, ", ".join(missing)))
            continue

        sample_id = str(record["sample_id"])
        if sample_id in sample_ids:
            errors.append("duplicate sample_id: %s" % sample_id)
        sample_ids.add(sample_id)

        patient_id = str(record["patient_id"])
        split = normalize_split(record["split"])
        if split not in ALLOWED_SPLITS:
            errors.append("%s has unsupported split %r" % (prefix, split))
        patient_splits[patient_id].add(split)
        split_counts[split] += 1
        modality_counts[str(record.get("modality", "unknown"))] += 1

        if "label" in record and record["label"] not in (0, 1, False, True):
            errors.append("%s label must be binary" % prefix)
        if require_mask and not record.get("mask"):
            errors.append("%s requires a lesion mask" % prefix)
        if require_semantics:
            semantic_missing = [field for field in ("organ", "finding") if not record.get(field)]
            if semantic_missing:
                errors.append("%s missing semantic fields: %s" % (prefix, ", ".join(semantic_missing)))

        if check_paths:
            for field in ("image", "mask", "normal_reference"):
                value = record.get(field)
                if value and not _resolve_record_path(str(value), manifest_path).exists():
                    errors.append("%s %s does not exist: %s" % (prefix, field, value))

    leaked = {patient: sorted(splits) for patient, splits in patient_splits.items() if len(splits) > 1}
    if leaked:
        preview = list(sorted(leaked.items()))[:10]
        errors.append("patient leakage across splits: %s" % preview)

    if not records:
        errors.append("manifest is empty")
    if len(split_counts) == 1:
        warnings.append("manifest contains only one split")

    result = {
        "valid": not errors,
        "errors": errors,
        "warnings": warnings,
        "num_samples": len(records),
        "num_patients": len(patient_splits),
        "split_counts": dict(sorted(split_counts.items())),
        "modality_counts": dict(sorted(modality_counts.items())),
    }
    return result


def assert_valid_manifest(records: Sequence[Mapping[str, Any]], **kwargs: Any) -> Dict[str, Any]:
    result = validate_manifest(records, **kwargs)
    if not result["valid"]:
        raise ValueError("Manifest validation failed:\n- " + "\n- ".join(result["errors"]))
    return result


def assign_patient_splits(
    records: Sequence[Mapping[str, Any]],
    ratios: Mapping[str, float],
    *,
    seed: int = 42,
    stratify_field: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """Assign whole patients to splits, optionally stratified by one field."""

    if not ratios or any(value < 0 for value in ratios.values()):
        raise ValueError("ratios must be non-negative")
    total = float(sum(ratios.values()))
    if total <= 0:
        raise ValueError("at least one split ratio must be positive")
    normalized = [(name, value / total) for name, value in ratios.items()]

    by_patient: MutableMapping[str, List[Mapping[str, Any]]] = defaultdict(list)
    for record in records:
        by_patient[str(record["patient_id"])].append(record)

    strata: MutableMapping[str, List[str]] = defaultdict(list)
    for patient_id, items in by_patient.items():
        if stratify_field:
            values = sorted({str(item.get(stratify_field, "unknown")) for item in items})
            stratum = "|".join(values)
        else:
            stratum = "all"
        strata[stratum].append(patient_id)

    rng = random.Random(seed)
    patient_to_split: Dict[str, str] = {}
    for patient_ids in strata.values():
        rng.shuffle(patient_ids)
        cumulative = []
        running = 0.0
        for name, ratio in normalized:
            running += ratio
            cumulative.append((name, running))
        for index, patient_id in enumerate(patient_ids):
            position = (index + 0.5) / len(patient_ids)
            selected = cumulative[-1][0]
            for name, boundary in cumulative:
                if position <= boundary:
                    selected = name
                    break
            patient_to_split[patient_id] = selected

    output: List[Dict[str, Any]] = []
    for record in records:
        item = dict(record)
        item["split"] = patient_to_split[str(record["patient_id"])]
        output.append(item)
    return output


def manifest_fingerprint(records: Sequence[Mapping[str, Any]]) -> str:
    """Stable SHA256 over clinically relevant identity and split fields."""

    canonical = []
    fields = ("sample_id", "patient_id", "split", "image", "mask", "modality", "label")
    for record in records:
        canonical.append({field: record.get(field) for field in fields})
    encoded = json.dumps(sorted(canonical, key=lambda item: str(item.get("sample_id"))), sort_keys=True).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()
