"""Prepare BMAD manifests for modality-specific LafitE training and evaluation."""

from __future__ import annotations

from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List, Mapping, MutableMapping, Sequence

from .manifest import manifest_fingerprint, read_jsonl, validate_manifest, write_jsonl


DATASETS = {
    "brain": "BMAD-Brain",
    "liver": "BMAD-Liver",
}


def _absolute_record(record: Mapping[str, Any], manifest_dir: Path) -> Dict[str, Any]:
    result = dict(record)
    for field in (
        "image",
        "mask",
        "nuisance_mask",
        "placebo_mask",
        "placebo_allowed_mask",
        "normal_reference",
        "style_reference",
    ):
        if not result.get(field):
            continue
        path = Path(str(result[field])).expanduser()
        if not path.is_absolute():
            path = manifest_dir / path
        result[field] = str(path.resolve())
    return result


def _lafite_row(record: Mapping[str, Any], clsname: str) -> Dict[str, Any]:
    row: Dict[str, Any] = {
        "filename": str(record["image"]),
        "label": int(record.get("label", 0)),
        "label_name": "defective" if int(record.get("label", 0)) else "good",
        "clsname": clsname,
        "sample_id": str(record["sample_id"]),
        "patient_id": str(record["patient_id"]),
    }
    if record.get("mask"):
        row["maskname"] = str(record["mask"])
    return row


def prepare_lafite_data(
    manifest_path: Path,
    output_dir: Path,
    *,
    strict_normal_patients: bool = True,
) -> Dict[str, Any]:
    """Create Brain/Liver diagnostic manifests and LafitE JSONL metadata.

    Training rows always contain only ``label=0`` slices.  With the default
    strict policy, a training slice is also excluded when any other slice from
    the same patient is anomalous.  This prevents patient-style leakage from a
    diseased patient into the unsupervised normal training cohort.
    """

    manifest_path = Path(manifest_path).expanduser().resolve()
    output_dir = Path(output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    records = read_jsonl(manifest_path)
    validation = validate_manifest(
        records,
        manifest_path=manifest_path,
        check_paths=True,
        require_mask=True,
        progress=True,
    )
    if not validation["valid"]:
        raise ValueError("medical manifest is invalid: %s" % validation["errors"])

    absolute = [_absolute_record(record, manifest_path.parent) for record in records]
    unknown = sorted(
        {
            str(record.get("source_dataset"))
            for record in absolute
            if str(record.get("source_dataset")) not in set(DATASETS.values())
        }
    )
    if unknown:
        raise ValueError("unsupported source_dataset values: %s" % unknown)

    result: Dict[str, Any] = {
        "source_manifest": str(manifest_path),
        "source_manifest_sha256": manifest_fingerprint(records),
        "output_dir": str(output_dir),
        "normal_training_policy": (
            "patient_strict" if strict_normal_patients else "normal_slices"
        ),
        "datasets": {},
    }
    for name, source_dataset in DATASETS.items():
        selected = [
            record
            for record in absolute
            if str(record.get("source_dataset")) == source_dataset
        ]
        if not selected:
            raise ValueError("no records found for %s" % source_dataset)
        patient_status: MutableMapping[str, int] = defaultdict(int)
        for record in selected:
            patient = str(record["patient_id"])
            patient_status[patient] = max(patient_status[patient], int(record.get("label", 0)))

        diagnostic_path = output_dir / (name + "-manifest.jsonl")
        print("Writing %s (%d records)" % (diagnostic_path.name, len(selected)), flush=True)
        write_jsonl(diagnostic_path, selected)
        # Every path here was already stat-ed by the whole-manifest validation
        # above, and re-checking costs another stat per image and mask -- minutes
        # on a network mount.  Re-validate the schema only.
        diagnostic_validation = validate_manifest(
            selected,
            manifest_path=diagnostic_path,
            check_paths=False,
            require_mask=True,
        )
        if not diagnostic_validation["valid"]:
            raise ValueError(
                "%s diagnostic manifest is invalid: %s"
                % (name, diagnostic_validation["errors"])
            )

        split_paths: Dict[str, str] = {}
        split_counts: Dict[str, int] = {}
        for split in ("train", "val", "test"):
            split_records = [record for record in selected if record.get("split") == split]
            if split == "train":
                split_records = [
                    record
                    for record in split_records
                    if int(record.get("label", 0)) == 0
                    and (
                        not strict_normal_patients
                        or patient_status[str(record["patient_id"])] == 0
                    )
                ]
            if not split_records:
                raise ValueError("%s %s cohort is empty" % (name, split))
            meta_path = output_dir / (name + "-" + split + ".jsonl")
            write_jsonl(
                meta_path,
                [_lafite_row(record, name) for record in split_records],
            )
            split_paths[split] = str(meta_path)
            split_counts[split] = len(split_records)

        result["datasets"][name] = {
            "source_dataset": source_dataset,
            "diagnostic_manifest": str(diagnostic_path),
            "lafite_meta": split_paths,
            "counts": split_counts,
            "patients": len({str(record["patient_id"]) for record in selected}),
            "label_counts": dict(
                sorted(Counter(int(record.get("label", 0)) for record in selected).items())
            ),
            "validation": diagnostic_validation,
        }
    return result
