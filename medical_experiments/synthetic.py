"""Deterministic synthetic artifacts for installation smoke tests only.

These arrays verify cache schemas, controls, and command wiring.  They are not
medical evidence and must never be included in scientific result tables.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Mapping

from .config import resolve_path, write_json
from .feature_cache import save_feature_record
from .manifest import write_jsonl


def generate_smoke_data(config: Mapping[str, Any]) -> Dict[str, Any]:
    try:
        import numpy as np
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("NumPy is required to generate smoke-test data") from exc

    output_dir = resolve_path(config.get("output_dir", "outputs/medical-smoke"), config)
    array_dir = output_dir / "arrays"
    cache_dir = output_dir / "detail_cache"
    mask_dir = output_dir / "predicted_masks"
    for directory in (array_dir, cache_dir, mask_dir):
        directory.mkdir(parents=True, exist_ok=True)
    seed = int(config.get("seed", 17))
    rng = np.random.RandomState(seed)
    size = int(config.get("image_size", 16))
    num_patients = int(config.get("num_patients", 36))
    if num_patients < 18:
        raise ValueError("smoke suite requires at least 18 patients")

    manifest = []
    intervention_rows = []
    yy, xx = np.mgrid[:size, :size]
    for index in range(num_patients):
        patient_id = "smoke-p%03d" % index
        sample_id = patient_id + "-s0"
        fraction = index / float(num_patients)
        split = "train" if fraction < 0.6 else ("val" if fraction < 0.8 else "test")
        positive = index % 3 != 0
        radius = 1 + index % 3
        center_y = 3 + (index * 3) % max(size - 6, 1)
        center_x = 3 + (index * 5) % max(size - 6, 1)
        mask = (((yy - center_y) ** 2 + (xx - center_x) ** 2) <= radius ** 2).astype("float32")
        if not positive:
            mask.fill(0)
        normal = rng.normal(0.0, 0.15, (size, size)).astype("float32")
        image = normal + mask * 1.2
        image_path = array_dir / (sample_id + ".npy")
        mask_path = array_dir / (sample_id + "-mask.npy")
        normal_path = array_dir / (sample_id + "-normal.npy")
        np.save(str(image_path), image, allow_pickle=False)
        np.save(str(mask_path), mask, allow_pickle=False)
        np.save(str(normal_path), normal, allow_pickle=False)
        record = {
            "sample_id": sample_id,
            "patient_id": patient_id,
            "image": str(image_path.resolve()),
            "mask": str(mask_path.resolve()),
            "normal_reference": str(normal_path.resolve()),
            "split": split,
            "modality": "CT",
            "label": int(positive),
        }
        manifest.append(record)

        noise = rng.normal(0.0, 0.25, (4, size, size)).astype("float32")
        encoder_early = noise.copy()
        encoder_early[0] += mask * 2.5
        projector = noise.copy()
        projector[0] += mask * 0.55
        llm_deep = rng.normal(0.0, 0.35, (4, size, size)).astype("float32")
        llm_deep[0] += mask * 0.2
        residual = rng.normal(0.0, 0.08, (2, size, size)).astype("float32")
        residual[0] += mask * 2.8
        save_feature_record(
            cache_dir,
            record,
            {
                "encoder_early": encoder_early,
                "projector": projector,
                "llm_deep": llm_deep,
                "lafite_residual": residual,
                "mask": mask,
            },
            overwrite=True,
        )

        if split == "test":
            prediction_mask_path = mask_dir / (sample_id + ".npy")
            np.save(str(prediction_mask_path), mask, allow_pickle=False)
            if positive:
                scores = {
                    "original": 0.85 + rng.normal(0, 0.01),
                    "lesion_removed": 0.22 + rng.normal(0, 0.01),
                    "nuisance_changed": 0.82 + rng.normal(0, 0.01),
                    "placebo": 0.84 + rng.normal(0, 0.01),
                }
            else:
                scores = {
                    "original": 0.14 + rng.normal(0, 0.01),
                    "nuisance_changed": 0.17 + rng.normal(0, 0.01),
                }
            for variant, score in scores.items():
                if positive:
                    entities = [] if variant == "lesion_removed" else ["synthetic_finding"]
                else:
                    entities = []
                intervention_rows.append(
                    {
                        "sample_id": sample_id,
                        "patient_id": patient_id,
                        "variant": variant,
                        "label": int(positive),
                        "score": float(score),
                        "mask": str(prediction_mask_path.resolve()),
                        "report_entities": entities,
                        "target_entity": "synthetic_finding" if positive else None,
                        "artifact_score": 0.01 if variant != "original" else 0.0,
                        "continuity_score": 0.99 if variant != "original" else 1.0,
                    }
                )

    manifest_path = output_dir / "manifest.jsonl"
    intervention_path = output_dir / "model_outputs.jsonl"
    write_jsonl(manifest_path, manifest)
    write_jsonl(intervention_path, intervention_rows)

    organs = ("organ_a", "organ_b", "organ_c")
    findings = ("finding_x", "finding_y", "finding_z")
    semantic_records = []
    semantic_index = 0
    for repeat in range(4):
        for organ_index, organ in enumerate(organs):
            for finding_index, finding in enumerate(findings):
                patient_id = "semantic-p%03d" % semantic_index
                region = "region_%d" % ((organ_index + finding_index + repeat) % 3)
                correct = {candidate: (-2.0 if candidate != finding else 2.0) for candidate in findings}
                no_image = {candidate: 0.0 for candidate in findings}
                shuffled_target = findings[(finding_index + 1) % len(findings)]
                shuffled = {candidate: (-2.0 if candidate != shuffled_target else 2.0) for candidate in findings}
                semantic_records.append(
                    {
                        "patient_id": patient_id,
                        "sample_id": patient_id + "-s0",
                        "organ": organ,
                        "region": region,
                        "finding": finding,
                        "scores": correct,
                        "scores_no_image": no_image,
                        "scores_shuffled_knowledge": shuffled,
                        "scores_no_image_shuffled_knowledge": no_image,
                        "source": "synthetic-smoke-only",
                    }
                )
                semantic_index += 1
    semantic_path = output_dir / "semantic_manifest.jsonl"
    write_jsonl(semantic_path, semantic_records)
    result = {
        "warning": "synthetic smoke data are not scientific evidence",
        "manifest": str(manifest_path),
        "detail_cache": str(cache_dir),
        "intervention_outputs": str(intervention_path),
        "semantic_manifest": str(semantic_path),
        "patients": num_patients,
    }
    write_json(output_dir / "smoke_data_summary.json", result)
    return result
