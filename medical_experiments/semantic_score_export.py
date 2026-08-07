"""Export four-condition scores from a real LLM/MLLM into experiment-three JSONL.

The model-specific factory returns a callable with signature::

    score(record, candidates, evidence_present, knowledge_intact) -> {finding: logit}

The exporter strips target ``finding`` and ``report`` fields before every call
and removes image/evidence paths for no-image conditions.  This makes leakage
controls part of the executable protocol instead of a documentation promise.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Mapping, Sequence

from .config import resolve_path
from .hooks import import_object
from .manifest import read_jsonl, write_jsonl


CONDITION_FIELDS = {
    (True, True): "scores",
    (False, True): "scores_no_image",
    (True, False): "scores_shuffled_knowledge",
    (False, False): "scores_no_image_shuffled_knowledge",
}


def _load_scorer(config: Mapping[str, Any]):
    factory = import_object(str(config["model_factory"]))
    bundle = factory(dict(config.get("factory_kwargs", {})))
    if callable(bundle):
        return bundle
    if isinstance(bundle, Mapping) and callable(bundle.get("score")):
        return bundle["score"]
    raise TypeError("model_factory must return a callable or {'score': callable}")


def _visible_record(record: Mapping[str, Any], evidence_present: bool, config: Mapping[str, Any]) -> Dict[str, Any]:
    forbidden = set(config.get("target_fields", ["finding", "report", "label", "target", "ground_truth"]))
    visible = {key: value for key, value in record.items() if key not in forbidden}
    if not evidence_present:
        evidence_fields = set(
            config.get(
                "evidence_fields",
                ["image", "mask", "feature_cache", "lesion_feature", "region", "slice_index"],
            )
        )
        visible = {key: value for key, value in visible.items() if key not in evidence_fields}
    visible["evidence_present"] = bool(evidence_present)
    return visible


def export_semantic_scores(config: Mapping[str, Any]) -> Dict[str, Any]:
    manifest_path = resolve_path(config["manifest"], config)
    output_path = resolve_path(config["output_manifest"], config)
    records = read_jsonl(manifest_path)
    if not records:
        raise ValueError("semantic manifest is empty")
    for index, record in enumerate(records):
        missing = [field for field in ("patient_id", "organ", "region", "finding") if not record.get(field)]
        if missing:
            raise ValueError("semantic record %d is missing: %s" % (index, ", ".join(missing)))
    candidates = config.get("candidates") or sorted({str(record["finding"]) for record in records})
    candidates = [str(value) for value in candidates]
    if len(candidates) < 2:
        raise ValueError("at least two candidate findings are required")
    scorer = _load_scorer(config)
    output = []
    for record in records:
        enriched = dict(record)
        for (evidence_present, knowledge_intact), field in CONDITION_FIELDS.items():
            visible = _visible_record(record, evidence_present, config)
            raw_scores = scorer(
                visible,
                list(candidates),
                evidence_present=bool(evidence_present),
                knowledge_intact=bool(knowledge_intact),
            )
            if not isinstance(raw_scores, Mapping):
                raise TypeError("semantic scorer must return a finding-to-score mapping")
            missing = sorted(set(candidates) - {str(key) for key in raw_scores})
            if missing:
                raise ValueError("semantic scorer omitted candidates: %s" % missing)
            enriched[field] = {candidate: float(raw_scores[candidate]) for candidate in candidates}
        output.append(enriched)
    write_jsonl(output_path, output)
    return {
        "output_manifest": str(output_path),
        "samples": len(output),
        "candidates": candidates,
        "conditions": list(CONDITION_FIELDS.values()),
        "target_fields_stripped": list(config.get("target_fields", ["finding", "report", "label", "target", "ground_truth"])),
    }
