"""Patient-safe compositional splits for the LLM-necessity experiment.

The manifest is JSONL with one object per patient and the required string fields
``patient_id``, ``organ``, ``finding`` and ``region``.  Optional ``split`` and
``source`` values are preserved but are *not* used as the generated evaluation
split.  Duplicate patient rows are resolved before any edge selection.  A
patient whose required fields (or source/split provenance) conflict is rejected
by default because silently choosing one row would hide patient leakage.

The generated ``unseen`` set contains complete held-out organ x finding edges.
Every organ and every finding in an unseen edge must still occur in ``fit``.
``seen`` is a patient-level validation subset drawn from the remaining edges;
each of its edges is retained in ``fit``.
"""

from __future__ import annotations

import json
import os
import random
from collections import Counter, defaultdict
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple


REQUIRED_FIELDS = ("patient_id", "organ", "finding", "region")
PROVENANCE_FIELDS = ("source", "split")
Edge = Tuple[str, str]
Record = Dict[str, Any]


def _normalise_record(record: Mapping[str, Any], line_number: Optional[int] = None) -> Record:
    if not isinstance(record, Mapping):
        where = "" if line_number is None else " on line %d" % line_number
        raise ValueError("Each manifest entry must be a JSON object%s." % where)

    result = dict(record)
    for field in REQUIRED_FIELDS:
        value = result.get(field)
        if not isinstance(value, str) or not value.strip():
            where = "" if line_number is None else " on line %d" % line_number
            raise ValueError("Field %r must be a non-empty string%s." % (field, where))
        result[field] = value.strip()

    for field in PROVENANCE_FIELDS:
        if field in result and result[field] is not None:
            if not isinstance(result[field], str) or not result[field].strip():
                where = "" if line_number is None else " on line %d" % line_number
                raise ValueError("Field %r must be a non-empty string%s." % (field, where))
            result[field] = result[field].strip()
    return result


def load_jsonl_records(manifest: Any) -> List[Record]:
    """Load and validate records from a JSONL path or an iterable of mappings."""

    if isinstance(manifest, (str, bytes, os.PathLike)):
        records = []
        with open(os.fspath(manifest), "r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, 1):
                if not line.strip():
                    continue
                try:
                    raw = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(
                        "Invalid JSON on line %d of %s: %s" % (line_number, manifest, exc)
                    ) from exc
                records.append(_normalise_record(raw, line_number=line_number))
        if not records:
            raise ValueError("Manifest %s contains no records." % manifest)
        return records

    if not isinstance(manifest, Iterable):
        raise TypeError("manifest must be a JSONL path or an iterable of records.")
    records = [_normalise_record(record, line_number=index) for index, record in enumerate(manifest, 1)]
    if not records:
        raise ValueError("Manifest contains no records.")
    return records


def deduplicate_patients(
    records: Sequence[Mapping[str, Any]], on_conflict: str = "error"
) -> Tuple[List[Record], Dict[str, int]]:
    """Return one deterministic row per patient before splitting.

    Exact/reconcilable duplicate rows retain the first row and record a
    ``duplicate_count``.  Conflicts in target fields or provenance raise unless
    ``on_conflict='first'`` is explicitly requested.
    """

    if on_conflict not in ("error", "first"):
        raise ValueError("on_conflict must be 'error' or 'first'.")

    by_patient: Dict[str, Record] = {}
    duplicate_counts: Counter = Counter()
    conflict_count = 0
    for raw in records:
        record = _normalise_record(raw)
        patient_id = record["patient_id"]
        if patient_id not in by_patient:
            by_patient[patient_id] = record
            duplicate_counts[patient_id] = 1
            continue

        duplicate_counts[patient_id] += 1
        previous = by_patient[patient_id]
        compared_fields = REQUIRED_FIELDS[1:] + PROVENANCE_FIELDS
        conflicts = [
            field
            for field in compared_fields
            if previous.get(field) != record.get(field)
        ]
        if conflicts:
            conflict_count += 1
            if on_conflict == "error":
                raise ValueError(
                    "Patient %r has conflicting duplicate rows in fields: %s. "
                    "Resolve the patient before generating a split."
                    % (patient_id, ", ".join(conflicts))
                )

    deduplicated = []
    for patient_id in sorted(by_patient):
        record = dict(by_patient[patient_id])
        record["duplicate_count"] = int(duplicate_counts[patient_id])
        deduplicated.append(record)

    report = {
        "input_records": int(len(records)),
        "unique_patients": int(len(deduplicated)),
        "duplicates_removed": int(len(records) - len(deduplicated)),
        "conflicting_duplicates_ignored": int(conflict_count if on_conflict == "first" else 0),
    }
    return deduplicated, report


def _edge(record: Mapping[str, Any]) -> Edge:
    return str(record["organ"]), str(record["finding"])


def _parse_edge(raw: Any) -> Edge:
    if isinstance(raw, Mapping):
        organ = raw.get("organ")
        finding = raw.get("finding")
    elif isinstance(raw, str):
        if "||" not in raw:
            raise ValueError("String edges must use the form 'organ||finding'.")
        organ, finding = raw.split("||", 1)
    elif isinstance(raw, Sequence) and not isinstance(raw, (str, bytes)) and len(raw) == 2:
        organ, finding = raw
    else:
        raise ValueError("Each held-out edge must be [organ, finding], a mapping, or 'organ||finding'.")

    if not isinstance(organ, str) or not organ.strip():
        raise ValueError("Held-out edge organ must be a non-empty string.")
    if not isinstance(finding, str) or not finding.strip():
        raise ValueError("Held-out edge finding must be a non-empty string.")
    return organ.strip(), finding.strip()


def _validate_node_coverage(records: Sequence[Mapping[str, Any]], held_out_edges: Sequence[Edge]) -> None:
    held = set(held_out_edges)
    all_edges = Counter(_edge(record) for record in records)
    missing = [edge for edge in held if edge not in all_edges]
    if missing:
        raise ValueError("Held-out edges are absent from the manifest: %s" % sorted(missing))

    train = [record for record in records if _edge(record) not in held]
    train_organs = {str(record["organ"]) for record in train}
    train_findings = {str(record["finding"]) for record in train}
    uncovered_organs = sorted({organ for organ, _ in held} - train_organs)
    uncovered_findings = sorted({finding for _, finding in held} - train_findings)
    if uncovered_organs or uncovered_findings:
        raise ValueError(
            "Held-out edges remove nodes from training; uncovered organs=%s, findings=%s."
            % (uncovered_organs, uncovered_findings)
        )


def select_held_out_edges(
    records: Sequence[Mapping[str, Any]],
    held_out_edges: Optional[Sequence[Any]] = None,
    num_edges: Optional[int] = None,
    seed: int = 0,
    min_edge_records: int = 1,
    max_attempts: int = 512,
) -> List[Edge]:
    """Select complete edges while retaining every incident node in training."""

    if min_edge_records < 1:
        raise ValueError("min_edge_records must be at least one.")
    if held_out_edges is not None:
        parsed = sorted(set(_parse_edge(edge) for edge in held_out_edges))
        if not parsed:
            raise ValueError("held_out_edges cannot be empty.")
        if num_edges is not None and int(num_edges) != len(parsed):
            raise ValueError("num_edges disagrees with the number of explicit held_out_edges.")
        _validate_node_coverage(records, parsed)
        return parsed

    if num_edges is None or int(num_edges) < 1:
        raise ValueError("Specify non-empty held_out_edges or num_edges >= 1.")
    requested = int(num_edges)
    edge_counts = Counter(_edge(record) for record in records)
    candidates = sorted(edge for edge, count in edge_counts.items() if count >= min_edge_records)
    if requested > len(candidates):
        raise ValueError("Requested %d held-out edges, but only %d are eligible." % (requested, len(candidates)))

    organ_totals = Counter(str(record["organ"]) for record in records)
    finding_totals = Counter(str(record["finding"]) for record in records)
    rng = random.Random(int(seed))
    for _ in range(max_attempts):
        order = list(candidates)
        rng.shuffle(order)
        remaining_organs = organ_totals.copy()
        remaining_findings = finding_totals.copy()
        selected = []
        for organ, finding in order:
            count = edge_counts[(organ, finding)]
            if remaining_organs[organ] - count < 1:
                continue
            if remaining_findings[finding] - count < 1:
                continue
            selected.append((organ, finding))
            remaining_organs[organ] -= count
            remaining_findings[finding] -= count
            if len(selected) == requested:
                result = sorted(selected)
                _validate_node_coverage(records, result)
                return result
    raise ValueError(
        "Could not select %d valid held-out edges after %d attempts; reduce num_edges "
        "or add cross-organ/finding support." % (requested, max_attempts)
    )


def _make_seen_validation(
    train_pool: Sequence[Record], fraction: float, seed: int
) -> Tuple[List[Record], List[Record]]:
    if fraction < 0.0 or fraction >= 1.0:
        raise ValueError("seen_eval_fraction must be in [0, 1).")
    if fraction == 0.0:
        return list(train_pool), []

    groups: Dict[Edge, List[Record]] = defaultdict(list)
    for record in train_pool:
        groups[_edge(record)].append(record)

    rng = random.Random(int(seed) + 104729)
    fit: List[Record] = []
    seen: List[Record] = []
    for edge in sorted(groups):
        group = sorted(groups[edge], key=lambda item: item["patient_id"])
        rng.shuffle(group)
        if len(group) < 2:
            fit.extend(group)
            continue
        eval_count = max(1, int(round(len(group) * fraction)))
        eval_count = min(eval_count, len(group) - 1)
        seen.extend(group[:eval_count])
        fit.extend(group[eval_count:])
    return sorted(fit, key=lambda item: item["patient_id"]), sorted(
        seen, key=lambda item: item["patient_id"]
    )


def assert_no_leakage(
    fit: Sequence[Mapping[str, Any]],
    seen: Sequence[Mapping[str, Any]],
    unseen: Sequence[Mapping[str, Any]],
    held_out_edges: Sequence[Any],
) -> Dict[str, Any]:
    """Raise on patient, edge, or node leakage and return a compact audit."""

    held = set(_parse_edge(edge) for edge in held_out_edges)
    patient_sets = {
        "fit": {str(record["patient_id"]) for record in fit},
        "seen": {str(record["patient_id"]) for record in seen},
        "unseen": {str(record["patient_id"]) for record in unseen},
    }
    overlaps = {
        "fit_seen": sorted(patient_sets["fit"] & patient_sets["seen"]),
        "fit_unseen": sorted(patient_sets["fit"] & patient_sets["unseen"]),
        "seen_unseen": sorted(patient_sets["seen"] & patient_sets["unseen"]),
    }
    if any(overlaps.values()):
        raise ValueError("Patient leakage detected: %s" % overlaps)

    fit_edges = {_edge(record) for record in fit}
    seen_edges = {_edge(record) for record in seen}
    unseen_edges = {_edge(record) for record in unseen}
    if fit_edges & held or seen_edges & held:
        raise ValueError("A held-out organ x finding edge occurs in fit/seen data.")
    if unseen_edges != held:
        raise ValueError(
            "Unseen edge set does not exactly match held-out edges: unseen=%s, held=%s."
            % (sorted(unseen_edges), sorted(held))
        )
    if seen and not seen_edges.issubset(fit_edges):
        raise ValueError("Every seen validation edge must remain represented in fit.")

    fit_organs = {str(record["organ"]) for record in fit}
    fit_findings = {str(record["finding"]) for record in fit}
    missing_organs = sorted({organ for organ, _ in held} - fit_organs)
    missing_findings = sorted({finding for _, finding in held} - fit_findings)
    if missing_organs or missing_findings:
        raise ValueError(
            "Held-out nodes disappeared after validation sampling; organs=%s, findings=%s."
            % (missing_organs, missing_findings)
        )

    return {
        "patient_overlap": overlaps,
        "held_out_edge_overlap_with_fit": 0,
        "held_out_edge_overlap_with_seen": 0,
        "all_unseen_nodes_present_in_fit": True,
        "fit_patients": len(patient_sets["fit"]),
        "seen_patients": len(patient_sets["seen"]),
        "unseen_patients": len(patient_sets["unseen"]),
    }


def build_compositional_split(
    records: Sequence[Mapping[str, Any]],
    held_out_edges: Optional[Sequence[Any]] = None,
    num_edges: Optional[int] = None,
    seed: int = 0,
    seen_eval_fraction: float = 0.2,
    min_edge_records: int = 1,
    patient_conflict: str = "error",
) -> Dict[str, Any]:
    """Deduplicate patients, select unseen edges, and create fit/seen/unseen sets."""

    normalised = [_normalise_record(record) for record in records]
    unique_records, deduplication = deduplicate_patients(normalised, on_conflict=patient_conflict)
    selected = select_held_out_edges(
        unique_records,
        held_out_edges=held_out_edges,
        num_edges=num_edges,
        seed=seed,
        min_edge_records=min_edge_records,
    )
    held = set(selected)
    unseen = sorted(
        [record for record in unique_records if _edge(record) in held],
        key=lambda item: item["patient_id"],
    )
    train_pool = [record for record in unique_records if _edge(record) not in held]
    fit, seen = _make_seen_validation(train_pool, seen_eval_fraction, seed)
    if seen_eval_fraction > 0.0 and not seen:
        raise ValueError(
            "No seen-validation patients could be sampled; each non-held-out edge needs "
            "at least two patients or seen_eval_fraction must be zero."
        )
    leakage = assert_no_leakage(fit, seen, unseen, selected)
    return {
        "fit": fit,
        "seen": seen,
        "unseen": unseen,
        "held_out_edges": [[organ, finding] for organ, finding in selected],
        "deduplication": deduplication,
        "leakage_audit": leakage,
    }
