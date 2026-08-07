"""Patient-clustered statistical utilities."""

from __future__ import annotations

from collections import defaultdict
from typing import Any, Callable, Dict, Iterable, List, Mapping, Optional, Sequence


def _numpy():
    try:
        import numpy as np
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("NumPy is required for statistical evaluation") from exc
    return np


def patient_cluster_bootstrap(
    records: Sequence[Mapping[str, Any]],
    statistic: Callable[[Sequence[Mapping[str, Any]]], float],
    *,
    patient_key: str = "patient_id",
    iterations: int = 2000,
    confidence: float = 0.95,
    seed: int = 42,
) -> Dict[str, float]:
    """Bootstrap patients, retaining every slice/lesion from a sampled patient."""

    np = _numpy()
    if not records:
        raise ValueError("records must not be empty")
    grouped = defaultdict(list)
    for record in records:
        if patient_key not in record:
            raise KeyError("missing patient key %r" % patient_key)
        grouped[str(record[patient_key])].append(record)
    patients = sorted(grouped)
    rng = np.random.default_rng(seed)
    estimates = []
    for _ in range(int(iterations)):
        chosen = rng.choice(patients, size=len(patients), replace=True)
        sample = []
        for patient in chosen:
            sample.extend(grouped[str(patient)])
        estimates.append(float(statistic(sample)))
    alpha = (1.0 - confidence) / 2.0
    low, high = np.quantile(np.asarray(estimates), [alpha, 1.0 - alpha])
    return {
        "estimate": float(statistic(records)),
        "ci_low": float(low),
        "ci_high": float(high),
        "confidence": float(confidence),
        "iterations": int(iterations),
        "num_patients": len(patients),
    }


def paired_patient_delta(
    records: Sequence[Mapping[str, Any]],
    *,
    left_key: str,
    right_key: str,
    patient_key: str = "patient_id",
    **bootstrap_kwargs: Any,
) -> Dict[str, float]:
    """Mean paired difference with a patient-clustered confidence interval."""

    def statistic(sample: Sequence[Mapping[str, Any]]) -> float:
        np = _numpy()
        deltas = [float(item[left_key]) - float(item[right_key]) for item in sample]
        return float(np.mean(deltas))

    return patient_cluster_bootstrap(
        records,
        statistic,
        patient_key=patient_key,
        **bootstrap_kwargs,
    )
