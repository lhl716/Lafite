"""Semantic controls and runner for the LLM-necessity diagnostic.

Minimal ``run_llm_necessity`` configuration::

    {
        "manifest": "/path/to/records.jsonl",
        "output_dir": "/path/to/output",
        "seed": 7,
        "held_out_edges": [["brain", "edema"]],  # or "num_edges": 1
        "top_k": [1, 3, 5],
        "bootstrap": {"iterations": 2000, "confidence": 0.95, "seed": 7},
        "baseline": {"type": "train_frequency", "kwargs": {}}
    }

``baseline`` may also be a list, allowing uniform, train-frequency, learned
embedding, and externally precomputed LLM scores to be evaluated in one run.
The runner writes ``per_sample.jsonl`` and ``summary.json`` and returns the same
JSON-serialisable summary dictionary.

For built-in controls, ``region`` is only a *proxy* for case-level visual/local
evidence.  ``evidence_present=False`` explicitly removes region while retaining
organ as shared metadata.  ``knowledge_intact=False`` applies a deterministic
derangement to finding associations while keeping score distributions and model
capacity fixed.  A real MLLM experiment should use ``precomputed`` with four
condition-specific score fields produced by actual image/no-image inference.
"""

from __future__ import annotations

import json
import math
import os
import random
from abc import ABC, abstractmethod
from collections import Counter, defaultdict
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

from medical_experiments.compositional_split import build_compositional_split, load_jsonl_records


Record = Mapping[str, Any]


def _condition_key(evidence_present: bool, knowledge_intact: bool) -> str:
    return "evidence=%d|knowledge=%d" % (int(bool(evidence_present)), int(bool(knowledge_intact)))


def _make_derangement(values: Sequence[str], seed: int) -> Dict[str, str]:
    values = list(values)
    if len(values) < 2:
        return {value: value for value in values}
    rng = random.Random(int(seed) + 65537)
    for _ in range(256):
        shuffled = list(values)
        rng.shuffle(shuffled)
        if all(source != target for source, target in zip(values, shuffled)):
            return dict(zip(values, shuffled))
    shift = 1 + rng.randrange(len(values) - 1)
    rotated = values[shift:] + values[:shift]
    return dict(zip(values, rotated))


class SemanticRankingBaseline(ABC):
    """Unified fit/score/predict interface for finding-ranking controls."""

    baseline_type = "abstract"

    def __init__(self, seed: int = 0) -> None:
        self.seed = int(seed)
        self.candidates: List[str] = []
        self.knowledge_permutation: Dict[str, str] = {}
        self._is_fitted = False

    def fit(self, records: Sequence[Record], candidates: Sequence[str]) -> "SemanticRankingBaseline":
        unique_candidates = sorted(set(str(value) for value in candidates))
        if not unique_candidates:
            raise ValueError("At least one candidate finding is required.")
        self.candidates = unique_candidates
        self.knowledge_permutation = _make_derangement(unique_candidates, self.seed)
        self._fit(records)
        self._is_fitted = True
        return self

    def _fit(self, records: Sequence[Record]) -> None:
        del records

    @abstractmethod
    def _base_scores(self, record: Record, evidence_present: bool) -> Dict[str, float]:
        raise NotImplementedError

    def score(
        self,
        record: Record,
        candidates: Optional[Sequence[str]] = None,
        evidence_present: bool = True,
        knowledge_intact: bool = True,
    ) -> Dict[str, float]:
        if not self._is_fitted:
            raise RuntimeError("fit must be called before score.")
        requested = self.candidates if candidates is None else [str(value) for value in candidates]
        unknown = sorted(set(requested) - set(self.candidates))
        if unknown:
            raise ValueError("Unknown candidate findings: %s" % unknown)
        base = self._base_scores(record, bool(evidence_present))
        if knowledge_intact:
            return {candidate: float(base[candidate]) for candidate in requested}
        return {
            candidate: float(base[self.knowledge_permutation[candidate]])
            for candidate in requested
        }

    def predict(
        self,
        record: Record,
        top_k: int = 1,
        candidates: Optional[Sequence[str]] = None,
        evidence_present: bool = True,
        knowledge_intact: bool = True,
    ) -> List[str]:
        if int(top_k) < 1:
            raise ValueError("top_k must be at least one.")
        scores = self.score(
            record,
            candidates=candidates,
            evidence_present=evidence_present,
            knowledge_intact=knowledge_intact,
        )
        ranked = sorted(scores, key=lambda candidate: (-scores[candidate], candidate))
        return ranked[: min(int(top_k), len(ranked))]


class UniformBaseline(SemanticRankingBaseline):
    baseline_type = "uniform"

    def _base_scores(self, record: Record, evidence_present: bool) -> Dict[str, float]:
        del record, evidence_present
        return {candidate: 0.0 for candidate in self.candidates}


class TrainFrequencyBaseline(SemanticRankingBaseline):
    """Smoothed global/organ frequency with optional region evidence."""

    baseline_type = "train_frequency"

    def __init__(
        self,
        seed: int = 0,
        alpha: float = 1.0,
        global_weight: float = 0.25,
        organ_weight: float = 1.0,
        region_weight: float = 1.0,
    ) -> None:
        super().__init__(seed=seed)
        if alpha <= 0:
            raise ValueError("alpha must be positive.")
        self.alpha = float(alpha)
        self.global_weight = float(global_weight)
        self.organ_weight = float(organ_weight)
        self.region_weight = float(region_weight)
        self.global_counts: Counter = Counter()
        self.organ_counts: Dict[str, Counter] = defaultdict(Counter)
        self.region_counts: Dict[str, Counter] = defaultdict(Counter)

    def _fit(self, records: Sequence[Record]) -> None:
        self.global_counts.clear()
        self.organ_counts = defaultdict(Counter)
        self.region_counts = defaultdict(Counter)
        for record in records:
            finding = str(record["finding"])
            self.global_counts[finding] += 1
            self.organ_counts[str(record["organ"])][finding] += 1
            self.region_counts[str(record["region"])][finding] += 1

    def _base_scores(self, record: Record, evidence_present: bool) -> Dict[str, float]:
        organ_count = self.organ_counts.get(str(record["organ"]), Counter())
        region_count = (
            self.region_counts.get(str(record["region"]), Counter())
            if evidence_present
            else Counter()
        )
        scores = {}
        for candidate in self.candidates:
            mass = self.alpha
            mass += self.global_weight * float(self.global_counts[candidate])
            mass += self.organ_weight * float(organ_count[candidate])
            if evidence_present:
                mass += self.region_weight * float(region_count[candidate])
            scores[candidate] = math.log(max(mass, 1e-300))
        return scores


class LearnedEmbeddingBaseline(SemanticRankingBaseline):
    """Small full-softmax organ/region-to-finding embedding ranker.

    This is intentionally a lightweight semantic control rather than a medical
    image model.  It tests whether training-set associations can explain gains
    attributed to a large language model.
    """

    baseline_type = "learned_embedding"

    def __init__(
        self,
        seed: int = 0,
        embedding_dim: int = 16,
        epochs: int = 80,
        learning_rate: float = 0.05,
        l2: float = 1e-4,
        region_dropout: float = 0.25,
    ) -> None:
        super().__init__(seed=seed)
        if int(embedding_dim) < 1 or int(epochs) < 1:
            raise ValueError("embedding_dim and epochs must be positive.")
        if float(learning_rate) <= 0 or float(l2) < 0:
            raise ValueError("learning_rate must be positive and l2 non-negative.")
        if not 0.0 <= float(region_dropout) < 1.0:
            raise ValueError("region_dropout must be in [0, 1).")
        self.embedding_dim = int(embedding_dim)
        self.epochs = int(epochs)
        self.learning_rate = float(learning_rate)
        self.l2 = float(l2)
        self.region_dropout = float(region_dropout)
        self.organ_to_index: Dict[str, int] = {}
        self.region_to_index: Dict[str, int] = {}
        self.finding_to_index: Dict[str, int] = {}
        self.organ_embeddings: List[List[float]] = []
        self.region_embeddings: List[List[float]] = []
        self.finding_embeddings: List[List[float]] = []
        self.bias: List[float] = []

    def _fit(self, records: Sequence[Record]) -> None:
        organs = sorted({str(record["organ"]) for record in records})
        regions = sorted({str(record["region"]) for record in records})
        self.organ_to_index = {value: index for index, value in enumerate(organs)}
        self.region_to_index = {value: index for index, value in enumerate(regions)}
        self.finding_to_index = {value: index for index, value in enumerate(self.candidates)}

        rng = random.Random(self.seed)
        scale = 1.0 / math.sqrt(float(self.embedding_dim))
        self.organ_embeddings = [
            [rng.gauss(0.0, scale) for _ in range(self.embedding_dim)]
            for _ in organs
        ]
        self.region_embeddings = [
            [rng.gauss(0.0, scale) for _ in range(self.embedding_dim)]
            for _ in regions
        ]
        self.finding_embeddings = [
            [rng.gauss(0.0, scale) for _ in range(self.embedding_dim)]
            for _ in self.candidates
        ]
        counts = Counter(str(record["finding"]) for record in records)
        total = float(sum(counts.values()) + len(self.candidates))
        self.bias = [math.log((counts[value] + 1.0) / total) for value in self.candidates]

        examples = [
            (
                self.organ_to_index[str(record["organ"])],
                self.region_to_index[str(record["region"])],
                self.finding_to_index[str(record["finding"])],
            )
            for record in records
        ]
        if not examples:
            raise ValueError("LearnedEmbeddingBaseline cannot fit an empty training set.")

        learning_rate = self.learning_rate
        for _ in range(self.epochs):
            order = list(range(len(examples)))
            rng.shuffle(order)
            for example_index in order:
                organ_index, region_index, finding_index = examples[example_index]
                include_region = bool(rng.random() >= self.region_dropout)
                organ_vector = list(self.organ_embeddings[organ_index])
                if include_region:
                    region_vector = list(self.region_embeddings[region_index])
                else:
                    region_vector = [0.0] * self.embedding_dim
                context = [
                    organ_vector[index] + region_vector[index]
                    for index in range(self.embedding_dim)
                ]
                logits = [
                    self.bias[finding] + sum(
                        self.finding_embeddings[finding][dimension] * context[dimension]
                        for dimension in range(self.embedding_dim)
                    )
                    for finding in range(len(self.candidates))
                ]
                maximum = max(logits)
                probabilities = [math.exp(max(-60.0, min(60.0, value - maximum))) for value in logits]
                normaliser = sum(probabilities)
                probabilities = [value / normaliser for value in probabilities]
                gradient_logits = list(probabilities)
                gradient_logits[finding_index] -= 1.0

                finding_before = [list(vector) for vector in self.finding_embeddings]
                gradient_context = [
                    sum(
                        gradient_logits[finding] * finding_before[finding][dimension]
                        for finding in range(len(self.candidates))
                    )
                    for dimension in range(self.embedding_dim)
                ]
                for finding in range(len(self.candidates)):
                    self.bias[finding] -= learning_rate * gradient_logits[finding]
                    for dimension in range(self.embedding_dim):
                        gradient = gradient_logits[finding] * context[dimension]
                        gradient += self.l2 * finding_before[finding][dimension]
                        self.finding_embeddings[finding][dimension] -= learning_rate * gradient
                for dimension in range(self.embedding_dim):
                    organ_gradient = gradient_context[dimension] + self.l2 * organ_vector[dimension]
                    self.organ_embeddings[organ_index][dimension] -= learning_rate * organ_gradient
                if include_region:
                    for dimension in range(self.embedding_dim):
                        region_gradient = (
                            gradient_context[dimension] + self.l2 * region_vector[dimension]
                        )
                        self.region_embeddings[region_index][dimension] -= (
                            learning_rate * region_gradient
                        )
            learning_rate *= 0.98

    def _base_scores(self, record: Record, evidence_present: bool) -> Dict[str, float]:
        context = [0.0] * self.embedding_dim
        organ_index = self.organ_to_index.get(str(record["organ"]))
        if organ_index is not None:
            context = [
                context[index] + self.organ_embeddings[organ_index][index]
                for index in range(self.embedding_dim)
            ]
        if evidence_present:
            region_index = self.region_to_index.get(str(record["region"]))
            if region_index is not None:
                context = [
                    context[index] + self.region_embeddings[region_index][index]
                    for index in range(self.embedding_dim)
                ]
        logits = [
            self.bias[finding] + sum(
                self.finding_embeddings[finding][dimension] * context[dimension]
                for dimension in range(self.embedding_dim)
            )
            for finding in range(len(self.candidates))
        ]
        return {
            candidate: float(logits[index])
            for candidate, index in self.finding_to_index.items()
        }


class PrecomputedScoreBaseline(SemanticRankingBaseline):
    """Adapter for scores produced by an external LLM or MLLM.

    ``score_fields`` maps the four explicit condition keys to record fields.  If
    corrupted-knowledge fields are omitted, the intact field is deranged by this
    class.  A no-image field is required for evidence-absent conditions.
    """

    baseline_type = "precomputed"

    def __init__(
        self,
        seed: int = 0,
        score_fields: Optional[Mapping[str, str]] = None,
    ) -> None:
        super().__init__(seed=seed)
        default_fields = {
            _condition_key(True, True): "scores",
            _condition_key(False, True): "scores_no_image",
            _condition_key(True, False): "scores_shuffled_knowledge",
            _condition_key(False, False): "scores_no_image_shuffled_knowledge",
        }
        if score_fields:
            default_fields.update({str(key): str(value) for key, value in score_fields.items()})
        self.score_fields = default_fields

    def _base_scores(self, record: Record, evidence_present: bool) -> Dict[str, float]:
        key = _condition_key(evidence_present, True)
        field = self.score_fields[key]
        raw = record.get(field)
        if not isinstance(raw, Mapping):
            raise ValueError("Record %r lacks mapping score field %r." % (record.get("patient_id"), field))
        missing = sorted(set(self.candidates) - set(str(value) for value in raw))
        if missing:
            raise ValueError("Score field %r is missing candidates: %s" % (field, missing))
        return {candidate: float(raw[candidate]) for candidate in self.candidates}

    def score(
        self,
        record: Record,
        candidates: Optional[Sequence[str]] = None,
        evidence_present: bool = True,
        knowledge_intact: bool = True,
    ) -> Dict[str, float]:
        if not self._is_fitted:
            raise RuntimeError("fit must be called before score.")
        requested = self.candidates if candidates is None else [str(value) for value in candidates]
        condition = _condition_key(evidence_present, knowledge_intact)
        field = self.score_fields.get(condition)
        raw = record.get(field) if field else None
        if isinstance(raw, Mapping):
            missing = sorted(set(requested) - set(str(value) for value in raw))
            if missing:
                raise ValueError("Score field %r is missing candidates: %s" % (field, missing))
            return {candidate: float(raw[candidate]) for candidate in requested}
        return super().score(
            record,
            candidates=requested,
            evidence_present=evidence_present,
            knowledge_intact=knowledge_intact,
        )


BASELINE_REGISTRY = {
    "uniform": UniformBaseline,
    "train_frequency": TrainFrequencyBaseline,
    "frequency": TrainFrequencyBaseline,
    "learned_embedding": LearnedEmbeddingBaseline,
    "embedding": LearnedEmbeddingBaseline,
    "precomputed": PrecomputedScoreBaseline,
}


def create_baseline(specification: Mapping[str, Any], default_seed: int = 0) -> SemanticRankingBaseline:
    if not isinstance(specification, Mapping):
        raise TypeError("Each baseline specification must be a mapping.")
    baseline_type = str(specification.get("type", "")).strip().lower()
    if baseline_type not in BASELINE_REGISTRY:
        raise ValueError(
            "Unknown baseline type %r; choose from %s."
            % (baseline_type, sorted(BASELINE_REGISTRY))
        )
    kwargs = dict(specification.get("kwargs", {}))
    kwargs.setdefault("seed", int(default_seed))
    return BASELINE_REGISTRY[baseline_type](**kwargs)


def _logsumexp(values: Iterable[float]) -> float:
    values = [float(value) for value in values]
    maximum = max(values)
    return maximum + math.log(sum(math.exp(value - maximum) for value in values))


def ranking_metrics(
    scores: Mapping[str, float], target: str, top_k: Sequence[int]
) -> Dict[str, Any]:
    """Compute tie-aware reciprocal rank, expected Recall@k, and NLL."""

    if target not in scores:
        raise ValueError("Target %r is absent from candidate scores." % target)
    target_score = float(scores[target])
    tolerance = 1e-12
    greater = sum(float(score) > target_score + tolerance for score in scores.values())
    tied = sum(abs(float(score) - target_score) <= tolerance for score in scores.values())
    average_rank = 1.0 + float(greater) + 0.5 * float(tied - 1)
    recalls = {}
    for value in top_k:
        k = int(value)
        if greater >= k:
            recall = 0.0
        elif greater + tied <= k:
            recall = 1.0
        else:
            recall = float(k - greater) / float(tied)
        recalls[str(k)] = float(recall)
    nll = _logsumexp(scores.values()) - target_score
    return {
        "rank": float(average_rank),
        "reciprocal_rank": float(1.0 / average_rank),
        "recall_at_k": recalls,
        "nll": float(nll),
    }


def _aggregate_rows(rows: Sequence[Mapping[str, Any]], top_k: Sequence[int]) -> Dict[str, Any]:
    if not rows:
        return {
            "n": 0,
            "mrr": None,
            "nll": None,
            "recall_at_k": {str(int(k)): None for k in top_k},
        }
    return {
        "n": int(len(rows)),
        "mrr": float(sum(float(row["reciprocal_rank"]) for row in rows) / len(rows)),
        "nll": float(sum(float(row["nll"]) for row in rows) / len(rows)),
        "recall_at_k": {
            str(int(k)): float(
                sum(float(row["recall_at_k"][str(int(k))]) for row in rows) / len(rows)
            )
            for k in top_k
        },
    }


def _condition_summaries(
    rows: Sequence[Mapping[str, Any]], top_k: Sequence[int]
) -> Dict[str, Dict[str, Any]]:
    grouped: Dict[str, List[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[_condition_key(row["evidence_present"], row["knowledge_intact"])].append(row)
    summaries = {}
    for evidence_present in (True, False):
        for knowledge_intact in (True, False):
            key = _condition_key(evidence_present, knowledge_intact)
            summaries[key] = _aggregate_rows(grouped.get(key, []), top_k)
    return summaries


def _interaction(
    conditions: Mapping[str, Mapping[str, Any]], top_k: Sequence[int]
) -> Dict[str, Any]:
    keys = {
        (evidence, knowledge): _condition_key(evidence, knowledge)
        for evidence in (True, False)
        for knowledge in (True, False)
    }
    if any(conditions[keys[index]]["n"] == 0 for index in keys):
        return {
            "definition": "q11 - q01 - q10 + q00",
            "mrr": None,
            "nll": None,
            "negative_nll": None,
            "recall_at_k": {str(int(k)): None for k in top_k},
        }

    def difference(metric_getter: Any) -> float:
        q11 = float(metric_getter(conditions[keys[(True, True)]]))
        q01 = float(metric_getter(conditions[keys[(False, True)]]))
        q10 = float(metric_getter(conditions[keys[(True, False)]]))
        q00 = float(metric_getter(conditions[keys[(False, False)]]))
        return q11 - q01 - q10 + q00

    return {
        "definition": "q11 - q01 - q10 + q00; positive means evidence and intact knowledge are complementary",
        "mrr": float(difference(lambda value: value["mrr"])),
        "nll": float(difference(lambda value: value["nll"])),
        "negative_nll": float(difference(lambda value: -float(value["nll"]))),
        "recall_at_k": {
            str(int(k)): float(
                difference(lambda value, key=str(int(k)): value["recall_at_k"][key])
            )
            for k in top_k
        },
        "evidence_gain_with_intact_knowledge": {
            "mrr": float(
                conditions[keys[(True, True)]]["mrr"]
                - conditions[keys[(False, True)]]["mrr"]
            ),
            "nll": float(
                conditions[keys[(True, True)]]["nll"]
                - conditions[keys[(False, True)]]["nll"]
            ),
            "negative_nll": float(
                -conditions[keys[(True, True)]]["nll"]
                + conditions[keys[(False, True)]]["nll"]
            ),
            "recall_at_k": {
                str(int(k)): float(
                    conditions[keys[(True, True)]]["recall_at_k"][str(int(k))]
                    - conditions[keys[(False, True)]]["recall_at_k"][str(int(k))]
                )
                for k in top_k
            },
        },
        "knowledge_gain_with_evidence": {
            "mrr": float(
                conditions[keys[(True, True)]]["mrr"]
                - conditions[keys[(True, False)]]["mrr"]
            ),
            "nll": float(
                conditions[keys[(True, True)]]["nll"]
                - conditions[keys[(True, False)]]["nll"]
            ),
            "negative_nll": float(
                -conditions[keys[(True, True)]]["nll"]
                + conditions[keys[(True, False)]]["nll"]
            ),
            "recall_at_k": {
                str(int(k)): float(
                    conditions[keys[(True, True)]]["recall_at_k"][str(int(k))]
                    - conditions[keys[(True, False)]]["recall_at_k"][str(int(k))]
                )
                for k in top_k
            },
        },
    }


def _gap(high: Optional[float], low: Optional[float]) -> Optional[float]:
    if high is None or low is None:
        return None
    return float(high - low)


def _seen_unseen_summary(
    seen_conditions: Mapping[str, Mapping[str, Any]],
    unseen_conditions: Mapping[str, Mapping[str, Any]],
    seen_interaction: Mapping[str, Any],
    unseen_interaction: Mapping[str, Any],
    top_k: Sequence[int],
) -> Dict[str, Any]:
    condition_gaps = {}
    for key in seen_conditions:
        seen = seen_conditions[key]
        unseen = unseen_conditions[key]
        condition_gaps[key] = {
            "mrr_unseen_minus_seen": _gap(unseen["mrr"], seen["mrr"]),
            "nll_unseen_minus_seen": _gap(unseen["nll"], seen["nll"]),
            "negative_nll_unseen_minus_seen": (
                None
                if unseen["nll"] is None or seen["nll"] is None
                else float(seen["nll"] - unseen["nll"])
            ),
            "recall_at_k_unseen_minus_seen": {
                str(int(k)): _gap(
                    unseen["recall_at_k"][str(int(k))],
                    seen["recall_at_k"][str(int(k))],
                )
                for k in top_k
            },
        }
    return {
        "condition_gaps": condition_gaps,
        "interaction_unseen_minus_seen": {
            "mrr": _gap(unseen_interaction.get("mrr"), seen_interaction.get("mrr")),
            "nll": _gap(unseen_interaction.get("nll"), seen_interaction.get("nll")),
            "negative_nll": _gap(
                unseen_interaction.get("negative_nll"), seen_interaction.get("negative_nll")
            ),
            "recall_at_k": {
                str(int(k)): _gap(
                    unseen_interaction.get("recall_at_k", {}).get(str(int(k))),
                    seen_interaction.get("recall_at_k", {}).get(str(int(k))),
                )
                for k in top_k
            },
        },
    }


def _bootstrap_options(config: Mapping[str, Any], default_seed: int) -> Dict[str, Any]:
    """Validate bootstrap settings while keeping a zero-iteration opt-out."""

    raw = config.get("bootstrap", {})
    if raw is False:
        raw = {"iterations": 0}
    elif raw is True or raw is None:
        raw = {}
    if not isinstance(raw, Mapping):
        raise TypeError("bootstrap must be a mapping or boolean.")
    iterations = int(raw.get("iterations", 1000))
    confidence = float(raw.get("confidence", 0.95))
    seed = int(raw.get("seed", default_seed))
    if iterations < 0:
        raise ValueError("bootstrap.iterations must be non-negative.")
    if not 0.0 < confidence < 1.0:
        raise ValueError("bootstrap.confidence must be in (0, 1).")
    return {"iterations": iterations, "confidence": confidence, "seed": seed}


def _patient_blocks(rows: Sequence[Mapping[str, Any]]) -> List[List[Mapping[str, Any]]]:
    grouped: Dict[str, List[Mapping[str, Any]]] = defaultdict(list)
    for index, row in enumerate(rows):
        patient_id = row.get("patient_id")
        key = str(patient_id) if patient_id is not None else "__row_%d" % index
        grouped[key].append(row)
    return [grouped[key] for key in sorted(grouped)]


def _row_edge_key(row: Mapping[str, Any]) -> Optional[str]:
    """Return an explicit edge id, or the organ x finding composition."""

    for field in ("edge_id", "edge"):
        raw = row.get(field)
        if raw is not None:
            if isinstance(raw, Sequence) and not isinstance(raw, (str, bytes)):
                return "\u241f".join(str(value) for value in raw)
            return str(raw)
    if row.get("organ") is not None and row.get("finding") is not None:
        return "%s\u241f%s" % (row["organ"], row["finding"])
    return None


def _bootstrap_resample_rows(
    rows: Sequence[Mapping[str, Any]],
    rng: random.Random,
    hierarchical_edges: bool,
) -> Tuple[List[Mapping[str, Any]], str]:
    """Resample paired condition rows by patient, optionally edge then patient."""

    blocks = _patient_blocks(rows)
    if not blocks:
        return [], "patient"

    if hierarchical_edges:
        edge_to_blocks: Dict[str, List[List[Mapping[str, Any]]]] = defaultdict(list)
        has_usable_edges = True
        for block in blocks:
            edges = {_row_edge_key(row) for row in block}
            edges.discard(None)
            if len(edges) != 1:
                has_usable_edges = False
                break
            edge_to_blocks[next(iter(edges))].append(block)
        if has_usable_edges and edge_to_blocks:
            edge_keys = sorted(edge_to_blocks)
            sampled: List[Mapping[str, Any]] = []
            for _ in range(len(edge_keys)):
                edge = edge_keys[rng.randrange(len(edge_keys))]
                edge_blocks = edge_to_blocks[edge]
                for _ in range(len(edge_blocks)):
                    sampled.extend(edge_blocks[rng.randrange(len(edge_blocks))])
            return sampled, "edge_then_patient"

    sampled = []
    for _ in range(len(blocks)):
        sampled.extend(blocks[rng.randrange(len(blocks))])
    return sampled, "patient"


def _quantile(values: Sequence[float], probability: float) -> Optional[float]:
    finite = sorted(float(value) for value in values if math.isfinite(float(value)))
    if not finite:
        return None
    if len(finite) == 1:
        return finite[0]
    position = (len(finite) - 1) * float(probability)
    lower_index = int(math.floor(position))
    upper_index = int(math.ceil(position))
    if lower_index == upper_index:
        return finite[lower_index]
    weight = position - lower_index
    return float(finite[lower_index] * (1.0 - weight) + finite[upper_index] * weight)


def _tree_value(tree: Mapping[str, Any], path: Sequence[str]) -> Optional[float]:
    value: Any = tree
    for key in path:
        if not isinstance(value, Mapping) or key not in value:
            return None
        value = value[key]
    if value is None or isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    value = float(value)
    return value if math.isfinite(value) else None


def _interval(
    estimate: Optional[float],
    values: Sequence[Optional[float]],
    confidence: float,
) -> Dict[str, Any]:
    finite = [
        float(value)
        for value in values
        if value is not None and math.isfinite(float(value))
    ]
    tail = (1.0 - confidence) / 2.0
    return {
        "estimate": None if estimate is None else float(estimate),
        "lower": _quantile(finite, tail),
        "upper": _quantile(finite, 1.0 - tail),
        "confidence": float(confidence),
        "valid_iterations": len(finite),
    }


def _interval_at(
    point: Mapping[str, Any],
    replicates: Sequence[Mapping[str, Any]],
    path: Sequence[str],
    confidence: float,
) -> Dict[str, Any]:
    return _interval(
        _tree_value(point, path),
        [_tree_value(replicate, path) for replicate in replicates],
        confidence,
    )


def _condition_intervals(
    point: Mapping[str, Any],
    replicates: Sequence[Mapping[str, Any]],
    top_k: Sequence[int],
    confidence: float,
) -> Dict[str, Any]:
    output = {}
    for condition in point:
        output[condition] = {
            "mrr": _interval_at(point, replicates, [condition, "mrr"], confidence),
            "nll": _interval_at(point, replicates, [condition, "nll"], confidence),
            "recall_at_k": {
                str(int(k)): _interval_at(
                    point,
                    replicates,
                    [condition, "recall_at_k", str(int(k))],
                    confidence,
                )
                for k in top_k
            },
        }
    return output


def _interaction_intervals(
    point: Mapping[str, Any],
    replicates: Sequence[Mapping[str, Any]],
    top_k: Sequence[int],
    confidence: float,
) -> Dict[str, Any]:
    output = {
        metric: _interval_at(point, replicates, [metric], confidence)
        for metric in ("mrr", "nll", "negative_nll")
    }
    output["recall_at_k"] = {
        str(int(k)): _interval_at(
            point, replicates, ["recall_at_k", str(int(k))], confidence
        )
        for k in top_k
    }
    for gain in (
        "evidence_gain_with_intact_knowledge",
        "knowledge_gain_with_evidence",
    ):
        output[gain] = {
            metric: _interval_at(point, replicates, [gain, metric], confidence)
            for metric in ("mrr", "nll", "negative_nll")
        }
        output[gain]["recall_at_k"] = {
            str(int(k)): _interval_at(
                point, replicates, [gain, "recall_at_k", str(int(k))], confidence
            )
            for k in top_k
        }
    return output


def _seen_unseen_intervals(
    point: Mapping[str, Any],
    replicates: Sequence[Mapping[str, Any]],
    top_k: Sequence[int],
    confidence: float,
) -> Dict[str, Any]:
    condition_gaps = {}
    for condition in point["condition_gaps"]:
        condition_gaps[condition] = {
            metric: _interval_at(
                point,
                replicates,
                ["condition_gaps", condition, metric],
                confidence,
            )
            for metric in (
                "mrr_unseen_minus_seen",
                "nll_unseen_minus_seen",
                "negative_nll_unseen_minus_seen",
            )
        }
        condition_gaps[condition]["recall_at_k_unseen_minus_seen"] = {
            str(int(k)): _interval_at(
                point,
                replicates,
                [
                    "condition_gaps",
                    condition,
                    "recall_at_k_unseen_minus_seen",
                    str(int(k)),
                ],
                confidence,
            )
            for k in top_k
        }
    interaction = {
        metric: _interval_at(
            point,
            replicates,
            ["interaction_unseen_minus_seen", metric],
            confidence,
        )
        for metric in ("mrr", "nll", "negative_nll")
    }
    interaction["recall_at_k"] = {
        str(int(k)): _interval_at(
            point,
            replicates,
            ["interaction_unseen_minus_seen", "recall_at_k", str(int(k))],
            confidence,
        )
        for k in top_k
    }
    return {
        "condition_gaps": condition_gaps,
        "interaction_unseen_minus_seen": interaction,
    }


def _bootstrap_confidence_intervals(
    seen_rows: Sequence[Mapping[str, Any]],
    unseen_rows: Sequence[Mapping[str, Any]],
    seen_conditions: Mapping[str, Mapping[str, Any]],
    unseen_conditions: Mapping[str, Mapping[str, Any]],
    seen_interaction: Mapping[str, Any],
    unseen_interaction: Mapping[str, Any],
    seen_vs_unseen: Mapping[str, Any],
    top_k: Sequence[int],
    options: Mapping[str, Any],
) -> Dict[str, Any]:
    """Percentile CIs with paired 2x2 rows kept inside sampling units."""

    iterations = int(options["iterations"])
    confidence = float(options["confidence"])
    rng = random.Random(int(options["seed"]))
    seen_condition_replicates = []
    unseen_condition_replicates = []
    seen_interaction_replicates = []
    unseen_interaction_replicates = []
    seen_vs_unseen_replicates = []
    seen_method = "patient"
    unseen_method = "edge_then_patient"
    for _ in range(iterations):
        sampled_seen, seen_method = _bootstrap_resample_rows(
            seen_rows, rng, hierarchical_edges=False
        )
        sampled_unseen, unseen_method = _bootstrap_resample_rows(
            unseen_rows, rng, hierarchical_edges=True
        )
        sampled_seen_conditions = _condition_summaries(sampled_seen, top_k)
        sampled_unseen_conditions = _condition_summaries(sampled_unseen, top_k)
        sampled_seen_interaction = _interaction(sampled_seen_conditions, top_k)
        sampled_unseen_interaction = _interaction(sampled_unseen_conditions, top_k)
        sampled_seen_vs_unseen = _seen_unseen_summary(
            sampled_seen_conditions,
            sampled_unseen_conditions,
            sampled_seen_interaction,
            sampled_unseen_interaction,
            top_k,
        )
        seen_condition_replicates.append(sampled_seen_conditions)
        unseen_condition_replicates.append(sampled_unseen_conditions)
        seen_interaction_replicates.append(sampled_seen_interaction)
        unseen_interaction_replicates.append(sampled_unseen_interaction)
        seen_vs_unseen_replicates.append(sampled_seen_vs_unseen)

    # Determine the actual fallback even when iterations is zero.
    if iterations == 0:
        _, seen_method = _bootstrap_resample_rows(
            seen_rows, random.Random(int(options["seed"])), hierarchical_edges=False
        )
        _, unseen_method = _bootstrap_resample_rows(
            unseen_rows, random.Random(int(options["seed"])), hierarchical_edges=True
        )
    return {
        "method": "percentile_cluster_bootstrap",
        "iterations": iterations,
        "confidence": confidence,
        "seed": int(options["seed"]),
        "sampling_unit": "patient; paired evidence x knowledge rows remain together",
        "resampling": {"seen": seen_method, "unseen": unseen_method},
        "metrics": {
            "seen": _condition_intervals(
                seen_conditions, seen_condition_replicates, top_k, confidence
            ),
            "unseen": _condition_intervals(
                unseen_conditions, unseen_condition_replicates, top_k, confidence
            ),
        },
        "interaction": {
            "seen": _interaction_intervals(
                seen_interaction, seen_interaction_replicates, top_k, confidence
            ),
            "unseen": _interaction_intervals(
                unseen_interaction, unseen_interaction_replicates, top_k, confidence
            ),
        },
        "seen_vs_unseen": _seen_unseen_intervals(
            seen_vs_unseen, seen_vs_unseen_replicates, top_k, confidence
        ),
    }


def _normalise_top_k(raw: Any) -> List[int]:
    values = [raw] if isinstance(raw, int) else list(raw)
    top_k = sorted(set(int(value) for value in values))
    if not top_k or any(value < 1 for value in top_k):
        raise ValueError("top_k must contain positive integers.")
    return top_k


def _baseline_specs(config: Mapping[str, Any]) -> List[Mapping[str, Any]]:
    raw = config.get("baselines", config.get("baseline"))
    if raw is None:
        raise ValueError("Config requires baseline or baselines.")
    if isinstance(raw, Mapping):
        return [raw]
    if isinstance(raw, Sequence) and not isinstance(raw, (str, bytes)):
        specs = list(raw)
        if not specs:
            raise ValueError("baselines cannot be empty.")
        return specs
    raise TypeError("baseline must be a mapping, or baselines must be a sequence of mappings.")


def _resolve_path_from_config(value: Any, config: Mapping[str, Any]) -> Any:
    if not isinstance(value, (str, bytes, os.PathLike)):
        return value
    path = os.path.expanduser(os.fspath(value))
    if not os.path.isabs(path) and config.get("__config_dir__"):
        path = os.path.join(os.fspath(config["__config_dir__"]), path)
    return os.path.abspath(path)


def _evaluate_baseline(
    name: str,
    baseline: SemanticRankingBaseline,
    split_name: str,
    records: Sequence[Record],
    candidates: Sequence[str],
    top_k: Sequence[int],
) -> List[Dict[str, Any]]:
    rows = []
    for record in records:
        for evidence_present, knowledge_intact in (
            (True, True),
            (False, True),
            (True, False),
            (False, False),
        ):
            scores = baseline.score(
                record,
                candidates=candidates,
                evidence_present=evidence_present,
                knowledge_intact=knowledge_intact,
            )
            metrics = ranking_metrics(scores, str(record["finding"]), top_k)
            row = {
                "baseline": name,
                "baseline_type": baseline.baseline_type,
                "evaluation_split": split_name,
                "patient_id": str(record["patient_id"]),
                "organ": str(record["organ"]),
                "finding": str(record["finding"]),
                "region": str(record["region"]),
                "source": record.get("source"),
                "original_split": record.get("split"),
                "evidence_present": bool(evidence_present),
                "knowledge_intact": bool(knowledge_intact),
                "predictions": baseline.predict(
                    record,
                    top_k=max(top_k),
                    candidates=candidates,
                    evidence_present=evidence_present,
                    knowledge_intact=knowledge_intact,
                ),
            }
            row.update(metrics)
            rows.append(row)
    return rows


def run_llm_necessity(config_dict: Mapping[str, Any]) -> Dict[str, Any]:
    """Run patient-safe semantic controls and return a JSON-serialisable summary.

    Besides the minimal config documented at module level, optional split keys
    are ``seen_eval_fraction``, ``min_edge_records`` and ``patient_conflict``.
    A list under ``baselines`` evaluates several controls in one output.
    """

    if not isinstance(config_dict, Mapping):
        raise TypeError("config_dict must be a mapping.")
    if "manifest" not in config_dict:
        raise ValueError("Config requires manifest.")
    if "output_dir" not in config_dict:
        raise ValueError("Config requires output_dir.")
    seed = int(config_dict.get("seed", 0))
    top_k = _normalise_top_k(config_dict.get("top_k", [1, 3, 5]))
    bootstrap_options = _bootstrap_options(config_dict, default_seed=seed)
    manifest_value = _resolve_path_from_config(config_dict["manifest"], config_dict)
    records = load_jsonl_records(manifest_value)
    split = build_compositional_split(
        records,
        held_out_edges=config_dict.get("held_out_edges"),
        num_edges=config_dict.get("num_edges"),
        seed=seed,
        seen_eval_fraction=float(config_dict.get("seen_eval_fraction", 0.2)),
        min_edge_records=int(config_dict.get("min_edge_records", 1)),
        patient_conflict=str(config_dict.get("patient_conflict", "error")),
    )
    candidates = sorted(
        {
            str(record["finding"])
            for group in (split["fit"], split["seen"], split["unseen"])
            for record in group
        }
    )
    if max(top_k) > len(candidates):
        top_k = sorted(set(min(value, len(candidates)) for value in top_k))

    per_sample_rows: List[Dict[str, Any]] = []
    baseline_summaries: Dict[str, Any] = {}
    used_names = set()
    for index, specification in enumerate(_baseline_specs(config_dict)):
        baseline = create_baseline(specification, default_seed=seed + index)
        requested_name = str(specification.get("name", baseline.baseline_type))
        name = requested_name
        suffix = 2
        while name in used_names:
            name = "%s_%d" % (requested_name, suffix)
            suffix += 1
        used_names.add(name)
        baseline.fit(split["fit"], candidates)
        seen_rows = _evaluate_baseline(
            name, baseline, "seen", split["seen"], candidates, top_k
        )
        unseen_rows = _evaluate_baseline(
            name, baseline, "unseen", split["unseen"], candidates, top_k
        )
        per_sample_rows.extend(seen_rows)
        per_sample_rows.extend(unseen_rows)
        seen_conditions = _condition_summaries(seen_rows, top_k)
        unseen_conditions = _condition_summaries(unseen_rows, top_k)
        seen_interaction = _interaction(seen_conditions, top_k)
        unseen_interaction = _interaction(unseen_conditions, top_k)
        seen_vs_unseen = _seen_unseen_summary(
            seen_conditions,
            unseen_conditions,
            seen_interaction,
            unseen_interaction,
            top_k,
        )
        baseline_summaries[name] = {
            "type": baseline.baseline_type,
            "kwargs": dict(specification.get("kwargs", {})),
            "metrics": {"seen": seen_conditions, "unseen": unseen_conditions},
            "interaction": {"seen": seen_interaction, "unseen": unseen_interaction},
            "seen_vs_unseen": seen_vs_unseen,
            "confidence_intervals": _bootstrap_confidence_intervals(
                seen_rows,
                unseen_rows,
                seen_conditions,
                unseen_conditions,
                seen_interaction,
                unseen_interaction,
                seen_vs_unseen,
                top_k,
                bootstrap_options,
            ),
        }

    output_dir = _resolve_path_from_config(config_dict["output_dir"], config_dict)
    os.makedirs(output_dir, exist_ok=True)
    per_sample_path = os.path.join(output_dir, "per_sample.jsonl")
    summary_path = os.path.join(output_dir, "summary.json")
    manifest_description = (
        os.fspath(manifest_value)
        if isinstance(manifest_value, (str, bytes, os.PathLike))
        else "<in-memory>"
    )
    summary = {
        "schema_version": 1,
        "config": {
            "manifest": manifest_description,
            "output_dir": output_dir,
            "seed": seed,
            "top_k": top_k,
            "seen_eval_fraction": float(config_dict.get("seen_eval_fraction", 0.2)),
            "bootstrap": dict(bootstrap_options),
        },
        "split": {
            "held_out_edges": split["held_out_edges"],
            "fit_patients": len(split["fit"]),
            "seen_patients": len(split["seen"]),
            "unseen_patients": len(split["unseen"]),
            "candidate_findings": candidates,
            "deduplication": split["deduplication"],
            "leakage_audit": split["leakage_audit"],
        },
        "baselines": baseline_summaries,
        "artifacts": {
            "per_sample_jsonl": per_sample_path,
            "summary_json": summary_path,
        },
        "limitations": [
            "Built-in evidence ablation removes region only; it is not a literal image ablation.",
            "Knowledge shuffling permutes finding associations, not internal LLM weights.",
            "The manifest is single-finding per patient; multi-label patients require prior consolidation.",
            "Percentile bootstrap intervals quantify sampled patient/edge variation, not external site shift.",
            "A causal LLM claim requires externally produced four-condition image/no-image scores.",
        ],
    }

    with open(per_sample_path, "w", encoding="utf-8") as handle:
        for row in per_sample_rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    with open(summary_path, "w", encoding="utf-8") as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
    return summary


__all__ = [
    "SemanticRankingBaseline",
    "UniformBaseline",
    "TrainFrequencyBaseline",
    "LearnedEmbeddingBaseline",
    "PrecomputedScoreBaseline",
    "create_baseline",
    "ranking_metrics",
    "run_llm_necessity",
]
