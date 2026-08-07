import json
import math
import os
import tempfile
import unittest

from medical_experiments.compositional_split import (
    build_compositional_split,
    deduplicate_patients,
)
from medical_experiments.semantic_baselines import (
    LearnedEmbeddingBaseline,
    PrecomputedScoreBaseline,
    TrainFrequencyBaseline,
    UniformBaseline,
    ranking_metrics,
    run_llm_necessity,
)


def make_records(include_duplicate=False):
    records = []
    patient_number = 0
    region_by_finding = {"edema": "diffuse", "tumor": "focal"}
    for organ, finding in (
        ("brain", "edema"),
        ("brain", "tumor"),
        ("liver", "edema"),
        ("liver", "tumor"),
    ):
        for repetition in range(3):
            patient_number += 1
            records.append(
                {
                    "patient_id": "p%02d" % patient_number,
                    "organ": organ,
                    "finding": finding,
                    "region": region_by_finding[finding],
                    "source": "site_%d" % (repetition % 2),
                    "split": "original",
                }
            )
    if include_duplicate:
        records.append(dict(records[0]))
    return records


class CompositionalSplitTest(unittest.TestCase):
    def test_patient_dedup_happens_before_edge_split(self):
        records = make_records(include_duplicate=True)
        result = build_compositional_split(
            records,
            held_out_edges=[["brain", "edema"]],
            seed=13,
            seen_eval_fraction=0.34,
        )

        self.assertEqual(result["deduplication"]["duplicates_removed"], 1)
        self.assertEqual(result["held_out_edges"], [["brain", "edema"]])
        self.assertTrue(result["leakage_audit"]["all_unseen_nodes_present_in_fit"])
        self.assertEqual(
            {(record["organ"], record["finding"]) for record in result["unseen"]},
            {("brain", "edema")},
        )
        fit_patients = {record["patient_id"] for record in result["fit"]}
        seen_patients = {record["patient_id"] for record in result["seen"]}
        unseen_patients = {record["patient_id"] for record in result["unseen"]}
        self.assertFalse(fit_patients & seen_patients)
        self.assertFalse(fit_patients & unseen_patients)
        self.assertFalse(seen_patients & unseen_patients)
        self.assertIn("brain", {record["organ"] for record in result["fit"]})
        self.assertIn("edema", {record["finding"] for record in result["fit"]})

    def test_conflicting_patient_rows_are_rejected(self):
        records = make_records()
        conflict = dict(records[0])
        conflict["finding"] = "tumor"
        records.append(conflict)
        with self.assertRaisesRegex(ValueError, "conflicting duplicate"):
            deduplicate_patients(records)

    def test_auto_selected_edge_retains_incident_nodes(self):
        result = build_compositional_split(
            make_records(),
            num_edges=1,
            seed=5,
            seen_eval_fraction=0.34,
        )
        organ, finding = result["held_out_edges"][0]
        self.assertIn(organ, {record["organ"] for record in result["fit"]})
        self.assertIn(finding, {record["finding"] for record in result["fit"]})

    def test_invalid_holdout_that_removes_nodes_is_rejected(self):
        records = [
            {"patient_id": "a", "organ": "brain", "finding": "edema", "region": "r1"},
            {"patient_id": "b", "organ": "liver", "finding": "tumor", "region": "r2"},
        ]
        with self.assertRaisesRegex(ValueError, "uncovered organs"):
            build_compositional_split(
                records,
                held_out_edges=[["brain", "edema"]],
                seen_eval_fraction=0.0,
            )


class SemanticBaselineTest(unittest.TestCase):
    def setUp(self):
        self.records = make_records()
        self.candidates = ["edema", "tumor"]

    def test_uniform_metrics_are_tie_aware(self):
        baseline = UniformBaseline(seed=1).fit(self.records, self.candidates)
        scores = baseline.score(self.records[0])
        metrics = ranking_metrics(scores, "edema", [1, 2])
        self.assertAlmostEqual(metrics["rank"], 1.5)
        self.assertAlmostEqual(metrics["reciprocal_rank"], 2.0 / 3.0)
        self.assertAlmostEqual(metrics["recall_at_k"]["1"], 0.5)
        self.assertAlmostEqual(metrics["recall_at_k"]["2"], 1.0)
        self.assertAlmostEqual(metrics["nll"], math.log(2.0))
        self.assertEqual(baseline.predict(self.records[0], top_k=2), ["edema", "tumor"])

    def test_frequency_knowledge_shuffle_preserves_scores_but_swaps_labels(self):
        baseline = TrainFrequencyBaseline(seed=3).fit(self.records, self.candidates)
        intact = baseline.score(self.records[0], evidence_present=True, knowledge_intact=True)
        shuffled = baseline.score(
            self.records[0], evidence_present=True, knowledge_intact=False
        )
        self.assertEqual(sorted(intact.values()), sorted(shuffled.values()))
        self.assertAlmostEqual(intact["edema"], shuffled["tumor"])
        self.assertAlmostEqual(intact["tumor"], shuffled["edema"])

    def test_learned_embedding_is_deterministic_and_uses_common_interface(self):
        kwargs = {
            "seed": 11,
            "embedding_dim": 4,
            "epochs": 6,
            "learning_rate": 0.05,
        }
        first = LearnedEmbeddingBaseline(**kwargs).fit(self.records, self.candidates)
        second = LearnedEmbeddingBaseline(**kwargs).fit(self.records, self.candidates)
        first_scores = first.score(self.records[0], evidence_present=False)
        second_scores = second.score(self.records[0], evidence_present=False)
        self.assertEqual(set(first_scores), set(self.candidates))
        for candidate in self.candidates:
            self.assertAlmostEqual(first_scores[candidate], second_scores[candidate], places=12)
        self.assertEqual(len(first.predict(self.records[0], top_k=1)), 1)

    def test_precomputed_adapter_reads_all_four_explicit_conditions(self):
        record = dict(self.records[0])
        record.update(
            {
                "scores": {"edema": 4.0, "tumor": 0.0},
                "scores_no_image": {"edema": 1.0, "tumor": 0.0},
                "scores_shuffled_knowledge": {"edema": 0.0, "tumor": 4.0},
                "scores_no_image_shuffled_knowledge": {"edema": 0.0, "tumor": 1.0},
            }
        )
        baseline = PrecomputedScoreBaseline(seed=0).fit([record], self.candidates)
        self.assertEqual(
            baseline.predict(record, evidence_present=True, knowledge_intact=True),
            ["edema"],
        )
        self.assertEqual(
            baseline.predict(record, evidence_present=False, knowledge_intact=False),
            ["tumor"],
        )

    def test_precomputed_adapter_can_derange_intact_score_fields(self):
        record = dict(self.records[0])
        record.update(
            {
                "scores": {"edema": 4.0, "tumor": 0.0},
                "scores_no_image": {"edema": 2.0, "tumor": 0.0},
            }
        )
        baseline = PrecomputedScoreBaseline(seed=0).fit([record], self.candidates)
        intact = baseline.score(record, evidence_present=False, knowledge_intact=True)
        shuffled = baseline.score(record, evidence_present=False, knowledge_intact=False)
        self.assertEqual(sorted(intact.values()), sorted(shuffled.values()))
        self.assertAlmostEqual(intact["edema"], shuffled["tumor"])


class RunnerTest(unittest.TestCase):
    def test_runner_writes_serialisable_summary_and_per_sample_conditions(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            manifest_path = os.path.join(temporary_directory, "records.jsonl")
            output_dir = os.path.join(temporary_directory, "output")
            with open(manifest_path, "w", encoding="utf-8") as handle:
                for record in make_records(include_duplicate=True):
                    handle.write(json.dumps(record) + "\n")

            config = {
                "manifest": manifest_path,
                "output_dir": output_dir,
                "seed": 17,
                "held_out_edges": [["brain", "edema"]],
                "seen_eval_fraction": 0.34,
                "top_k": [1, 2],
                "bootstrap": {"iterations": 31, "confidence": 0.8, "seed": 991},
                "baselines": [
                    {"type": "uniform"},
                    {"type": "train_frequency"},
                    {
                        "type": "learned_embedding",
                        "kwargs": {"embedding_dim": 4, "epochs": 4},
                    },
                ],
            }
            summary = run_llm_necessity(config)

            json.dumps(summary)
            self.assertEqual(summary["split"]["deduplication"]["duplicates_removed"], 1)
            self.assertEqual(set(summary["baselines"]), {"uniform", "train_frequency", "learned_embedding"})
            for baseline_summary in summary["baselines"].values():
                self.assertIn("seen", baseline_summary["metrics"])
                self.assertIn("unseen", baseline_summary["metrics"])
                self.assertIn("negative_nll", baseline_summary["interaction"]["unseen"])
                self.assertIn("interaction_unseen_minus_seen", baseline_summary["seen_vs_unseen"])
                intervals = baseline_summary["confidence_intervals"]
                self.assertEqual(intervals["iterations"], 31)
                self.assertEqual(intervals["confidence"], 0.8)
                self.assertEqual(intervals["seed"], 991)
                self.assertEqual(intervals["resampling"]["seen"], "patient")
                self.assertEqual(intervals["resampling"]["unseen"], "edge_then_patient")
                condition = "evidence=1|knowledge=1"
                mrr_interval = intervals["metrics"]["unseen"][condition]["mrr"]
                self.assertEqual(mrr_interval["valid_iterations"], 31)
                self.assertLessEqual(mrr_interval["lower"], mrr_interval["upper"])
                self.assertAlmostEqual(
                    mrr_interval["estimate"],
                    baseline_summary["metrics"]["unseen"][condition]["mrr"],
                )
                interaction = intervals["interaction"]["unseen"]
                self.assertEqual(interaction["nll"]["valid_iterations"], 31)
                self.assertEqual(interaction["recall_at_k"]["1"]["valid_iterations"], 31)
                self.assertEqual(
                    interaction["evidence_gain_with_intact_knowledge"]["recall_at_k"]["1"]["valid_iterations"],
                    31,
                )

            per_sample_path = summary["artifacts"]["per_sample_jsonl"]
            summary_path = summary["artifacts"]["summary_json"]
            self.assertTrue(os.path.isfile(per_sample_path))
            self.assertTrue(os.path.isfile(summary_path))
            with open(per_sample_path, "r", encoding="utf-8") as handle:
                rows = [json.loads(line) for line in handle if line.strip()]
            expected_cases = summary["split"]["seen_patients"] + summary["split"]["unseen_patients"]
            self.assertEqual(len(rows), expected_cases * 4 * 3)
            self.assertEqual({row["evidence_present"] for row in rows}, {True, False})
            self.assertEqual({row["knowledge_intact"] for row in rows}, {True, False})
            self.assertEqual({row["evaluation_split"] for row in rows}, {"seen", "unseen"})

            repeat_config = dict(config)
            repeat_config["output_dir"] = os.path.join(temporary_directory, "repeat")
            repeated = run_llm_necessity(repeat_config)
            self.assertEqual(
                summary["baselines"]["train_frequency"]["confidence_intervals"],
                repeated["baselines"]["train_frequency"]["confidence_intervals"],
            )

    def test_bootstrap_is_robust_when_seen_split_is_empty(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            manifest_path = os.path.join(temporary_directory, "records.jsonl")
            with open(manifest_path, "w", encoding="utf-8") as handle:
                for record in make_records():
                    handle.write(json.dumps(record) + "\n")
            summary = run_llm_necessity(
                {
                    "manifest": manifest_path,
                    "output_dir": os.path.join(temporary_directory, "output"),
                    "seed": 4,
                    "held_out_edges": [["brain", "edema"]],
                    "seen_eval_fraction": 0.0,
                    "top_k": [1, 2],
                    "bootstrap": {"iterations": 9, "confidence": 0.9},
                    "baseline": {"type": "uniform"},
                }
            )
            baseline = summary["baselines"]["uniform"]
            condition = "evidence=1|knowledge=1"
            seen_ci = baseline["confidence_intervals"]["metrics"]["seen"][condition]["mrr"]
            unseen_ci = baseline["confidence_intervals"]["metrics"]["unseen"][condition]["mrr"]
            self.assertIsNone(seen_ci["estimate"])
            self.assertIsNone(seen_ci["lower"])
            self.assertEqual(seen_ci["valid_iterations"], 0)
            self.assertEqual(unseen_ci["valid_iterations"], 9)
            gap_ci = baseline["confidence_intervals"]["seen_vs_unseen"][
                "interaction_unseen_minus_seen"
            ]["mrr"]
            self.assertEqual(gap_ci["valid_iterations"], 0)


if __name__ == "__main__":
    unittest.main()
