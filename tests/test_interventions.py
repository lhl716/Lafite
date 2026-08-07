import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from medical_experiments.interventions import (
    apply_nuisance,
    dilate_soft_mask,
    generate_intervention_variants,
    intervention_quality_metrics,
    make_placebo_mask,
    remove_lesion,
)
from medical_experiments.shortcut_eval import (
    bootstrap_ci,
    run_shortcut_evaluation,
    soft_dice,
)


class InterventionTests(unittest.TestCase):
    def test_normal_reference_replacement_respects_soft_mask(self):
        image = np.full((9, 9), 10.0, dtype=np.float32)
        reference = np.full((9, 9), 6.0, dtype=np.float32)
        mask = np.zeros((9, 9), dtype=np.float32)
        mask[3:6, 3:6] = 0.5

        removed = remove_lesion(
            image,
            mask,
            normal_reference=reference,
            method="normal_reference",
        )

        np.testing.assert_allclose(removed[3:6, 3:6], 8.0)
        np.testing.assert_allclose(removed[mask == 0], image[mask == 0])
        self.assertGreater(float(np.min(removed)), 0.0)  # No constant black occlusion.
        boolean_removed = remove_lesion(
            image,
            mask.astype(bool),
            normal_reference=reference,
            method="normal_reference",
        )
        np.testing.assert_allclose(boolean_removed[3:6, 3:6], 6.0)

    def test_local_gaussian_removal_supports_3d(self):
        grid = np.indices((13, 13, 13)).sum(axis=0).astype(np.float32)
        image = grid.copy()
        image[5:8, 5:8, 5:8] += 20.0
        mask = np.zeros_like(image)
        mask[5:8, 5:8, 5:8] = 1.0

        removed = remove_lesion(image, mask, method="local_gaussian", gaussian_sigma=2.0)

        self.assertEqual(removed.shape, image.shape)
        self.assertTrue(np.all(np.isfinite(removed)))
        np.testing.assert_allclose(removed[mask == 0], image[mask == 0])
        self.assertLess(float(np.mean(removed[mask == 1])), float(np.mean(image[mask == 1])))

    def test_nuisance_modes_protect_dilated_roi(self):
        image = np.linspace(-1.0, 1.0, 21 * 21).reshape(21, 21).astype(np.float32)
        lesion = np.zeros_like(image)
        lesion[9:12, 9:12] = 1.0
        protected = dilate_soft_mask(lesion, radius=2)

        for mode in ("gaussian_noise", "gaussian_blur", "style"):
            changed = apply_nuisance(
                image,
                lesion,
                mode=mode,
                strength=0.5,
                dilation_radius=2,
                seed=7,
            )
            np.testing.assert_allclose(changed[protected == 1], image[protected == 1], atol=1e-7)
            self.assertGreater(float(np.mean(np.abs(changed[protected == 0] - image[protected == 0]))), 0.0)

        soft_lesion = lesion * 0.4
        soft_protected = dilate_soft_mask(soft_lesion, radius=2) > 0
        soft_changed = apply_nuisance(
            image,
            soft_lesion,
            mode="gaussian_noise",
            strength=0.5,
            dilation_radius=2,
            seed=8,
        )
        np.testing.assert_allclose(
            soft_changed[soft_protected], image[soft_protected], atol=1e-7
        )
        dense_soft = np.full_like(image, 1e-4)
        dense_soft[9:12, 9:12] = 0.4
        dense_changed = apply_nuisance(
            image, dense_soft, strength=0.5, dilation_radius=2, seed=9
        )
        self.assertGreater(float(np.mean(np.abs(dense_changed - image))), 0.0)

    def test_placebo_is_shape_matched_and_non_overlapping(self):
        lesion = np.zeros((24, 24), dtype=np.float32)
        lesion[9:13, 10:15] = np.linspace(0.2, 1.0, 20).reshape(4, 5)
        protected = dilate_soft_mask(lesion, radius=2)
        placebo = make_placebo_mask(lesion, protected_mask=protected, seed=12)

        self.assertAlmostEqual(float(np.sum(placebo)), float(np.sum(lesion)), places=6)
        overlap = float(np.sum(np.minimum(placebo, protected))) / float(np.sum(placebo))
        self.assertLessEqual(overlap, 0.05)
        self.assertFalse(np.array_equal(placebo, lesion))

    def test_generate_all_variants_and_quality(self):
        image = np.indices((24, 24)).sum(axis=0).astype(np.float32)
        lesion = np.zeros_like(image)
        lesion[9:13, 9:13] = 1.0
        reference = image.copy()
        reference[9:13, 9:13] -= 3.0

        variants = generate_intervention_variants(
            image,
            lesion,
            normal_reference=reference,
            nuisance_mode="gaussian_noise",
            nuisance_strength=0.1,
            dilation_radius=2,
            seed=4,
        )

        for name in ("original", "lesion_removed", "nuisance_changed", "placebo"):
            self.assertIn(name, variants)
            self.assertEqual(variants[name].shape, image.shape)
            self.assertIn(name, variants["quality"])
        protected = variants["masks"]["protected_lesion"]
        np.testing.assert_allclose(
            variants["nuisance_changed"][protected == 1],
            image[protected == 1],
            atol=1e-7,
        )
        for metrics in variants["quality"].values():
            for value in metrics.values():
                self.assertTrue(np.isfinite(value))
        self.assertEqual(variants["quality"]["original"]["artifact_score"], 0.0)
        self.assertEqual(variants["quality"]["original"]["continuity_score"], 1.0)
        tiny_placebo = np.zeros_like(lesion)
        tiny_placebo[1, 1] = 1.0
        with self.assertRaises(ValueError):
            generate_intervention_variants(
                image,
                lesion,
                normal_reference=reference,
                placebo_mask=tiny_placebo,
            )

    def test_quality_reports_edit_locality(self):
        image = np.indices((12, 12)).sum(axis=0).astype(np.float32)
        mask = np.zeros_like(image)
        mask[4:8, 4:8] = 1.0
        changed = image.copy()
        changed[4:8, 4:8] += 0.25
        metrics = intervention_quality_metrics(image, changed, mask)

        self.assertAlmostEqual(metrics["inside_mae"], 0.25, places=6)
        self.assertAlmostEqual(metrics["outside_mae"], 0.0, places=6)
        self.assertGreaterEqual(metrics["continuity_score"], 0.0)
        self.assertLessEqual(metrics["continuity_score"], 1.0)

    def test_torch_input_is_restored_when_available(self):
        try:
            import torch
        except ImportError:
            self.skipTest("torch unavailable")
        image = torch.ones((10, 10), dtype=torch.float32)
        mask = torch.zeros((10, 10), dtype=torch.float32)
        mask[4:6, 4:6] = 1.0
        reference = torch.full((10, 10), 0.75, dtype=torch.float32)
        removed = remove_lesion(image, mask, normal_reference=reference)
        self.assertIsInstance(removed, torch.Tensor)
        self.assertEqual(removed.dtype, image.dtype)
        self.assertEqual(removed.device, image.device)


class ShortcutEvaluationTests(unittest.TestCase):
    @staticmethod
    def _mask(offset=0):
        mask = np.zeros((8, 8), dtype=np.float32)
        mask[2 + offset : 5 + offset, 2:5] = 1.0
        return mask

    def test_jsonl_patient_level_pairing_bootstrap_and_pass(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            path = Path(temporary_directory) / "predictions.jsonl"
            mask = self._mask().tolist()
            with path.open("w", encoding="utf-8") as handle:
                for patient_index in range(30):
                    base = 0.8 + 0.001 * patient_index
                    rows = {
                        "original": base,
                        "lesion_removed": base - 0.30,
                        "nuisance_changed": base + (0.01 if patient_index % 2 else -0.01),
                        "placebo": base + 0.005,
                    }
                    for variant, score in rows.items():
                        predicted_mask = np.zeros((8, 8), dtype=np.float32).tolist() if variant == "lesion_removed" else mask
                        handle.write(
                            json.dumps(
                                {
                                    "patient_id": "p%02d" % patient_index,
                                    "variant": variant,
                                    "score": score,
                                    "mask": predicted_mask,
                                }
                            )
                            + "\n"
                        )

            result = run_shortcut_evaluation(
                {
                    "input_path": str(path),
                    "bootstrap": {"iterations": 200, "confidence": 0.90, "seed": 3},
                }
            )

        self.assertEqual(result["n_patients"], 30)
        lesion = result["paired_score_deltas"]["lesion_removed"]
        self.assertEqual(lesion["n_patients"], 30)
        self.assertAlmostEqual(lesion["mean_delta"], -0.30, places=6)
        self.assertEqual(result["mask_stability"]["nuisance_changed"]["mean_soft_dice"], 1.0)
        self.assertEqual(
            result["report_entity_endpoints"]["entity_jaccard"]["nuisance_changed"]["status"],
            "not_evaluable",
        )
        self.assertEqual(result["normal_cohort_fpr"]["status"], "not_evaluable")
        self.assertEqual(result["red_flag_assessment"]["status"], "pass")
        json.dumps(result, allow_nan=False)

    def test_structured_report_mapping_is_exposed_as_entities(self):
        from medical_experiments.intervention_pipeline import _prediction_payload

        payload = _prediction_payload(
            {"score": 0.7, "report": {"entities": ["nodule", "effusion"]}}
        )
        self.assertEqual(payload["report_entities"], ["nodule", "effusion"])

    def test_structured_report_entity_endpoints(self):
        records = []
        for patient_index in range(4):
            common = {
                "patient_id": "report-%d" % patient_index,
                "sample_id": "study-1",
                "target_entity": "nodule",
            }
            records.extend(
                [
                    dict(
                        common,
                        variant="original",
                        score=0.8,
                        report_entities=["nodule", "effusion"],
                    ),
                    dict(
                        common,
                        variant="lesion_removed",
                        score=0.5,
                        entities=["effusion"],
                    ),
                    dict(
                        common,
                        variant="nuisance_changed",
                        score=0.8,
                        entities=["effusion", "nodule"],
                    ),
                    dict(
                        common,
                        variant="placebo",
                        score=0.8,
                        entities=["nodule"],
                    ),
                ]
            )

        result = run_shortcut_evaluation(
            {
                "records": records,
                "bootstrap_iterations": 50,
                "red_flag_rules": {"min_paired_patients": 4},
            }
        )
        endpoints = result["report_entity_endpoints"]
        nuisance = endpoints["entity_jaccard"]["nuisance_changed"]
        placebo = endpoints["entity_jaccard"]["placebo"]
        disappearance = endpoints[
            "target_entity_disappearance_after_lesion_removal"
        ]
        self.assertEqual(nuisance["status"], "evaluable")
        self.assertEqual(nuisance["mean_jaccard"], 1.0)
        self.assertEqual(placebo["mean_jaccard"], 0.5)
        self.assertEqual(disappearance["n_patients"], 4)
        self.assertEqual(disappearance["disappearance_rate"], 1.0)
        self.assertEqual(disappearance["disappearance_rate_ci"]["lower"], 1.0)

    def test_normal_fpr_and_positive_only_lesion_metrics(self):
        records = []
        for patient_index in range(3):
            for variant, score in (
                ("original", 0.8),
                ("lesion_removed", 0.5),
                ("nuisance_changed", 0.8),
                ("placebo", 0.8),
            ):
                records.append(
                    {
                        "patient_id": "abnormal-%d" % patient_index,
                        "variant": variant,
                        "score": score,
                        "label": 1,
                    }
                )

        normal_original = [0.2, 0.4, 0.6, 0.1]
        normal_nuisance = [0.7, 0.4, 0.6, 0.1]
        for patient_index, (original, nuisance) in enumerate(
            zip(normal_original, normal_nuisance)
        ):
            for variant, score in (
                ("original", original),
                ("lesion_removed", original),
                ("nuisance_changed", nuisance),
                ("placebo", original),
            ):
                records.append(
                    {
                        "patient_id": "normal-%d" % patient_index,
                        "variant": variant,
                        "score": score,
                        "label": 0,
                    }
                )

        result = run_shortcut_evaluation(
            {
                "records": records,
                "decision_threshold": 0.5,
                "bootstrap_iterations": 100,
                "seed": 13,
                "red_flag_rules": {"min_paired_patients": 3},
            }
        )

        lesion = result["paired_score_deltas"]["lesion_removed"]
        self.assertEqual(lesion["n_patients"], 3)
        self.assertAlmostEqual(lesion["mean_delta"], -0.3, places=6)
        fpr = result["normal_cohort_fpr"]
        self.assertEqual(fpr["status"], "evaluable")
        self.assertEqual(fpr["n_normal_patients"], 4)
        self.assertEqual(fpr["by_variant"]["original"]["fpr"], 0.25)
        self.assertEqual(fpr["by_variant"]["nuisance_changed"]["fpr"], 0.5)
        self.assertEqual(
            fpr["paired_change_vs_original"]["nuisance_changed"][
                "mean_change_percentage_points"
            ],
            25.0,
        )
        self.assertEqual(
            fpr["paired_change_vs_original"]["placebo"][
                "mean_change_percentage_points"
            ],
            0.0,
        )
        self.assertEqual(result["analysis_cohorts"]["n_label_1_patients"], 3)
        self.assertEqual(result["analysis_cohorts"]["n_label_0_patients"], 4)
        flagged = {
            decision["rule"]
            for decision in result["red_flag_assessment"]["decisions"]
            if decision["triggered"]
        }
        self.assertIn("normal_nuisance_fpr_increase", flagged)

    def test_missing_optional_normal_placebo_does_not_block_gate(self):
        records = []
        positive_mask = self._mask().tolist()
        for patient_index in range(3):
            for variant, score in (
                ("original", 0.8),
                ("lesion_removed", 0.4),
                ("nuisance_changed", 0.8),
                ("placebo", 0.8),
            ):
                records.append(
                    {
                        "patient_id": "abnormal-%d" % patient_index,
                        "variant": variant,
                        "score": score,
                        "label": 1,
                        "mask": positive_mask,
                    }
                )
        for patient_index in range(3):
            for variant, score in (("original", 0.1), ("nuisance_changed", 0.1)):
                records.append(
                    {
                        "patient_id": "normal-%d" % patient_index,
                        "variant": variant,
                        "score": score,
                        "label": 0,
                    }
                )

        result = run_shortcut_evaluation(
            {
                "records": records,
                "decision_threshold": 0.5,
                "bootstrap_iterations": 20,
                "red_flag_rules": {"min_paired_patients": 3},
            }
        )

        normal_placebo = next(
            item
            for item in result["red_flag_assessment"]["decisions"]
            if item["rule"] == "normal_placebo_fpr_increase"
        )
        self.assertEqual(normal_placebo["status"], "not_evaluable")
        self.assertEqual(result["red_flag_assessment"]["status"], "pass")

    def test_multiple_samples_are_aggregated_before_patient_delta(self):
        records = []
        for patient in ("a", "b"):
            for sample_index, original in enumerate((0.6, 1.0)):
                records.extend(
                    [
                        {
                            "patient_id": patient,
                            "sample_id": str(sample_index),
                            "variant": "original",
                            "score": original,
                        },
                        {
                            "patient_id": patient,
                            "sample_id": str(sample_index),
                            "variant": "lesion_removed",
                            "score": original - 0.2,
                        },
                        {
                            "patient_id": patient,
                            "sample_id": str(sample_index),
                            "variant": "nuisance_changed",
                            "score": original,
                        },
                        {
                            "patient_id": patient,
                            "sample_id": str(sample_index),
                            "variant": "placebo",
                            "score": original,
                        },
                    ]
                )
        result = run_shortcut_evaluation(
            {
                "records": records,
                "bootstrap_iterations": 50,
                "red_flag_rules": {"min_paired_patients": 2},
            }
        )
        lesion = result["paired_score_deltas"]["lesion_removed"]
        self.assertEqual(lesion["n_patients"], 2)
        self.assertAlmostEqual(lesion["mean_delta"], -0.2, places=6)

    def test_long_npz_and_preregistered_red_flags(self):
        patients = []
        variants = []
        scores = []
        masks = []
        for patient_index in range(6):
            original_mask = self._mask()
            shifted_mask = self._mask(offset=1)
            for variant, score, mask in (
                ("original", 0.80, original_mask),
                ("lesion_removed", 0.76, np.zeros_like(original_mask)),
                ("nuisance_changed", 0.45, shifted_mask),
                ("placebo", 0.80, original_mask),
            ):
                patients.append("case-%d" % patient_index)
                variants.append(variant)
                scores.append(score)
                masks.append(mask)

        with tempfile.TemporaryDirectory() as temporary_directory:
            path = Path(temporary_directory) / "outputs.npz"
            np.savez(
                str(path),
                patient_id=np.asarray(patients),
                variant=np.asarray(variants),
                score=np.asarray(scores),
                mask=np.asarray(masks),
            )
            result = run_shortcut_evaluation(
                {
                    "input_path": str(path),
                    "bootstrap_iterations": 100,
                    "seed": 9,
                    "red_flag_rules": {"min_paired_patients": 5},
                }
            )

        self.assertTrue(result["red_flag_assessment"]["any_red_flag"])
        flagged = {
            item["rule"]
            for item in result["red_flag_assessment"]["decisions"]
            if item["triggered"]
        }
        self.assertIn("lesion_response_too_small", flagged)
        self.assertIn("nuisance_score_sensitivity", flagged)
        self.assertIn("nuisance_mask_instability", flagged)

    def test_wide_npz_schema(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            path = Path(temporary_directory) / "wide.npz"
            np.savez(
                str(path),
                patient_id=np.asarray(["p1", "p2", "p3"]),
                original_score=np.asarray([0.8, 0.7, 0.9]),
                lesion_removed_score=np.asarray([0.5, 0.4, 0.6]),
                nuisance_changed_score=np.asarray([0.8, 0.7, 0.9]),
                placebo_score=np.asarray([0.8, 0.7, 0.9]),
            )
            result = run_shortcut_evaluation(
                {
                    "input_path": str(path),
                    "bootstrap_iterations": 50,
                    "red_flag_rules": {"min_paired_patients": 3},
                }
            )
        self.assertEqual(result["n_records"], 12)
        self.assertAlmostEqual(
            result["paired_score_deltas"]["lesion_removed"]["mean_delta"],
            -0.3,
            places=6,
        )

    def test_bootstrap_and_soft_dice_edge_cases(self):
        first = bootstrap_ci([0.1, 0.2, 0.3], iterations=100, seed=5)
        second = bootstrap_ci([0.1, 0.2, 0.3], iterations=100, seed=5)
        self.assertEqual(first, second)
        self.assertEqual(soft_dice(np.zeros((3, 3)), np.zeros((3, 3))), 1.0)
        self.assertEqual(soft_dice(np.ones((3, 3)), np.zeros((3, 3))), 0.0)


if __name__ == "__main__":
    unittest.main()
