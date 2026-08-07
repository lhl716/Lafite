import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from medical_experiments.detail_probe import (
    RepresentationRecord,
    _split_indices,
    load_representation_cache,
    run_detail_probe,
)
from medical_experiments.probes import (
    evaluate_segmentation,
    fixed_random_channel_projection,
    spatial_shuffle_features,
)


def _synthetic_batch(path: Path, duplicate_patient_across_splits: bool = False) -> None:
    rng = np.random.default_rng(7)
    sample_count = 12
    height = width = 8
    masks = np.zeros((sample_count, 1, height, width), dtype=np.float32)
    for index in range(sample_count):
        if index % 3 != 0:
            top = 1 + (index % 4)
            left = 1 + ((index * 2) % 4)
            masks[index, 0, top : top + 2, left : left + 2] = 1.0

    stage_nchw = rng.normal(0.0, 0.08, size=(sample_count, 2, height, width)).astype(
        np.float32
    )
    stage_nchw[:, 0] += masks[:, 0] * 4.0
    stage_nchw[:, 1] += masks[:, 0] * -2.0

    pooled = masks[:, 0].reshape(sample_count, 4, 2, 4, 2).max(axis=(2, 4))
    stage_nlc = rng.normal(0.0, 0.08, size=(sample_count, 17, 3)).astype(np.float32)
    stage_nlc[:, 1:, 0] += pooled.reshape(sample_count, 16) * 3.0
    stage_nlc[:, 1:, 1] += pooled.reshape(sample_count, 16) * -1.5

    splits = np.asarray(["train"] * 6 + ["val"] * 3 + ["test"] * 3)
    patients = np.asarray(["patient_%02d" % index for index in range(sample_count)])
    if duplicate_patient_across_splits:
        patients[-1] = patients[0]
    np.savez(
        path,
        sample_id=np.asarray(["sample_%02d" % index for index in range(sample_count)]),
        patient_id=patients,
        split=splits,
        coarse_label=np.asarray(["ct" if index % 2 == 0 else "mri" for index in range(sample_count)]),
        mask=masks,
        fine=stage_nchw,
        tokens=stage_nlc,
    )


class DetailProbeTests(unittest.TestCase):
    def test_run_detail_probe_supports_nchw_nlc_and_json(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            cache_path = Path(directory) / "representations.npz"
            _synthetic_batch(cache_path)
            result = run_detail_probe(
                {
                    "cache_paths": [str(cache_path)],
                    "stage_keys": ["fine", "tokens"],
                    "stage_layouts": {"fine": "NCHW", "tokens": "NLC"},
                    "stage_grids": {"tokens": [4, 4]},
                    "prefix_tokens": {"tokens": 1},
                    "target_grid": [8, 8],
                    "feature_sets": {"fused": ["fine", "tokens"]},
                    "coarse_label_key": "coarse_label",
                    "channel_budget": 4,
                    "channel_projection": {"type": "orthogonal", "seed": 123},
                    "device": "cpu",
                    "torch_num_threads": 1,
                    "seed": 11,
                    "probe": {
                        "type": "linear",
                        "epochs": 8,
                        "batch_size": 3,
                        "patience": 8,
                        "learning_rate": 0.03,
                        "threshold_candidates": [0.25, 0.5, 0.75],
                    },
                    "small_lesion": {"max_pixels": 8, "min_overlap": 0.1},
                    "spatial_shuffle_control": {"enabled": True, "seed": 99},
                }
            )

            json.dumps(result, allow_nan=False)
            self.assertEqual(result["data"]["num_samples"], 12)
            self.assertEqual(result["data"]["target_grid"], [8, 8])
            self.assertEqual(result["feature_sets"]["fused"]["original_input_channels"], 5)
            self.assertEqual(result["feature_sets"]["fused"]["input_channels"], 4)
            self.assertTrue(result["feature_sets"]["fused"]["channel_projection"]["enabled"])
            self.assertEqual(
                result["feature_sets"]["fused"]["training"]["parameter_count"], 5
            )
            self.assertEqual(
                result["feature_sets"]["fused"]["metrics"]["test"]["num_patients"],
                3,
            )
            self.assertTrue(
                result["feature_sets"]["fused"]["spatial_shuffle_control"]["enabled"]
            )
            self.assertTrue(result["feature_sets"]["fused"]["coordinate_control"]["enabled"])
            self.assertTrue(
                result["feature_sets"]["fused"]["random_feature_control"]["enabled"]
            )
            observed_parameters = result["feature_sets"]["fused"]["training"][
                "parameter_count"
            ]
            self.assertEqual(
                result["feature_sets"]["fused"]["coordinate_control"]["training"][
                    "parameter_count"
                ],
                observed_parameters,
            )
            self.assertEqual(
                result["feature_sets"]["fused"]["random_feature_control"]["training"][
                    "parameter_count"
                ],
                observed_parameters,
            )
            self.assertEqual(result["coarse_label_probe"]["status"], "ok")
            self.assertEqual(
                set(result["coarse_label_probe"]["stages"]), {"fine", "tokens"}
            )
            self.assertIn(
                "patient_dice_mean",
                result["feature_sets"]["fused"]["spatial_shuffle_control"][
                    "metric_residuals"
                ]["test"],
            )

    def test_loader_accepts_one_sample_per_npz(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for index, split in enumerate(("train", "val", "test")):
                mask = np.zeros((6, 6), dtype=np.uint8)
                mask[2:4, 2:4] = 255
                np.savez(
                    root / ("sample_%d.npz" % index),
                    sample_id=np.asarray("s%d" % index),
                    patient_id=np.asarray("p%d" % index),
                    split=np.asarray(split),
                    modality=np.asarray("CT"),
                    label=np.asarray(1),
                    metadata_json=np.asarray('{"cache_version": 1, "organ": "liver"}'),
                    mask=mask,
                    fine=np.ones((2, 6, 6), dtype=np.float32) * index,
                    tokens=np.ones((10, 3), dtype=np.float32) * index,
                )
            records = load_representation_cache(
                {
                    "cache_paths": str(root),
                    "stage_layouts": {"fine": "NCHW", "tokens": "NLC"},
                    "stage_grids": {"tokens": [3, 3]},
                    "prefix_tokens": {"tokens": 1},
                    "coarse_label_key": "organ",
                    "mask_positive_threshold": 127,
                }
            )
            self.assertEqual(len(records), 3)
            self.assertEqual(records[0].stages["fine"].shape, (2, 6, 6))
            self.assertEqual(records[0].stages["tokens"].shape, (3, 3, 3))
            self.assertEqual(set(records[0].stages), {"fine", "tokens"})
            self.assertEqual(records[0].coarse_label, "liver")
            self.assertEqual(int(records[0].mask.sum()), 4)

    def test_patient_overlap_is_rejected(self) -> None:
        records = [
            RepresentationRecord(
                sample_id="a",
                patient_id="same",
                split="train",
                mask=np.zeros((2, 2), dtype=np.float32),
                stages={"x": np.zeros((1, 2, 2), dtype=np.float32)},
                source_path="a.npz",
            ),
            RepresentationRecord(
                sample_id="b",
                patient_id="v",
                split="val",
                mask=np.zeros((2, 2), dtype=np.float32),
                stages={"x": np.zeros((1, 2, 2), dtype=np.float32)},
                source_path="b.npz",
            ),
            RepresentationRecord(
                sample_id="c",
                patient_id="same",
                split="test",
                mask=np.zeros((2, 2), dtype=np.float32),
                stages={"x": np.zeros((1, 2, 2), dtype=np.float32)},
                source_path="c.npz",
            ),
        ]
        with self.assertRaisesRegex(ValueError, "multiple splits"):
            _split_indices(records, {})

    def test_spatial_shuffle_is_stable_and_preserves_values(self) -> None:
        features = torch.arange(2 * 3 * 4 * 4, dtype=torch.float32).reshape(2, 3, 4, 4)
        first = spatial_shuffle_features(features, ["a", "b"], seed=17)
        second = spatial_shuffle_features(features, ["a", "b"], seed=17)
        self.assertTrue(torch.equal(first, second))
        self.assertFalse(torch.equal(first, features))
        for sample in range(2):
            for channel in range(3):
                self.assertTrue(
                    torch.equal(
                        torch.sort(first[sample, channel].reshape(-1)).values,
                        torch.sort(features[sample, channel].reshape(-1)).values,
                    )
                )

    def test_channel_budget_expands_without_trainable_parameters(self) -> None:
        features = torch.randn(2, 2, 3, 3)
        projected, summary = fixed_random_channel_projection(
            features, channel_budget=5, seed=23, projection_type="orthogonal"
        )
        projected_again, _ = fixed_random_channel_projection(
            features, channel_budget=5, seed=23, projection_type="orthogonal"
        )
        self.assertEqual(tuple(projected.shape), (2, 5, 3, 3))
        self.assertTrue(torch.allclose(projected, projected_again))
        self.assertEqual(summary["relation"], "expand")
        self.assertFalse(summary["trainable"])

    def test_patient_dice_and_small_lesion_sensitivity(self) -> None:
        masks = np.zeros((3, 6, 6), dtype=np.float32)
        masks[0, 1:3, 1:3] = 1.0
        masks[1, 4:6, 4:6] = 1.0
        probabilities = masks * 0.9 + (1.0 - masks) * 0.1
        probabilities[1] = 0.1
        metrics = evaluate_segmentation(
            probabilities,
            masks,
            patient_ids=["p1", "p1", "p2"],
            threshold=0.5,
            small_lesion_max_pixels=4,
            small_lesion_min_overlap=0.25,
        )
        self.assertEqual(metrics["num_patients"], 2)
        self.assertEqual(len(metrics["per_patient"]), 2)
        self.assertEqual(metrics["small_lesion_count"], 2)
        self.assertEqual(metrics["small_lesion_detected"], 1)
        self.assertAlmostEqual(metrics["small_lesion_sensitivity_micro"], 0.5)


if __name__ == "__main__":
    unittest.main()
