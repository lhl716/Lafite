import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

from medical_experiments.lafite_calibration import (
    select_mask_threshold,
    select_score_threshold,
)
from medical_experiments.lafite_medical import prepare_lafite_data
from medical_experiments.manifest import write_jsonl
from medical_experiments.medical_io import prepare_model_image


class LafiteMedicalPreparationTest(unittest.TestCase):
    def _write_png(self, path, value):
        path.parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray(np.full((8, 8), value, dtype=np.uint8), mode="L").save(path)

    def _manifest(self, root):
        records = []
        for dataset, modality in (("Brain", "MRI"), ("Liver", "CT_PNG")):
            cases = (
                ("normal-train", "train", 0, 0),
                ("mixed-train-normal", "train", 0, 0),
                ("mixed-train-positive", "train", 1, 255),
                ("normal-val", "val", 0, 0),
                ("positive-val", "val", 1, 255),
                ("normal-test", "test", 0, 0),
            )
            for name, split, label, mask_value in cases:
                patient_name = name.replace("-normal", "").replace("-positive", "")
                patient = "%s-%s" % (dataset.lower(), patient_name)
                image = root / "images" / dataset.lower() / (name + ".png")
                mask = root / "masks" / dataset.lower() / (name + ".png")
                self._write_png(image, 40 + 10 * label)
                self._write_png(mask, mask_value)
                records.append(
                    {
                        "sample_id": "%s-%s" % (dataset.lower(), name),
                        "patient_id": patient,
                        "image": str(image),
                        "mask": str(mask),
                        "split": split,
                        "modality": modality,
                        "label": label,
                        "source_dataset": "BMAD-" + dataset,
                    }
                )
        path = root / "medical-manifest.jsonl"
        write_jsonl(path, records)
        return path

    def test_strict_training_excludes_normal_slice_from_anomalous_patient(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            result = prepare_lafite_data(self._manifest(root), root / "lafite")
            self.assertEqual(result["normal_training_policy"], "patient_strict")
            for name in ("brain", "liver"):
                train_path = Path(result["datasets"][name]["lafite_meta"]["train"])
                rows = [json.loads(line) for line in train_path.read_text().splitlines()]
                self.assertEqual(len(rows), 1)
                self.assertIn("normal-train", rows[0]["sample_id"])
                self.assertEqual(rows[0]["label"], 0)
                self.assertTrue(Path(rows[0]["filename"]).is_absolute())

    def test_slice_policy_keeps_normal_slice_from_anomalous_patient(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            result = prepare_lafite_data(
                self._manifest(root),
                root / "lafite",
                strict_normal_patients=False,
            )
            for name in ("brain", "liver"):
                self.assertEqual(result["datasets"][name]["counts"]["train"], 2)


class LafiteMedicalCalibrationTest(unittest.TestCase):
    def test_uint8_passthrough_matches_tensor_scaling(self):
        image = np.asarray([[0, 64], [128, 255]], dtype=np.uint8)
        prepared = prepare_model_image(
            image,
            "MRI",
            {"input_scaling": "uint8_passthrough"},
        )
        self.assertEqual(prepared.shape, (2, 2, 3))
        np.testing.assert_allclose(prepared[..., 0], image.astype(np.float32) / 255.0)

    def test_score_threshold_uses_patient_balanced_accuracy(self):
        threshold, metrics = select_score_threshold(
            {"n1": 0.1, "n2": 0.2, "p1": 0.8, "p2": 0.9},
            {"n1": 0, "n2": 0, "p1": 1, "p2": 1},
        )
        self.assertGreater(threshold, 0.2)
        self.assertLess(threshold, 0.8)
        self.assertEqual(metrics["balanced_accuracy"], 1.0)

    def test_mask_threshold_is_global_and_deterministic(self):
        predictions = [
            np.asarray([[0.1, 0.2], [0.8, 0.9]], dtype=np.float32),
            np.asarray([[0.05, 0.15], [0.7, 0.95]], dtype=np.float32),
        ]
        masks = [
            np.asarray([[0, 0], [1, 1]], dtype=np.uint8),
            np.asarray([[0, 0], [1, 1]], dtype=np.uint8),
        ]
        threshold, metrics = select_mask_threshold(predictions, masks, candidate_count=21)
        self.assertGreater(threshold, 0.2)
        self.assertLessEqual(threshold, 0.7)
        self.assertEqual(metrics["mean_positive_slice_dice"], 1.0)


if __name__ == "__main__":
    unittest.main()
