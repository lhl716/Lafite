import tempfile
import unittest
import zipfile
from pathlib import Path

import numpy as np
from PIL import Image

from medical_experiments.bmad_prepare import (
    extract_bmad_subsets,
    prepare_bmad_manifests,
)
from medical_experiments.manifest import read_jsonl, validate_manifest


def _image(path: Path, *, lesion: bool = False) -> None:
    array = np.zeros((24, 24), dtype=np.uint8)
    array[3:21, 3:21] = 80
    if lesion:
        array[9:14, 10:15] = 220
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(array).save(path)


def _mask(path: Path) -> None:
    array = np.zeros((24, 24), dtype=np.uint8)
    array[9:14, 10:15] = 255
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(array).save(path)


class BmadPrepareTests(unittest.TestCase):
    def _make_subset(self, root: Path, dataset: str) -> None:
        if dataset == "Brain":
            normal_names = ["00001_10", "00002_10", "00003_10", "00004_10", "00005_10"]
            positive_names = ["00101_10", "00102_10", "00103_10", "00104_10"]
        else:
            normal_names = ["img0001_10", "img0002_10", "3_10", "4_10", "5_10"]
            positive_names = ["11_10", "12_10", "13_10", "14_10"]

        for name in normal_names[:2]:
            _image(root / dataset / "train" / "good" / f"{name}.png")
        for index, name in enumerate(normal_names[2:]):
            split = "valid" if index == 0 else "test"
            _image(root / dataset / split / "good" / "img" / f"{name}.png")
        for index, name in enumerate(positive_names):
            split = "valid" if index < 2 else "test"
            class_name = "Ungood" if dataset == "Brain" else "ungood"
            _image(root / dataset / split / class_name / "img" / f"{name}.png", lesion=True)
            _mask(root / dataset / split / class_name / "label" / f"{name}.png")

    def test_prepare_creates_patient_safe_manifests_and_masks(self):
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            raw = base / "raw"
            self._make_subset(raw, "Brain")
            self._make_subset(raw, "Liver")

            summary = prepare_bmad_manifests(raw, base / "prepared", seed=7)
            medical_path = Path(summary["medical_manifest"])
            semantic_path = Path(summary["semantic_manifest"])
            medical = read_jsonl(medical_path)
            semantic = read_jsonl(semantic_path)

            validation = validate_manifest(
                medical,
                manifest_path=medical_path,
                check_paths=True,
                require_mask=True,
                require_semantics=True,
            )
            self.assertTrue(validation["valid"], validation["errors"])
            self.assertEqual({"train", "val", "test"}, {row["split"] for row in medical})
            self.assertEqual({"MRI", "CT_PNG"}, {row["modality"] for row in medical})
            self.assertEqual(len({row["patient_id"] for row in medical}), len(semantic))
            self.assertEqual(
                {"brain×normal", "brain×tumor", "liver×normal", "liver×tumor"},
                set(summary["semantic_edge_counts"]),
            )

            for row in medical:
                mask = np.asarray(Image.open(medical_path.parent / row["mask"]))
                nuisance = np.asarray(Image.open(medical_path.parent / row["nuisance_mask"]))
                placebo = np.asarray(Image.open(medical_path.parent / row["placebo_allowed_mask"]))
                self.assertEqual(bool(mask.any()), bool(row["label"]))
                self.assertTrue(nuisance.any())
                self.assertTrue(placebo.any())

    def test_slice_limits_preserve_positive_patient_representative(self):
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            raw = base / "raw"
            self._make_subset(raw, "Brain")
            self._make_subset(raw, "Liver")
            summary = prepare_bmad_manifests(
                raw,
                base / "prepared",
                seed=11,
                max_slices_per_patient=1,
                semantic_patients_per_edge=1,
            )
            medical = read_jsonl(summary["medical_manifest"])
            semantic = read_jsonl(summary["semantic_manifest"])
            self.assertEqual(len(medical), len({row["patient_id"] for row in medical}))
            self.assertEqual(4, len(semantic))
            self.assertEqual({0, 1}, {row["label"] for row in semantic})

    def test_selective_extraction_rejects_path_traversal(self):
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            archive = base / "bad.zip"
            with zipfile.ZipFile(archive, "w") as handle:
                handle.writestr("Brain/../../escape.png", b"bad")
            with self.assertRaises(ValueError):
                extract_bmad_subsets(archive, base / "raw")


if __name__ == "__main__":
    unittest.main()
