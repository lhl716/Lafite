import json
import tempfile
import unittest
from pathlib import Path

from medical_experiments.manifest import (
    assign_patient_splits,
    manifest_fingerprint,
    normalize_split,
    read_jsonl,
    validate_manifest,
    write_jsonl,
)


class ManifestTest(unittest.TestCase):
    def _records(self):
        return [
            {
                "sample_id": "p1-s1",
                "patient_id": "p1",
                "image": "p1.npy",
                "mask": "p1-mask.npy",
                "split": "train",
                "modality": "CT",
                "label": 1,
            },
            {
                "sample_id": "p1-s2",
                "patient_id": "p1",
                "image": "p1.npy",
                "mask": "p1-mask.npy",
                "split": "train",
                "modality": "CT",
                "label": 1,
            },
            {
                "sample_id": "p2-s1",
                "patient_id": "p2",
                "image": "p2.npy",
                "mask": "p2-mask.npy",
                "split": "test",
                "modality": "MRI",
                "label": 0,
            },
        ]

    def test_round_trip_and_fingerprint(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "manifest.jsonl"
            write_jsonl(path, self._records())
            loaded = read_jsonl(path)
        self.assertEqual(loaded, self._records())
        self.assertEqual(manifest_fingerprint(loaded), manifest_fingerprint(list(reversed(loaded))))

    def test_patient_leakage_is_rejected(self):
        records = self._records()
        records[1]["split"] = "test"
        result = validate_manifest(records)
        self.assertFalse(result["valid"])
        self.assertTrue(any("patient leakage" in error for error in result["errors"]))

    def test_split_validation_is_case_insensitive_and_canonical(self):
        records = self._records()
        records[0]["split"] = " Train "
        records[1]["split"] = "TRAIN"
        result = validate_manifest(records)
        self.assertTrue(result["valid"])
        self.assertEqual(result["split_counts"], {"test": 1, "train": 2})
        self.assertEqual(normalize_split(" Train "), "train")

    def test_patient_split_keeps_slices_together(self):
        records = []
        for patient_index in range(12):
            for slice_index in range(2):
                records.append(
                    {
                        "sample_id": "p%d-s%d" % (patient_index, slice_index),
                        "patient_id": "p%d" % patient_index,
                        "image": "placeholder.npy",
                        "split": "train",
                        "modality": "CT" if patient_index % 2 else "MRI",
                    }
                )
        split_records = assign_patient_splits(
            records,
            {"train": 0.5, "val": 0.25, "test": 0.25},
            seed=7,
            stratify_field="modality",
        )
        self.assertTrue(validate_manifest(split_records)["valid"])
        patient_splits = {}
        for record in split_records:
            patient_splits.setdefault(record["patient_id"], set()).add(record["split"])
        self.assertTrue(all(len(splits) == 1 for splits in patient_splits.values()))


if __name__ == "__main__":
    unittest.main()
