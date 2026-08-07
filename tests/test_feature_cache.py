import importlib.util
import tempfile
import unittest
from pathlib import Path


@unittest.skipUnless(importlib.util.find_spec("numpy"), "NumPy is not installed")
class FeatureCacheTest(unittest.TestCase):
    def test_round_trip(self):
        import numpy as np

        from medical_experiments.feature_cache import load_feature_record, save_feature_record

        with tempfile.TemporaryDirectory() as directory:
            path = save_feature_record(
                directory,
                {"sample_id": "patient/one", "patient_id": "patient"},
                {"stage": np.ones((2, 3), dtype=np.float32)},
            )
            metadata, arrays = load_feature_record(path)
        self.assertEqual(metadata["sample_id"], "patient/one")
        np.testing.assert_array_equal(arrays["stage"], np.ones((2, 3), dtype=np.float32))

    def test_merge_by_sample_id(self):
        import numpy as np

        from medical_experiments.feature_cache import (
            load_feature_record,
            merge_feature_caches,
            safe_sample_name,
            save_feature_record,
        )

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            metadata = {"sample_id": "s1", "patient_id": "p1", "split": "test"}
            save_feature_record(root / "a", metadata, {"vision": np.ones((2, 2, 2))})
            save_feature_record(root / "b", metadata, {"residual": np.zeros((1, 2, 2)), "mask": np.ones((2, 2))})
            result = merge_feature_caches(
                {
                    "sources": [
                        {"cache_dir": str(root / "a")},
                        {"cache_dir": str(root / "b")},
                    ],
                    "output_dir": str(root / "merged"),
                }
            )
            merged_metadata, arrays = load_feature_record(
                root / "merged" / (safe_sample_name("s1") + ".npz")
            )
        self.assertEqual(result["exported"], 1)
        self.assertEqual(merged_metadata["patient_id"], "p1")
        self.assertEqual(set(arrays), {"sample_id", "patient_id", "split", "vision", "residual", "mask"})

    def test_safe_names_do_not_collide_after_sanitization(self):
        import numpy as np

        from medical_experiments.feature_cache import (
            load_feature_record,
            safe_sample_name,
            save_feature_record,
        )

        self.assertNotEqual(safe_sample_name("case/a"), safe_sample_name("case?a"))
        with tempfile.TemporaryDirectory() as directory:
            first = save_feature_record(
                directory,
                {"sample_id": "case/a", "patient_id": "case", "split": "Train"},
                {"stage": np.ones((1,), dtype=np.float32)},
                overwrite=True,
            )
            second = save_feature_record(
                directory,
                {"sample_id": "case?a", "patient_id": "case", "split": "Train"},
                {"stage": np.zeros((1,), dtype=np.float32)},
                overwrite=True,
            )
            first_metadata, _ = load_feature_record(first)
            second_metadata, _ = load_feature_record(second)

        self.assertNotEqual(first, second)
        self.assertEqual(first_metadata["sample_id"], "case/a")
        self.assertEqual(second_metadata["sample_id"], "case?a")
        self.assertEqual(first_metadata["split"], "train")


if __name__ == "__main__":
    unittest.main()
