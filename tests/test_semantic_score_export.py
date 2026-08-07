import tempfile
import unittest
from pathlib import Path
from unittest import mock

from medical_experiments.manifest import read_jsonl, write_jsonl
from medical_experiments.semantic_score_export import export_semantic_scores


class SemanticScoreExportTest(unittest.TestCase):
    def test_target_and_evidence_fields_are_stripped(self):
        calls = []

        def scorer(record, candidates, evidence_present, knowledge_intact):
            calls.append((dict(record), evidence_present, knowledge_intact))
            return {candidate: float(index) for index, candidate in enumerate(candidates)}

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "input.jsonl"
            output = root / "output.jsonl"
            write_jsonl(
                source,
                [
                    {
                        "patient_id": "p1",
                        "organ": "lung",
                        "region": "upper",
                        "finding": "nodule",
                        "report": "target report",
                        "image": "secret-image.npy",
                    },
                    {
                        "patient_id": "p2",
                        "organ": "lung",
                        "region": "lower",
                        "finding": "opacity",
                        "image": "other-image.npy",
                    },
                ],
            )
            config = {
                "manifest": str(source),
                "output_manifest": str(output),
                "model_factory": "unused:factory",
            }
            with mock.patch(
                "medical_experiments.semantic_score_export._load_scorer",
                return_value=scorer,
            ):
                result = export_semantic_scores(config)
            exported = read_jsonl(output)

        self.assertEqual(result["samples"], 2)
        self.assertEqual(len(calls), 8)
        self.assertTrue(all("finding" not in record and "report" not in record for record, _, _ in calls))
        with_image_records = [record for record, evidence, _ in calls if evidence]
        self.assertTrue(all(Path(record["image"]).is_absolute() for record in with_image_records))
        no_image_records = [record for record, evidence, _ in calls if not evidence]
        self.assertTrue(all("image" not in record and "region" not in record for record in no_image_records))
        self.assertIn("scores_no_image_shuffled_knowledge", exported[0])


if __name__ == "__main__":
    unittest.main()
