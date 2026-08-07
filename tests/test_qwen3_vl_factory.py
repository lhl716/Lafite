import unittest

from medical_experiments.factories.qwen3_vl import (
    _append_candidate,
    _deranged_descriptions,
)


class Qwen3VLFactoryTest(unittest.TestCase):
    def test_knowledge_derangement_has_no_fixed_points(self):
        candidates = ["a", "b", "c"]
        descriptions = {value: "description-" + value for value in candidates}
        shuffled = _deranged_descriptions(candidates, descriptions, seed=7)
        self.assertEqual(set(shuffled), set(candidates))
        self.assertEqual(set(shuffled.values()), set(descriptions.values()))
        self.assertTrue(
            all(shuffled[value] != descriptions[value] for value in candidates)
        )

    def test_candidate_tokens_extend_text_fields_only(self):
        try:
            import torch
        except ImportError:
            self.skipTest("PyTorch is unavailable")
        base = {
            "input_ids": torch.tensor([[1, 2, 3]]),
            "attention_mask": torch.tensor([[1, 1, 1]]),
            "pixel_values": torch.ones(2, 4),
            "image_grid_thw": torch.tensor([[1, 2, 2]]),
        }
        result, prompt_length, candidate_length = _append_candidate(
            base, torch.tensor([[4, 5]]), torch
        )
        self.assertEqual((prompt_length, candidate_length), (3, 2))
        self.assertEqual(result["input_ids"].tolist(), [[1, 2, 3, 4, 5]])
        self.assertEqual(result["attention_mask"].tolist(), [[1, 1, 1, 1, 1]])
        self.assertIs(result["pixel_values"], base["pixel_values"])


if __name__ == "__main__":
    unittest.main()
