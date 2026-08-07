import json
import os
import tempfile
import unittest
from pathlib import Path

from medical_experiments.config import load_config, resolve_known_paths, resolve_path, write_json


class ConfigTest(unittest.TestCase):
    def test_json_config_resolves_relative_paths(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config_path = root / "config.json"
            config_path.write_text(json.dumps({"manifest": "data/manifest.jsonl"}), encoding="utf-8")
            config = load_config(config_path)
            self.assertEqual(resolve_path(config["manifest"], config), (root / "data/manifest.jsonl").resolve())

    def test_write_json_serializes_paths(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "result.json"
            write_json(output, {"path": Path("sample")})
            self.assertEqual(json.loads(output.read_text(encoding="utf-8"))["path"], "sample")

    def test_environment_values_and_defaults_are_expanded(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "config.json"
            path.write_text(
                '{"manifest":"${TEST_MEDICAL_MANIFEST}","model_id":"${MISSING_MODEL:-Qwen/model}"}',
                encoding="utf-8",
            )
            with unittest.mock.patch.dict(
                os.environ, {"TEST_MEDICAL_MANIFEST": "/data/manifest.jsonl"}, clear=False
            ):
                config = load_config(path)
        self.assertEqual(config["manifest"], "/data/manifest.jsonl")
        self.assertEqual(config["model_id"], "Qwen/model")

    def test_missing_required_environment_value_fails(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "config.json"
            path.write_text('{"manifest":"${DEFINITELY_MISSING_LAFITE_VAR}"}', encoding="utf-8")
            with unittest.mock.patch.dict(os.environ, {}, clear=True):
                with self.assertRaisesRegex(ValueError, "DEFINITELY_MISSING"):
                    load_config(path)

    def test_model_id_is_not_resolved_as_local_path(self):
        config = {"factory_kwargs": {"model_id": "Qwen/Qwen3-VL-8B-Instruct"}}
        self.assertEqual(
            resolve_known_paths(config)["factory_kwargs"]["model_id"],
            "Qwen/Qwen3-VL-8B-Instruct",
        )


if __name__ == "__main__":
    unittest.main()
