import json
import tempfile
import unittest
from pathlib import Path

from medical_experiments.config import load_config, resolve_path, write_json


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


if __name__ == "__main__":
    unittest.main()
