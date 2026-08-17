import tempfile
import unittest
from pathlib import Path


class BenchmarkEnvironmentTests(unittest.TestCase):
    def test_discovers_huggingface_cache_revision(self):
        from benchmarks.environment import discover_model_revision

        revision = "b968826d9c46dd6066d109eabc6255188de91218"
        with tempfile.TemporaryDirectory() as directory:
            metadata = (
                Path(directory)
                / ".cache/huggingface/download/config.json.metadata"
            )
            metadata.parent.mkdir(parents=True)
            metadata.write_text(f"{revision}\nunused\n")

            self.assertEqual(discover_model_revision(directory), revision)


if __name__ == "__main__":
    unittest.main()
