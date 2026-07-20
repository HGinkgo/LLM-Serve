import json
import tempfile
import unittest
from pathlib import Path

from scripts.summarize_benchmarks import build_row, sanitize_result


class SummarizeBenchmarksTest(unittest.TestCase):

    def make_result(self):
        return {
            "config": {
                "model": "/private/models/Qwen3-8B",
                "speculative_model": "/private/models/Qwen3-8B-speculator.eagle3",
                "prompt_file": "/private/prompts/natural.txt",
                "arrival": "all",
                "num_requests": 4,
                "input_len": 128,
                "output_len": 256,
                "speculative_gamma": 3,
                "enable_chunked_prefill": False,
            },
            "metadata": {"git_commit": "abc123"},
            "metrics": {
                "summary": {
                    "throughput": 100.0,
                    "ttft": {"p50": 0.1, "p99": 0.2},
                    "itl": {"p50": 0.03, "p99": 0.05},
                    "tpot": {"p50": 0.04, "p99": 0.06},
                    "request_latency": {"p50": 2.0, "p99": 3.0},
                    "speculative": {
                        "acceptance_rate": 0.5,
                        "acceptance_length": 2.0,
                    },
                },
                "requests": [],
            },
        }

    def test_build_row_keeps_old_itl_as_burst_only(self):
        row = build_row(Path("run1.json"), self.make_result())

        self.assertEqual(row["burst_itl_p50_ms"], 30.0)
        self.assertEqual(row["output_event_latency_p50_ms"], "")
        self.assertEqual(row["speculative_step_latency_p50_ms"], "")
        self.assertEqual(row["git_commit"], "abc123")

    def test_sanitize_result_removes_private_paths_without_mutating_input(self):
        result = self.make_result()

        sanitized = sanitize_result(result)

        self.assertEqual(sanitized["config"]["model"], "Qwen3-8B")
        self.assertEqual(
            sanitized["config"]["speculative_model"],
            "Qwen3-8B-speculator.eagle3",
        )
        self.assertEqual(sanitized["config"]["prompt_file"], "natural.txt")
        self.assertEqual(result["config"]["model"], "/private/models/Qwen3-8B")

    def test_sanitized_result_can_be_serialized(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "result.json"
            path.write_text(json.dumps(sanitize_result(self.make_result()), indent=2))

            loaded = json.loads(path.read_text())

        self.assertEqual(loaded["config"]["model"], "Qwen3-8B")


if __name__ == "__main__":
    unittest.main()
