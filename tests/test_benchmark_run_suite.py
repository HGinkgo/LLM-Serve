import json
import subprocess
import tempfile
import unittest
from pathlib import Path


SUITE = {
    "schema_version": 1,
    "name": "test-suite",
    "runs": 1,
    "profiles": {
        "tiny": {
            "classes": [
                {"name": "short", "weight": 1, "input_len": 2, "output_len": 1}
            ]
        }
    },
    "experiments": [
        {
            "name": "poisson",
            "arrival": "poisson",
            "profile": "tiny",
            "num_requests": 2,
            "request_rates": [1.0],
            "variants": [
                {"name": "baseline"},
                {"name": "chunked", "enable_chunked_prefill": True},
            ],
            "runtime": {"max_model_len": 32, "max_num_batched_tokens": 8},
        }
    ],
}


def successful_result(point, commit):
    return {
        "schema_version": 2,
        "complete": True,
        "point_id": point["point_id"],
        "git_commit": commit,
        "metadata": {"git_commit": commit, "git_dirty": False},
        "config": point,
        "metrics": {
            "completed": 2,
            "failed": 0,
            "throughput": {
                "requests_per_second": 1.0,
                "input_tokens_per_second": 2.0,
                "output_tokens_per_second": 1.0,
                "total_tokens_per_second": 3.0,
            },
            "latency": {"overall": {}},
            "speculative": {"acceptance_rate": None},
        },
        "requests": [],
    }


class BenchmarkRunSuiteTests(unittest.TestCase):
    def test_execute_suite_writes_manifest_runs_and_csv_outputs(self):
        from benchmarks.run_suite import execute_suite

        commands = []

        def command_runner(command, **kwargs):
            commands.append(command)
            point_path = Path(command[command.index("--point-config") + 1])
            output_path = Path(command[command.index("--output") + 1])
            point = json.loads(point_path.read_text())
            output_path.write_text(json.dumps(successful_result(point, "abc123")))
            return subprocess.CompletedProcess(command, 0, "ok", "")

        with tempfile.TemporaryDirectory() as directory:
            output_dir = Path(directory)
            exit_code = execute_suite(
                SUITE,
                output_dir,
                model="/private/Qwen3-8B",
                speculative_model=None,
                metadata={"git_commit": "abc123", "git_dirty": False},
                distributed_init_method="tcp://localhost:2444",
                command_runner=command_runner,
            )

            self.assertEqual(exit_code, 0)
            self.assertEqual(len(commands), 2)
            self.assertTrue(all(
                command[command.index("--distributed-init-method") + 1]
                == "tcp://localhost:2444"
                for command in commands
            ))
            manifest = json.loads((output_dir / "manifest.json").read_text())
            self.assertTrue(manifest["complete"])
            self.assertEqual(manifest["completed_points"], 2)
            self.assertEqual(manifest["model"], "Qwen3-8B")
            self.assertEqual(
                manifest["distributed_init_method"],
                "tcp://localhost:2444",
            )
            self.assertTrue((output_dir / "summary.csv").exists())
            self.assertTrue((output_dir / "aggregate.csv").exists())
            self.assertNotIn(b"\r", (output_dir / "summary.csv").read_bytes())
            self.assertNotIn(b"\r", (output_dir / "aggregate.csv").read_bytes())
            self.assertEqual(len(list((output_dir / "runs").glob("*.json"))), 2)

    def test_execute_suite_records_failure_and_continues(self):
        from benchmarks.run_suite import execute_suite

        calls = 0

        def command_runner(command, **kwargs):
            nonlocal calls
            calls += 1
            if calls == 1:
                return subprocess.CompletedProcess(command, 2, "", "out of memory")
            point_path = Path(command[command.index("--point-config") + 1])
            output_path = Path(command[command.index("--output") + 1])
            point = json.loads(point_path.read_text())
            output_path.write_text(json.dumps(successful_result(point, "abc123")))
            return subprocess.CompletedProcess(command, 0, "ok", "")

        with tempfile.TemporaryDirectory() as directory:
            output_dir = Path(directory)
            exit_code = execute_suite(
                SUITE,
                output_dir,
                model="/models/Qwen3-8B",
                speculative_model=None,
                metadata={"git_commit": "abc123", "git_dirty": False},
                command_runner=command_runner,
            )

            self.assertEqual(exit_code, 1)
            self.assertEqual(calls, 2)
            manifest = json.loads((output_dir / "manifest.json").read_text())
            self.assertFalse(manifest["complete"])
            self.assertEqual(manifest["failed_points"], 1)
            failures = [
                json.loads(path.read_text())
                for path in (output_dir / "runs").glob("*.json")
                if not json.loads(path.read_text())["complete"]
            ]
            self.assertEqual(failures[0]["error"]["returncode"], 2)

    def test_execute_suite_resume_skips_matching_complete_result(self):
        from benchmarks.run_suite import execute_suite
        from benchmarks.suite import expand_suite

        one_variant_suite = json.loads(json.dumps(SUITE))
        one_variant_suite["experiments"][0]["variants"] = [
            {"name": "baseline"}
        ]
        point = expand_suite(one_variant_suite)[0]

        with tempfile.TemporaryDirectory() as directory:
            output_dir = Path(directory)
            (output_dir / "runs").mkdir()
            result_path = output_dir / "runs" / f"{point['point_id']}.json"
            result_path.write_text(json.dumps(successful_result(point, "abc123")))

            exit_code = execute_suite(
                one_variant_suite,
                output_dir,
                model="/models/Qwen3-8B",
                speculative_model=None,
                metadata={"git_commit": "abc123", "git_dirty": False},
                resume=True,
                command_runner=lambda *args, **kwargs: self.fail(
                    "matching result should have been resumed"
                ),
            )

            self.assertEqual(exit_code, 0)

    def test_execute_suite_records_worker_start_failure(self):
        from benchmarks.run_suite import execute_suite

        one_variant_suite = json.loads(json.dumps(SUITE))
        one_variant_suite["experiments"][0]["variants"] = [
            {"name": "baseline"}
        ]

        with tempfile.TemporaryDirectory() as directory:
            exit_code = execute_suite(
                one_variant_suite,
                Path(directory),
                model="/models/Qwen3-8B",
                speculative_model=None,
                metadata={"git_commit": "abc123", "git_dirty": False},
                command_runner=lambda *args, **kwargs: (_ for _ in ()).throw(
                    OSError("worker unavailable")
                ),
            )

            self.assertEqual(exit_code, 1)
            result_path = next((Path(directory) / "runs").glob("*.json"))
            result = json.loads(result_path.read_text())
            self.assertEqual(result["error"]["type"], "WorkerStartError")
