import os
import sys
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from io import StringIO
from types import SimpleNamespace
from unittest.mock import patch

import bench_serving


class FakeLLM:
    calls = []
    instances = []

    def __init__(self, model, **kwargs):
        self.model = model
        self.kwargs = kwargs
        self.pending = False
        self.model_runner = SimpleNamespace(sampler=None)
        FakeLLM.calls.append((model, kwargs))
        FakeLLM.instances.append(self)

    def add_request(self, prompt, sampling_params):
        self.pending = True
        return 1

    def is_finished(self):
        return not self.pending

    def step(self):
        self.pending = False
        return [], -1

    def get_metrics(self):
        return {
            "summary": {
                "num_requests": 1,
                "num_finished": 1,
                "num_failed": 0,
                "total_output_tokens": 1,
                "wall_time": 1.0,
                "throughput": 1.0,
                "ttft": {"mean": 0.1, "p50": 0.1, "p99": 0.1, "max": 0.1},
                "itl": {"mean": None, "p50": None, "p99": None, "max": None},
                "tpot": {"mean": 0.0, "p50": 0.0, "p99": 0.0, "max": 0.0},
                "request_latency": {"mean": 0.2, "p50": 0.2, "p99": 0.2, "max": 0.2},
                "speculative": {
                    "steps": 1,
                    "draft_tokens": 3,
                    "accepted_tokens": 2,
                    "emitted_tokens": 3,
                    "acceptance_rate": 2 / 3,
                    "acceptance_length": 3.0,
                    "accepted_length": 2.0,
                    "draft_tokens_per_step": 3.0,
                    "accept_all_count": 0,
                    "gamma_counts": {"3": 1},
                    "timing": {
                        "draft_proposal_time": {"total": 0.02, "mean": 0.02},
                        "draft_pack_time": {"total": 0.001, "mean": 0.001},
                        "draft_forward_time": {"total": 0.012, "mean": 0.012},
                        "draft_sample_time": {"total": 0.002, "mean": 0.002},
                        "draft_compact_time": {"total": 0.004, "mean": 0.004},
                        "target_verify_time": {"total": 0.03, "mean": 0.03},
                        "accept_time": {"total": 0.004, "mean": 0.004},
                        "kv_update_time": {"total": 0.005, "mean": 0.005},
                        "trace_time": {"total": 0.006, "mean": 0.006},
                        "total_time": {"total": 0.075, "mean": 0.075},
                    },
                },
            },
            "requests": [],
        }

    def exit(self):
        pass


class BenchServingSpeculativeTest(unittest.TestCase):

    def test_build_environment_metadata_records_commit_and_cpu_runtime(self):
        with patch.object(
            bench_serving,
            "git_output",
            side_effect=["abc123", ""],
        ), patch.object(bench_serving.torch.cuda, "is_available", return_value=False):
            metadata = bench_serving.build_environment_metadata()

        self.assertEqual(metadata["schema_version"], 1)
        self.assertEqual(metadata["git_commit"], "abc123")
        self.assertFalse(metadata["git_dirty"])
        self.assertFalse(metadata["cuda_available"])
        self.assertIsNone(metadata["gpu_name"])
        self.assertIsNone(metadata["gpu_memory_bytes"])
        self.assertIn("python_version", metadata)
        self.assertIn("torch_version", metadata)

    def test_discover_model_revision_reads_huggingface_metadata(self):
        with tempfile.TemporaryDirectory() as directory:
            metadata = os.path.join(
                directory,
                ".cache",
                "huggingface",
                "download",
                "config.json.metadata",
            )
            os.makedirs(os.path.dirname(metadata))
            with open(metadata, "w") as f:
                f.write("a" * 40 + "\netag\n123\n")

            revision = bench_serving.discover_model_revision(directory)

        self.assertEqual(revision, "a" * 40)

    def test_parse_args_accepts_closed_loop_flags(self):
        with patch.object(
            sys,
            "argv",
            [
                "bench_serving.py",
                "--arrival",
                "closed-loop",
                "--max-concurrency",
                "4",
                "--warmup-seconds",
                "2",
                "--measurement-seconds",
                "5",
            ],
        ):
            args = bench_serving.parse_args()

        self.assertEqual(args.arrival, "closed-loop")
        self.assertEqual(args.max_concurrency, 4)
        self.assertEqual(args.warmup_seconds, 2.0)
        self.assertEqual(args.measurement_seconds, 5.0)

    def test_build_steady_state_summary_uses_measurement_window(self):
        metrics = {
            "requests": [
                {
                    "arrival_time": 9.0,
                    "first_token_time": 10.0,
                    "token_times": [10.0, 11.0],
                    "output_event_times": [10.0, 11.0],
                    "burst_itl": [1.0],
                    "output_event_latency": [1.0],
                    "speculative_step_latency": [],
                    "finish_time": 11.0,
                    "output_tokens": 2,
                },
                {
                    "arrival_time": 12.0,
                    "first_token_time": 13.0,
                    "token_times": [13.0, 14.0],
                    "output_event_times": [13.0, 13.4],
                    "burst_itl": [1.0],
                    "output_event_latency": [0.4],
                    "speculative_step_latency": [0.4, 0.5],
                    "finish_time": 14.0,
                    "output_tokens": 2,
                    "speculative_steps": 2,
                    "speculative_draft_tokens": 6,
                    "speculative_accepted_tokens": 3,
                    "speculative_emitted_tokens": 4,
                    "speculative_accept_all_count": 1,
                    "speculative_timing": {
                        "draft_proposal_time": 0.1,
                        "target_verify_time": 0.2,
                        "total_time": 0.4,
                    },
                },
                {
                    "arrival_time": 18.0,
                    "first_token_time": 19.0,
                    "token_times": [19.0, 20.0],
                    "output_event_times": [19.0, 20.0],
                    "burst_itl": [1.0],
                    "output_event_latency": [1.0],
                    "speculative_step_latency": [],
                    "finish_time": 20.0,
                    "output_tokens": 2,
                },
            ],
        }

        summary = bench_serving.build_steady_state_summary(
            metrics,
            measurement_start=10.0,
            measurement_end=20.0,
            scheduled_batch_sizes=[4, 3],
            num_admitted=6,
        )

        self.assertEqual(summary["output_tokens"], 5)
        self.assertEqual(summary["throughput"], 0.5)
        self.assertEqual(summary["num_requests_admitted"], 6)
        self.assertEqual(summary["num_requests_completed"], 2)
        self.assertEqual(summary["num_requests_fully_measured"], 1)
        self.assertEqual(summary["mean_scheduled_batch_size"], 3.5)
        self.assertEqual(summary["max_scheduled_batch_size"], 4)
        self.assertEqual(summary["ttft"]["mean"], 1.0)
        self.assertEqual(summary["tpot"]["mean"], 1.0)
        self.assertEqual(summary["request_latency"]["mean"], 2.0)
        self.assertEqual(summary["output_event_latency"]["mean"], 0.4)
        self.assertEqual(summary["speculative_step_latency"]["mean"], 0.45)
        self.assertEqual(summary["speculative"]["steps"], 2)
        self.assertEqual(summary["speculative"]["acceptance_rate"], 0.5)
        self.assertEqual(summary["speculative"]["acceptance_length"], 2.0)
        self.assertEqual(
            summary["speculative"]["timing"]["draft_proposal_time"]["mean"],
            0.05,
        )

    def test_parse_args_accepts_speculative_flags(self):
        with patch.object(
            sys,
            "argv",
            [
                "bench_serving.py",
                "--speculative-model",
                "/models/eagle3",
                "--speculative-gamma",
                "4",
                "--speculative-tree-nodes",
                "6",
                "--speculative-accept-mode",
                "rejection",
                "--speculative-trace",
                "--argmax-sampler",
            ],
        ):
            args = bench_serving.parse_args()

        self.assertEqual(args.speculative_model, "/models/eagle3")
        self.assertEqual(args.speculative_gamma, 4)
        self.assertEqual(args.speculative_tree_nodes, 6)
        self.assertEqual(args.speculative_accept_mode, "rejection")
        self.assertTrue(args.speculative_trace)
        self.assertTrue(args.argmax_sampler)

    def test_parse_args_allows_explicitly_disabling_env_speculative_model(self):
        with patch.dict(os.environ, {"SPECULATIVE_MODEL": "/env/eagle"}), patch.object(
            sys,
            "argv",
            ["bench_serving.py", "--speculative-model", ""],
        ):
            args = bench_serving.parse_args()

        self.assertEqual(args.speculative_model, "")

    def test_parse_args_rejects_completed_tree_kv_ablation_flag(self):
        with patch.object(
            sys,
            "argv",
            ["bench_serving.py", "--speculative-tree-kv-mode", "layerwise"],
        ), redirect_stderr(StringIO()), self.assertRaises(SystemExit):
            bench_serving.parse_args()

    def test_parse_args_rejects_completed_draft_batching_ablation_flag(self):
        with patch.object(
            sys,
            "argv",
            ["bench_serving.py", "--disable-batched-draft"],
        ), redirect_stderr(StringIO()), self.assertRaises(SystemExit):
            bench_serving.parse_args()

    def test_summarize_speculative_requests_merges_gamma_counts(self):
        summary = bench_serving.summarize_speculative_requests([
            {
                "speculative_steps": 2,
                "speculative_draft_tokens": 4,
                "speculative_gamma_counts": {"1": 1, "3": 1},
            },
            {
                "speculative_steps": 2,
                "speculative_draft_tokens": 7,
                "speculative_gamma_counts": {"3": 1, "4": 1},
            },
        ])

        self.assertEqual(summary["gamma_counts"], {"1": 1, "3": 2, "4": 1})

    def test_build_workload_can_use_builtin_natural_prompts(self):
        args = SimpleNamespace(
            num_requests=3,
            input_len=4,
            output_len=1,
            arrival="all",
            request_rate=1.0,
            temperature=1.0,
            seed=0,
            prompt_mode="natural",
            prompt_file=None,
        )

        workload = bench_serving.build_workload(args)

        prompts = [item[1] for item in workload]
        self.assertEqual(len(prompts), 3)
        self.assertTrue(all(isinstance(prompt, str) for prompt in prompts))
        self.assertIn("Explain", prompts[0])

    def test_build_workload_can_use_prompt_file(self):
        with tempfile.NamedTemporaryFile("w", delete=False) as f:
            f.write("First natural prompt.\n\nSecond natural prompt.\n")
            prompt_file = f.name
        args = SimpleNamespace(
            num_requests=3,
            input_len=4,
            output_len=1,
            arrival="all",
            request_rate=1.0,
            temperature=1.0,
            seed=0,
            prompt_mode="random-token",
            prompt_file=prompt_file,
        )

        workload = bench_serving.build_workload(args)

        prompts = [item[1] for item in workload]
        self.assertEqual(prompts, ["First natural prompt.", "Second natural prompt.", "First natural prompt."])

    def test_build_workload_injects_one_long_prompt_after_background_requests(self):
        args = SimpleNamespace(
            num_requests=4,
            input_len=8,
            long_input_len=32,
            injection_delay=0.5,
            output_len=4,
            arrival="prefill-injection",
            request_rate=1.0,
            temperature=1.0,
            seed=0,
            prompt_mode="random-token",
            prompt_file=None,
        )

        workload = list(bench_serving.build_workload(args))

        self.assertEqual([item[0] for item in workload], [0.0, 0.0, 0.0, 0.5])
        self.assertEqual([len(item[1]) for item in workload], [8, 8, 8, 32])

    def test_run_benchmark_passes_speculative_config_to_engine(self):
        FakeLLM.calls.clear()
        FakeLLM.instances.clear()
        args = SimpleNamespace(
            model="/models/target",
            num_requests=1,
            input_len=4,
            output_len=1,
            arrival="all",
            request_rate=1.0,
            enforce_eager=True,
            enable_chunked_prefill=False,
            max_model_len=64,
            max_num_batched_tokens=64,
            temperature=1.0,
            seed=0,
            speculative_model="/models/eagle3",
            speculative_gamma=4,
            speculative_accept_mode="greedy",
            speculative_trace=True,
            argmax_sampler=True,
            prompt_mode="natural",
            prompt_file=None,
        )

        with patch.object(bench_serving, "LLM", FakeLLM):
            result = bench_serving.run_benchmark(args)

        self.assertEqual(FakeLLM.calls[0][0], "/models/target")
        self.assertEqual(FakeLLM.calls[0][1]["speculative_model"], "/models/eagle3")
        self.assertEqual(FakeLLM.calls[0][1]["speculative_gamma"], 4)
        self.assertEqual(FakeLLM.calls[0][1]["speculative_accept_mode"], "greedy")
        self.assertTrue(FakeLLM.calls[0][1]["speculative_trace"])
        self.assertIsInstance(FakeLLM.instances[0].model_runner.sampler, bench_serving.ArgmaxSampler)
        self.assertEqual(result["config"]["speculative_model"], "/models/eagle3")
        self.assertEqual(result["config"]["speculative_gamma"], 4)
        self.assertEqual(result["config"]["speculative_accept_mode"], "greedy")
        self.assertTrue(result["config"]["speculative_trace"])
        self.assertTrue(result["config"]["argmax_sampler"])
        self.assertEqual(result["config"]["prompt_mode"], "natural")

    def test_run_benchmark_closed_loop_refills_until_measurement_end(self):
        class FakeClock:
            def __init__(self):
                self.now = 0.0

            def perf_counter(self):
                return self.now

        class FakeClosedLoopLLM:
            instance = None

            def __init__(self, model, **kwargs):
                self.clock = clock
                self.model_runner = SimpleNamespace(sampler=None)
                self.active = []
                self.requests = []
                self.last_step_events = {}
                FakeClosedLoopLLM.instance = self

            def add_request(self, prompt, sampling_params):
                seq_id = len(self.requests)
                self.active.append(seq_id)
                self.requests.append({
                    "seq_id": seq_id,
                    "arrival_time": self.clock.now,
                    "first_token_time": None,
                    "token_times": [],
                    "finish_time": None,
                    "output_tokens": 0,
                    "itl": [],
                })
                return seq_id

            def is_finished(self):
                return not self.active

            def step(self):
                scheduled = list(self.active)
                self.clock.now += 0.5
                seq_id = self.active.pop(0)
                request = self.requests[seq_id]
                request["first_token_time"] = self.clock.now
                request["token_times"] = [self.clock.now]
                request["finish_time"] = self.clock.now
                request["output_tokens"] = 1
                self.last_step_events = {
                    "step_end": self.clock.now,
                    "scheduled_seq_ids": scheduled,
                }
                return [(seq_id, [1])], -1

            def get_metrics(self):
                return {
                    "summary": {
                        "num_requests": len(self.requests),
                        "num_finished": len(self.requests),
                        "num_failed": 0,
                        "total_output_tokens": len(self.requests),
                        "wall_time": self.clock.now,
                        "throughput": len(self.requests) / self.clock.now,
                        "ttft": {"mean": 0.5, "p50": 0.5, "p99": 0.5, "max": 0.5},
                        "itl": {"mean": None, "p50": None, "p99": None, "max": None},
                        "tpot": {"mean": 0.0, "p50": 0.0, "p99": 0.0, "max": 0.0},
                        "request_latency": {"mean": 0.5, "p50": 0.5, "p99": 0.5, "max": 0.5},
                        "speculative": {"steps": 0},
                    },
                    "requests": self.requests,
                }

            def exit(self):
                pass

        clock = FakeClock()
        args = SimpleNamespace(
            model="/models/target",
            num_requests=1,
            input_len=4,
            output_len=1,
            arrival="closed-loop",
            request_rate=1.0,
            max_concurrency=2,
            warmup_seconds=0.5,
            measurement_seconds=1.5,
            enforce_eager=True,
            enable_chunked_prefill=False,
            max_model_len=64,
            max_num_batched_tokens=64,
            temperature=1.0,
            seed=0,
            speculative_model=None,
            speculative_gamma=3,
            speculative_accept_mode="greedy",
            speculative_trace=False,
            argmax_sampler=False,
            prompt_mode="natural",
            prompt_file=None,
        )

        with patch.object(bench_serving, "LLM", FakeClosedLoopLLM), patch.object(
            bench_serving.time,
            "perf_counter",
            clock.perf_counter,
        ):
            result = bench_serving.run_benchmark(args)

        steady = result["metrics"]["steady_state"]
        self.assertEqual(steady["output_tokens"], 3)
        self.assertEqual(steady["throughput"], 2.0)
        self.assertEqual(steady["num_requests_admitted"], 5)
        self.assertEqual(steady["num_requests_completed"], 3)
        self.assertEqual(steady["num_requests_fully_measured"], 1)
        self.assertEqual(steady["mean_scheduled_batch_size"], 2.0)
        self.assertEqual(steady["max_scheduled_batch_size"], 2)
        self.assertEqual(len(FakeClosedLoopLLM.instance.requests), 5)
        self.assertTrue(FakeClosedLoopLLM.instance.is_finished())
        self.assertEqual(result["config"]["max_concurrency"], 2)
        self.assertEqual(result["config"]["warmup_seconds"], 0.5)
        self.assertEqual(result["config"]["measurement_seconds"], 1.5)

    def test_run_benchmark_validates_closed_loop_before_loading_engine(self):
        FakeLLM.calls.clear()
        FakeLLM.instances.clear()
        args = SimpleNamespace(
            model="/models/target",
            arrival="closed-loop",
            max_concurrency=8,
            warmup_seconds=1.0,
            measurement_seconds=0.0,
            speculative_model=None,
        )

        with patch.object(bench_serving, "LLM", FakeLLM):
            with self.assertRaisesRegex(ValueError, "--measurement-seconds must be positive"):
                bench_serving.run_benchmark(args)

        self.assertEqual(FakeLLM.calls, [])

    def test_print_summary_includes_speculative_metrics_when_present(self):
        result = {
            "metrics": FakeLLM("/unused").get_metrics(),
        }
        output = StringIO()

        with redirect_stdout(output):
            bench_serving.print_summary(result)

        text = output.getvalue()
        self.assertIn("Speculative Metrics", text)
        self.assertIn("steps:          1", text)
        self.assertIn("draft tokens:   3", text)
        self.assertIn("accepted:       2", text)
        self.assertIn("acceptance:     66.67%", text)
        self.assertIn("accept length:  3.00", text)
        self.assertIn("accepted/step:  2.00", text)
        self.assertIn("draft/step:     3.00", text)
        self.assertIn("gamma counts:   3:1", text)
        self.assertIn("Speculative Timing (ms)", text)
        self.assertIn("draft proposal", text)
        self.assertIn("draft pack", text)
        self.assertIn("draft forward", text)
        self.assertIn("draft sample", text)
        self.assertIn("draft compact", text)
        self.assertIn("target verify", text)
        self.assertIn("total", text)


if __name__ == "__main__":
    unittest.main()
