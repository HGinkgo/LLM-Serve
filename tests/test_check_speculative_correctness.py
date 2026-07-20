import unittest
from types import SimpleNamespace
from unittest.mock import patch

import check_speculative_correctness as check


def make_metrics(output_tokens, speculative_summary=None):
    summary = {
        "num_requests": 1,
        "num_finished": 1,
        "num_failed": 0,
        "total_output_tokens": output_tokens,
        "wall_time": 1.0,
        "throughput": output_tokens,
        "ttft": {"mean": 0.1, "p50": 0.1, "p99": 0.1, "max": 0.1},
        "itl": {"mean": 0.1, "p50": 0.1, "p99": 0.1, "max": 0.1},
        "tpot": {"mean": 0.1, "p50": 0.1, "p99": 0.1, "max": 0.1},
        "request_latency": {"mean": 0.2, "p50": 0.2, "p99": 0.2, "max": 0.2},
        "speculative": speculative_summary or {
            "steps": 0,
            "draft_tokens": 0,
            "accepted_tokens": 0,
            "emitted_tokens": 0,
            "acceptance_rate": None,
            "accept_all_count": 0,
        },
    }
    return {
        "summary": summary,
        "requests": [
            {
                "seq_id": 1,
                "output_tokens": output_tokens,
                "success": True,
                "failure_reason": None,
            }
        ],
    }


class FakeLLM:
    calls = []
    outputs = []
    metrics = []

    def __init__(self, model, **kwargs):
        self.model = model
        self.kwargs = kwargs
        self.pending = True
        self.output = FakeLLM.outputs.pop(0)
        self.metric = FakeLLM.metrics.pop(0)
        self.last_step_events = {"scheduled_seq_ids": [1]}
        self.model_runner = SimpleNamespace(sampler=None)
        FakeLLM.calls.append((model, kwargs))

    def add_request(self, prompt, sampling_params):
        self.prompt = prompt
        self.sampling_params = sampling_params
        return 1

    def is_finished(self):
        return not self.pending

    def step(self):
        self.pending = False
        return [(1, self.output)], -len(self.output)

    def get_metrics(self):
        return self.metric

    def exit(self):
        pass


class CheckSpeculativeCorrectnessTest(unittest.TestCase):

    def test_parse_token_ids(self):
        self.assertEqual(check.parse_token_ids("1, 2,3"), [1, 2, 3])
        with self.assertRaises(ValueError):
            check.parse_token_ids("")

    def test_parse_batch_sizes(self):
        self.assertEqual(check.parse_batch_sizes("4,2,4"), [2, 4])
        with self.assertRaises(ValueError):
            check.parse_batch_sizes("0,2")

    def test_build_checks_records_token_match_as_observation(self):
        baseline = check.RunResult(
            token_ids=[10, 11],
            metrics=make_metrics(2),
            step_events=[],
        )
        speculative = check.RunResult(
            token_ids=[10, 12],
            metrics=make_metrics(
                2,
                {
                    "steps": 1,
                    "draft_tokens": 3,
                    "accepted_tokens": 1,
                    "emitted_tokens": 1,
                    "acceptance_rate": 1 / 3,
                    "accept_all_count": 0,
                },
            ),
            step_events=[],
        )

        checks = check.build_checks(baseline, speculative, max_tokens=2)

        self.assertTrue(checks["baseline_finished"])
        self.assertTrue(checks["speculative_finished"])
        self.assertTrue(checks["speculative_metrics_present"])
        self.assertTrue(checks["speculative_metrics_consistent"])
        self.assertFalse(checks["token_ids_match"])

    def test_build_batch_checks_requires_exact_outputs_and_real_batch(self):
        batched = check.BatchRunResult(
            token_ids=[[10, 11], [20, 21]],
            metrics={
                "summary": {
                    "num_finished": 2,
                    "num_failed": 0,
                    "speculative": {"steps": 2, "max_batch_size": 2},
                }
            },
            step_events=[],
        )

        checks = check.build_batch_checks(
            [[10, 11], [20, 21]],
            batched,
            expected_batch_size=2,
            output_len=2,
        )

        self.assertTrue(checks["token_ids_match"])
        self.assertTrue(checks["multi_request_batch_used"])
        self.assertTrue(checks["required_checks_pass"])

    def test_run_comparison_passes_baseline_and_speculative_configs(self):
        FakeLLM.calls.clear()
        FakeLLM.outputs = [[10, 11], [10, 12]]
        FakeLLM.metrics = [
            make_metrics(2),
            make_metrics(
                2,
                {
                    "steps": 1,
                    "draft_tokens": 3,
                    "accepted_tokens": 1,
                    "emitted_tokens": 1,
                    "acceptance_rate": 1 / 3,
                    "accept_all_count": 0,
                },
            ),
        ]
        args = SimpleNamespace(
            model="/models/target",
            speculative_model="/models/eagle3",
            prompt="Explain speculative decoding briefly.",
            prompt_token_ids=[1, 2, 3],
            output_len=2,
            temperature=1.0,
            seed=0,
            speculative_gamma=3,
            speculative_accept_mode="greedy",
            speculative_trace=True,
            argmax_sampler=True,
            max_model_len=64,
            max_num_batched_tokens=64,
            max_steps=8,
            require_token_match=False,
        )

        with patch.object(check, "LLM", FakeLLM), patch.object(check, "cleanup_cuda_memory") as cleanup:
            report = check.run_comparison(args)

        self.assertIsNone(FakeLLM.calls[0][1]["speculative_model"])
        self.assertEqual(FakeLLM.calls[1][1]["speculative_model"], "/models/eagle3")
        self.assertEqual(FakeLLM.calls[1][1]["speculative_accept_mode"], "greedy")
        self.assertTrue(FakeLLM.calls[1][1]["speculative_trace"])
        self.assertEqual(FakeLLM.calls[0][0], "/models/target")
        self.assertEqual(report["baseline"]["token_ids"], [10, 11])
        self.assertEqual(report["speculative"]["token_ids"], [10, 12])
        self.assertEqual(report["config"]["prompt"], "Explain speculative decoding briefly.")
        self.assertEqual(report["config"]["speculative_accept_mode"], "greedy")
        self.assertTrue(report["config"]["speculative_trace"])
        self.assertTrue(report["config"]["argmax_sampler"])
        self.assertIsNone(report["config"]["prompt_token_ids"])
        self.assertFalse(report["checks"]["token_ids_match"])
        self.assertEqual(cleanup.call_count, 2)


if __name__ == "__main__":
    unittest.main()
