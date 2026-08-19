import unittest

from llmserve.sampling_params import SamplingParams
from tests.test_service_runtime import FailingStepEngine, FakeEngine, FakeTokenizer


class _DirectEngine:
    def __init__(self):
        self.exited = False

    def generate(self, prompts, sampling_params, *, use_tqdm):
        self.prompts = prompts
        self.sampling_params = sampling_params
        self.use_tqdm = use_tqdm
        return [{"text": "a", "token_ids": [97]}]

    def exit(self):
        self.exited = True


class ServiceStartupTests(unittest.TestCase):
    def setUp(self):
        self.metadata = {
            "git_commit": "commit",
            "git_dirty": True,
            "cuda_available": False,
        }
        self.config = {
            "max_model_len": 512,
            "max_num_batched_tokens": 1024,
            "max_num_seqs": 64,
            "enforce_eager": False,
        }
        self.memory_snapshot = lambda: {"free_bytes": 1, "total_bytes": 2}
        self.sampling_params = SamplingParams(
            temperature=0.01,
            max_tokens=1,
            ignore_eos=True,
        )

    def test_direct_probe_records_success_schema(self):
        from benchmarks.service_startup import run_startup_probe

        engine = _DirectEngine()
        result = run_startup_probe(
            "direct",
            engine_factory=lambda: engine,
            tokenizer=None,
            prompt_token_ids=(1, 2),
            sampling_params=self.sampling_params,
            timeout_seconds=1.0,
            model="Qwen3-8B",
            config=self.config,
            metadata=self.metadata,
            memory_snapshot=self.memory_snapshot,
        )

        self.assertEqual(result["schema_version"], 1)
        self.assertEqual(result["mode"], "direct")
        self.assertEqual(result["model"], "Qwen3-8B")
        self.assertEqual(result["git_commit"], "commit")
        self.assertTrue(result["git_dirty"])
        self.assertTrue(result["startup_ok"])
        self.assertTrue(result["request_ok"])
        self.assertIsNone(result["fatal_error"])
        self.assertEqual(result["output_tokens"], 1)
        self.assertEqual(result["memory_snapshot"], {
            "free_bytes": 1,
            "total_bytes": 2,
        })
        self.assertTrue(engine.exited)

    def test_service_probe_records_driver_failure_without_raising(self):
        from benchmarks.service_startup import run_startup_probe

        result = run_startup_probe(
            "service",
            engine_factory=FailingStepEngine,
            tokenizer=FakeTokenizer(),
            prompt_token_ids=(1, 2),
            sampling_params=self.sampling_params,
            timeout_seconds=1.0,
            model="Qwen3-8B",
            config=self.config,
            metadata=self.metadata,
            memory_snapshot=self.memory_snapshot,
        )

        self.assertTrue(result["startup_ok"])
        self.assertFalse(result["request_ok"])
        self.assertEqual(result["fatal_error"], {
            "type": "RuntimeError",
            "message": "synthetic engine failure",
        })
        self.assertEqual(result["terminal_event"], "failed")


if __name__ == "__main__":
    unittest.main()
