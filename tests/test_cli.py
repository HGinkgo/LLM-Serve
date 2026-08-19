import io
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

from llmserve import cli


class _FakeCuda:
    def __init__(self, available: bool):
        self._available = available

    def is_available(self):
        return self._available

    def device_count(self):
        return 1 if self._available else 0

    def get_device_name(self, index):
        return "Test GPU"


class _FakeTorch:
    __version__ = "2.7.1+cu128"

    def __init__(self, cuda_available: bool):
        self.cuda = _FakeCuda(cuda_available)
        self.version = type("Version", (), {"cuda": "12.8"})()


class CLITest(unittest.TestCase):
    def test_check_reports_all_items_and_fails_without_cuda(self):
        stdout = io.StringIO()
        with (
            patch("llmserve.cli._import_torch", return_value=_FakeTorch(False)),
            patch("llmserve.cli.platform.system", return_value="Linux"),
            patch("llmserve.cli.metadata.version", return_value="4.57.6"),
        ):
            status = cli.main(["check"], stdout=stdout)

        output = stdout.getvalue()
        self.assertEqual(status, 1)
        self.assertIn("[ok] Python", output)
        self.assertIn("[ok] PyTorch: 2.7.1+cu128", output)
        self.assertIn("[failed] CUDA", output)
        self.assertIn("[ok] Transformers: 4.57.6", output)
        self.assertIn("[ok] Triton: 4.57.6", output)
        self.assertIn("[ok] FlashAttention: 4.57.6", output)

    def test_check_rejects_unsupported_transformers_major_version(self):
        stdout = io.StringIO()

        def package_version(package):
            if package == "transformers":
                return "5.14.1"
            return "2.8.3"

        with (
            patch("llmserve.cli._import_torch", return_value=_FakeTorch(True)),
            patch("llmserve.cli.platform.system", return_value="Linux"),
            patch("llmserve.cli.metadata.version", side_effect=package_version),
        ):
            status = cli.main(["check"], stdout=stdout)

        self.assertEqual(status, 1)
        self.assertIn(
            "[failed] Transformers: 5.14.1 (requires >=4.51,<5)",
            stdout.getvalue(),
        )

    def test_generate_maps_arguments_and_prints_completion(self):
        calls = {}

        class FakeTokenizer:
            @classmethod
            def from_pretrained(cls, model):
                calls["tokenizer_model"] = model
                return cls()

            def apply_chat_template(self, messages, **kwargs):
                calls["messages"] = messages
                calls["chat_kwargs"] = kwargs
                return "formatted prompt"

        class FakeSamplingParams:
            def __init__(self, **kwargs):
                calls["sampling"] = kwargs

        class FakeLLM:
            def __init__(self, model, **kwargs):
                calls["llm"] = (model, kwargs)

            def generate(self, prompts, sampling_params, **kwargs):
                calls["generate"] = (prompts, sampling_params, kwargs)
                return [{"text": "generated text", "token_ids": [1, 2]}]

            def exit(self):
                calls["exited"] = True

        stdout = io.StringIO()
        with tempfile.TemporaryDirectory() as model_dir:
            with (
                patch("llmserve.cli._cuda_is_available", return_value=True),
                patch(
                    "llmserve.cli._load_runtime",
                    return_value=(FakeLLM, FakeSamplingParams, FakeTokenizer),
                ),
            ):
                status = cli.main(
                    [
                        "generate",
                        "--model",
                        model_dir,
                        "--prompt",
                        "hello",
                        "--max-tokens",
                        "32",
                        "--temperature",
                        "0.7",
                        "--enforce-eager",
                    ],
                    stdout=stdout,
                )

            self.assertEqual(calls["tokenizer_model"], model_dir)
            self.assertEqual(calls["llm"], (model_dir, {"enforce_eager": True}))

        self.assertEqual(status, 0)
        self.assertEqual(calls["messages"], [{"role": "user", "content": "hello"}])
        self.assertEqual(
            calls["chat_kwargs"],
            {"tokenize": False, "add_generation_prompt": True},
        )
        self.assertEqual(calls["sampling"], {"temperature": 0.7, "max_tokens": 32})
        self.assertEqual(calls["generate"][0], ["formatted prompt"])
        self.assertEqual(calls["generate"][2], {"use_tqdm": False})
        self.assertTrue(calls["exited"])
        self.assertEqual(stdout.getvalue(), "generated text\n")

    def test_generate_rejects_missing_model_before_runtime_import(self):
        stderr = io.StringIO()
        missing_model = str(Path(tempfile.gettempdir()) / "missing-llmserve-model")
        with patch("llmserve.cli._load_runtime") as load_runtime:
            status = cli.main(
                ["generate", "--model", missing_model, "--prompt", "hello"],
                stderr=stderr,
            )

        self.assertEqual(status, 2)
        self.assertIn("model directory does not exist", stderr.getvalue())
        load_runtime.assert_not_called()

    def test_generate_releases_runtime_and_reports_runtime_error(self):
        calls = {}

        class FakeTokenizer:
            @classmethod
            def from_pretrained(cls, model):
                return cls()

            def apply_chat_template(self, messages, **kwargs):
                return "formatted prompt"

        class FakeSamplingParams:
            def __init__(self, **kwargs):
                pass

        class FakeLLM:
            def __init__(self, model, **kwargs):
                pass

            def generate(self, prompts, sampling_params, **kwargs):
                raise RuntimeError("generation failed")

            def exit(self):
                calls["exited"] = True

        stderr = io.StringIO()
        with tempfile.TemporaryDirectory() as model_dir:
            with (
                patch("llmserve.cli._cuda_is_available", return_value=True),
                patch(
                    "llmserve.cli._load_runtime",
                    return_value=(FakeLLM, FakeSamplingParams, FakeTokenizer),
                ),
            ):
                status = cli.main(
                    ["generate", "--model", model_dir, "--prompt", "hello"],
                    stderr=stderr,
                )

        self.assertEqual(status, 1)
        self.assertTrue(calls["exited"])
        self.assertIn("RuntimeError: generation failed", stderr.getvalue())

    def test_serve_builds_pd_shared_runtime_and_closes_it_after_server_exit(self):
        runtime = Mock()
        app = object()
        with tempfile.TemporaryDirectory() as model_dir:
            with (
                patch("llmserve.cli._cuda_is_available", return_value=True),
                patch("llmserve.cli._build_service_runtime", return_value=runtime) as build_runtime,
                patch("llmserve.cli._create_service_app", return_value=app) as create_app,
                patch("llmserve.cli._run_uvicorn") as run_uvicorn,
            ):
                status = cli.main([
                    "serve",
                    "--model", model_dir,
                    "--served-model-name", "Qwen3-8B",
                    "--mode", "pd-shared",
                    "--host", "0.0.0.0",
                    "--port", "18000",
                    "--max-model-len", "2048",
                    "--max-num-batched-tokens", "1024",
                    "--max-num-seqs", "64",
                    "--pd-prefill-gpu", "2",
                    "--pd-decode-gpus", "3",
                    "--pd-prefill-batch-size", "4",
                    "--pd-kv-slot-capacity-tokens", "8192",
                ])

        self.assertEqual(status, 0)
        launch_config = build_runtime.call_args.args[0]
        self.assertEqual(launch_config.mode, "pd-shared")
        self.assertEqual(launch_config.model, model_dir)
        self.assertEqual(launch_config.max_model_len, 2048)
        self.assertEqual(launch_config.max_num_batched_tokens, 1024)
        self.assertEqual(launch_config.max_num_seqs, 64)
        self.assertEqual(launch_config.prefill_gpu, 2)
        self.assertEqual(launch_config.decode_gpus, (3,))
        self.assertEqual(launch_config.prefill_batch_size, 4)
        self.assertEqual(launch_config.kv_slot_capacity_tokens, 8192)
        runtime.start.assert_called_once_with()
        create_app.assert_called_once_with(runtime, model_name="Qwen3-8B")
        run_uvicorn.assert_called_once_with(app, host="0.0.0.0", port=18000)
        runtime.close.assert_called_once_with()


if __name__ == "__main__":
    unittest.main()
