import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch

from llmserve.config import Config


class ConfigSpeculativeTest(unittest.TestCase):

    def make_config(self, **kwargs):
        with tempfile.TemporaryDirectory() as model_dir, patch(
            "llmserve.config.AutoConfig.from_pretrained",
            return_value=SimpleNamespace(max_position_embeddings=4096),
        ):
            return Config(model_dir, **kwargs)

    def test_fixed_gamma_must_be_positive(self):
        with self.assertRaises(AssertionError):
            self.make_config(speculative_gamma=0)

    def test_completed_tree_kv_ablation_is_not_a_runtime_config(self):
        config = self.make_config()

        self.assertFalse(hasattr(config, "speculative_tree_kv_mode"))
        self.assertFalse(hasattr(config, "speculative_batched_draft"))

    def test_distributed_init_method_is_configurable(self):
        config = self.make_config(
            distributed_init_method="tcp://localhost:2444"
        )

        self.assertEqual(
            config.distributed_init_method, "tcp://localhost:2444"
        )

    def test_awq_checkpoint_enables_native_bfloat16_runtime(self):
        hf_config = SimpleNamespace(
            max_position_embeddings=4096,
            dtype=torch.bfloat16,
            quantization_config={
                "quant_method": "awq",
                "backend": "autoawq",
                "bits": 4,
                "group_size": 128,
                "version": "gemm",
                "zero_point": True,
            },
        )
        with tempfile.TemporaryDirectory() as model_dir, patch(
            "llmserve.config.AutoConfig.from_pretrained",
            return_value=hf_config,
        ):
            config = Config(model_dir, enforce_eager=True)

        self.assertIsNotNone(config.quant_config)
        self.assertEqual(config.quant_config.activation_dtype, torch.bfloat16)
        self.assertEqual(config.quant_config.execution_backend, "cuda")

    def test_awq_runtime_can_select_reference_backend(self):
        hf_config = SimpleNamespace(
            max_position_embeddings=4096,
            dtype=torch.bfloat16,
            quantization_config={
                "quant_method": "awq",
                "backend": "autoawq",
                "bits": 4,
                "group_size": 128,
                "version": "gemm",
                "zero_point": True,
            },
        )
        with tempfile.TemporaryDirectory() as model_dir, patch(
            "llmserve.config.AutoConfig.from_pretrained",
            return_value=hf_config,
        ):
            config = Config(
                model_dir,
                enforce_eager=True,
                awq_backend="reference",
            )

        self.assertEqual(config.quant_config.execution_backend, "reference")

    def test_awq_runtime_can_select_cuda_backend(self):
        hf_config = SimpleNamespace(
            max_position_embeddings=4096,
            dtype=torch.bfloat16,
            quantization_config={
                "quant_method": "awq",
                "backend": "autoawq",
                "bits": 4,
                "group_size": 128,
                "version": "gemm",
                "zero_point": True,
            },
        )
        with tempfile.TemporaryDirectory() as model_dir, patch(
            "llmserve.config.AutoConfig.from_pretrained",
            return_value=hf_config,
        ):
            config = Config(
                model_dir,
                enforce_eager=True,
                awq_backend="cuda",
            )

        self.assertEqual(config.quant_config.execution_backend, "cuda")

    def test_awq_reference_backend_requires_eager_execution(self):
        hf_config = SimpleNamespace(
            max_position_embeddings=4096,
            dtype=torch.bfloat16,
            quantization_config={
                "quant_method": "awq",
                "backend": "autoawq",
                "bits": 4,
                "group_size": 128,
                "version": "gemm",
                "zero_point": True,
            },
        )
        with tempfile.TemporaryDirectory() as model_dir, patch(
            "llmserve.config.AutoConfig.from_pretrained",
            return_value=hf_config,
        ), self.assertRaisesRegex(ValueError, "enforce_eager"):
            Config(model_dir)

    def test_awq_checkpoint_rejects_tensor_parallel_runtime(self):
        hf_config = SimpleNamespace(
            max_position_embeddings=4096,
            dtype=torch.bfloat16,
            quantization_config={
                "quant_method": "awq",
                "backend": "autoawq",
                "bits": 4,
                "group_size": 128,
                "version": "gemm",
                "zero_point": True,
            },
        )
        with tempfile.TemporaryDirectory() as model_dir, patch(
            "llmserve.config.AutoConfig.from_pretrained",
            return_value=hf_config,
        ), self.assertRaisesRegex(ValueError, "tensor parallel"):
            Config(model_dir, tensor_parallel_size=2)


if __name__ == "__main__":
    unittest.main()
