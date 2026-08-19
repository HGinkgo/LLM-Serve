import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import patch


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

    def test_speculative_cuda_graph_is_disabled_by_default(self):
        config = self.make_config()

        self.assertFalse(config.enable_speculative_cuda_graph)

    def test_tree_speculation_is_not_a_runtime_option(self):
        with self.assertRaises(TypeError):
            self.make_config(speculative_tree_nodes=6)

    def test_speculative_cuda_graph_requires_greedy_acceptance(self):
        with tempfile.TemporaryDirectory() as model_dir, tempfile.TemporaryDirectory() as draft_dir, patch(
            "llmserve.config.AutoConfig.from_pretrained",
            return_value=SimpleNamespace(max_position_embeddings=4096),
        ), self.assertRaisesRegex(ValueError, "greedy acceptance"):
            Config(
                model_dir,
                speculative_model=draft_dir,
                speculative_accept_mode="rejection",
                enable_speculative_cuda_graph=True,
            )

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

    def test_quantized_checkpoint_is_rejected(self):
        hf_config = SimpleNamespace(
            max_position_embeddings=4096,
            quantization_config={
                "format": "int4",
            },
        )
        with tempfile.TemporaryDirectory() as model_dir, patch(
            "llmserve.config.AutoConfig.from_pretrained",
            return_value=hf_config,
        ), self.assertRaisesRegex(ValueError, "quantized checkpoints are not supported"):
            Config(model_dir, enforce_eager=True)


if __name__ == "__main__":
    unittest.main()
