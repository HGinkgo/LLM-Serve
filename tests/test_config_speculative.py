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

    def test_completed_tree_kv_ablation_is_not_a_runtime_config(self):
        config = self.make_config()

        self.assertFalse(hasattr(config, "speculative_tree_kv_mode"))
        self.assertFalse(hasattr(config, "speculative_batched_draft"))


if __name__ == "__main__":
    unittest.main()
