import subprocess
import sys
import unittest


class OptionalGpuDependenciesTest(unittest.TestCase):

    def test_engine_import_does_not_require_gpu_only_dependencies(self):
        code = """
import sys
import torch
import transformers

sys.modules['flash_attn'] = None
sys.modules['triton'] = None
sys.modules['triton.language'] = None

from llmserve.engine.llm_engine import LLMEngine

assert LLMEngine is not None
"""
        result = subprocess.run(
            [sys.executable, "-c", code],
            capture_output=True,
            text=True,
        )

        self.assertEqual(result.returncode, 0, result.stderr)


if __name__ == "__main__":
    unittest.main()
