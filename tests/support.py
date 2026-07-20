import os
import unittest


TARGET_MODEL_PATH = os.environ.get("LLMSERVE_TEST_TARGET_MODEL")
SPECULATIVE_MODEL_PATH = os.environ.get("LLMSERVE_TEST_SPECULATIVE_MODEL")

_models_available = bool(
    TARGET_MODEL_PATH
    and SPECULATIVE_MODEL_PATH
    and os.path.isdir(TARGET_MODEL_PATH)
    and os.path.isdir(SPECULATIVE_MODEL_PATH)
)

requires_eagle3_models = unittest.skipUnless(
    _models_available,
    "set LLMSERVE_TEST_TARGET_MODEL and LLMSERVE_TEST_SPECULATIVE_MODEL",
)
