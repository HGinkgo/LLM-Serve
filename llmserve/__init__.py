from llmserve.sampling_params import SamplingParams

__all__ = ["LLM", "SamplingParams"]


def __getattr__(name):
    if name == "LLM":
        from llmserve.llm import LLM

        return LLM
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
