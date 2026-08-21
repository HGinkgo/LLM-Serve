"""Provenance shared by the MoE GPTQ benchmark reports."""

from __future__ import annotations

import re
from pathlib import Path


_VLLM_VERSION_PATTERN = re.compile(r"(?:vllm[-_]?)(\d+\.\d+(?:\.\d+)?)", re.IGNORECASE)


def infer_vllm_version(source: str | Path | None) -> str | None:
    """Infer a version from a reference directory name when available."""
    if source is None:
        return None
    match = _VLLM_VERSION_PATTERN.search(str(source))
    return None if match is None else match.group(1)


def build_vllm_marlin_baseline(
    *,
    source: str | Path | None = None,
    runtime_version: str | None = None,
    role: str = "external_reference",
) -> dict:
    """Return stable provenance for the vLLM GPTQ-Marlin reference.

    LLM-Serve throughput rows remain candidate measurements; this metadata
    records the reference contract without claiming they were measured by
    vLLM's Python runtime.
    """
    version = runtime_version or infer_vllm_version(source)
    return {
        "runtime": "vllm",
        "backend": "gptq_marlin",
        "runtime_version": version,
        "source": None if source is None else str(source),
        "role": role,
        "token_contract": "greedy_token_parity",
    }


def build_moe_experiment_metadata(
    *,
    optimization: str,
    enabled: bool,
    baseline: dict,
) -> dict:
    """Describe one optimization variant and its fixed comparison baseline."""
    return {
        "baseline": dict(baseline),
        "optimization": {
            "name": optimization,
            "enabled": bool(enabled),
        },
    }
