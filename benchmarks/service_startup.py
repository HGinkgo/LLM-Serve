"""Isolated direct-versus-service startup diagnostic for one model request.

The overload benchmark deliberately creates a large amount of concurrent
state.  This diagnostic instead makes one short request through either the
direct ``LLM`` interface or ``EngineServiceRuntime`` and records both the
terminal state and CUDA memory.  It is intended to localize failures before
interpreting a load-test result.
"""

from __future__ import annotations

import argparse
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from time import monotonic, sleep
from typing import Any

from benchmarks.environment import (
    atomic_write_json,
    build_environment_metadata,
    discover_model_revision,
)
from llmserve.service.runtime import EngineServiceRuntime


_TERMINAL_KINDS = frozenset({"completed", "cancelled", "failed", "timed_out"})


def _error_snapshot(error: BaseException | None) -> dict[str, str] | None:
    if error is None:
        return None
    return {"type": type(error).__name__, "message": str(error)}


def cuda_memory_snapshot() -> dict[str, Any]:
    """Capture the visible CUDA process view without assuming CUDA exists."""
    try:
        import torch
    except ImportError:
        return {"cuda_available": False}
    if not torch.cuda.is_available():
        return {"cuda_available": False}
    device = torch.cuda.current_device()
    free_bytes, total_bytes = torch.cuda.mem_get_info(device)
    return {
        "cuda_available": True,
        "device_index": device,
        "device_name": torch.cuda.get_device_name(device),
        "free_bytes": free_bytes,
        "total_bytes": total_bytes,
        "allocated_bytes": torch.cuda.memory_allocated(device),
        "reserved_bytes": torch.cuda.memory_reserved(device),
    }


def _output_token_count(outputs: Any) -> int:
    if not outputs:
        return 0
    first = outputs[0]
    if isinstance(first, Mapping):
        return len(first.get("token_ids") or ())
    return len(getattr(first, "token_ids", ()) or ())


def run_startup_probe(
    mode: str,
    *,
    engine_factory: Callable[[], Any],
    tokenizer: Any | None,
    prompt_token_ids: Sequence[int],
    sampling_params: Any,
    timeout_seconds: float,
    model: str,
    config: Mapping[str, Any],
    metadata: Mapping[str, Any],
    memory_snapshot: Callable[[], Mapping[str, Any]] = cuda_memory_snapshot,
) -> dict[str, Any]:
    """Run one request and return a serializable result even when it fails."""
    if mode not in {"direct", "service"}:
        raise ValueError("mode must be 'direct' or 'service'")
    if timeout_seconds <= 0:
        raise ValueError("timeout_seconds must be positive")

    result: dict[str, Any] = {
        "schema_version": 1,
        "mode": mode,
        "model": model,
        "git_commit": metadata.get("git_commit"),
        "git_dirty": metadata.get("git_dirty"),
        "environment": dict(metadata),
        "config": dict(config),
        "startup_ok": False,
        "request_ok": False,
        "terminal_event": None,
        "output_tokens": 0,
        "fatal_error": None,
        "memory_snapshot": None,
    }
    engine = None
    runtime = None
    try:
        if mode == "direct":
            engine = engine_factory()
            result["startup_ok"] = True
            outputs = engine.generate(
                [list(prompt_token_ids)],
                sampling_params,
                use_tqdm=False,
            )
            result["output_tokens"] = _output_token_count(outputs)
            result["request_ok"] = True
            result["terminal_event"] = "completed"
        else:
            runtime = EngineServiceRuntime(
                engine_factory,
                tokenizer=tokenizer,
                submit_timeout_seconds=timeout_seconds,
            )
            runtime.start(timeout=timeout_seconds)
            result["startup_ok"] = True
            request = runtime.submit(list(prompt_token_ids), sampling_params)
            deadline = monotonic() + timeout_seconds
            while monotonic() < deadline:
                event = request.poll_event()
                if event is None:
                    sleep(0.001)
                    continue
                if event.kind == "token":
                    result["output_tokens"] += len(event.token_ids)
                    continue
                if event.kind in _TERMINAL_KINDS:
                    result["terminal_event"] = event.kind
                    result["request_ok"] = event.kind == "completed"
                    if event.kind == "failed":
                        result["fatal_error"] = _error_snapshot(runtime.fatal_error)
                    break
            else:
                raise TimeoutError("timed out waiting for service request completion")
            if result["terminal_event"] is None:
                raise TimeoutError("service request ended without a terminal event")
    except BaseException as error:
        result["fatal_error"] = _error_snapshot(error)
    finally:
        if runtime is not None:
            try:
                runtime.close(timeout=timeout_seconds)
            except BaseException as error:
                if result["fatal_error"] is None:
                    result["fatal_error"] = _error_snapshot(error)
        elif engine is not None:
            try:
                engine.exit()
            except BaseException as error:
                if result["fatal_error"] is None:
                    result["fatal_error"] = _error_snapshot(error)
        result["memory_snapshot"] = dict(memory_snapshot())
    return result


def _probe_prompt(tokenizer: Any, input_len: int) -> list[int]:
    base = tokenizer.encode("Service startup probe. ", add_special_tokens=False)
    if not base:
        raise RuntimeError("tokenizer returned an empty probe prompt")
    repetitions = (input_len + len(base) - 1) // len(base)
    return (base * repetitions)[:input_len]


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("direct", "service"), required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--input-len", type=int, default=128)
    parser.add_argument("--output-len", type=int, default=1)
    parser.add_argument("--timeout-seconds", type=float, default=120.0)
    parser.add_argument("--max-model-len", type=int, default=512)
    parser.add_argument("--max-num-batched-tokens", type=int, default=1024)
    parser.add_argument("--max-num-seqs", type=int, default=64)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.9)
    parser.add_argument("--enforce-eager", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    if args.input_len <= 0 or args.output_len <= 0:
        raise ValueError("input-len and output-len must be positive")
    if args.timeout_seconds <= 0:
        raise ValueError("timeout-seconds must be positive")

    from transformers import AutoTokenizer

    from llmserve.sampling_params import SamplingParams
    from llmserve.service.factory import ServiceLaunchConfig, build_engine_factory

    config = ServiceLaunchConfig(
        model=args.model,
        mode="collocated",
        max_model_len=args.max_model_len,
        max_num_batched_tokens=args.max_num_batched_tokens,
        max_num_seqs=args.max_num_seqs,
        gpu_memory_utilization=args.gpu_memory_utilization,
        enable_chunked_prefill=True,
        enable_kv_capacity_admission=True,
        enforce_eager=args.enforce_eager,
    )
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    metadata = build_environment_metadata()
    result = run_startup_probe(
        args.mode,
        engine_factory=build_engine_factory(config),
        tokenizer=tokenizer,
        prompt_token_ids=_probe_prompt(tokenizer, args.input_len),
        sampling_params=SamplingParams(
            temperature=0.01,
            max_tokens=args.output_len,
            ignore_eos=True,
        ),
        timeout_seconds=args.timeout_seconds,
        model=args.model,
        config={
            "max_model_len": args.max_model_len,
            "max_num_batched_tokens": args.max_num_batched_tokens,
            "max_num_seqs": args.max_num_seqs,
            "gpu_memory_utilization": args.gpu_memory_utilization,
            "enforce_eager": args.enforce_eager,
        },
        metadata={
            **metadata,
            "model_revision": discover_model_revision(args.model),
        },
    )
    atomic_write_json(Path(args.output), result)
    return 0 if result["request_ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
