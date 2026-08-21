"""Short torch.profiler runs for the Qwen3-MoE GPTQ runtime."""

from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
from hashlib import sha256
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile

import torch
from torch.profiler import ProfilerActivity, profile, record_function

from benchmarks.environment import atomic_write_json, build_environment_metadata
from benchmarks.moe_baseline import (
    build_moe_experiment_metadata,
    build_vllm_marlin_baseline,
)
from benchmarks.moe_backend import SUPPORTED_BACKENDS, parse_backends
from benchmarks.moe_reference import load_prompt_cases


NATIVE_EVENT_PARTS = (
    "_weight_int4pack_mm",
    "gptq_marlin_gemm",
    "index_select",
    "index_add",
    "sort",
    "unique",
    "topk",
    "softmax",
    "scaled_dot_product",
    "flash",
)


def build_profile_engine_kwargs(config: Mapping, *, backend: str) -> dict:
    if backend not in SUPPORTED_BACKENDS:
        raise ValueError(f"unknown GPTQ backend: {backend}")
    kwargs = {
        "max_model_len": int(config["max_model_len"]),
        "max_num_batched_tokens": int(config["max_num_batched_tokens"]),
        "max_num_seqs": int(config["max_num_seqs"]),
        "gpu_memory_utilization": float(config["gpu_memory_utilization"]),
        "enforce_eager": True,
        "random_seed": int(config.get("seed", 20260820)),
        "gptq_backend": backend,
        "enable_moe_gate_up_fusion": bool(config.get("enable_moe_gate_up_fusion", False)),
    }
    if backend == "marlin":
        library = config.get("marlin_library")
        if not library:
            raise ValueError("Marlin profiling requires marlin_library")
        kwargs["marlin_library"] = str(library)
    return kwargs


def _time_ms(event, attribute: str) -> float:
    value = getattr(event, attribute, 0.0)
    return round(float(value or 0.0) / 1000.0, 6)


def _keep_event(name: str) -> bool:
    return name.startswith("llmserve.") or any(
        part in name for part in NATIVE_EVENT_PARTS
    )


def _event_device_kind(event):
    """Return True for CUDA, False for CPU, or None for old/fake events."""
    device_type = getattr(event, "device_type", None)
    if device_type is None:
        return None
    value = str(device_type).upper()
    if value.endswith("CUDA") or value in {"1", "DEVICE_TYPE.CUDA"}:
        return True
    if value.endswith("CPU") or value in {"0", "DEVICE_TYPE.CPU"}:
        return False
    return None


def summarize_profile_events(events: Sequence) -> list[dict]:
    """Convert profiler averages into a compact, stable JSON table."""
    events_by_name = {}
    for event in events:
        name = str(getattr(event, "key", ""))
        if not _keep_event(name):
            continue
        events_by_name.setdefault(name, []).append(event)

    rows = []
    for name, named_events in events_by_name.items():
        row = {
            "name": name,
            "calls": 0,
            "self_cpu_ms": 0.0,
            "cpu_total_ms": 0.0,
            "self_cuda_ms": 0.0,
            "cuda_total_ms": 0.0,
        }
        # Some torch.profiler versions expose one average per device for a
        # record_function marker. Counts are therefore not additive. CPU
        # events carry an associated device duration on some versions, but
        # that duration is the only available CUDA value for many native ops.
        cuda_events = [
            event for event in named_events if _event_device_kind(event) is True
        ]
        for event in named_events:
            row["calls"] = max(row["calls"], int(getattr(event, "count", 1)))
            device_kind = _event_device_kind(event)
            if device_kind is not True:
                row["self_cpu_ms"] += _time_ms(event, "self_cpu_time_total")
                row["cpu_total_ms"] += _time_ms(event, "cpu_time_total")
            if cuda_events:
                if device_kind is True:
                    row["self_cuda_ms"] += _time_ms(event, "self_device_time_total")
                    row["cuda_total_ms"] += _time_ms(event, "device_time_total")
            else:
                row["self_cuda_ms"] += _time_ms(event, "self_device_time_total")
                row["cuda_total_ms"] += _time_ms(event, "device_time_total")
        rows.append(row)
    for row in rows:
        for key in ("self_cpu_ms", "cpu_total_ms", "self_cuda_ms", "cuda_total_ms"):
            row[key] = round(row[key], 6)
    rows.sort(
        key=lambda row: (
            row["cuda_total_ms"],
            row["cpu_total_ms"],
            row["name"],
        ),
        reverse=True,
    )
    return rows


class _ArgmaxSampler:
    def __call__(self, logits, temperatures):
        del temperatures
        return logits.argmax(dim=-1)


def _run_requests(engine, prompt_token_ids: Sequence[Sequence[int]], *, batch_size: int, max_new_tokens: int) -> int:
    from llmserve import SamplingParams

    sampling_params = SamplingParams(
        temperature=1.0,
        max_tokens=max_new_tokens,
        ignore_eos=True,
    )
    for index in range(batch_size):
        engine.add_request(
            list(prompt_token_ids[index % len(prompt_token_ids)]),
            sampling_params,
        )
    steps = 0
    while not engine.is_finished():
        with record_function("llmserve.engine.step"):
            engine.step()
        steps += 1
    return steps


def _run_worker(payload: Mapping) -> dict:
    from llmserve import LLM

    if not torch.cuda.is_available():
        raise RuntimeError("MoE profiler requires CUDA")
    backend = str(payload["backend"])
    engine = None
    trace_path = payload.get("trace_path")
    try:
        engine = LLM(
            payload["model"],
            **build_profile_engine_kwargs(payload, backend=backend),
        )
        engine.model_runner.sampler = _ArgmaxSampler()
        warmup_tokens = int(payload["warmup_tokens"])
        if warmup_tokens > 0:
            _run_requests(
                engine,
                payload["prompt_token_ids"],
                batch_size=1,
                max_new_tokens=warmup_tokens,
            )
        engine.reset_metrics()
        torch.cuda.synchronize()
        activities = [ProfilerActivity.CPU, ProfilerActivity.CUDA]
        with profile(
            activities=activities,
            record_shapes=False,
            profile_memory=False,
            with_stack=False,
        ) as profiler:
            steps = _run_requests(
                engine,
                payload["prompt_token_ids"],
                batch_size=int(payload["batch_size"]),
                max_new_tokens=int(payload["max_new_tokens"]),
            )
            profiler.step()
        torch.cuda.synchronize()
        if trace_path:
            trace_file = Path(trace_path)
            trace_file.parent.mkdir(parents=True, exist_ok=True)
            profiler.export_chrome_trace(str(trace_file))
        return {
            "backend": backend,
            "batch_size": int(payload["batch_size"]),
            "max_new_tokens": int(payload["max_new_tokens"]),
            "steps": steps,
            "event_count": len(profiler.key_averages()),
            "events": summarize_profile_events(profiler.key_averages()),
            "trace_path": None if trace_path is None else str(trace_path),
        }
    finally:
        if engine is not None:
            engine.exit()


def _worker_command(payload_path: Path, output_path: Path) -> list[str]:
    return [
        sys.executable,
        "-m",
        "benchmarks.moe_profile",
        "--worker-payload",
        str(payload_path),
        "--worker-output",
        str(output_path),
    ]


def _run_worker_subprocess(payload: Mapping) -> dict:
    with tempfile.TemporaryDirectory(prefix="llmserve-moe-profile-") as directory:
        workdir = Path(directory)
        payload_path = workdir / "payload.json"
        output_path = workdir / "result.json"
        atomic_write_json(payload_path, dict(payload))
        completed = subprocess.run(
            _worker_command(payload_path, output_path),
            cwd=Path.cwd(),
            env=os.environ.copy(),
            text=True,
            capture_output=True,
        )
        if completed.returncode != 0:
            raise RuntimeError(
                f"MoE profiler worker failed ({completed.returncode}):\n"
                f"stdout:\n{completed.stdout}\n"
                f"stderr:\n{completed.stderr}"
            )
        with output_path.open(encoding="utf-8") as handle:
            return json.load(handle)


def run_profile(args: argparse.Namespace) -> dict:
    from transformers import AutoTokenizer

    model = Path(args.model).expanduser().resolve()
    if not model.is_dir():
        raise ValueError(f"model directory does not exist: {model}")
    backends = parse_backends(args.backends)
    if args.batch_size < 1:
        raise ValueError("batch-size must be positive")
    if args.max_new_tokens < 1:
        raise ValueError("max-new-tokens must be positive")
    if args.warmup_tokens < 0:
        raise ValueError("warmup-tokens must be non-negative")
    marlin_library = None
    if "marlin" in backends:
        if not args.marlin_library:
            raise ValueError("--marlin-library is required when using Marlin")
        marlin_library = Path(args.marlin_library).expanduser().resolve()
        if not marlin_library.is_file():
            raise ValueError(f"Marlin library does not exist: {marlin_library}")
    elif args.marlin_library:
        raise ValueError("--marlin-library requires the marlin backend")

    tokenizer = AutoTokenizer.from_pretrained(model, use_fast=True)
    cases = load_prompt_cases(
        tokenizer=tokenizer,
        prompt=args.prompt,
        prompts_file=args.prompts_file,
    )
    prompt_token_ids = [case["prompt_token_ids"] for case in cases]
    trace_payload = json.dumps(prompt_token_ids, separators=(",", ":")).encode()
    trace_dir = None if args.trace_dir is None else Path(args.trace_dir).expanduser().resolve()
    config = {
        "max_model_len": args.max_model_len,
        "max_num_batched_tokens": args.max_num_batched_tokens,
        "max_num_seqs": args.batch_size,
        "gpu_memory_utilization": args.gpu_memory_utilization,
        "seed": args.seed,
        "max_new_tokens": args.max_new_tokens,
        "warmup_tokens": args.warmup_tokens,
        "batch_size": args.batch_size,
        "marlin_library": None if marlin_library is None else str(marlin_library),
        "enable_moe_gate_up_fusion": args.enable_gate_up_fusion,
    }
    results = []
    for backend in backends:
        trace_path = None
        if trace_dir is not None:
            trace_path = str(trace_dir / f"{backend}.json")
        payload = {
            "model": str(model),
            "backend": backend,
            "prompt_token_ids": prompt_token_ids,
            "trace_path": trace_path,
            **config,
        }
        results.append(_run_worker_subprocess(payload))
    return {
        "schema_version": 1,
        "model": str(model),
        "metadata": build_environment_metadata(),
        **build_moe_experiment_metadata(
            optimization="moe_gate_up_fusion",
            enabled=args.enable_gate_up_fusion,
            baseline=build_vllm_marlin_baseline(source=marlin_library),
        ),
        "input": {
            "case_count": len(cases),
            "prompt_token_sha256": sha256(trace_payload).hexdigest(),
            "cases": [
                {
                    "id": case["id"],
                    "prompt_token_count": len(case["prompt_token_ids"]),
                }
                for case in cases
            ],
        },
        "runtime_config": {
            **config,
            "backends": list(backends),
            "native_event_parts": list(NATIVE_EVENT_PARTS),
        },
        "results": results,
    }


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model")
    parser.add_argument("--prompt")
    parser.add_argument("--prompts-file", type=Path)
    parser.add_argument("--backends", default="tinygemm,marlin")
    parser.add_argument("--marlin-library")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--max-new-tokens", type=int, default=4)
    parser.add_argument("--warmup-tokens", type=int, default=2)
    parser.add_argument("--max-model-len", type=int, default=512)
    parser.add_argument("--max-num-batched-tokens", type=int, default=128)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.85)
    parser.add_argument("--seed", type=int, default=20260820)
    parser.add_argument(
        "--enable-gate-up-fusion",
        action="store_true",
        help="enable the experimental fused Gate/Up projection",
    )
    parser.add_argument("--trace-dir", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--worker-payload", type=Path)
    parser.add_argument("--worker-output", type=Path)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    if args.worker_payload is not None or args.worker_output is not None:
        if args.worker_payload is None or args.worker_output is None:
            raise ValueError("worker mode requires both payload and output")
        with args.worker_payload.open(encoding="utf-8") as handle:
            payload = json.load(handle)
        atomic_write_json(args.worker_output, _run_worker(payload))
        return 0
    if args.model is None:
        raise ValueError("comparison mode requires --model")
    if args.output is None:
        raise ValueError("comparison mode requires --output")
    report = run_profile(args)
    atomic_write_json(args.output, report)
    print(json.dumps({"result_count": len(report["results"])}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
