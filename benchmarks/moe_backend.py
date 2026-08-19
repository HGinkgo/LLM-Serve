"""Sequential TinyGEMM/Marlin baseline for the Qwen3-MoE GPTQ runtime."""

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
from time import perf_counter

import torch

from benchmarks.environment import atomic_write_json, build_environment_metadata
from benchmarks.metrics import summarize_values
from benchmarks.moe_reference import load_prompt_cases


SUPPORTED_BACKENDS = ("tinygemm", "marlin")


def parse_backends(value: str) -> tuple[str, ...]:
    backends = []
    for item in value.split(","):
        backend = item.strip()
        if not backend:
            continue
        if backend not in SUPPORTED_BACKENDS:
            raise ValueError(f"unknown GPTQ backend: {backend}")
        if backend not in backends:
            backends.append(backend)
    if not backends:
        raise ValueError("at least one GPTQ backend is required")
    return tuple(backends)


def parse_positive_int_list(value: str, *, name: str) -> tuple[int, ...]:
    values = []
    for item in value.split(","):
        try:
            parsed = int(item.strip())
        except ValueError as error:
            raise ValueError(f"{name} must contain positive integers") from error
        if parsed < 1:
            raise ValueError(f"{name} must contain positive integers")
        if parsed not in values:
            values.append(parsed)
    if not values:
        raise ValueError(f"{name} must not be empty")
    return tuple(values)


def build_engine_kwargs(config: Mapping, *, backend: str) -> dict:
    if backend not in SUPPORTED_BACKENDS:
        raise ValueError(f"unknown GPTQ backend: {backend}")
    kwargs = {
        "max_model_len": int(config["max_model_len"]),
        "max_num_batched_tokens": int(config["max_num_batched_tokens"]),
        "max_num_seqs": int(config["max_num_seqs"]),
        "gpu_memory_utilization": float(config["gpu_memory_utilization"]),
        "enforce_eager": True,
        "random_seed": int(config.get("seed", 20260819)),
        "gptq_backend": backend,
    }
    if backend == "marlin":
        library = config.get("marlin_library")
        if not library:
            raise ValueError("Marlin baseline requires marlin_library")
        kwargs["marlin_library"] = str(library)
    return kwargs


def _percentile_summary(values: Sequence[float]) -> dict:
    return summarize_values(list(values))


def summarize_batch_metrics(
    requests: Sequence[Mapping],
    *,
    wall_time: float,
    peak_allocated_bytes: int,
    peak_reserved_bytes: int,
) -> dict:
    if wall_time <= 0:
        raise ValueError("wall_time must be positive")
    successful = [request for request in requests if request.get("success")]
    ttft_ms = []
    tpot_ms = []
    output_tokens = 0
    compact_requests = []
    for request in successful:
        arrival_time = request["arrival_time"]
        first_token_time = request["first_token_time"]
        finish_time = request["finish_time"]
        token_count = int(request["output_tokens"])
        token_times = list(request.get("token_times", ()))
        ttft = (first_token_time - arrival_time) * 1000
        ttft_ms.append(ttft)
        tpot = None
        if token_count > 1:
            last_token_time = token_times[-1] if token_times else finish_time
            tpot = (last_token_time - first_token_time) * 1000 / (token_count - 1)
            tpot_ms.append(tpot)
        output_tokens += token_count
        compact_requests.append({
            "prompt_tokens": int(request.get("prompt_tokens", 0)),
            "output_tokens": token_count,
            "ttft_ms": ttft,
            "tpot_ms": tpot,
        })
    return {
        "request_count": len(requests),
        "completed": len(successful),
        "failed": len(requests) - len(successful),
        "output_tokens": output_tokens,
        "wall_time_seconds": wall_time,
        "request_throughput": len(successful) / wall_time,
        "output_throughput": output_tokens / wall_time,
        "ttft_ms": _percentile_summary(ttft_ms),
        "tpot_ms": _percentile_summary(tpot_ms),
        "peak_allocated_bytes": int(peak_allocated_bytes),
        "peak_reserved_bytes": int(peak_reserved_bytes),
        "requests": compact_requests,
    }


class _ArgmaxSampler:
    def __call__(self, logits, temperatures):
        del temperatures
        return logits.argmax(dim=-1)


def _run_batch(engine, prompt_token_ids: Sequence[Sequence[int]], output_len: int, batch_size: int) -> dict:
    from llmserve import SamplingParams

    engine.reset_metrics()
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    sampling_params = SamplingParams(
        temperature=1.0,
        max_tokens=output_len,
        ignore_eos=True,
    )
    for index in range(batch_size):
        engine.add_request(
            list(prompt_token_ids[index % len(prompt_token_ids)]),
            sampling_params,
        )
    while not engine.is_finished():
        engine.step()
    torch.cuda.synchronize()
    metrics = engine.get_metrics()
    summary = metrics["summary"]
    return summarize_batch_metrics(
        metrics["requests"],
        wall_time=summary["wall_time"],
        peak_allocated_bytes=torch.cuda.max_memory_allocated(),
        peak_reserved_bytes=torch.cuda.max_memory_reserved(),
    )


def _run_worker(payload: Mapping) -> dict:
    from llmserve import LLM

    backend = payload["backend"]
    engine = None
    try:
        engine = LLM(
            payload["model"],
            **build_engine_kwargs(payload, backend=backend),
        )
        engine.model_runner.sampler = _ArgmaxSampler()
        warmup_len = min(int(payload["max_new_tokens"]), 4)
        _run_batch(engine, payload["prompt_token_ids"], warmup_len, 1)
        results = []
        for repeat in range(int(payload["repeats"])):
            for batch_size in payload["batch_sizes"]:
                result = _run_batch(
                    engine,
                    payload["prompt_token_ids"],
                    int(payload["max_new_tokens"]),
                    int(batch_size),
                )
                results.append({
                    "backend": backend,
                    "batch_size": int(batch_size),
                    "repeat": repeat,
                    **result,
                })
        return {"backend": backend, "results": results}
    finally:
        if engine is not None:
            engine.exit()


def _worker_command(payload_path: Path, output_path: Path) -> list[str]:
    return [
        sys.executable,
        "-m",
        "benchmarks.moe_backend",
        "--worker-payload",
        str(payload_path),
        "--worker-output",
        str(output_path),
    ]


def _run_worker_subprocess(payload: Mapping) -> dict:
    with tempfile.TemporaryDirectory(prefix="llmserve-moe-backend-") as directory:
        workdir = Path(directory)
        payload_path = workdir / "payload.json"
        output_path = workdir / "result.json"
        atomic_write_json(payload_path, payload)
        completed = subprocess.run(
            _worker_command(payload_path, output_path),
            cwd=Path.cwd(),
            env=os.environ.copy(),
            text=True,
            capture_output=True,
        )
        if completed.returncode != 0:
            raise RuntimeError(
                f"MoE backend worker failed ({completed.returncode}):\n"
                f"stdout:\n{completed.stdout}\n"
                f"stderr:\n{completed.stderr}"
            )
        with output_path.open(encoding="utf-8") as handle:
            return json.load(handle)


def run_baseline(args: argparse.Namespace) -> dict:
    from transformers import AutoTokenizer

    model = Path(args.model).expanduser().resolve()
    if not model.is_dir():
        raise ValueError(f"model directory does not exist: {model}")
    backends = parse_backends(args.backends)
    batch_sizes = parse_positive_int_list(args.batch_sizes, name="batch-sizes")
    if args.repeats < 1:
        raise ValueError("repeats must be positive")
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
    config = {
        "max_model_len": args.max_model_len,
        "max_num_batched_tokens": args.max_num_batched_tokens,
        "max_num_seqs": max(batch_sizes),
        "gpu_memory_utilization": args.gpu_memory_utilization,
        "seed": args.seed,
        "max_new_tokens": args.max_new_tokens,
        "marlin_library": None if marlin_library is None else str(marlin_library),
    }
    results = []
    for backend in backends:
        payload = {
            "model": str(model),
            "backend": backend,
            "prompt_token_ids": prompt_token_ids,
            "batch_sizes": batch_sizes,
            "repeats": args.repeats,
            **config,
        }
        results.extend(_run_worker_subprocess(payload)["results"])
    return {
        "schema_version": 1,
        "model": str(model),
        "metadata": build_environment_metadata(),
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
            "batch_sizes": list(batch_sizes),
            "repeats": args.repeats,
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
    parser.add_argument("--batch-sizes", default="1,4")
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--max-new-tokens", type=int, default=16)
    parser.add_argument("--max-model-len", type=int, default=512)
    parser.add_argument("--max-num-batched-tokens", type=int, default=128)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.85)
    parser.add_argument("--seed", type=int, default=20260819)
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
    report = run_baseline(args)
    atomic_write_json(args.output, report)
    print(json.dumps({"result_count": len(report["results"])}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
