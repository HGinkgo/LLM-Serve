"""Optional external-reference parity checks for Qwen3-MoE GPTQ models."""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from hashlib import sha256
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile

from benchmarks.environment import atomic_write_json

DEFAULT_PROMPT = "What is 2 + 2?"


def compare_token_traces(
    *,
    prompt_token_ids: Sequence[int],
    reference_token_ids: Sequence[int],
    candidate_token_ids: Sequence[int],
    candidate_prompt_token_ids: Sequence[int] | None = None,
) -> dict:
    """Compare greedy outputs without hiding a length or token mismatch."""
    prompt = list(prompt_token_ids)
    candidate_prompt = (
        prompt if candidate_prompt_token_ids is None else list(candidate_prompt_token_ids)
    )
    if prompt != candidate_prompt:
        raise ValueError("reference and candidate prompt token ids must match")

    reference = list(reference_token_ids)
    candidate = list(candidate_token_ids)
    compared_token_count = min(len(reference), len(candidate))
    first_mismatch_index = next(
        (
            index
            for index, (reference_token, candidate_token) in enumerate(
                zip(reference, candidate)
            )
            if reference_token != candidate_token
        ),
        None,
    )
    if first_mismatch_index is None and len(reference) != len(candidate):
        first_mismatch_index = compared_token_count

    report = {
        "matches": first_mismatch_index is None,
        "prompt_token_count": len(prompt),
        "reference_token_count": len(reference),
        "candidate_token_count": len(candidate),
        "compared_token_count": compared_token_count,
        "first_mismatch_index": first_mismatch_index,
    }
    if first_mismatch_index is not None:
        report["reference_token_id"] = (
            reference[first_mismatch_index]
            if first_mismatch_index < len(reference)
            else None
        )
        report["candidate_token_id"] = (
            candidate[first_mismatch_index]
            if first_mismatch_index < len(candidate)
            else None
        )
    return report


def build_worker_command(
    *,
    worker: str,
    payload_path: Path,
    output_path: Path,
    reference_package_dir: Path | None = None,
    inherited_environment: dict[str, str] | None = None,
) -> tuple[list[str], dict[str, str]]:
    """Build an isolated worker invocation without leaking vLLM to the candidate."""
    environment = dict(os.environ if inherited_environment is None else inherited_environment)
    if reference_package_dir is not None:
        previous_path = environment.get("PYTHONPATH")
        environment["PYTHONPATH"] = (
            f"{reference_package_dir}{os.pathsep}{previous_path}"
            if previous_path
            else str(reference_package_dir)
        )
    return [
        sys.executable,
        "-m",
        "benchmarks.moe_reference",
        "--payload",
        str(payload_path),
        "--output",
        str(output_path),
        "--worker",
        worker,
    ], environment


def load_prompt_cases(
    *,
    tokenizer,
    prompt: str | None,
    prompts_file: Path | None,
) -> list[dict]:
    """Encode an ordered, named prompt fixture for both runtime workers."""
    if prompt is not None and prompts_file is not None:
        raise ValueError("--prompt and --prompts-file cannot be combined")
    if prompts_file is None:
        raw_cases = [{"id": "cli-prompt", "prompt": prompt or DEFAULT_PROMPT}]
    else:
        with prompts_file.open(encoding="utf-8") as handle:
            raw_cases = json.load(handle)
        if not isinstance(raw_cases, list) or not raw_cases:
            raise ValueError("prompt fixture must be a non-empty JSON list")

    cases = []
    seen_ids = set()
    for raw_case in raw_cases:
        if not isinstance(raw_case, dict):
            raise ValueError("each prompt fixture entry must be an object")
        case_id = raw_case.get("id")
        case_prompt = raw_case.get("prompt")
        if not isinstance(case_id, str) or not case_id:
            raise ValueError("each prompt fixture entry needs a non-empty id")
        if case_id in seen_ids:
            raise ValueError(f"duplicate prompt fixture id: {case_id}")
        if not isinstance(case_prompt, str) or not case_prompt:
            raise ValueError(f"prompt fixture {case_id!r} needs a non-empty prompt")
        prompt_token_ids = tokenizer.encode(case_prompt, add_special_tokens=False)
        if not prompt_token_ids:
            raise ValueError(f"prompt fixture {case_id!r} encodes to no tokens")
        seen_ids.add(case_id)
        cases.append({"id": case_id, "prompt_token_ids": list(prompt_token_ids)})
    return cases


def compare_case_traces(
    *,
    reference_cases: Sequence[dict],
    candidate_cases: Sequence[dict],
) -> dict:
    """Compare all named cases while retaining the first failure location."""
    reference_by_id = {case["id"]: case for case in reference_cases}
    candidate_by_id = {case["id"]: case for case in candidate_cases}
    reference_ids = [case["id"] for case in reference_cases]
    candidate_ids = [case["id"] for case in candidate_cases]
    if reference_ids != candidate_ids:
        raise ValueError("reference and candidate case ids must match in order")
    if len(reference_by_id) != len(reference_ids):
        raise ValueError("reference case ids must be unique")
    if len(candidate_by_id) != len(candidate_ids):
        raise ValueError("candidate case ids must be unique")

    case_reports = []
    for case_id in reference_ids:
        reference = reference_by_id[case_id]
        candidate = candidate_by_id[case_id]
        comparison = compare_token_traces(
            prompt_token_ids=reference["prompt_token_ids"],
            candidate_prompt_token_ids=candidate["prompt_token_ids"],
            reference_token_ids=reference["generated_token_ids"],
            candidate_token_ids=candidate["generated_token_ids"],
        )
        case_reports.append({"id": case_id, **comparison})

    first_failure = next(
        (case for case in case_reports if not case["matches"]),
        None,
    )
    return {
        "matches": first_failure is None,
        "case_count": len(case_reports),
        "matched_case_count": sum(case["matches"] for case in case_reports),
        "first_mismatch_case_id": None if first_failure is None else first_failure["id"],
        "cases": case_reports,
    }


class _ArgmaxSampler:
    def __call__(self, logits, temperatures):
        del temperatures
        return logits.argmax(dim=-1)


def _load_payload(path: Path) -> dict:
    with path.open(encoding="utf-8") as handle:
        payload = json.load(handle)
    cases = payload.get("cases")
    if not isinstance(cases, list) or not cases:
        raise ValueError("payload must contain non-empty cases")
    case_ids = set()
    for case in cases:
        if not isinstance(case, dict) or not case.get("prompt_token_ids"):
            raise ValueError("each payload case must contain prompt_token_ids")
        case_id = case.get("id")
        if not isinstance(case_id, str) or not case_id:
            raise ValueError("each payload case must contain a non-empty id")
        if case_id in case_ids:
            raise ValueError(f"duplicate payload case id: {case_id}")
        case_ids.add(case_id)
    if payload.get("max_new_tokens", 0) <= 0:
        raise ValueError("payload max_new_tokens must be positive")
    return payload


def _run_vllm_worker(payload: dict) -> dict:
    from vllm import LLM, SamplingParams
    import vllm

    llm = LLM(
        model=payload["model"],
        quantization="gptq_marlin",
        dtype="half",
        tensor_parallel_size=1,
        enforce_eager=True,
        max_model_len=payload["max_model_len"],
        max_num_batched_tokens=payload["max_num_batched_tokens"],
        max_num_seqs=1,
        gpu_memory_utilization=payload["gpu_memory_utilization"],
        disable_log_stats=True,
    )
    sampling_params = SamplingParams(
        temperature=0.0,
        max_tokens=payload["max_new_tokens"],
        min_tokens=payload["max_new_tokens"],
        ignore_eos=True,
    )
    case_outputs = []
    for case in payload["cases"]:
        output = llm.generate(
            [{"prompt_token_ids": case["prompt_token_ids"]}],
            sampling_params,
            use_tqdm=False,
        )[0]
        case_outputs.append((case, output))
    return {
        "runtime": "vllm",
        "runtime_version": vllm.__version__,
        "cases": [
            {
                "id": case["id"],
                "prompt_token_ids": case["prompt_token_ids"],
                "generated_token_ids": list(output.outputs[0].token_ids),
            }
            for case, output in case_outputs
        ],
    }


def _run_llmserve_worker(payload: dict) -> dict:
    from llmserve import LLM, SamplingParams

    engine = LLM(
        payload["model"],
        enforce_eager=True,
        max_model_len=payload["max_model_len"],
        max_num_batched_tokens=payload["max_num_batched_tokens"],
        max_num_seqs=1,
        gpu_memory_utilization=payload["gpu_memory_utilization"],
        random_seed=payload["seed"],
        gptq_backend=payload.get("candidate_gptq_backend", "tinygemm"),
        marlin_library=payload.get("candidate_marlin_library"),
    )
    try:
        engine.model_runner.sampler = _ArgmaxSampler()
        sampling_params = SamplingParams(
            temperature=1.0,
            max_tokens=payload["max_new_tokens"],
            ignore_eos=True,
        )
        case_outputs = []
        for case in payload["cases"]:
            output = engine.generate(
                [case["prompt_token_ids"]],
                sampling_params,
                use_tqdm=False,
            )[0]
            case_outputs.append((case, output))
        return {
            "runtime": "llmserve",
            "cases": [
                {
                    "id": case["id"],
                    "prompt_token_ids": case["prompt_token_ids"],
                    "generated_token_ids": list(output["token_ids"]),
                }
                for case, output in case_outputs
            ],
        }
    finally:
        engine.exit()


def _run_worker(worker: str, payload_path: Path, output_path: Path) -> int:
    payload = _load_payload(payload_path)
    if worker == "vllm":
        result = _run_vllm_worker(payload)
    elif worker == "llmserve":
        result = _run_llmserve_worker(payload)
    else:
        raise ValueError(f"unknown worker: {worker}")
    atomic_write_json(output_path, result)
    return 0


def _worker_result(command: list[str], environment: dict[str, str]) -> None:
    completed = subprocess.run(
        command,
        cwd=Path.cwd(),
        env=environment,
        text=True,
        capture_output=True,
    )
    if completed.returncode == 0:
        return
    raise RuntimeError(
        f"reference worker failed ({completed.returncode}):\n"
        f"stdout:\n{completed.stdout}\n"
        f"stderr:\n{completed.stderr}"
    )


def run_comparison(args: argparse.Namespace) -> dict:
    from transformers import AutoTokenizer

    model = Path(args.model).resolve()
    reference_package_dir = Path(args.reference_package_dir).resolve()
    if not model.is_dir():
        raise ValueError(f"model directory does not exist: {model}")
    if not reference_package_dir.is_dir():
        raise ValueError(
            f"vLLM reference package directory does not exist: {reference_package_dir}"
        )
    if args.max_new_tokens <= 0:
        raise ValueError("max_new_tokens must be positive")
    if args.candidate_gptq_backend == "marlin":
        if args.candidate_marlin_library is None:
            raise ValueError(
                "candidate_gptq_backend=marlin requires --candidate-marlin-library"
            )
        candidate_marlin_library = Path(args.candidate_marlin_library).resolve()
        if not candidate_marlin_library.is_file():
            raise ValueError(
                f"candidate Marlin library does not exist: {candidate_marlin_library}"
            )
    else:
        if args.candidate_marlin_library is not None:
            raise ValueError(
                "--candidate-marlin-library requires --candidate-gptq-backend marlin"
            )
        candidate_marlin_library = None

    tokenizer = AutoTokenizer.from_pretrained(model, use_fast=True)
    prompt_cases = load_prompt_cases(
        tokenizer=tokenizer,
        prompt=args.prompt,
        prompts_file=args.prompts_file,
    )
    payload = {
        "model": str(model),
        "cases": prompt_cases,
        "max_new_tokens": args.max_new_tokens,
        "max_model_len": args.max_model_len,
        "max_num_batched_tokens": args.max_num_batched_tokens,
        "gpu_memory_utilization": args.gpu_memory_utilization,
        "seed": args.seed,
        "candidate_gptq_backend": args.candidate_gptq_backend,
        "candidate_marlin_library": (
            None if candidate_marlin_library is None else str(candidate_marlin_library)
        ),
    }

    with tempfile.TemporaryDirectory(prefix="llmserve-moe-reference-") as directory:
        workdir = Path(directory)
        payload_path = workdir / "payload.json"
        reference_path = workdir / "vllm.json"
        candidate_path = workdir / "llmserve.json"
        atomic_write_json(payload_path, payload)
        reference_command, reference_environment = build_worker_command(
            worker="vllm",
            payload_path=payload_path,
            output_path=reference_path,
            reference_package_dir=reference_package_dir,
        )
        candidate_command, candidate_environment = build_worker_command(
            worker="llmserve",
            payload_path=payload_path,
            output_path=candidate_path,
        )
        _worker_result(reference_command, reference_environment)
        _worker_result(candidate_command, candidate_environment)
        with reference_path.open(encoding="utf-8") as handle:
            reference = json.load(handle)
        with candidate_path.open(encoding="utf-8") as handle:
            candidate = json.load(handle)

    comparison = compare_case_traces(
        reference_cases=reference["cases"],
        candidate_cases=candidate["cases"],
    )
    return {
        "schema_version": 1,
        "model": str(model),
        "reference_package_dir": str(reference_package_dir),
        "input": {
            "case_count": len(prompt_cases),
            "cases": [
                {
                    "id": case["id"],
                    "prompt_token_count": len(case["prompt_token_ids"]),
                    "prompt_token_sha256": sha256(
                        json.dumps(
                            case["prompt_token_ids"], separators=(",", ":")
                        ).encode()
                    ).hexdigest(),
                }
                for case in prompt_cases
            ],
            "max_new_tokens": args.max_new_tokens,
        },
        "runtime_config": {
            "max_model_len": args.max_model_len,
            "max_num_batched_tokens": args.max_num_batched_tokens,
            "gpu_memory_utilization": args.gpu_memory_utilization,
            "seed": args.seed,
            "candidate_gptq_backend": args.candidate_gptq_backend,
            "candidate_marlin_library": (
                None if candidate_marlin_library is None else str(candidate_marlin_library)
            ),
        },
        "reference": reference,
        "candidate": candidate,
        "comparison": comparison,
    }


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--worker", choices=("vllm", "llmserve"))
    parser.add_argument("--payload", type=Path)
    parser.add_argument("--model")
    parser.add_argument("--reference-package-dir")
    parser.add_argument("--prompt")
    parser.add_argument("--prompts-file", type=Path)
    parser.add_argument("--max-new-tokens", type=int, default=4)
    parser.add_argument("--max-model-len", type=int, default=512)
    parser.add_argument("--max-num-batched-tokens", type=int, default=128)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.85)
    parser.add_argument("--seed", type=int, default=20260819)
    parser.add_argument(
        "--candidate-gptq-backend",
        choices=("tinygemm", "marlin"),
        default="tinygemm",
    )
    parser.add_argument("--candidate-marlin-library")
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    if args.worker is not None:
        if args.payload is None:
            raise ValueError("worker mode requires --payload")
        return _run_worker(args.worker, args.payload, args.output)
    if args.model is None or args.reference_package_dir is None:
        raise ValueError("comparison mode requires --model and --reference-package-dir")
    report = run_comparison(args)
    atomic_write_json(args.output, report)
    print(json.dumps(report["comparison"], sort_keys=True))
    return 0 if report["comparison"]["matches"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
