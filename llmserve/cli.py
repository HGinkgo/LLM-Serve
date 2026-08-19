import argparse
from importlib import metadata
import platform
from pathlib import Path
import sys
from typing import TextIO

from packaging.specifiers import SpecifierSet
from packaging.version import InvalidVersion, Version


class CLIError(Exception):
    pass


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed < 1:
        raise argparse.ArgumentTypeError("must be at least 1")
    return parsed


def _positive_float(value: str) -> float:
    parsed = float(value)
    if parsed <= 1e-10:
        raise argparse.ArgumentTypeError("must be greater than 0")
    return parsed


def _non_negative_int(value: str) -> int:
    parsed = int(value)
    if parsed < 0:
        raise argparse.ArgumentTypeError("must be non-negative")
    return parsed


def _gpu_ids(value: str) -> tuple[int, ...]:
    try:
        gpu_ids = tuple(_non_negative_int(item.strip()) for item in value.split(","))
    except ValueError as error:
        raise argparse.ArgumentTypeError("must be a comma-separated GPU id list") from error
    if not gpu_ids:
        raise argparse.ArgumentTypeError("must contain at least one GPU id")
    return gpu_ids


def _endpoints(value: str) -> tuple[str, ...]:
    endpoints = tuple(item.strip() for item in value.split(",") if item.strip())
    if not endpoints:
        raise argparse.ArgumentTypeError("must contain at least one endpoint")
    return endpoints


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="llmserve",
        description="Run and inspect the LLM-Serve inference runtime.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    check = subparsers.add_parser(
        "check",
        help="check the local Python, CUDA, and GPU environment",
    )
    check.set_defaults(handler=_run_check)

    generate = subparsers.add_parser(
        "generate",
        help="generate one completion with a local Qwen3 model",
    )
    generate.add_argument("--model", required=True, help="local model directory")
    generate.add_argument("--prompt", required=True, help="user prompt")
    generate.add_argument("--max-tokens", type=_positive_int, default=64)
    generate.add_argument("--temperature", type=_positive_float, default=0.6)
    generate.add_argument(
        "--enforce-eager",
        action="store_true",
        help="disable CUDA Graph execution",
    )
    generate.set_defaults(handler=_run_generate)

    serve = subparsers.add_parser(
        "serve",
        help="start an OpenAI-compatible single-host HTTP service",
    )
    serve.add_argument("--model", required=True, help="local model directory")
    serve.add_argument(
        "--served-model-name",
        help="model name exposed by the HTTP API (defaults to the directory name)",
    )
    serve.add_argument(
        "--mode",
        choices=("collocated", "pd-shared"),
        default="collocated",
        help="single-GPU Runtime or explicit Prefill/Decode Shared-KV deployment",
    )
    serve.add_argument("--host", default="127.0.0.1")
    serve.add_argument("--port", type=_positive_int, default=8000)
    serve.add_argument("--max-model-len", type=_positive_int, default=4096)
    serve.add_argument("--max-num-batched-tokens", type=_positive_int, default=1024)
    serve.add_argument("--max-num-seqs", type=_positive_int, default=128)
    serve.add_argument(
        "--max-inflight-requests",
        type=_positive_int,
        help="service-level limit for queued and active requests (defaults to serving capacity)",
    )
    serve.add_argument(
        "--request-timeout-seconds",
        type=_positive_float,
        help="optional end-to-end request deadline; expiry is enforced at an Engine step boundary",
    )
    serve.add_argument("--gpu-memory-utilization", type=float, default=0.9)
    serve.add_argument("--disable-chunked-prefill", action="store_true")
    serve.add_argument("--disable-kv-capacity-admission", action="store_true")
    serve.add_argument("--enforce-eager", action="store_true")
    serve.add_argument("--speculative-model", help="local EAGLE3 draft-model directory")
    serve.add_argument("--speculative-gamma", type=_positive_int, default=3)
    serve.add_argument("--pd-prefill-gpu", type=_non_negative_int, default=0)
    serve.add_argument("--pd-decode-gpus", type=_gpu_ids, default=(1,))
    serve.add_argument("--pd-prefill-batch-size", type=_positive_int, default=4)
    serve.add_argument("--pd-kv-slot-count", type=_positive_int, default=2)
    serve.add_argument(
        "--pd-kv-slot-capacity-tokens",
        type=_positive_int,
        default=8192,
    )
    serve.add_argument(
        "--pd-prefill-init-method",
        default="tcp://127.0.0.1:24431",
    )
    serve.add_argument("--pd-decode-init-methods", type=_endpoints, default=())
    serve.add_argument(
        "--pd-startup-timeout-seconds",
        type=_positive_float,
        default=300.0,
    )
    serve.set_defaults(handler=_run_serve)
    return parser


def _import_torch():
    import torch

    return torch


def _cuda_is_available() -> bool:
    try:
        return bool(_import_torch().cuda.is_available())
    except (ImportError, RuntimeError):
        return False


def _package_version(
    package: str,
    requirement: str | None = None,
) -> tuple[bool, str]:
    try:
        version = metadata.version(package)
    except metadata.PackageNotFoundError:
        return False, "not installed"
    if requirement is None:
        return True, version
    try:
        supported = Version(version) in SpecifierSet(requirement)
    except InvalidVersion:
        supported = False
    detail = version if supported else f"{version} (requires {requirement})"
    return supported, detail


def _environment_checks() -> list[tuple[str, bool, str]]:
    checks = []

    system = platform.system()
    checks.append(("Platform", system == "Linux", system))

    python_version = platform.python_version()
    python_supported = (3, 10) <= sys.version_info[:2] < (3, 13)
    checks.append(("Python", python_supported, python_version))

    try:
        torch = _import_torch()
    except ImportError:
        checks.extend(
            [
                ("PyTorch", False, "not installed"),
                ("CUDA", False, "PyTorch is unavailable"),
                ("GPU", False, "PyTorch is unavailable"),
            ]
        )
    else:
        checks.append(("PyTorch", True, str(torch.__version__)))
        cuda_available = bool(torch.cuda.is_available())
        cuda_version = getattr(getattr(torch, "version", None), "cuda", None)
        checks.append(
            (
                "CUDA",
                cuda_available,
                str(cuda_version) if cuda_available else "unavailable",
            )
        )
        if cuda_available:
            gpu_names = [
                torch.cuda.get_device_name(index)
                for index in range(torch.cuda.device_count())
            ]
            checks.append(("GPU", bool(gpu_names), ", ".join(gpu_names)))
        else:
            checks.append(("GPU", False, "no NVIDIA GPU visible"))

    for distribution, label, requirement in (
        ("transformers", "Transformers", ">=4.51,<5"),
        ("triton", "Triton", None),
        ("flash-attn", "FlashAttention", None),
    ):
        installed, version = _package_version(distribution, requirement)
        checks.append((label, installed, version))
    return checks


def _run_check(args: argparse.Namespace, stdout: TextIO) -> int:
    checks = _environment_checks()
    for label, passed, detail in checks:
        status = "ok" if passed else "failed"
        print(f"[{status}] {label}: {detail}", file=stdout)
    return 0 if all(passed for _, passed, _ in checks) else 1


def _load_runtime():
    from transformers import AutoTokenizer

    from llmserve import LLM, SamplingParams

    return LLM, SamplingParams, AutoTokenizer


def _run_generate(args: argparse.Namespace, stdout: TextIO) -> int:
    model_path = str(Path(args.model).expanduser())
    if not Path(model_path).is_dir():
        raise CLIError(f"model directory does not exist: {model_path}")
    if not _cuda_is_available():
        raise CLIError("CUDA is unavailable; run 'llmserve check' for details")

    llm_class, sampling_params_class, tokenizer_class = _load_runtime()
    tokenizer = tokenizer_class.from_pretrained(model_path)
    formatted_prompt = tokenizer.apply_chat_template(
        [{"role": "user", "content": args.prompt}],
        tokenize=False,
        add_generation_prompt=True,
    )
    sampling_params = sampling_params_class(
        temperature=args.temperature,
        max_tokens=args.max_tokens,
    )
    llm = llm_class(model_path, enforce_eager=args.enforce_eager)
    try:
        outputs = llm.generate(
            [formatted_prompt],
            sampling_params,
            use_tqdm=False,
        )
    finally:
        llm.exit()
    print(outputs[0]["text"], file=stdout)
    return 0


def _build_service_runtime(launch_config):
    from llmserve.service.factory import build_service_runtime

    return build_service_runtime(launch_config)


def _create_service_app(runtime, *, model_name: str):
    from llmserve.service.api import create_app

    return create_app(runtime, model_name=model_name)


def _run_uvicorn(app, *, host: str, port: int):
    import uvicorn

    uvicorn.run(app, host=host, port=port, log_level="info")


def _run_serve(args: argparse.Namespace, stdout: TextIO) -> int:
    model_path = str(Path(args.model).expanduser())
    if not Path(model_path).is_dir():
        raise CLIError(f"model directory does not exist: {model_path}")
    if not _cuda_is_available():
        raise CLIError("CUDA is unavailable; run 'llmserve check' for details")

    from llmserve.service.factory import ServiceLaunchConfig

    try:
        launch_config = ServiceLaunchConfig(
            model=model_path,
            mode=args.mode,
            max_model_len=args.max_model_len,
            max_num_batched_tokens=args.max_num_batched_tokens,
            max_num_seqs=args.max_num_seqs,
            max_inflight_requests=args.max_inflight_requests,
            request_timeout_seconds=args.request_timeout_seconds,
            gpu_memory_utilization=args.gpu_memory_utilization,
            enable_chunked_prefill=not args.disable_chunked_prefill,
            enable_kv_capacity_admission=not args.disable_kv_capacity_admission,
            enforce_eager=args.enforce_eager,
            speculative_model=args.speculative_model,
            speculative_gamma=args.speculative_gamma,
            prefill_gpu=args.pd_prefill_gpu,
            decode_gpus=tuple(args.pd_decode_gpus),
            prefill_batch_size=args.pd_prefill_batch_size,
            kv_slot_count=args.pd_kv_slot_count,
            kv_slot_capacity_tokens=args.pd_kv_slot_capacity_tokens,
            prefill_init_method=args.pd_prefill_init_method,
            decode_init_methods=tuple(args.pd_decode_init_methods),
            startup_timeout_seconds=args.pd_startup_timeout_seconds,
        )
    except ValueError as error:
        raise CLIError(str(error)) from error

    runtime = _build_service_runtime(launch_config)
    try:
        runtime.start()
        served_model_name = args.served_model_name or Path(model_path).name
        app = _create_service_app(runtime, model_name=served_model_name)
        print(
            f"serving {served_model_name} at http://{args.host}:{args.port}",
            file=stdout,
        )
        _run_uvicorn(app, host=args.host, port=args.port)
    finally:
        runtime.close()
    return 0


def main(
    argv: list[str] | None = None,
    *,
    stdout: TextIO | None = None,
    stderr: TextIO | None = None,
) -> int:
    stdout = sys.stdout if stdout is None else stdout
    stderr = sys.stderr if stderr is None else stderr
    args = build_parser().parse_args(argv)
    try:
        return args.handler(args, stdout)
    except CLIError as error:
        print(f"error: {error}", file=stderr)
        return 2
    except KeyboardInterrupt:
        print("error: interrupted", file=stderr)
        return 130
    except Exception as error:
        print(f"error: {type(error).__name__}: {error}", file=stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
