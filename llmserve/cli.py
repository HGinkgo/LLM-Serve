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
