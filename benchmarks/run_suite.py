import argparse
import csv
import json
import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

from benchmarks.environment import (
    atomic_write_json,
    build_environment_metadata,
    discover_model_revision,
)
from benchmarks.suite import (
    aggregate_results,
    build_summary_rows,
    can_resume_result,
    expand_suite,
)


AGGREGATE_METRICS = (
    ("metrics.throughput.requests_per_second", 1.0, "req/s"),
    ("metrics.throughput.input_tokens_per_second", 1.0, "tok/s"),
    ("metrics.throughput.output_tokens_per_second", 1.0, "tok/s"),
    ("metrics.throughput.total_tokens_per_second", 1.0, "tok/s"),
    ("metrics.goodput.requests_per_second", 1.0, "req/s"),
    ("metrics.latency.overall.ttft.p50", 1000.0, "ms"),
    ("metrics.latency.overall.ttft.p99", 1000.0, "ms"),
    ("metrics.latency.overall.tpot.p50", 1000.0, "ms"),
    ("metrics.latency.overall.tpot.p99", 1000.0, "ms"),
    ("metrics.latency.overall.burst_itl.p50", 1000.0, "ms"),
    ("metrics.latency.overall.burst_itl.p99", 1000.0, "ms"),
    (
        "metrics.latency.overall.output_event_latency.p50",
        1000.0,
        "ms",
    ),
    (
        "metrics.latency.overall.output_event_latency.p99",
        1000.0,
        "ms",
    ),
    (
        "metrics.latency.overall.speculative_step_latency.p50",
        1000.0,
        "ms",
    ),
    (
        "metrics.latency.overall.speculative_step_latency.p99",
        1000.0,
        "ms",
    ),
    ("metrics.latency.overall.e2e.p50", 1000.0, "ms"),
    ("metrics.latency.overall.e2e.p99", 1000.0, "ms"),
    ("metrics.latency.short.ttft.p50", 1000.0, "ms"),
    ("metrics.latency.short.ttft.p99", 1000.0, "ms"),
    ("metrics.latency.short.tpot.p50", 1000.0, "ms"),
    ("metrics.latency.short.tpot.p99", 1000.0, "ms"),
    ("metrics.latency.short.e2e.p50", 1000.0, "ms"),
    ("metrics.latency.short.e2e.p99", 1000.0, "ms"),
    ("metrics.latency.long.ttft.p50", 1000.0, "ms"),
    ("metrics.latency.long.ttft.p99", 1000.0, "ms"),
    ("metrics.latency.long.tpot.p50", 1000.0, "ms"),
    ("metrics.latency.long.tpot.p99", 1000.0, "ms"),
    ("metrics.latency.long.e2e.p50", 1000.0, "ms"),
    ("metrics.latency.long.e2e.p99", 1000.0, "ms"),
    ("metrics.speculative.acceptance_rate", 1.0, "ratio"),
    ("metrics.speculative.acceptance_length", 1.0, "tokens/step"),
    ("metrics.scheduled_batch_size.mean", 1.0, "requests"),
    ("metrics.speculative_batch_size.mean", 1.0, "requests"),
    ("metrics.waiting_queue_size.p99", 1.0, "requests"),
    ("metrics.running_queue_size.p99", 1.0, "requests"),
)


def _write_csv(path: Path, rows: list[dict]):
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("")
        return
    with path.open("w", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _load_result(path: Path):
    try:
        return json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return None


def _redact_error(message: str, paths):
    for path in paths:
        if path:
            message = message.replace(path, Path(path).name)
    return message


def _failure_result(
    point: dict,
    metadata: dict,
    process,
    *,
    error_type="WorkerProcessError",
    redacted_paths=(),
):
    stderr = (getattr(process, "stderr", "") or "").strip()
    return {
        "schema_version": 2,
        "complete": False,
        "point_id": point["point_id"],
        "git_commit": metadata.get("git_commit"),
        "metadata": metadata,
        "config": point,
        "metrics": None,
        "requests": [],
        "error": {
            "type": error_type,
            "returncode": getattr(process, "returncode", None),
            "message": _redact_error(
                stderr[-2000:] or "worker did not write a result",
                redacted_paths,
            ),
        },
    }


def _aggregate_all(results):
    rows = []
    for metric_path, scale, unit in AGGREGATE_METRICS:
        rows.extend(
            aggregate_results(
                results, metric_path, scale=scale, unit=unit
            )
        )
    return rows


def execute_suite(
    suite: dict,
    output_dir: Path,
    model: str,
    speculative_model: str | None,
    metadata: dict,
    *,
    model_revision: str | None = None,
    speculative_model_revision: str | None = None,
    resume: bool = False,
    timeout_seconds: float | None = None,
    command_runner=subprocess.run,
):
    output_dir = Path(output_dir)
    runs_dir = output_dir / "runs"
    points_dir = output_dir / "points"
    runs_dir.mkdir(parents=True, exist_ok=True)
    points_dir.mkdir(parents=True, exist_ok=True)
    points = expand_suite(suite)
    commit = metadata.get("git_commit")
    model_revision = model_revision or discover_model_revision(model)
    speculative_model_revision = (
        speculative_model_revision
        or discover_model_revision(speculative_model)
    )
    manifest = {
        "schema_version": 1,
        "suite": suite["name"],
        "complete": False,
        "git_commit": commit,
        "metadata": metadata,
        "model": Path(model).name,
        "model_revision": model_revision,
        "speculative_model": (
            Path(speculative_model).name if speculative_model else None
        ),
        "speculative_model_revision": speculative_model_revision,
        "total_points": len(points),
        "completed_points": 0,
        "failed_points": 0,
    }
    atomic_write_json(output_dir / "manifest.json", manifest)

    results = []
    for index, point in enumerate(points, start=1):
        point_path = points_dir / f"{point['point_id']}.json"
        result_path = runs_dir / f"{point['point_id']}.json"
        atomic_write_json(point_path, point)
        if resume and can_resume_result(result_path, point, commit):
            result = _load_result(result_path)
            print(
                f"[{index}/{len(points)}] RESUME {point['point_id']}",
                flush=True,
            )
        else:
            result_path.unlink(missing_ok=True)
            command = [
                sys.executable,
                "-m",
                "benchmarks.serve",
                "--point-config",
                str(point_path),
                "--model",
                model,
                "--output",
                str(result_path),
            ]
            if speculative_model:
                command.extend(["--speculative-model", speculative_model])
            if model_revision:
                command.extend(["--model-revision", model_revision])
            if speculative_model_revision:
                command.extend([
                    "--speculative-model-revision",
                    speculative_model_revision,
                ])
            if commit:
                command.extend(["--expected-git-commit", commit])
            print(
                f"[{index}/{len(points)}] RUN {point['point_id']}",
                flush=True,
            )
            worker_error_type = "WorkerProcessError"
            try:
                process = command_runner(
                    command,
                    capture_output=True,
                    text=True,
                    timeout=timeout_seconds,
                )
            except subprocess.TimeoutExpired as error:
                process = error
                process.returncode = None
                process.stderr = f"worker timed out after {timeout_seconds}s"
            except OSError as error:
                worker_error_type = "WorkerStartError"
                process = SimpleNamespace(
                    returncode=None,
                    stderr=str(error),
                )
            result = _load_result(result_path)
            if result is None:
                result = _failure_result(
                    point,
                    metadata,
                    process,
                    error_type=worker_error_type,
                    redacted_paths=(model, speculative_model),
                )
                atomic_write_json(result_path, result)
            elif result.get("complete") and not can_resume_result(
                result_path, point, commit
            ):
                result["complete"] = False
                result["error"] = {
                    "type": "InvalidWorkerResult",
                    "returncode": process.returncode,
                    "message": "worker result identity does not match suite point",
                }
                atomic_write_json(result_path, result)
            if process.returncode != 0 and result.get("complete"):
                result["complete"] = False
                result["error"] = _failure_result(
                    point,
                    metadata,
                    process,
                    redacted_paths=(model, speculative_model),
                )["error"]
                atomic_write_json(result_path, result)
        results.append(result)

    summary_rows = build_summary_rows(results)
    aggregate_rows = _aggregate_all(results)
    _write_csv(output_dir / "summary.csv", summary_rows)
    _write_csv(output_dir / "aggregate.csv", aggregate_rows)
    failed = sum(not result.get("complete") for result in results)
    manifest.update({
        "complete": failed == 0,
        "completed_points": len(results) - failed,
        "failed_points": failed,
    })
    atomic_write_json(output_dir / "manifest.json", manifest)
    return 0 if failed == 0 else 1


def _parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Run a reproducible LLM-Serve benchmark suite"
    )
    parser.add_argument("--suite", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--model", default=os.environ.get("MODEL_PATH"))
    parser.add_argument(
        "--speculative-model", default=os.environ.get("SPECULATIVE_MODEL")
    )
    parser.add_argument("--model-revision")
    parser.add_argument("--speculative-model-revision")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--allow-dirty", action="store_true")
    parser.add_argument("--timeout-seconds", type=float)
    args = parser.parse_args(argv)
    if not args.model:
        parser.error("--model or MODEL_PATH is required")
    return args


def main(argv=None):
    args = _parse_args(argv)
    suite = json.loads(Path(args.suite).read_text())
    metadata = build_environment_metadata()
    if metadata.get("git_dirty") and not args.allow_dirty:
        print(
            "refusing formal benchmark on a dirty worktree; "
            "use --allow-dirty for smoke/pilot runs",
            file=sys.stderr,
        )
        return 2
    return execute_suite(
        suite,
        Path(args.output_dir),
        model=os.path.expanduser(args.model),
        speculative_model=(
            os.path.expanduser(args.speculative_model)
            if args.speculative_model
            else None
        ),
        metadata=metadata,
        model_revision=args.model_revision,
        speculative_model_revision=args.speculative_model_revision,
        resume=args.resume,
        timeout_seconds=args.timeout_seconds,
    )


if __name__ == "__main__":
    raise SystemExit(main())
