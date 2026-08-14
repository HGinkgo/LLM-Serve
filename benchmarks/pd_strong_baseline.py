"""Build the formal A-versus-C report from this run's raw benchmark JSON."""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path

from benchmarks.metrics import summarize_values


def _read_json(path: Path):
    return json.loads(Path(path).read_text())


def _write_csv(path: Path, rows: list[dict]):
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("")
        return
    fieldnames = sorted({field for row in rows for field in row})
    with path.open("w", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def _ms(start, end):
    if start is None or end is None:
        return None
    return (end - start) * 1000


def _measurement_requests(result: dict):
    window = result.get("telemetry", {}).get("measurement_window", {})
    start, end = window.get("start"), window.get("end")
    for request in result.get("requests", []):
        timeline = request.get("timeline")
        if not timeline:
            continue
        if any(timeline.get(name) is None for name in (
            "t_submit", "t_prefill_first_scheduled", "t_prefill_finish",
            "t_first_token", "t_finish",
        )):
            continue
        if start is not None and timeline["t_submit"] < start:
            continue
        if end is not None and timeline["t_finish"] >= end:
            continue
        yield request


def _request_rows(results: list[dict]):
    rows = []
    for result in results:
        config = result["config"]
        for request in _measurement_requests(result):
            timeline = request["timeline"]
            submit = timeline["t_submit"]
            row = {
                "point_id": result["point_id"],
                "variant": config["variant"],
                "run": int(config["run"]) + 1,
                "request_id": request.get("request_id"),
                "request_class": request.get("request_class"),
                "engine_ttft_ms": request.get("engine_ttft_ms") or _ms(
                    submit, timeline.get("t_first_token")
                ),
                "decode_ready_ms": request.get("decode_ready_ms") or _ms(
                    submit, timeline.get("t_handoff_finish")
                ),
                "prefill_queue_wait_ms": _ms(
                    submit, timeline.get("t_prefill_first_scheduled")
                ),
                "prefill_forward_wall_ms": _ms(
                    timeline.get("t_prefill_first_scheduled"),
                    timeline.get("t_prefill_finish"),
                ),
                "kv_handoff_ms": _ms(
                    timeline.get("t_prefill_finish"),
                    timeline.get("t_handoff_finish"),
                ),
                "decode_admission_offset_ms": _ms(
                    timeline.get("t_prefill_finish"),
                    timeline.get("t_decode_admitted"),
                ),
                "tpot_ms": request.get("tpot_ms"),
                "e2e_ms": request.get("e2e_ms"),
                "first_token_writer": timeline.get("writers", {}).get(
                    "t_first_token"
                ),
                "submit_writer": timeline.get("writers", {}).get("t_submit"),
            }
            rows.append(row)
    return rows


def _summaries(rows: list[dict], results: list[dict]):
    by_variant = defaultdict(list)
    for row in rows:
        by_variant[row["variant"]].append(row)
    by_variant_results = defaultdict(list)
    for result in results:
        by_variant_results[result["config"]["variant"]].append(result)

    latency_names = (
        "engine_ttft_ms", "decode_ready_ms", "prefill_queue_wait_ms",
        "prefill_forward_wall_ms", "kv_handoff_ms", "tpot_ms", "e2e_ms",
    )
    rows_out = []
    for variant, requests in sorted(by_variant.items()):
        runs = by_variant_results[variant]
        request_tps = [
            item["metrics"]["throughput"]["requests_per_second"]
            for item in runs
        ]
        output_tps = [
            item["metrics"]["throughput"]["output_tokens_per_second"]
            for item in runs
        ]
        for metric in latency_names:
            values = [row[metric] for row in requests if row.get(metric) is not None]
            pooled = summarize_values(values)
            per_run_p50 = []
            per_run_p99 = []
            for run in range(1, 4):
                run_values = [
                    row[metric] for row in requests
                    if row["run"] == run and row.get(metric) is not None
                ]
                run_summary = summarize_values(run_values)
                if run_summary["p50"] is not None:
                    per_run_p50.append(run_summary["p50"])
                    per_run_p99.append(run_summary["p99"])
            rows_out.append({
                "variant": variant,
                "metric": metric,
                **pooled,
                "run_p50_min": min(per_run_p50) if per_run_p50 else None,
                "run_p50_max": max(per_run_p50) if per_run_p50 else None,
                "run_p99_min": min(per_run_p99) if per_run_p99 else None,
                "run_p99_max": max(per_run_p99) if per_run_p99 else None,
                "request_tps_mean": summarize_values(request_tps)["mean"],
                "request_tps_min": min(request_tps) if request_tps else None,
                "request_tps_max": max(request_tps) if request_tps else None,
                "output_tps_mean": summarize_values(output_tps)["mean"],
                "output_tps_min": min(output_tps) if output_tps else None,
                "output_tps_max": max(output_tps) if output_tps else None,
            })
    return rows_out


def _scheduler_rows(results: list[dict]):
    rows = []
    for result in results:
        config = result["config"]
        for index, step in enumerate(result.get("telemetry", {}).get("scheduler_steps", [])):
            row = {
                "point_id": result["point_id"],
                "variant": config["variant"],
                "run": int(config["run"]) + 1,
                "step": index,
            }
            row.update(step)
            rows.append(row)
    return rows


def _handoff_rows(results: list[dict]):
    rows = []
    for result in results:
        config = result["config"]
        for index, batch in enumerate(
            result.get("metrics", {}).get("pd", {}).get("prefill_batches_detail", [])
        ):
            row = {
                "point_id": result["point_id"],
                "variant": config["variant"],
                "run": int(config["run"]) + 1,
                "batch": index,
            }
            for name, value in batch.items():
                if isinstance(value, (list, dict)):
                    row[name] = json.dumps(value, sort_keys=True)
                else:
                    row[name] = value
            rows.append(row)
    return rows


def _validate_shared_transport(results: list[dict]):
    shared = [
        result for result in results
        if result["config"].get("variant") == "pd-shared"
    ]
    if len(shared) != 3:
        raise ValueError("expected three complete pd-shared runs")
    for result in shared:
        runtime = result["config"]["runtime"]
        if not runtime.get("pd") or runtime.get("kv_slot_count") != 2:
            raise ValueError("pd-shared runtime does not use two shared KV slots")
        pd_metrics = result.get("metrics", {}).get("pd", {})
        if pd_metrics.get("fatal_error"):
            raise ValueError("pd-shared worker reported a fatal error")
        batches = pd_metrics.get("prefill_batches_detail", [])
        if not batches:
            raise ValueError("pd-shared run recorded no KV handoffs")
        transports = []
        for batch in batches:
            transports.extend(batch.get("transports") or [batch.get("transport")])
        if any(transport != "shared_slot" for transport in transports):
            raise ValueError("pd-shared run used inline or unknown KV transport")
        releases = pd_metrics.get("slot_release_samples", [])
        if not releases:
            raise ValueError("pd-shared run did not record slot ACK release")
        final = releases[-1].get("slot_stats", {})
        if (
            final.get("free_slots") != runtime["kv_slot_count"]
            or final.get("ready_slots") != 0
            or final.get("consuming_slots") != 0
            or final.get("pending_transfers") != 0
        ):
            raise ValueError("pd-shared final slot state is not fully released")


def _chunked_validation_text(results_dir: Path | None):
    if results_dir is None:
        return "No separate chunked-prefill validation was supplied."
    results = [
        _read_json(path)
        for path in sorted((Path(results_dir) / "runs").glob("*.json"))
    ]
    results = [item for item in results if item.get("complete")]
    if not results:
        return "No complete chunked-prefill validation result was supplied."
    result = results[0]
    steps = result.get("telemetry", {}).get("scheduler_steps", [])
    partial_lengths = [
        length
        for step in steps
        for length in step.get("prefill_chunk_lengths", [])
    ]
    mixed_steps = sum(
        bool(step.get("prefill_request_count")) and bool(step.get("decode_request_count"))
        for step in steps
    )
    short_requests = [
        request for request in result.get("requests", [])
        if request.get("request_class") == "short" and request.get("engine_ttft_ms") is not None
    ]
    ttft = summarize_values([item["engine_ttft_ms"] for item in short_requests])
    tpot = summarize_values([
        item["tpot_ms"] for item in short_requests if item.get("tpot_ms") is not None
    ])
    _write_csv(
        Path(results_dir) / "chunked_scheduler_steps.csv",
        [dict(step=index, **step) for index, step in enumerate(steps)],
    )
    return (
        f"- `chunked_prefill=true`; partial chunks: **{len(partial_lengths)}**; "
        f"lengths: `{partial_lengths}`.\n"
        f"- Mixed prefill/decode rounds: **{mixed_steps}**.\n"
        f"- Short requests: Engine TTFT P50/P99 = "
        f"{ttft['p50']:.2f}/{ttft['p99']:.2f} ms; TPOT P50/P99 = "
        f"{tpot['p50']:.2f}/{tpot['p99']:.2f} ms."
    )


def _summary_table(summary_rows: list[dict]):
    by_variant = defaultdict(dict)
    for row in summary_rows:
        by_variant[row["variant"]][row["metric"]] = row
    lines = [
        "| Variant | Output tok/s mean (min-max) | Req/s mean (min-max) | Engine TTFT P50/P99 | TPOT P50/P99 | E2E P50/P99 |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for variant, metrics in sorted(by_variant.items()):
        throughput = metrics["engine_ttft_ms"]
        def pair(name):
            item = metrics.get(name, {})
            return f"{item.get('p50', 0):.2f}/{item.get('p99', 0):.2f} ms"
        lines.append(
            f"| {variant} | {throughput['output_tps_mean']:.2f} "
            f"({throughput['output_tps_min']:.2f}-{throughput['output_tps_max']:.2f}) | "
            f"{throughput['request_tps_mean']:.2f} "
            f"({throughput['request_tps_min']:.2f}-{throughput['request_tps_max']:.2f}) | "
            f"{pair('engine_ttft_ms')} | {pair('tpot_ms')} | {pair('e2e_ms')} |"
        )
    return "\n".join(lines)


def build_report(
    results_dir: Path,
    *,
    chunked_results_dir: Path | None = None,
    reproduction_command: str | None = None,
):
    results_dir = Path(results_dir)
    results = [
        _read_json(path)
        for path in sorted((results_dir / "runs").glob("*.json"))
    ]
    if not results or any(not item.get("complete") for item in results):
        raise ValueError("formal result directory does not contain only complete runs")
    variants = {item["config"].get("variant") for item in results}
    if variants != {"strong-collocated", "pd-shared"}:
        raise ValueError("report requires only strong-collocated and pd-shared runs")
    _validate_shared_transport(results)
    request_rows = _request_rows(results)
    if not request_rows:
        raise ValueError("no complete measurement-window request timelines found")
    summary_rows = _summaries(request_rows, results)
    scheduler_rows = _scheduler_rows(results)
    handoff_rows = _handoff_rows(results)
    _write_csv(results_dir / "request_timeline.csv", request_rows)
    _write_csv(results_dir / "pd_handoff.csv", handoff_rows)
    _write_csv(results_dir / "scheduler_steps.csv", scheduler_rows)
    _write_csv(results_dir / "derived_summary.csv", summary_rows)

    config_rows = []
    for result in results:
        config_rows.append({
            "point_id": result["point_id"],
            "variant": result["config"]["variant"],
            "run": int(result["config"]["run"]) + 1,
            "runtime": json.dumps(result["config"]["runtime"], sort_keys=True),
            "effective_runtime": json.dumps(
                result.get("telemetry", {}).get("effective_runtime", {}), sort_keys=True
            ),
            "gpu_clocks": json.dumps(
                result.get("telemetry", {}).get("gpu_clocks", {}), sort_keys=True
            ),
        })
    _write_csv(results_dir / "runtime_configuration.csv", config_rows)

    manifest = _read_json(results_dir / "manifest.json")
    metadata = results[0].get("metadata", {})
    command = reproduction_command or "see reproduction_commands.txt"
    (results_dir / "reproduction_commands.txt").write_text(
        command.rstrip() + "\n"
    )
    chunked_text = _chunked_validation_text(chunked_results_dir)
    report = f"""# PD + Shared KV vs Strong Collocated Baseline

## Scope

This report is generated from the raw JSON under this directory only.

- Git commit: `{manifest.get('git_commit')}`
- Model: `{manifest.get('model')}` revision `{manifest.get('model_revision')}`
- Complete points: `{manifest.get('completed_points')}/{manifest.get('total_points')}`
- Execution order seed: `{manifest.get('execution_order_seed')}`; the exact order is in `manifest.json`.
- Reproduction command: `{command}`

## Deployments

- **A / strong-collocated**: GPU0, one BF16 Qwen3-8B process. Paged KV and block allocation/reclamation, continuous batching, decode-first chunked prefill, capacity admission, structured scheduler output, ordinary Decode CUDA Graph, and lifecycle telemetry are enabled.
- **C / pd-shared**: Prefill on GPU0 and Decode on GPU1, two BF16 model copies. KV handoff uses two pinned shared-memory slots, descriptor-only messages, ACK release/backpressure, and Decode CUDA Graph. Prefill also has chunked-prefill enabled; because it serves no decode sequences, it does not form mixed Prefill/Decode batches itself.
- Both use `128 input / 64 output`, `temperature=0.01`, `ignore_eos=true`, `max_num_batched_tokens=1024`, `max_num_seqs=128`, capacity admission, `max_model_len=512`, and c=64 closed-loop (30 s warmup, 60 s measurement). Workload prompts are independently generated; no shared prefix is used.

Topology and P2P metadata are retained in each run JSON and `runtime_configuration.csv`. Detected topology: `{metadata.get('nvidia_smi_topology')}`. CUDA peer access: `{metadata.get('cuda_peer_access')}`.

## Metric Definitions

- **Engine TTFT** = `t_first_token - t_submit`: host-monotonic time from the benchmark calling `add_request` to the model sampling the first output token. It is an engine-internal first-token measurement, **not a client-receipt timestamp** because this API is non-streaming.
- **E2E** = `t_finish - t_submit`.
- **TPOT** = `(t_finish - first-token time) / (output tokens - 1)`; output length one uses 0 ms.
- Output throughput is generated output tokens within the 60-second measurement window divided by 60; request throughput is completed requests divided by 60.
- `decode_ready_ms`, KV handoff, and Decode admission are internal diagnosis fields only. In C the scheduler admission currently precedes the blocking KV import, so their raw ordering is preserved rather than forced into an invalid serial TTFT equation.

## Main Results

{_summary_table(summary_rows)}

P50/P99 pool measurement-window requests. Per-run P50/P99 ranges and three-run throughput min/max are in `derived_summary.csv`; raw request records are in `request_timeline.csv`.

## Shared KV Validation

All C runs passed the report validator: every recorded handoff used `shared_slot`; each run recorded ACK release; final slot state was fully free with no ready, consuming, or pending transfer. `pd_handoff.csv` retains every batch's prefill, export, admission, handoff, descriptor and slot record. No alternative tensor transport is present in this comparison.

## Scheduling And Chunked Prefill

`scheduler_steps.csv` records per-round waiting/running lengths, prefill/decode token and request counts, remaining budget, actual partial chunks, and CUDA Event forward time. The short main workload enables chunked prefill but has 128-token prompts below its 1024-token budget; whether it actually emitted a partial chunk is therefore reported separately from whether the capability was enabled.

### Long/Short Validation (A Only)

{chunked_text}

## Conclusion And Limits

**C relative to A is the complete deployment gain of PD+Shared KV relative to the strong single-GPU Collocated Runtime. It includes the additional GPU and model copy, resource isolation, the PD pipeline, and Shared KV; it is not attributed as a pure PD, pure Shared KV, or pure hardware-bandwidth gain.**

This is a complete deployment comparison, not an isolation experiment for any one subsystem.

This run records host queueing, CUDA Event model-forward durations, KV handoff, Decode admission, and GPU clock ranges. It does not include NCU counters, so it does not claim SM or DRAM saturation.
"""
    report_path = results_dir / "pd_strong_baseline_report.md"
    report_path.write_text(report)
    return report_path


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", required=True)
    parser.add_argument("--chunked-results-dir")
    parser.add_argument("--reproduction-command")
    args = parser.parse_args(argv)
    build_report(
        Path(args.results_dir),
        chunked_results_dir=(
            Path(args.chunked_results_dir) if args.chunked_results_dir else None
        ),
        reproduction_command=args.reproduction_command,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
