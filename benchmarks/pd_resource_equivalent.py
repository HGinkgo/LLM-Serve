"""Summarize equal-resource Dual Collocated versus PD + Shared results."""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from statistics import mean, median

from benchmarks.metrics import summarize_values


VARIANTS = ("strong-collocated", "dual-collocated", "pd-shared")
CORE_RUNTIME_FIELDS = (
    "max_model_len",
    "max_num_batched_tokens",
    "max_num_seqs",
    "gpu_memory_utilization",
    "enable_chunked_prefill",
    "enable_kv_capacity_admission",
    "random_seed",
)


def _read_json(path: Path):
    return json.loads(path.read_text())


def _write_csv(path: Path, rows: list[dict]):
    if not rows:
        path.write_text("")
        return
    fields = sorted({field for row in rows for field in row})
    with path.open("w", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def _nested(value: dict, path: tuple[str, ...]):
    for key in path:
        value = value.get(key) if isinstance(value, dict) else None
        if value is None:
            return None
    return value


def _mean_range(values):
    values = [float(value) for value in values if value is not None]
    return {
        "mean": mean(values) if values else None,
        "median": median(values) if values else None,
        "min": min(values) if values else None,
        "max": max(values) if values else None,
    }


def _format(value, digits=2, suffix=""):
    return "n/a" if value is None else f"{value:.{digits}f}{suffix}"


def _validate_results(results):
    if not results or any(not result.get("complete") for result in results):
        raise ValueError("result directory must contain only complete runs")
    variants = {result["config"].get("variant") for result in results}
    if variants != set(VARIANTS):
        raise ValueError("result directory must contain A, B, and C variants")

    for result in results:
        runtime = result["config"]["runtime"]
        variant = result["config"]["variant"]
        graph = result.get("metrics", {}).get("cuda_graph", {})
        if (
            not graph.get("enabled")
            or not graph.get("captured_graphs")
            or not graph.get("replays")
        ):
            raise ValueError(
                f"{variant} result has no verified CUDA Graph capture/replay"
            )
        if variant == "dual-collocated":
            if not runtime.get("dual_collocated") or runtime.get("pd"):
                raise ValueError("dual-collocated result is not an independent replica deployment")
            if result.get("telemetry", {}).get("effective_runtime", {}).get("kind") != "dual_collocated":
                raise ValueError("dual-collocated effective runtime was not selected")
            replicas = graph.get("replicas", {})
            if set(replicas) != {"replica-0", "replica-1"} or any(
                not replica.get("enabled")
                or not replica.get("captured_graphs")
                or not replica.get("replays")
                for replica in replicas.values()
            ):
                raise ValueError(
                    "dual-collocated replicas require verified CUDA Graph replay"
                )
        if variant == "pd-shared":
            if not runtime.get("pd") or runtime.get("dual_collocated"):
                raise ValueError("pd-shared result is not a PD deployment")
            batches = result.get("metrics", {}).get("pd", {}).get("prefill_batches_detail", [])
            if not batches:
                raise ValueError("pd-shared result did not record any handoff")
            transports = [
                transport
                for batch in batches
                for transport in batch.get("transports", [batch.get("transport")])
            ]
            if any(transport != "shared_slot" for transport in transports):
                raise ValueError("pd-shared result must use shared_slot only")
            releases = result["metrics"]["pd"].get("slot_release_samples", [])
            if not releases:
                raise ValueError("pd-shared result did not record slot ACK release")
            final = releases[-1].get("slot_stats", {})
            if any(final.get(field) != expected for field, expected in (
                ("free_slots", runtime["kv_slot_count"]),
                ("ready_slots", 0),
                ("consuming_slots", 0),
                ("pending_transfers", 0),
            )):
                raise ValueError("pd-shared result has unreleased shared slots")

    grouped = defaultdict(dict)
    for result in results:
        key = (result["config"]["experiment"], result["config"]["run"])
        grouped[key][result["config"]["variant"]] = result
    for variants_by_run in grouped.values():
        if set(variants_by_run) != set(VARIANTS):
            raise ValueError("each experiment/run must include A, B, and C")
        baseline_config = variants_by_run["strong-collocated"]["config"]
        baseline_runtime = baseline_config["runtime"]
        for variant in ("dual-collocated", "pd-shared"):
            runtime = variants_by_run[variant]["config"]["runtime"]
            for field in CORE_RUNTIME_FIELDS:
                if baseline_runtime.get(field) != runtime.get(field):
                    raise ValueError(f"A/{variant} differ on runtime field {field}")
            candidate_config = variants_by_run[variant]["config"]
            for field in ("model", "model_revision", "sampling"):
                if baseline_config.get(field) != candidate_config.get(field):
                    raise ValueError(f"A/{variant} differ on config field {field}")


def _trace_audit(results):
    grouped = defaultdict(dict)
    for result in results:
        key = (result["config"]["experiment"], result["config"]["run"])
        grouped[key][result["config"]["variant"]] = result
    audits = []
    for (experiment, run), variants in sorted(grouped.items()):
        traces = {
            name: value.get("telemetry", {}).get("request_trace", {})
            for name, value in variants.items()
        }
        reference = variants["strong-collocated"]["config"]
        if any(
            value["config"].get("workload_seed") != reference.get("workload_seed")
            or value["config"].get("workload") != reference.get("workload")
            for value in variants.values()
        ):
            raise ValueError("A/B/C do not use the same deterministic workload")
        common = min(len(trace.get("entries", [])) for trace in traces.values())
        reference_entries = traces["strong-collocated"].get("entries", [])[:common]
        if any(
            trace.get("entries", [])[:common] != reference_entries
            for trace in traces.values()
        ):
            raise ValueError("A/B/C request trace prefixes differ")
        classes = reference.get("workload", {}).get("classes", [])
        trace_order = reference.get("workload", {}).get("trace_order")
        if (
            trace_order in {"interleaved", "balanced_interleaved"}
            and len(classes) == 2
            and classes[0].get("weight") == classes[1].get("weight")
        ):
            pattern = (0, 1) if trace_order == "interleaved" else (0, 1, 1, 0)
            expected_classes = [
                classes[pattern[index % len(pattern)]]["name"]
                for index in range(common)
            ]
            actual_classes = [
                entry.get("request_class") for entry in reference_entries
            ]
            if actual_classes != expected_classes:
                raise ValueError(
                    "interleaved trace does not preserve the declared class order"
                )
        audits.append({
            "experiment": experiment,
            "run": int(run) + 1,
            "workload_seed": reference.get("workload_seed"),
            "trace_order": reference.get("workload", {}).get("trace_order"),
            "common_prefix_requests": common,
            "strong_collocated_trace_sha256": traces["strong-collocated"].get("sha256"),
            "dual_collocated_trace_sha256": traces["dual-collocated"].get("sha256"),
            "pd_shared_trace_sha256": traces["pd-shared"].get("sha256"),
        })
    return audits


def _summary_rows(results):
    rows = []
    groups = defaultdict(list)
    for result in results:
        groups[(result["config"]["experiment"], result["config"]["variant"])].append(result)
    for (experiment, variant), runs in sorted(groups.items()):
        metric_paths = {
            "request_throughput_rps": ("metrics", "throughput", "requests_per_second"),
            "output_throughput_tps": ("metrics", "throughput", "output_tokens_per_second"),
        }
        for request_class in ("overall", "short", "long"):
            for metric in ("ttft", "tpot", "e2e"):
                for percentile in ("p50", "p99"):
                    metric_paths[f"{request_class}_{metric}_{percentile}_ms"] = (
                        "metrics", "latency", request_class, metric, percentile
                    )
        for name, path in metric_paths.items():
            scale = 1000 if name.endswith("_ms") else 1
            values = [
                _nested(result, path) * scale
                for result in runs
                if _nested(result, path) is not None
            ]
            summary = _mean_range(values)
            rows.append({
                "experiment": experiment,
                "variant": variant,
                "metric": name,
                "runs": len(values),
                **summary,
            })
    return rows


def _summary_lookup(rows):
    return {
        (row["experiment"], row["variant"], row["metric"]): row
        for row in rows
    }


def _ratio(summary, experiment, numerator, denominator, metric):
    top = summary[(experiment, numerator, metric)]["mean"]
    bottom = summary[(experiment, denominator, metric)]["mean"]
    return top / bottom if top is not None and bottom not in (None, 0) else None


def _pd_stage_rows(results):
    rows = []
    for result in results:
        if result["config"]["variant"] != "pd-shared":
            continue
        window = result.get("telemetry", {}).get("measurement_window", {})
        start, end = window.get("start"), window.get("end")
        duration = end - start if start is not None and end is not None else None
        requests = result.get("requests", [])
        waits = []
        prefill_finished = 0
        long_prefill = []
        short_decode = []
        for request in requests:
            timeline = request.get("timeline") or {}
            submit = timeline.get("t_submit")
            first_scheduled = timeline.get("t_prefill_first_scheduled")
            prefill_finish = timeline.get("t_prefill_finish")
            decode_admitted = timeline.get("t_decode_admitted")
            finish = timeline.get("t_finish")
            if submit is not None and first_scheduled is not None and start <= submit < end:
                waits.append((first_scheduled - submit) * 1000)
            if prefill_finish is not None and start <= prefill_finish < end:
                prefill_finished += 1
            if (
                request.get("request_class") == "long"
                and first_scheduled is not None
                and prefill_finish is not None
            ):
                long_prefill.append((
                    max(start, first_scheduled), min(end, prefill_finish)
                ))
            if (
                request.get("request_class") == "short"
                and decode_admitted is not None
                and finish is not None
            ):
                short_decode.append((
                    max(start, decode_admitted), min(end, finish)
                ))
        overlap_pairs = sum(
            max(prefill_start, decode_start) < min(prefill_end, decode_end)
            for prefill_start, prefill_end in long_prefill
            for decode_start, decode_end in short_decode
        )
        pd = result["metrics"]["pd"]
        queue_samples = pd.get("queue_samples", [])
        idle = pd.get("decode_idle", {})
        idle_in_window = defaultdict(float)
        for interval in idle.get("intervals", []):
            clipped_start = max(start, interval.get("started_at", start))
            clipped_end = min(end, interval.get("finished_at", end))
            if clipped_end > clipped_start:
                idle_in_window[interval.get("reason", "unknown")] += (
                    clipped_end - clipped_start
                ) * 1000
        rows.append({
            "experiment": result["config"]["experiment"],
            "run": int(result["config"]["run"]) + 1,
            "prefill_queue_wait_p50_ms": summarize_values(waits)["p50"],
            "prefill_queue_wait_p99_ms": summarize_values(waits)["p99"],
            "prefill_service_rps": prefill_finished / duration if duration else None,
            "decode_output_service_tps": _nested(
                result, ("metrics", "throughput", "output_tokens_per_second")
            ),
            "pending_prefill_p99": summarize_values([
                item.get("pending_prefill_requests", 0) for item in queue_samples
            ])["p99"],
            "active_decode_p99": summarize_values([
                item.get("active_decode_requests", 0) for item in queue_samples
            ])["p99"],
            "decode_idle_ms_by_reason": json.dumps(dict(idle_in_window), sort_keys=True),
            "long_prefill_short_decode_overlap_pairs": overlap_pairs,
        })
    return rows


def _replica_rows(results):
    rows = []
    for result in results:
        if result["config"]["variant"] != "dual-collocated":
            continue
        routing = result["metrics"].get("dual_collocated", {})
        assigned = routing.get("assigned_requests", {})
        states = defaultdict(lambda: {"running": [], "decode": []})
        for step in result.get("telemetry", {}).get("scheduler_steps", []):
            for worker_id, state in step.get("replica_queue_state", {}).items():
                states[worker_id]["running"].append(state.get("running_queue_size", 0))
                states[worker_id]["decode"].append(state.get("decode_request_count", 0))
        for worker_id in ("replica-0", "replica-1"):
            replica_requests = [
                request
                for request in result.get("requests", [])
                if request.get("replica_id") == worker_id
            ]
            rows.append({
                "experiment": result["config"]["experiment"],
                "run": int(result["config"]["run"]) + 1,
                "replica": worker_id,
                "assigned_requests": assigned.get(worker_id, 0),
                "short_requests": sum(
                    request.get("request_class") == "short"
                    for request in replica_requests
                ),
                "long_requests": sum(
                    request.get("request_class") == "long"
                    for request in replica_requests
                ),
                "running_sequences_mean": summarize_values(states[worker_id]["running"])["mean"],
                "decode_scheduled_sequences_mean": summarize_values(states[worker_id]["decode"])["mean"],
            })
    return rows


def _chunk_rows(results):
    rows = []
    for result in results:
        steps = result.get("telemetry", {}).get("scheduler_steps", [])
        pd_batches = result.get("metrics", {}).get("pd", {}).get(
            "prefill_batches_detail", []
        )
        replica_events = [
            event
            for step in steps
            for event in step.get("replica_events", {}).values()
        ]
        scheduler_events = replica_events or steps
        rows.append({
            "experiment": result["config"]["experiment"],
            "variant": result["config"]["variant"],
            "run": int(result["config"]["run"]) + 1,
            "chunked_prefill_enabled": result["config"]["runtime"].get(
                "enable_chunked_prefill"
            ),
            "scheduler_partial_chunk_count": sum(
                int(step.get("partial_prefill_chunk_count", 0))
                for step in scheduler_events
            ),
            "scheduler_partial_chunk_lengths": json.dumps([
                length
                for step in scheduler_events
                for length in (step.get("prefill_chunk_lengths") or [])
            ]),
            "scheduler_local_mixed_prefill_decode_rounds": sum(
                bool(step.get("prefill_request_count"))
                and bool(step.get("decode_request_count"))
                for step in scheduler_events
            ),
            "frontend_concurrent_prefill_decode_rounds": sum(
                bool(step.get("prefill_request_count"))
                and bool(step.get("decode_request_count"))
                for step in steps
            ),
            "pd_prefill_partial_chunk_count": sum(
                int(batch.get("partial_prefill_chunk_count") or 0)
                for batch in pd_batches
            ),
            "pd_prefill_chunk_lengths": json.dumps([
                length
                for batch in pd_batches
                for length in (batch.get("partial_prefill_chunk_lengths") or [])
            ]),
        })
    return rows


def _replica_table(rows):
    lines = [
        "| Experiment | Run | Replica | Assigned requests | Short/long requests | Mean running sequences | Mean Decode-scheduled sequences |",
        "|---|---:|---|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| {row['experiment']} | {row['run']} | {row['replica']} | "
            f"{row['assigned_requests']} | {row['short_requests']}/{row['long_requests']} | "
            f"{_format(row['running_sequences_mean'])} | "
            f"{_format(row['decode_scheduled_sequences_mean'])} |"
        )
    return "\n".join(lines)


def _pd_stage_table(rows):
    lines = [
        "| Experiment | Run | Prefill queue wait P50/P99 | Prefill service req/s | Decode output tok/s | Pending Prefill P99 | Active Decode P99 | Decode idle ms by reason | Long Prefill / short Decode overlap pairs |",
        "|---|---:|---:|---:|---:|---:|---:|---|---:|",
    ]
    for row in rows:
        lines.append(
            f"| {row['experiment']} | {row['run']} | "
            f"{_format(row['prefill_queue_wait_p50_ms'])}/{_format(row['prefill_queue_wait_p99_ms'])} ms | "
            f"{_format(row['prefill_service_rps'])} | "
            f"{_format(row['decode_output_service_tps'])} | "
            f"{_format(row['pending_prefill_p99'])} | "
            f"{_format(row['active_decode_p99'])} | "
            f"{row['decode_idle_ms_by_reason']} | "
            f"{row['long_prefill_short_decode_overlap_pairs']} |"
        )
    return "\n".join(lines)


def _chunk_table(rows):
    lines = [
        "| Experiment | Variant | Run | Scheduler partial chunks | Scheduler-local mixed rounds | Frontend concurrent P/D rounds | PD Prefill partial chunks |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| {row['experiment']} | {row['variant']} | {row['run']} | "
            f"{row['scheduler_partial_chunk_count']} | "
            f"{row['scheduler_local_mixed_prefill_decode_rounds']} | "
            f"{row['frontend_concurrent_prefill_decode_rounds']} | "
            f"{row['pd_prefill_partial_chunk_count']} |"
        )
    return "\n".join(lines)


def _table(summary, experiment):
    lookup = _summary_lookup(summary)
    lines = [
        "| Variant | Output tok/s | Req/s | Engine TTFT P50/P99 | TPOT P50/P99 | E2E P50/P99 |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for variant in VARIANTS:
        def value(metric):
            return lookup[(experiment, variant, metric)]
        def pair(prefix):
            return f"{_format(value(f'{prefix}_p50_ms')['mean'])}/{_format(value(f'{prefix}_p99_ms')['mean'])} ms"
        output = value("output_throughput_tps")
        request = value("request_throughput_rps")
        lines.append(
            f"| {variant} | {_format(output['mean'])} [med {_format(output['median'])}; {_format(output['min'])}-{_format(output['max'])}] | "
            f"{_format(request['mean'])} [med {_format(request['median'])}; {_format(request['min'])}-{_format(request['max'])}] | "
            f"{pair('overall_ttft')} | {pair('overall_tpot')} | {pair('overall_e2e')} |"
        )
    return "\n".join(lines)


def _class_table(summary, experiment):
    lookup = _summary_lookup(summary)
    lines = [
        "| Variant | Short TTFT P50/P99 | Short TPOT P50/P99 | Short E2E P50/P99 | Long TTFT P50/P99 | Long TPOT P50/P99 | Long E2E P50/P99 |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for variant in VARIANTS:
        def pair(request_class, metric):
            p50 = lookup[(experiment, variant, f"{request_class}_{metric}_p50_ms")]["mean"]
            p99 = lookup[(experiment, variant, f"{request_class}_{metric}_p99_ms")]["mean"]
            return f"{_format(p50)}/{_format(p99)} ms"
        lines.append(
            f"| {variant} | {pair('short', 'ttft')} | {pair('short', 'tpot')} | "
            f"{pair('short', 'e2e')} | {pair('long', 'ttft')} | "
            f"{pair('long', 'tpot')} | {pair('long', 'e2e')} |"
        )
    return "\n".join(lines)


def build_report(results_dir: Path):
    results_dir = Path(results_dir)
    results = [_read_json(path) for path in sorted((results_dir / "runs").glob("*.json"))]
    _validate_results(results)
    summary = _summary_rows(results)
    stage_rows = _pd_stage_rows(results)
    replica_rows = _replica_rows(results)
    chunk_rows = _chunk_rows(results)
    trace_audit = _trace_audit(results)
    _write_csv(results_dir / "resource_equivalent_summary.csv", summary)
    _write_csv(results_dir / "pd_stage_diagnostics.csv", stage_rows)
    _write_csv(results_dir / "dual_collocated_replicas.csv", replica_rows)
    _write_csv(results_dir / "chunked_prefill_evidence.csv", chunk_rows)
    (results_dir / "trace_audit.json").write_text(
        json.dumps(trace_audit, indent=2, sort_keys=True) + "\n"
    )

    experiments = sorted({result["config"]["experiment"] for result in results})
    run_count = len({result["config"]["run"] for result in results})
    lookup = _summary_lookup(summary)
    comparison_lines = []
    for experiment in experiments:
        for label, numerator, denominator in (
            ("B vs A", "dual-collocated", "strong-collocated"),
            ("C vs B", "pd-shared", "dual-collocated"),
            ("C vs A", "pd-shared", "strong-collocated"),
        ):
            comparison_lines.append(
                f"- {experiment} {label}: output throughput "
                f"`{_format(_ratio(lookup, experiment, numerator, denominator, 'output_throughput_tps'), 3)}x`; "
                f"request throughput `{_format(_ratio(lookup, experiment, numerator, denominator, 'request_throughput_rps'), 3)}x`."
            )
    mixed_experiment = next(
        (name for name in experiments if name.endswith("mixed")), None
    )
    mixed_class_table = _class_table(summary, mixed_experiment) if mixed_experiment else "n/a"
    report = f"""# PD + Shared Resource-Equivalent Benchmark

## Scope

This report is generated only from the raw JSON in this directory. It compares:

- **A**: one GPU, one complete Collocated Runtime.
- **B**: two independent complete Collocated replicas, one on each GPU, with frontend `round_robin` routing. There is no PD role, KV handoff, Shared Slot, or Inline Queue path.
- **C**: one Prefill GPU and one Decode GPU with the existing descriptor-only, pinned Shared KV slots and ACK/backpressure path. Every validated C handoff is `shared_slot`.

`C vs A` is a complete deployment comparison; it includes the extra GPU/model copy, resource isolation, PD pipeline, and Shared KV. `C vs B` is the equal-resource comparison. Neither is attributed to Inline or to pure Shared KV. The c=64 cap is enforced once at the frontend and is shared by both B replicas; it is not 64 requests per replica.

## Aggregate Results

{chr(10).join(f'### {experiment}{chr(10)}{_table(summary, experiment)}' for experiment in experiments)}

Values are {run_count}-run means; `med` is the {run_count}-run median and the trailing range is min/max. Engine TTFT is the engine's submit-to-first-sampled-token interval, TPOT is between first and final generated token, and E2E is submit-to-finish. Latency samples are requests submitted after warmup and finished before the measurement-window end; output throughput counts tokens emitted within the 60-second window.

## Resource-Equivalent Comparisons

{chr(10).join(comparison_lines)}

## Long/Short Mixed Results

The mixed experiment uses a deterministic 50:50 `short, long, long, short, ...` trace: short is `128 input / 64 output`; long is `2048 input / 64 output`. The fixed four-request pattern is deliberately de-correlated from B's `replica-0, replica-1, ...` routing, so each replica receives one short and one long request per pattern. Each run uses global c=64 closed-loop, 30-second warmup and 60-second measurement. C reserves `8192` tokens per Shared KV slot so a four-request all-long Prefill batch stays on the Shared Slot path.

{mixed_class_table}

## B Replica Balance

{_replica_table(replica_rows)}

`dual_collocated_replicas.csv` records the same data, including short/long request allocation. The fixed request order prevents round-robin routing from pinning one class to one GPU. A nonzero imbalance or missing replica state is reported as data, not hidden by global aggregation.

## C Stage Diagnostics

{_pd_stage_table(stage_rows)}

`pd_stage_diagnostics.csv` additionally records Decode-idle duration by reason. A positive overlap count proves that a long request was in Prefill while a short request was already admitted to Decode during the measurement window; it does not itself prove a latency improvement.

## Chunked Prefill Evidence

{_chunk_table(chunk_rows)}

`chunked_prefill_evidence.csv` retains every run's actual chunk counts and lengths. `Scheduler-local mixed` means a single Collocated scheduler step contains both Prefill and Decode; `frontend concurrent` also counts simultaneous Prefill/Decode work across B's independent replicas. In C, the role-separated Prefill worker completes its own chunk loop while Decode runs independently, so the direct coexistence evidence is the long-Prefill/short-Decode overlap count in the preceding table.

## Trace And Validity Checks

`trace_audit.json` verifies the same workload seed, profile and ordered common trace prefix for A/B/C on every run. It stores prompt hashes rather than raw prompt tokens. The report refuses to run when model revision, sampling, or core runtime settings differ; CUDA Graph has no verified capture/replay; C records a non-`shared_slot` transport; or C has a missing ACK release sample or a non-free final slot state.

## Interpretation

Use **B vs A** only for direct two-replica horizontal scaling. Use **C vs B** for the net architectural result of complete PD+Shared against direct two-GPU replica scaling. Use **C vs A** only as full-deployment gain. Neither comparison is a pure PD or pure Shared KV effect. A C loss to B means the KV handoff plus 1P+1D stage balance outweighed resource isolation for this workload; a C improvement in short-request TPOT P99 or lower Decode-idle instability is evidence for isolation, not a claim about SM/DRAM saturation.

The benchmark does not use Inline Queue, EAGLE, Tree speculation, AWQ, or a shared prefix. GPU clocks and topology remain in each raw result; clocks are not locked and no NCU counters are collected.
"""
    report_path = results_dir / "pd_resource_equivalent_report.md"
    report_path.write_text(report)
    return report_path


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", required=True)
    args = parser.parse_args(argv)
    build_report(Path(args.results_dir))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
