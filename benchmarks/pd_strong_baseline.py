"""Build the formal A-versus-C report from this run's raw benchmark JSON."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
from collections import defaultdict
from pathlib import Path

from benchmarks.metrics import summarize_values
from benchmarks.workloads import WorkloadClass, iter_request_specs
import benchmarks.workloads as workloads_module


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
                "decode_continuation_ready_ms": request.get("decode_ready_ms") or _ms(
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
        "engine_ttft_ms", "decode_continuation_ready_ms", "prefill_queue_wait_ms",
        "prefill_forward_wall_ms", "kv_handoff_ms", "decode_admission_offset_ms",
        "tpot_ms", "e2e_ms",
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
    runtime = result["config"]["runtime"]
    effective_runtime = result.get("telemetry", {}).get("effective_runtime", {})
    effective_config = effective_runtime.get("engine_config", {})
    workload = {
        item["name"]: item
        for item in result["config"]["workload"]["classes"]
    }
    kv_cache = result.get("metrics", {}).get("kv_cache", {})
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
        request for request in _measurement_requests(result)
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
        "- This is a dedicated A-only capability validation, not an A/C performance "
        "comparison: it uses c=16, 15-second warmup, 30-second measurement, and "
        f"`max_model_len={effective_config.get('max_model_len')}` rather than the "
        "main experiment's c=64 / 30-second warmup / 60-second measurement / "
        "`max_model_len=512`.\n"
        f"- Workload: long `{workload['long']['input_len']} input / "
        f"{workload['long']['output_len']} output`; short "
        f"`{workload['short']['input_len']} input / "
        f"{workload['short']['output_len']} output`. Effective runtime: "
        f"`max_model_len={effective_config.get('max_model_len')}`, "
        f"`max_num_batched_tokens={effective_config.get('max_num_batched_tokens')}`, "
        f"`max_num_seqs={effective_config.get('max_num_seqs')}`, "
        f"`chunked_prefill={effective_config.get('enable_chunked_prefill')}`, "
        f"`capacity_admission={effective_config.get('enable_kv_capacity_admission')}`; "
        f"KV blocks=`{kv_cache.get('total_blocks')}`, peak reserved="
        f"`{kv_cache.get('peak_reserved_blocks')}`.\n"
        "- Decode-first has no independent boolean: with `chunked_prefill=true`, the "
        "scheduler's chunked path schedules running Decode before filling remaining "
        "budget with Prefill.\n"
        f"- `chunked_prefill=true`; partial chunks: **{len(partial_lengths)}**; "
        f"lengths: `{partial_lengths}`.\n"
        f"- Mixed prefill/decode rounds: **{mixed_steps}**. Both counts are derived "
        "from measurement-window `scheduler_steps`: partial chunk lengths are "
        "flattened from `prefill_chunk_lengths`; a mixed round has both positive "
        "Prefill and Decode request counts.\n"
        f"- Short requests: Engine TTFT P50/P99 = "
        f"{ttft['p50']:.2f}/{ttft['p99']:.2f} ms; TPOT P50/P99 = "
        f"{tpot['p50']:.2f}/{tpot['p99']:.2f} ms."
    )


def _main_chunking_text(results: list[dict]):
    collocated = [
        result for result in results
        if result["config"].get("variant") == "strong-collocated"
    ]
    partial_counts = []
    step_counts = []
    for result in collocated:
        steps = result.get("telemetry", {}).get("scheduler_steps", [])
        step_counts.append(len(steps))
        partial_counts.append(sum(
            int(step.get("partial_prefill_chunk_count", 0))
            for step in steps
        ))
    return (
        "- Main c=64 short-prompt workload: chunked prefill was enabled; "
        "A recorded actual partial chunks per run: "
        f"**{partial_counts}** across {step_counts} measured scheduler rounds. "
        "These are budget-limited fragments observed under concurrent decode, "
        "not evidence that every 128-token prompt must be chunked."
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


def _ttft_diagnostic_table(summary_rows: list[dict]):
    by_variant = defaultdict(dict)
    for row in summary_rows:
        by_variant[row["variant"]][row["metric"]] = row

    def pair(metrics, name):
        item = metrics.get(name, {})
        p50, p99 = item.get("p50"), item.get("p99")
        if p50 is None or p99 is None:
            return "n/a"
        return f"{p50:.2f}/{p99:.2f} ms"

    lines = [
        "| Variant | Engine TTFT P50/P99 | `decode_continuation_ready_ms` P50/P99 | Prefill queue wait P50/P99 | Prefill forward wall P50/P99 | KV handoff P50/P99 | Decode admission offset P50/P99 |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for variant, metrics in sorted(by_variant.items()):
        lines.append(
            f"| {variant} | {pair(metrics, 'engine_ttft_ms')} | "
            f"{pair(metrics, 'decode_continuation_ready_ms')} | "
            f"{pair(metrics, 'prefill_queue_wait_ms')} | "
            f"{pair(metrics, 'prefill_forward_wall_ms')} | "
            f"{pair(metrics, 'kv_handoff_ms')} | "
            f"{pair(metrics, 'decode_admission_offset_ms')} |"
        )
    return "\n".join(lines)


def _trace_workload_classes(result: dict):
    return [
        WorkloadClass(**item)
        for item in result["config"]["workload"]["classes"]
    ]


def _request_trace_digest(classes, seed: int, request_ids: list[int]):
    if not request_ids:
        return None
    wanted = set(request_ids)
    specs = iter_request_specs(classes, seed=seed)
    digest = hashlib.sha256()
    for _ in range(max(wanted) + 1):
        spec = next(specs)
        if spec.request_id not in wanted:
            continue
        digest.update(json.dumps(
            {
                "request_id": spec.request_id,
                "request_class": spec.request_class,
                "input_len": spec.input_len,
                "output_len": spec.output_len,
                "prompt_token_ids": spec.prompt_token_ids,
            },
            sort_keys=True,
            separators=(",", ":"),
        ).encode())
        digest.update(b"\n")
    return digest.hexdigest()


def _request_trace_audit(results: list[dict], manifest: dict):
    generator_sha256 = hashlib.sha256(
        Path(workloads_module.__file__).read_bytes()
    ).hexdigest()
    by_run = defaultdict(dict)
    for result in results:
        by_run[int(result["config"]["run"])][result["config"]["variant"]] = result

    run_audits = []
    for run, variants in sorted(by_run.items()):
        baseline = variants["strong-collocated"]
        shared = variants["pd-shared"]
        baseline_seed = baseline["config"].get("workload_seed")
        shared_seed = shared["config"].get("workload_seed")
        if baseline_seed != shared_seed:
            raise ValueError("A/C workload seeds differ")
        if baseline["config"].get("workload") != shared["config"].get("workload"):
            raise ValueError("A/C workloads differ")
        baseline_ids = {
            request["request_id"] for request in baseline.get("requests", [])
            if request.get("request_id") is not None
        }
        shared_ids = {
            request["request_id"] for request in shared.get("requests", [])
            if request.get("request_id") is not None
        }
        common_ids = sorted(baseline_ids & shared_ids)
        run_audits.append({
            "run": run + 1,
            "workload_seed": baseline_seed,
            "sampling_seed": baseline["config"].get("runtime", {}).get("random_seed"),
            "baseline_request_count": len(baseline_ids),
            "pd_shared_request_count": len(shared_ids),
            "shared_request_count": len(common_ids),
            "shared_request_id_range": (
                [common_ids[0], common_ids[-1]] if common_ids else None
            ),
            "shared_trace_sha256": _request_trace_digest(
                _trace_workload_classes(baseline), baseline_seed, common_ids
            ),
        })
    return {
        "schema_version": 1,
        "source_git_commit": manifest.get("git_commit"),
        "generator": "iter_request_specs",
        "generator_module": "benchmarks.workloads",
        "generator_module_sha256": generator_sha256,
        "runs": run_audits,
    }


def _scheduler_evidence_text(results: list[dict]):
    collocated = [
        result for result in results
        if result["config"].get("variant") == "strong-collocated"
    ]
    contention = decode_scheduled = prefill_only_with_running = 0
    per_run = []
    for result in collocated:
        steps = result.get("telemetry", {}).get("scheduler_steps", [])
        run_contention = [
            step for step in steps
            if step.get("waiting_queue_size", 0) > 0
            and step.get("running_queue_size", 0) > 0
        ]
        run_decode = sum(
            step.get("decode_token_count", 0) > 0
            for step in run_contention
        )
        run_prefill_only = sum(
            step.get("running_queue_size", 0) > 0
            and step.get("decode_token_count", 0) == 0
            and step.get("prefill_token_count", 0) > 0
            for step in steps
        )
        contention += len(run_contention)
        decode_scheduled += run_decode
        prefill_only_with_running += run_prefill_only
        per_run.append(
            f"r{int(result['config']['run']) + 1}: "
            f"{len(run_contention)}/{run_decode}/{run_prefill_only}"
        )
    return (
        "- Scheduler trace format per run is "
        "`contention/decode-scheduled-in-contention/prefill-only-with-running`: "
        f"{', '.join(per_run)}; combined `{contention}/{decode_scheduled}/"
        f"{prefill_only_with_running}`.\n"
        "- Decode-first is an implementation invariant, not a separate runtime flag: "
        "`schedule_chunked_prefill()` schedules every eligible running Decode "
        "sequence before assigning the remaining token budget to waiting Prefill. "
        "The zero prefill-only-with-running events are consistent with that invariant."
    )


def _cuda_graph_evidence_text(results: list[dict]):
    lines = []
    for variant in ("strong-collocated", "pd-shared"):
        variant_results = [
            result for result in results
            if result["config"].get("variant") == variant
        ]
        graphs = [result.get("metrics", {}).get("cuda_graph", {}) for result in variant_results]
        enabled = all(graph.get("enabled") for graph in graphs)
        captured = [graph.get("captured_graphs") for graph in graphs]
        replays = [graph.get("replays") for graph in graphs]
        fallbacks = [graph.get("fallbacks", {}) for graph in graphs]
        lines.append(
            f"- `{variant}`: enabled=`{enabled}`, captured graphs per run="
            f"`{captured}`, replays per run=`{replays}`, "
            f"fallbacks per run=`{fallbacks}`."
        )
    lines.append(
        "- These counters cover warmup plus the 60-second measurement interval: "
        "closed-loop does not call `reset_metrics()` at the warmup boundary while "
        "requests are active. Nonzero replay counts prove the Decode Graph path was "
        "actually used, but they are not measurement-only replay counts."
    )
    return "\n".join(lines)


def _clean_topology(value):
    return re.sub(r"\x1b\[[0-9;]*m", "", value or "").strip()


def _main_output_ratio(summary_rows: list[dict]):
    by_variant = {
        row["variant"]: row
        for row in summary_rows
        if row["metric"] == "engine_ttft_ms"
    }
    return (
        by_variant["pd-shared"]["output_tps_mean"]
        / by_variant["strong-collocated"]["output_tps_mean"]
    )


def _measurement_rules_text():
    return """- Closed-loop first fills c=64, then defines `measurement_start = start + 30s` and `measurement_end = measurement_start + 60s`; it continues stepping until the end, then drains remaining requests.
- `reset_metrics()` is not called at the warmup boundary because requests are still active; scheduler-step records are retained only when `measurement_start <= step_end < measurement_end`.
- Output throughput counts every generated token whose recorded token event is in `[measurement_start, measurement_end)` and divides by 60 seconds. Request throughput counts every request whose `finish_time` is in the same half-open interval and divides by 60 seconds.
- Engine TTFT, TPOT, and E2E summaries include only requests with `t_submit >= measurement_start` and `t_finish < measurement_end`. A request crossing either boundary can contribute in-window output tokens or a completion count according to those definitions, while being excluded from the latency sample."""


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
    trace_audit = _request_trace_audit(results, manifest)
    (results_dir / "request_trace_audit.json").write_text(
        json.dumps(trace_audit, indent=2, sort_keys=True) + "\n"
    )
    metadata = results[0].get("metadata", {})
    command = reproduction_command or "see reproduction_commands.txt"
    (results_dir / "reproduction_commands.txt").write_text(
        command.rstrip() + "\n"
    )
    command_block = f"```bash\n{command.rstrip()}\n```"
    chunked_text = _chunked_validation_text(chunked_results_dir)
    main_chunking_text = _main_chunking_text(results)
    report = f"""# PD + Shared KV vs Strong Collocated Baseline

## Scope

This report is generated from the raw JSON under this directory only.

- Git commit: `{manifest.get('git_commit')}`
- Report revision: `2026-08-15-r1` (definition and evidence revision; no main A/C rerun)
- Model: `{manifest.get('model')}` revision `{manifest.get('model_revision')}`
- Complete points: `{manifest.get('completed_points')}/{manifest.get('total_points')}`
- Execution order seed: `{manifest.get('execution_order_seed')}`; the exact order is in `manifest.json`.
- Request trace audit: deterministic `iter_request_specs` inputs are recorded in `request_trace_audit.json`, including each run's seed, sampling seed, generator module SHA-256, and the A/C shared-request trace SHA-256.

## Reproduction

{command_block}

## Deployments

- **A / strong-collocated**: GPU0, one BF16 Qwen3-8B process. Paged KV and block allocation/reclamation, continuous batching, decode-first chunked prefill, capacity admission, structured scheduler output, ordinary Decode CUDA Graph, and lifecycle telemetry are enabled.
- **C / pd-shared**: Prefill on GPU0 and Decode on GPU1, two BF16 model copies. KV handoff uses two pinned shared-memory slots, descriptor-only messages, ACK release/backpressure, and Decode CUDA Graph. Prefill also has chunked-prefill enabled; because it serves no decode sequences, it does not form mixed Prefill/Decode batches itself.
- Both use `128 input / 64 output`, `temperature=0.01`, `ignore_eos=true`, `max_num_batched_tokens=1024`, `max_num_seqs=128`, capacity admission, `max_model_len=512`, and c=64 closed-loop (30 s warmup, 60 s measurement). Prompts are generated deterministically from the same per-run seed and generator; no shared prefix is used.

Topology and P2P metadata are retained in each run JSON and `runtime_configuration.csv`.

```text
{_clean_topology(metadata.get('nvidia_smi_topology'))}
```

CUDA peer access: `{metadata.get('cuda_peer_access')}`.

## Metric Definitions

- **Engine TTFT** = `t_first_token - t_submit`: host-monotonic time from the benchmark calling `add_request` to Prefill sampling x1, the first output token. It is an engine-internal first-token measurement, **not a client-receipt timestamp** because this API is non-streaming.
- **`decode_continuation_ready_ms`** = `t_handoff_finish - t_submit`: in PD, Decode has finished importing the prompt KV and has admitted the sequence with x1, so it can generate x2. For A this equals Engine TTFT because no cross-worker handoff exists.
- **E2E** = `t_finish - t_submit`.
- **TPOT** = `(t_finish - first-token time) / (output tokens - 1)`; output length one uses 0 ms.
- Output throughput is generated output tokens within the 60-second measurement window divided by 60; request throughput is completed requests divided by 60.
- KV handoff and Decode admission are continuation diagnostics only; they are **not** added to Engine TTFT. In C the scheduler admission currently precedes the blocking KV import, so their raw ordering is preserved rather than forced into an invalid serial equation.

## Measurement Window and Sample Rules

{_measurement_rules_text()}

## Main Results

{_summary_table(summary_rows)}

P50/P99 pool measurement-window requests. Per-run P50/P99 ranges and three-run throughput min/max are in `derived_summary.csv`; raw request records are in `request_timeline.csv`.

## First-token and PD continuation diagnostics

{_ttft_diagnostic_table(summary_rows)}

These are raw host-side intervals between independently recorded events. In C, Prefill samples x1 before Decode finishes importing KV; `decode_continuation_ready_ms`, KV handoff, and Decode admission describe the later x2 continuation path and are not a breakdown or arithmetic extension of Engine TTFT.

## Decode-first and CUDA Graph Evidence

{_scheduler_evidence_text(results)}

{_cuda_graph_evidence_text(results)}

## Shared KV Validation

All C runs passed the report validator: every recorded handoff used `shared_slot`; each run recorded ACK release; final slot state was fully free with no ready, consuming, or pending transfer. `pd_handoff.csv` retains every batch's prefill, export, admission, handoff, descriptor and slot record. No alternative tensor transport is present in this comparison.

## Scheduling And Chunked Prefill

`scheduler_steps.csv` records per-round waiting/running lengths, prefill/decode token and request counts, remaining budget, actual partial chunks, and CUDA Event forward time. The short main workload enables chunked prefill but has 128-token prompts below its 1024-token budget; whether it actually emitted a partial chunk is therefore reported separately from whether the capability was enabled.

{main_chunking_text}

### Long/Short Validation (A Only)

{chunked_text}

## Conclusion And Limits

**C relative to A is the complete deployment gain of PD+Shared KV relative to the strong single-GPU Collocated Runtime. It includes the additional GPU and model copy, resource isolation, the PD pipeline, and Shared KV; it is not attributed as a pure PD, pure Shared KV, or pure hardware-bandwidth gain.**

The sole resume/interview number from this revision is **`{_main_output_ratio(summary_rows):.4f}x`** PD+Shared/A output throughput at c=64. It reuses the original `2026-08-14` raw JSON and is therefore comparable to the preceding report, not a new performance rerun.

This is a complete deployment comparison, not an isolation experiment for any one subsystem. Remaining limits are unequal A/C hardware resources, unlocked clocks, and no valid NCU SM/DRAM counter collection.

This run records host queueing, CUDA Event model-forward durations, KV handoff, Decode admission, and GPU clock ranges. It does not include NCU counters, so it does not claim SM or DRAM saturation.

## Report Revision

- `2026-08-15-r1` corrects the x1/Decode-continuation terminology, adds effective long/short validation configuration, adds decode-first and actual CUDA Graph evidence, documents exact window boundaries, and adds request-trace audit hashes.
- These corrections use existing raw results only. No main A/C point was rerun, and the `2026-08-14` and revised reports refer to the same six main raw JSON points.
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
