"""Summarize raw PD transport telemetry without assigning a performance cause."""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

from benchmarks.metrics import summarize_values
from llmserve.pd.observability import DecodeIdleReason


_TIMELINE_FIELDS = (
    "t_submit",
    "t_prefill_first_scheduled",
    "t_prefill_finish",
    "t_first_token",
    "t_slot_acquired",
    "t_kv_export_started",
    "t_kv_export_finished",
    "t_slot_ready",
    "t_decode_descriptor_received",
    "t_slot_consuming",
    "t_decode_admission_started",
    "t_decode_admitted",
    "t_handoff_finish",
    "t_decode_admission_returned",
    "t_finish",
)


def _read_json(path: Path) -> dict:
    return json.loads(path.read_text())


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


def _ms(start: float | None, end: float | None) -> float | None:
    if start is None or end is None:
        return None
    return (end - start) * 1000


def _load_complete_pd_results(results_dir: Path) -> list[dict]:
    results = [
        _read_json(path)
        for path in sorted((results_dir / "runs").glob("*.json"))
    ]
    if not results:
        raise ValueError("result directory contains no run JSON")
    if any(not result.get("complete") for result in results):
        raise ValueError("transport report requires only complete run JSON")
    if any(
        not result.get("config", {}).get("runtime", {}).get("pd", False)
        for result in results
    ):
        raise ValueError("transport report requires only PD shared runs")
    return results


def _measurement_requests(result: dict):
    window = result.get("telemetry", {}).get("measurement_window", {})
    start, end = window.get("start"), window.get("end")
    for request in result.get("requests", []):
        timeline = request.get("timeline") or {}
        missing = [name for name in _TIMELINE_FIELDS if timeline.get(name) is None]
        if missing:
            continue
        if start is not None and timeline["t_submit"] < start:
            continue
        if end is not None and timeline["t_finish"] >= end:
            continue
        yield request


def _timeline_rows(results: list[dict]) -> list[dict]:
    rows = []
    for result in results:
        config = result["config"]
        for request in _measurement_requests(result):
            timeline = request["timeline"]
            row = {
                "point_id": result["point_id"],
                "run": int(config.get("run", 0)) + 1,
                "request_id": request.get("request_id"),
                "first_token_ms": _ms(
                    timeline["t_submit"], timeline["t_first_token"]
                ),
                "prefill_queue_wait_ms": _ms(
                    timeline["t_submit"],
                    timeline["t_prefill_first_scheduled"],
                ),
                "prefill_forward_wall_ms": _ms(
                    timeline["t_prefill_first_scheduled"],
                    timeline["t_prefill_finish"],
                ),
                "slot_acquire_offset_ms": _ms(
                    timeline["t_submit"], timeline["t_slot_acquired"]
                ),
                "prefill_to_export_start_ms": _ms(
                    timeline["t_prefill_finish"],
                    timeline["t_kv_export_started"],
                ),
                "kv_export_wall_ms": _ms(
                    timeline["t_kv_export_started"],
                    timeline["t_kv_export_finished"],
                ),
                "slot_publish_ms": _ms(
                    timeline["t_kv_export_finished"], timeline["t_slot_ready"]
                ),
                "descriptor_queue_ms": _ms(
                    timeline["t_slot_ready"],
                    timeline["t_decode_descriptor_received"],
                ),
                "slot_ready_to_consuming_ms": _ms(
                    timeline["t_slot_ready"], timeline["t_slot_consuming"]
                ),
                "decode_admission_wall_ms": _ms(
                    timeline["t_decode_admission_started"],
                    timeline["t_decode_admission_returned"],
                ),
                "decode_scheduler_admit_ms": _ms(
                    timeline["t_decode_admission_started"],
                    timeline["t_decode_admitted"],
                ),
                "kv_import_enqueue_wall_ms": _ms(
                    timeline["t_decode_admitted"], timeline["t_handoff_finish"]
                ),
                "decode_continuation_enqueue_ready_ms": _ms(
                    timeline["t_submit"],
                    timeline["t_decode_admission_returned"],
                ),
                "e2e_ms": _ms(timeline["t_submit"], timeline["t_finish"]),
            }
            row.update({name: timeline[name] for name in _TIMELINE_FIELDS})
            rows.append(row)
    return rows


def _timeline_summary(rows: list[dict]) -> list[dict]:
    metrics = (
        "first_token_ms",
        "prefill_queue_wait_ms",
        "prefill_forward_wall_ms",
        "slot_acquire_offset_ms",
        "prefill_to_export_start_ms",
        "kv_export_wall_ms",
        "slot_publish_ms",
        "descriptor_queue_ms",
        "slot_ready_to_consuming_ms",
        "decode_admission_wall_ms",
        "decode_scheduler_admit_ms",
        "kv_import_enqueue_wall_ms",
        "decode_continuation_enqueue_ready_ms",
        "e2e_ms",
    )
    by_run = defaultdict(list)
    for row in rows:
        by_run[row["run"]].append(row)
    summary = []
    for metric in metrics:
        values = [row[metric] for row in rows if row.get(metric) is not None]
        pooled = summarize_values(values)
        run_p50 = []
        for run_rows in by_run.values():
            p50 = summarize_values([
                row[metric] for row in run_rows if row.get(metric) is not None
            ])["p50"]
            if p50 is not None:
                run_p50.append(p50)
        summary.append({
            "metric": metric,
            **pooled,
            "run_p50_min": min(run_p50) if run_p50 else None,
            "run_p50_max": max(run_p50) if run_p50 else None,
        })
    return summary


def _decode_idle_rows(results: list[dict]) -> list[dict]:
    rows = []
    for result in results:
        idle = result.get("metrics", {}).get("pd", {}).get("decode_idle", {})
        counts = idle.get("counts", {})
        durations = idle.get("duration_ms", {})
        for reason in DecodeIdleReason:
            rows.append({
                "point_id": result["point_id"],
                "run": int(result["config"].get("run", 0)) + 1,
                "reason": reason.value,
                "count": counts.get(reason.value, 0),
                "duration_ms": durations.get(reason.value, 0.0),
            })
    return rows


def _slot_rows(results: list[dict]) -> list[dict]:
    rows = []
    seen = set()
    for result in results:
        pd = result.get("metrics", {}).get("pd", {})
        run = int(result["config"].get("run", 0)) + 1
        for sample in pd.get("slot_release_samples", []):
            events = sample.get("slot_stats", {}).get("observability", {}).get(
                "events", []
            )
            for event in events:
                key = (
                    run,
                    event.get("slot_id"),
                    event.get("generation"),
                    event.get("state"),
                    event.get("at"),
                )
                if key in seen:
                    continue
                seen.add(key)
                rows.append({
                    "point_id": result["point_id"],
                    "run": run,
                    **event,
                })
    return rows


def _slot_environment_rows(results: list[dict]) -> list[dict]:
    rows = []
    for result in results:
        batches = result.get("metrics", {}).get("pd", {}).get(
            "prefill_batches_detail", []
        )
        environment = None
        for batch in batches:
            environment = batch.get("slot_observability_after_ready", {}).get(
                "environment"
            )
            if environment:
                break
        rows.append({
            "point_id": result["point_id"],
            "run": int(result["config"].get("run", 0)) + 1,
            "environment": json.dumps(environment, sort_keys=True),
        })
    return rows


def _worker_environment_rows(results: list[dict]) -> list[dict]:
    rows = []
    for result in results:
        worker_health = result.get("metrics", {}).get("pd", {}).get(
            "worker_health", {}
        )
        for role in ("prefill", "decode"):
            rows.append({
                "point_id": result["point_id"],
                "run": int(result["config"].get("run", 0)) + 1,
                "role": role,
                "environment": json.dumps(
                    worker_health.get(role, {}).get("environment"),
                    sort_keys=True,
                ),
            })
    return rows


def _validate_shared_slots(results: list[dict]):
    for result in results:
        runtime = result["config"].get("runtime", {})
        expected_slots = runtime.get("kv_slot_count")
        pd = result.get("metrics", {}).get("pd", {})
        batches = pd.get("prefill_batches_detail", [])
        if not batches:
            raise ValueError("PD run recorded no prefill batches")
        for batch in batches:
            if any(item != "shared_slot" for item in batch.get("transports", [])):
                raise ValueError("PD run used non-shared KV transport")
        releases = pd.get("slot_release_samples", [])
        if not releases:
            raise ValueError("PD run recorded no shared-slot ACK releases")
        final = releases[-1].get("slot_stats", {})
        if any(final.get(name) != expected for name, expected in (
            ("free_slots", expected_slots),
            ("ready_slots", 0),
            ("consuming_slots", 0),
            ("pending_transfers", 0),
        )):
            raise ValueError("PD run did not return every shared slot to FREE")


def _markdown_table(rows: list[dict]) -> str:
    lines = ["| Metric | Mean (ms) | P50 (ms) | P99 (ms) | Run P50 range (ms) |", "|---|---:|---:|---:|---:|"]
    for row in rows:
        def fmt(value):
            return "-" if value is None else f"{value:.3f}"
        range_text = "-"
        if row["run_p50_min"] is not None:
            range_text = f"{row['run_p50_min']:.3f}-{row['run_p50_max']:.3f}"
        lines.append(
            f"| {row['metric']} | {fmt(row['mean'])} | {fmt(row['p50'])} | "
            f"{fmt(row['p99'])} | {range_text} |"
        )
    return "\n".join(lines)


def build_report(results_dir: Path, *, reproduction_command: str | None = None) -> Path:
    """Write derived telemetry views while preserving raw run JSON unchanged."""

    results_dir = Path(results_dir)
    results = _load_complete_pd_results(results_dir)
    _validate_shared_slots(results)
    timeline_rows = _timeline_rows(results)
    if not timeline_rows:
        raise ValueError("no complete measurement-window transport timelines found")
    timeline_summary = _timeline_summary(timeline_rows)
    idle_rows = _decode_idle_rows(results)
    slot_rows = _slot_rows(results)
    environment_rows = _slot_environment_rows(results)
    worker_environment_rows = _worker_environment_rows(results)
    _write_csv(results_dir / "request_transport_timeline.csv", timeline_rows)
    _write_csv(results_dir / "transport_timing_summary.csv", timeline_summary)
    _write_csv(results_dir / "decode_idle_summary.csv", idle_rows)
    _write_csv(results_dir / "slot_lifecycle.csv", slot_rows)
    _write_csv(results_dir / "slot_environment.csv", environment_rows)
    _write_csv(results_dir / "worker_environment.csv", worker_environment_rows)

    manifest = _read_json(results_dir / "manifest.json")
    command = reproduction_command or "see reproduction_commands.txt"
    (results_dir / "reproduction_commands.txt").write_text(command.rstrip() + "\n")
    idle_totals = defaultdict(lambda: {"count": 0, "duration_ms": 0.0})
    for row in idle_rows:
        idle_totals[row["reason"]]["count"] += row["count"]
        idle_totals[row["reason"]]["duration_ms"] += row["duration_ms"]
    idle_lines = "\n".join(
        f"- `{reason.value}`: {idle_totals[reason.value]['count']} intervals, "
        f"{idle_totals[reason.value]['duration_ms']:.3f} ms total"
        for reason in DecodeIdleReason
    )
    report = f"""# PD Transport Observability Baseline

## Scope

- Git commit: `{manifest.get('git_commit')}`
- Complete points: `{manifest.get('completed_points')}/{manifest.get('total_points')}`
- Raw inputs: `runs/*.json`; derived files in this directory preserve those inputs unchanged.

This is an observation baseline for the existing synchronous pinned-CPU relay. It
does not claim a transport optimization or a root cause for end-to-end latency.

## Reproduction

```bash
{command.rstrip()}
```

## Request Timeline

{_markdown_table(timeline_summary)}

`first_token_ms` remains Prefill's model-generated first token. The later
`decode_continuation_enqueue_ready_ms` is a handoff diagnostic, not part of
first-token latency.

`kv_import_enqueue_wall_ms` ends when Decode's KV import call returns. **H2D is enqueued, not completed.** This version has no CUDA Event completion boundary and therefore cannot claim copy/compute overlap or use this timestamp as a safe asynchronous ACK boundary.

## Decode Idle Reasons

{idle_lines}

The four reason codes are mutually exclusive. In the current synchronous path,
`waiting_kv_h2d` may remain zero because an admitted request does not become
visible until its import call returns; that is a limitation of this baseline,
not evidence that H2D is free.

## Shared-slot lifecycle and NUMA Snapshot

`slot_lifecycle.csv` records owner-side `FILLING -> READY -> CONSUMING -> FREE`
transitions from the slot pool, including its writer. The owner records
`CONSUMING` and `FREE` while it processes an ACK; use per-request
`t_slot_consuming` for Decode's actual first shared-slot read. `slot_environment.csv` records the Prefill
process CPU affinity and the Linux NUMA placement visible for the shared backing
mapping; `worker_environment.csv` records affinity snapshots for both workers.
A shared page cannot be simultaneously local to both GPUs on this SYS topology;
placement is measured here and will become an explicit variable in the NUMA
experiment.

## Limits and Next Gate

The current relay synchronizes source-side D2H export. Before changing slot
count or adding Prefill workers, the next implementation must add stream-aware
CUDA Events, defer slot ACK until target-side H2D and scatter are complete, and
report copy/compute overlap plus critical-path reduction. Only then can this
dataset decide whether Decode starvation is due to Prefill supply, KV import, or
scheduler behavior.
"""
    report_path = results_dir / "pd_transport_observability_report.md"
    report_path.write_text(report)
    return report_path


def _parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", required=True, type=Path)
    parser.add_argument("--reproduction-command")
    return parser.parse_args(argv)


def main(argv=None):
    args = _parse_args(argv)
    build_report(args.results_dir, reproduction_command=args.reproduction_command)


if __name__ == "__main__":
    main()
