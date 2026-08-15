"""Spawn entry point for isolated Prefill and Decode workers."""

from __future__ import annotations

import os
import signal
from time import perf_counter
import traceback


def _handle_termination(signum, _frame):
    """Convert SIGTERM into normal interpreter unwinding for worker cleanup."""
    raise SystemExit(128 + signum)


def _destroy_process_group(distributed=None):
    """Release a partially initialized worker process group."""
    if distributed is None:
        import torch.distributed as distributed
    if distributed.is_initialized():
        distributed.destroy_process_group()


def _reply(
    response_queue,
    *,
    result=None,
    error=None,
    worker_received_at: float | None = None,
):
    response_queue.put(
        {
            "ok": error is None,
            "result": result,
            "error": error,
            "timing": (
                {
                    "worker_received_at": worker_received_at,
                    "worker_reply_enqueued_at": perf_counter(),
                }
                if worker_received_at is not None
                else None
            ),
        }
    )


def _cleanup_worker_resources(engine, runtime):
    """Synchronize target imports before unregistering shared pinned memory."""
    try:
        if engine is not None:
            engine.exit()
    finally:
        if runtime is not None:
            transports = [
                getattr(runtime, "slot_reader", None),
                getattr(runtime, "slot_pool", None),
                *getattr(runtime, "slot_pools", {}).values(),
            ]
            closed = set()
            for transport in transports:
                if transport is None or id(transport) in closed:
                    continue
                transport.close()
                closed.add(id(transport))


def _create_prefill_slot_pools(
    engine,
    transport_config: dict,
    *,
    register_cuda: bool = True,
):
    """Create one Prefill-owned shared-memory pool per Decode Worker."""
    if not transport_config["slot_count"]:
        return {}
    from llmserve.pd.shared_slots import SharedKVSlotPool

    target_workers = tuple(transport_config.get("target_workers", ("decode",)))
    if not target_workers or len(set(target_workers)) != len(target_workers):
        raise ValueError("shared KV target workers must be unique and non-empty")
    kv_cache = engine.model_runner.kv_cache
    return {
        worker_id: SharedKVSlotPool.create(
            slot_count=transport_config["slot_count"],
            capacity_tokens=transport_config["capacity_tokens"],
            num_layers=kv_cache.size(1),
            num_kv_heads=kv_cache.size(4),
            head_dim=kv_cache.size(5),
            dtype=kv_cache.dtype,
            register_cuda=register_cuda,
        )
        for worker_id in target_workers
    }


def worker_main(
    role: str,
    model: str,
    gpu_id: int,
    engine_kwargs: dict,
    command_queue,
    response_queue,
    transport_config: dict,
):
    """Run one long-lived worker process.

    CUDA_VISIBLE_DEVICES is set before importing the engine so each worker sees
    its assigned physical GPU as local device 0.
    """
    signal.signal(signal.SIGTERM, _handle_termination)
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    from llmserve import LLM
    from llmserve.pd.observability import collect_process_numa_observability
    from llmserve.pd.protocol import RequestEnvelope
    from llmserve.pd.runtime import DecodeWorkerRuntime, PrefillWorkerRuntime

    engine = None
    runtime = None
    try:
        engine = LLM(model, **engine_kwargs)
        if role == "prefill":
            slot_pools = _create_prefill_slot_pools(engine, transport_config)
            slot_handles = {
                worker_id: pool.handle for worker_id, pool in slot_pools.items()
            }
            ready_result = {
                "ready": True,
                "role": role,
                "kv_slot_handle": (
                    next(iter(slot_handles.values())) if len(slot_handles) == 1 else None
                ),
                "kv_slot_handles": slot_handles,
                "environment": collect_process_numa_observability(),
                "slot_environments": {
                    worker_id: collect_process_numa_observability(
                        shared_memory_address=pool.handle.backing.data_ptr()
                    )
                    for worker_id, pool in slot_pools.items()
                },
            }
            runtime = PrefillWorkerRuntime(engine, slot_pools=slot_pools)
        else:
            runtime = DecodeWorkerRuntime(engine, worker_id=role)
            ready_result = {
                "ready": True,
                "role": role,
                "environment": collect_process_numa_observability(),
            }
        _reply(
            response_queue,
            result=ready_result,
        )
        while True:
            command = command_queue.get()
            worker_received_at = perf_counter()
            command_type = command.get("type")
            if command_type == "shutdown":
                _reply(
                    response_queue,
                    result={"stopped": True},
                    worker_received_at=worker_received_at,
                )
                break
            if role.startswith("decode") and command_type == "attach_shared_slots":
                runtime.attach_shared_slots(command["handle"])
                _reply(
                    response_queue,
                    result={"attached": True},
                    worker_received_at=worker_received_at,
                )
                continue
            if role == "prefill" and command_type == "prefill_batch":
                runtime.release_transfers(command.get("release_transfer_ids", ()))
                envelopes = [
                    RequestEnvelope.from_payload(payload)
                    for payload in command["envelopes"]
                ]
                _reply(
                    response_queue,
                    result=runtime.prefill_batch(envelopes),
                    worker_received_at=worker_received_at,
                )
                continue
            if role == "prefill" and command_type == "release_transfers":
                _reply(
                    response_queue,
                    result=runtime.release_transfers(command["transfer_ids"]),
                    worker_received_at=worker_received_at,
                )
                continue
            if role.startswith("decode") and command_type == "admit_batch":
                _reply(
                    response_queue,
                    result=runtime.admit_batch(command["handoffs"]),
                    worker_received_at=worker_received_at,
                )
                continue
            if role.startswith("decode") and command_type == "step":
                outputs, num_tokens, completed_transfers = runtime.step()
                _reply(
                    response_queue,
                    result={
                        "outputs": outputs,
                        "num_tokens": num_tokens,
                        "completed_transfers": completed_transfers,
                        "last_step_events": engine.last_step_events,
                        "step_diagnostics": runtime.last_step_diagnostics,
                    },
                    worker_received_at=worker_received_at,
                )
                continue
            if role.startswith("decode") and command_type == "step_with_handoffs":
                (
                    admissions,
                    outputs,
                    num_tokens,
                    completed_transfers,
                ) = runtime.admit_and_step(command["handoffs"])
                _reply(
                    response_queue,
                    result={
                        "admissions": admissions,
                        "outputs": outputs,
                        "num_tokens": num_tokens,
                        "completed_transfers": completed_transfers,
                        "last_step_events": engine.last_step_events,
                        "step_diagnostics": runtime.last_step_diagnostics,
                        "admission_timing": runtime.last_admission_timing,
                    },
                    worker_received_at=worker_received_at,
                )
                continue
            if role.startswith("decode") and command_type == "collect_completed_transfers":
                _reply(
                    response_queue,
                    result=runtime.collect_completed_transfers(
                        wait=bool(command.get("wait", False))
                    ),
                    worker_received_at=worker_received_at,
                )
                continue
            if role.startswith("decode") and command_type == "abort_request":
                _reply(
                    response_queue,
                    result=runtime.abort_request(command["seq_id"]),
                    worker_received_at=worker_received_at,
                )
                continue
            if role.startswith("decode") and command_type == "metrics":
                _reply(
                    response_queue,
                    result=engine.get_metrics(),
                    worker_received_at=worker_received_at,
                )
                continue
            if role.startswith("decode") and command_type == "reset_metrics":
                engine.reset_metrics()
                _reply(
                    response_queue,
                    result={"reset": True},
                    worker_received_at=worker_received_at,
                )
                continue
            raise ValueError(f"unsupported {role} worker command: {command_type}")
    except Exception as error:
        _reply(
            response_queue,
            error={
                "type": type(error).__name__,
                "message": str(error),
                "traceback": traceback.format_exc(),
            },
        )
    finally:
        try:
            _cleanup_worker_resources(engine, runtime)
        finally:
            _destroy_process_group()
