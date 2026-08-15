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
            slot_pool = None
            if transport_config["slot_count"]:
                from llmserve.pd.shared_slots import SharedKVSlotPool

                kv_cache = engine.model_runner.kv_cache
                slot_pool = SharedKVSlotPool.create(
                    slot_count=transport_config["slot_count"],
                    capacity_tokens=transport_config["capacity_tokens"],
                    num_layers=kv_cache.size(1),
                    num_kv_heads=kv_cache.size(4),
                    head_dim=kv_cache.size(5),
                    dtype=kv_cache.dtype,
                )
            runtime = PrefillWorkerRuntime(engine, slot_pool=slot_pool)
            ready_result = {
                "ready": True,
                "role": role,
                "kv_slot_handle": slot_pool.handle if slot_pool is not None else None,
                "environment": collect_process_numa_observability(
                    shared_memory_address=(
                        slot_pool.handle.backing.data_ptr()
                        if slot_pool is not None
                        else None
                    ),
                ),
            }
        else:
            runtime = DecodeWorkerRuntime(engine)
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
            if role == "decode" and command_type == "attach_shared_slots":
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
            if role == "decode" and command_type == "admit_batch":
                _reply(
                    response_queue,
                    result=runtime.admit_batch(command["handoffs"]),
                    worker_received_at=worker_received_at,
                )
                continue
            if role == "decode" and command_type == "step":
                outputs, num_tokens = engine.step()
                _reply(
                    response_queue,
                    result={
                        "outputs": outputs,
                        "num_tokens": num_tokens,
                        "last_step_events": engine.last_step_events,
                    },
                    worker_received_at=worker_received_at,
                )
                continue
            if role == "decode" and command_type == "abort_request":
                _reply(
                    response_queue,
                    result=runtime.abort_request(command["seq_id"]),
                    worker_received_at=worker_received_at,
                )
                continue
            if role == "decode" and command_type == "metrics":
                _reply(
                    response_queue,
                    result=engine.get_metrics(),
                    worker_received_at=worker_received_at,
                )
                continue
            if role == "decode" and command_type == "reset_metrics":
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
            if runtime is not None:
                transport = getattr(runtime, "slot_reader", None) or getattr(
                    runtime, "slot_pool", None
                )
                if transport is not None:
                    transport.close()
            if engine is not None:
                engine.exit()
        finally:
            _destroy_process_group()
