"""Spawn entry point for isolated Prefill and Decode workers."""

from __future__ import annotations

import os
import traceback


def _destroy_process_group(distributed=None):
    """Release a partially initialized worker process group."""
    if distributed is None:
        import torch.distributed as distributed
    if distributed.is_initialized():
        distributed.destroy_process_group()


def _reply(response_queue, *, result=None, error=None):
    response_queue.put(
        {
            "ok": error is None,
            "result": result,
            "error": error,
        }
    )


def worker_main(
    role: str,
    model: str,
    gpu_id: int,
    engine_kwargs: dict,
    command_queue,
    response_queue,
):
    """Run one long-lived worker process.

    CUDA_VISIBLE_DEVICES is set before importing the engine so each worker sees
    its assigned physical GPU as local device 0.
    """
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    from llmserve import LLM
    from llmserve.pd.protocol import RequestEnvelope
    from llmserve.pd.runtime import DecodeWorkerRuntime, PrefillWorkerRuntime

    engine = None
    try:
        engine = LLM(model, **engine_kwargs)
        runtime = (
            PrefillWorkerRuntime(engine)
            if role == "prefill"
            else DecodeWorkerRuntime(engine)
        )
        _reply(
            response_queue,
            result={"ready": True, "role": role},
        )
        while True:
            command = command_queue.get()
            command_type = command.get("type")
            if command_type == "shutdown":
                _reply(response_queue, result={"stopped": True})
                break
            if role == "prefill" and command_type == "prefill_batch":
                envelopes = [
                    RequestEnvelope.from_payload(payload)
                    for payload in command["envelopes"]
                ]
                _reply(response_queue, result=runtime.prefill_batch(envelopes))
                continue
            if role == "decode" and command_type == "admit_batch":
                _reply(response_queue, result=runtime.admit_batch(command["handoffs"]))
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
                )
                continue
            if role == "decode" and command_type == "metrics":
                _reply(response_queue, result=engine.get_metrics())
                continue
            if role == "decode" and command_type == "reset_metrics":
                engine.reset_metrics()
                _reply(response_queue, result={"reset": True})
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
            if engine is not None:
                engine.exit()
        finally:
            _destroy_process_group()
