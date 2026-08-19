"""OpenAI-compatible HTTP endpoints over the step-wise serving runtime."""

from __future__ import annotations

from collections.abc import Iterator
from time import time
from typing import Literal
from uuid import uuid4

from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse, PlainTextResponse, StreamingResponse
from pydantic import BaseModel, ConfigDict, Field

from llmserve import SamplingParams
from llmserve.service.runtime import (
    EngineServiceRuntime,
    GenerationEvent,
    GenerationRequest,
    ServiceOverloadedError,
    ServiceRuntimeError,
)


class _CompletionRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    model: str | None = None
    prompt: str = Field(min_length=1)
    max_tokens: int = Field(default=16, ge=1, le=4096)
    temperature: float = Field(default=0.6, gt=0.0)
    ignore_eos: bool = False
    stream: bool = False


class _ChatMessage(BaseModel):
    model_config = ConfigDict(extra="forbid")

    role: Literal["system", "user", "assistant"]
    content: str = Field(min_length=1)


class _ChatCompletionRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    model: str | None = None
    messages: list[_ChatMessage] = Field(min_length=1)
    max_tokens: int = Field(default=16, ge=1, le=4096)
    temperature: float = Field(default=0.6, gt=0.0)
    ignore_eos: bool = False
    stream: bool = False


def create_app(runtime: EngineServiceRuntime, *, model_name: str) -> FastAPI:
    """Build an HTTP app around an already-started service runtime."""
    if not model_name:
        raise ValueError("model_name must not be empty")

    app = FastAPI(title="LLM-Serve", version="0.3.0")

    @app.get("/health/live")
    def liveness():
        return {"status": "live"}

    @app.get("/health/ready")
    def readiness():
        if runtime.ready:
            return {"status": "ready"}
        return JSONResponse(status_code=503, content={"status": "not_ready"})

    @app.get("/v1/models")
    def models():
        return {
            "object": "list",
            "data": [{
                "id": model_name,
                "object": "model",
                "owned_by": "llmserve",
            }],
        }

    @app.get("/metrics")
    def metrics():
        if not runtime.ready:
            return PlainTextResponse(
                _render_metrics({"ready": False}),
                media_type="text/plain; version=0.0.4; charset=utf-8",
            )
        try:
            snapshot = runtime.metrics_snapshot()
        except ServiceRuntimeError as error:
            return _error_response(str(error), status_code=503)
        return PlainTextResponse(
            _render_metrics(snapshot),
            media_type="text/plain; version=0.0.4; charset=utf-8",
        )

    @app.post("/v1/completions")
    def completions(payload: _CompletionRequest):
        _check_model(payload.model, model_name)
        prompt_token_ids = _encode_prompt(runtime.tokenizer, payload.prompt)
        request = _submit(runtime, prompt_token_ids, payload)
        request_id = f"cmpl-{uuid4().hex}"
        created = int(time())
        if payload.stream:
            return StreamingResponse(
                _completion_stream(runtime, request, request_id, created, model_name),
                media_type="text/event-stream",
                headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
            )
        return _completion_response(request, request_id, created, model_name, len(prompt_token_ids))

    @app.post("/v1/chat/completions")
    def chat_completions(payload: _ChatCompletionRequest):
        _check_model(payload.model, model_name)
        messages = [message.model_dump() for message in payload.messages]
        try:
            prompt = runtime.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
        except Exception as error:
            return _error_response(f"failed to format chat prompt: {error}", status_code=400)
        prompt_token_ids = _encode_prompt(runtime.tokenizer, prompt)
        request = _submit(runtime, prompt_token_ids, payload)
        request_id = f"chatcmpl-{uuid4().hex}"
        created = int(time())
        if payload.stream:
            return StreamingResponse(
                _chat_stream(runtime, request, request_id, created, model_name),
                media_type="text/event-stream",
                headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
            )
        return _chat_response(request, request_id, created, model_name, len(prompt_token_ids))

    return app


def _check_model(requested_model: str | None, model_name: str):
    if requested_model is not None and requested_model != model_name:
        raise HTTPException(
            status_code=404,
            detail={
                "message": f"model {requested_model!r} is not served by this runtime",
                "type": "invalid_request_error",
                "code": "model_not_found",
            },
        )


def _encode_prompt(tokenizer, prompt: str) -> list[int]:
    try:
        token_ids = list(tokenizer.encode(prompt, add_special_tokens=False))
    except Exception as error:
        raise HTTPException(
            status_code=400,
            detail={
                "message": f"failed to tokenize prompt: {error}",
                "type": "invalid_request_error",
            },
        ) from error
    if not token_ids:
        raise HTTPException(
            status_code=400,
            detail={
                "message": "prompt must produce at least one token",
                "type": "invalid_request_error",
            },
        )
    return token_ids


def _submit(runtime: EngineServiceRuntime, prompt_token_ids: list[int], payload) -> GenerationRequest:
    try:
        return runtime.submit(
            prompt_token_ids,
            SamplingParams(
                temperature=payload.temperature,
                max_tokens=payload.max_tokens,
                ignore_eos=payload.ignore_eos,
            ),
        )
    except ServiceOverloadedError as error:
        raise HTTPException(
            status_code=429,
            detail={
                "message": str(error),
                "type": "server_error",
                "code": "overloaded",
            },
            headers={"Retry-After": "1"},
        ) from error
    except ServiceRuntimeError as error:
        raise HTTPException(
            status_code=503,
            detail={"message": str(error), "type": "server_error"},
        ) from error


def _completion_response(
    request: GenerationRequest,
    request_id: str,
    created: int,
    model_name: str,
    prompt_tokens: int,
):
    text, finish_reason = _consume(request)
    if finish_reason == "error":
        return _error_response("generation failed", status_code=500)
    completion_tokens = len(request._emitted_token_ids)
    return {
        "id": request_id,
        "object": "text_completion",
        "created": created,
        "model": model_name,
        "choices": [{"text": text, "index": 0, "logprobs": None, "finish_reason": finish_reason}],
        "usage": _usage(prompt_tokens, completion_tokens),
    }


def _chat_response(
    request: GenerationRequest,
    request_id: str,
    created: int,
    model_name: str,
    prompt_tokens: int,
):
    text, finish_reason = _consume(request)
    if finish_reason == "error":
        return _error_response("generation failed", status_code=500)
    completion_tokens = len(request._emitted_token_ids)
    return {
        "id": request_id,
        "object": "chat.completion",
        "created": created,
        "model": model_name,
        "choices": [{
            "index": 0,
            "message": {"role": "assistant", "content": text},
            "finish_reason": finish_reason,
        }],
        "usage": _usage(prompt_tokens, completion_tokens),
    }


def _completion_stream(
    runtime: EngineServiceRuntime,
    request: GenerationRequest,
    request_id: str,
    created: int,
    model_name: str,
) -> Iterator[str]:
    try:
        for event in request.iter_events():
            if event.kind == "token" and event.text:
                yield _sse({
                    "id": request_id,
                    "object": "text_completion",
                    "created": created,
                    "model": model_name,
                    "choices": [{"text": event.text, "index": 0, "logprobs": None, "finish_reason": None}],
                })
            elif event.kind in {"completed", "cancelled", "failed", "timed_out"}:
                yield _sse({
                    "id": request_id,
                    "object": "text_completion",
                    "created": created,
                    "model": model_name,
                    "choices": [{
                        "text": "",
                        "index": 0,
                        "logprobs": None,
                        "finish_reason": _finish_reason(event),
                    }],
                })
                break
        yield "data: [DONE]\n\n"
    finally:
        _cancel_unfinished(runtime, request)


def _chat_stream(
    runtime: EngineServiceRuntime,
    request: GenerationRequest,
    request_id: str,
    created: int,
    model_name: str,
) -> Iterator[str]:
    first_chunk = True
    try:
        for event in request.iter_events():
            if event.kind == "token" and event.text:
                delta = {"content": event.text}
                if first_chunk:
                    delta = {"role": "assistant", **delta}
                    first_chunk = False
                yield _sse({
                    "id": request_id,
                    "object": "chat.completion.chunk",
                    "created": created,
                    "model": model_name,
                    "choices": [{"index": 0, "delta": delta, "finish_reason": None}],
                })
            elif event.kind in {"completed", "cancelled", "failed", "timed_out"}:
                yield _sse({
                    "id": request_id,
                    "object": "chat.completion.chunk",
                    "created": created,
                    "model": model_name,
                    "choices": [{
                        "index": 0,
                        "delta": {},
                        "finish_reason": _finish_reason(event),
                    }],
                })
                break
        yield "data: [DONE]\n\n"
    finally:
        _cancel_unfinished(runtime, request)


def _consume(request: GenerationRequest) -> tuple[str, str]:
    text_parts = []
    terminal_event = None
    for event in request.iter_events():
        if event.kind == "token":
            text_parts.append(event.text)
        elif event.kind in {"completed", "cancelled", "failed", "timed_out"}:
            terminal_event = event
            break
    if terminal_event is None:
        return "".join(text_parts), "error"
    return "".join(text_parts), _finish_reason(terminal_event)


def _cancel_unfinished(runtime: EngineServiceRuntime, request: GenerationRequest):
    # The HTTP server closes this generator on client disconnect. Cancellation
    # remains step-boundary because the Engine driver owns abort_request().
    if request._terminal:
        return
    try:
        runtime.cancel(request)
    except ServiceRuntimeError:
        # The runtime may already be shutting down after the client disconnects.
        pass


def _finish_reason(event: GenerationEvent) -> str:
    if event.kind == "completed":
        return "stop"
    if event.kind == "cancelled":
        return "cancelled"
    if event.kind == "timed_out":
        return "timeout"
    return "error"


def _usage(prompt_tokens: int, completion_tokens: int) -> dict[str, int]:
    return {
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": prompt_tokens + completion_tokens,
    }


def _sse(payload: dict) -> str:
    import json

    return f"data: {json.dumps(payload, ensure_ascii=False)}\n\n"


def _render_metrics(snapshot: dict) -> str:
    ready = 1 if snapshot.get("ready") else 0
    lines = [
        "# HELP llmserve_runtime_ready Whether the Engine driver accepts requests.",
        "# TYPE llmserve_runtime_ready gauge",
        f"llmserve_runtime_ready {ready}",
    ]
    for key, metric in (
        ("accepted_requests", "llmserve_requests_accepted_total"),
        ("rejected_requests", "llmserve_requests_rejected_total"),
        ("completed_requests", "llmserve_requests_completed_total"),
        ("cancelled_requests", "llmserve_requests_cancelled_total"),
        ("failed_requests", "llmserve_requests_failed_total"),
        ("timed_out_requests", "llmserve_requests_timed_out_total"),
    ):
        lines.extend((
            f"# TYPE {metric} counter",
            f"{metric} {int(snapshot.get(key, 0))}",
        ))
    for key, metric in (
        ("inflight_requests", "llmserve_requests_inflight"),
        ("admission_reserved", "llmserve_requests_admission_reserved"),
        ("inflight_high_watermark", "llmserve_requests_inflight_high_watermark"),
        ("queue_waiting", "llmserve_scheduler_queue_waiting"),
        ("queue_running", "llmserve_scheduler_queue_running"),
    ):
        lines.extend((
            f"# TYPE {metric} gauge",
            f"{metric} {int(snapshot.get(key, 0))}",
        ))
    inflight_limit = snapshot.get("inflight_limit")
    if inflight_limit is not None:
        lines.extend((
            "# TYPE llmserve_requests_inflight_limit gauge",
            f"llmserve_requests_inflight_limit {int(inflight_limit)}",
        ))
    return "\n".join(lines) + "\n"


def _error_response(message: str, *, status_code: int) -> JSONResponse:
    return JSONResponse(
        status_code=status_code,
        content={"error": {"message": message, "type": "server_error"}},
    )
