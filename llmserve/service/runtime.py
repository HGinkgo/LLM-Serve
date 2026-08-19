"""Thread-safe request streaming over the step-wise inference engines."""

from __future__ import annotations

from concurrent.futures import Future
from dataclasses import dataclass, field
from queue import Empty, Queue
from threading import Event, Lock, Thread, current_thread
from time import monotonic
from typing import Any, Callable, Iterator


class ServiceRuntimeError(RuntimeError):
    """Raised when the serving runtime cannot accept a request."""


class ServiceOverloadedError(ServiceRuntimeError):
    """Raised when the service-level admission limit has been reached."""


_TERMINAL_EVENT_KINDS = frozenset({
    "completed",
    "cancelled",
    "failed",
    "timed_out",
})


@dataclass(frozen=True, slots=True)
class GenerationEvent:
    """One output transition emitted by the Engine driver."""

    kind: str
    request_id: int
    text: str = ""
    token_ids: tuple[int, ...] = ()
    message: str | None = None


@dataclass(slots=True)
class GenerationRequest:
    """Caller-owned stream handle backed by a driver-owned Engine request."""

    _events: Queue[GenerationEvent] = field(default_factory=Queue)
    _assigned: Event = field(default_factory=Event)
    _engine_request_id: int | None = None
    _assignment_error: BaseException | None = None
    _emitted_token_ids: list[int] = field(default_factory=list)
    _emitted_text: str = ""
    _terminal: bool = False
    _deadline: float | None = None

    @property
    def engine_request_id(self) -> int:
        if self._engine_request_id is None:
            raise ServiceRuntimeError("request has not been admitted")
        return self._engine_request_id

    def iter_events(self, *, timeout: float | None = None) -> Iterator[GenerationEvent]:
        """Yield output events until a terminal transition or timeout."""
        deadline = None if timeout is None else monotonic() + timeout
        while True:
            remaining = None if deadline is None else deadline - monotonic()
            if remaining is not None and remaining <= 0:
                raise TimeoutError("timed out waiting for generation event")
            try:
                event = self._events.get(timeout=remaining)
            except Empty as error:
                raise TimeoutError("timed out waiting for generation event") from error
            yield event
            if event.kind in _TERMINAL_EVENT_KINDS:
                return

    def poll_event(self) -> GenerationEvent | None:
        """Return one already available event without waiting for the driver."""
        try:
            return self._events.get_nowait()
        except Empty:
            return None


@dataclass(slots=True)
class _SubmitCommand:
    request: GenerationRequest
    prompt: str | list[int]
    sampling_params: Any


@dataclass(slots=True)
class _CancelCommand:
    request: GenerationRequest
    acknowledged: Future[bool]


@dataclass(slots=True)
class _MetricsCommand:
    response: Future[dict[str, Any]]


class EngineServiceRuntime:
    """Own one Engine on a driver thread and expose event-stream requests.

    The engine and its scheduler are intentionally only touched by the driver
    thread. HTTP workers may enqueue commands and consume event queues, but
    never call ``step()``, ``add_request()``, or ``abort_request()`` directly.
    """

    def __init__(
        self,
        engine_factory: Callable[[], Any],
        *,
        tokenizer: Any | None = None,
        idle_wait_seconds: float = 0.01,
        submit_timeout_seconds: float = 30.0,
        max_inflight_requests: int | None = None,
        request_timeout_seconds: float | None = None,
    ):
        if idle_wait_seconds <= 0:
            raise ValueError("idle_wait_seconds must be positive")
        if submit_timeout_seconds <= 0:
            raise ValueError("submit_timeout_seconds must be positive")
        if max_inflight_requests is not None and max_inflight_requests <= 0:
            raise ValueError("max_inflight_requests must be positive when set")
        if request_timeout_seconds is not None and request_timeout_seconds <= 0:
            raise ValueError("request_timeout_seconds must be positive when set")
        self._engine_factory = engine_factory
        self._tokenizer = tokenizer
        self._idle_wait_seconds = idle_wait_seconds
        self._submit_timeout_seconds = submit_timeout_seconds
        self._max_inflight_requests = max_inflight_requests
        self._request_timeout_seconds = request_timeout_seconds
        self._commands: Queue[_SubmitCommand | _CancelCommand | _MetricsCommand] = Queue()
        self._wake = Event()
        self._started = Event()
        self._stopped = Event()
        self._stop_requested = Event()
        self._state_lock = Lock()
        self._admission_lock = Lock()
        self._ready = False
        self._startup_error: BaseException | None = None
        self._fatal_error: BaseException | None = None
        self._thread: Thread | None = None
        self._engine: Any | None = None
        self._active: dict[int, GenerationRequest] = {}
        self._admission_reserved = 0
        self._inflight_high_watermark = 0
        self._accepted_requests = 0
        self._rejected_requests = 0
        self._cancelled_requests = 0
        self._failed_requests = 0
        self._timed_out_requests = 0
        self._completed_requests = 0

    @property
    def ready(self) -> bool:
        with self._state_lock:
            return self._ready and self._fatal_error is None

    @property
    def startup_error(self) -> BaseException | None:
        with self._state_lock:
            return self._startup_error

    @property
    def fatal_error(self) -> BaseException | None:
        with self._state_lock:
            return self._fatal_error

    @property
    def tokenizer(self) -> Any:
        if self._tokenizer is None:
            raise ServiceRuntimeError("serving tokenizer is not initialized")
        return self._tokenizer

    def start(self, *, timeout: float = 300.0):
        """Start the driver and wait until the Engine is ready or fails."""
        with self._state_lock:
            if self._thread is not None:
                if self._ready:
                    return
                raise ServiceRuntimeError("serving runtime has already stopped")
            self._thread = Thread(
                target=self._run,
                name="llmserve-engine-driver",
                daemon=True,
            )
            self._thread.start()
        if not self._started.wait(timeout):
            raise ServiceRuntimeError("timed out while starting the serving runtime")
        if self._startup_error is not None:
            raise ServiceRuntimeError(
                f"failed to start serving runtime: {self._startup_error}"
            ) from self._startup_error

    def submit(self, prompt: str | list[int], sampling_params: Any) -> GenerationRequest:
        """Admit a request through the driver and return its event stream."""
        if not self.ready:
            raise self._unavailable_error()
        request = GenerationRequest()
        self._reserve_admission(request)
        self._commands.put(_SubmitCommand(request, prompt, sampling_params))
        self._wake.set()
        if not request._assigned.wait(self._submit_timeout_seconds):
            raise ServiceRuntimeError("timed out while admitting request")
        if request._assignment_error is not None:
            raise ServiceRuntimeError(
                f"failed to admit request: {request._assignment_error}"
            ) from request._assignment_error
        return request

    def cancel(self, request: GenerationRequest) -> bool:
        """Request step-boundary cancellation and wait for driver acknowledgement."""
        if request._terminal:
            return False
        acknowledged: Future[bool] = Future()
        self._commands.put(_CancelCommand(request, acknowledged))
        self._wake.set()
        try:
            return acknowledged.result(timeout=self._submit_timeout_seconds)
        except TimeoutError as error:
            raise ServiceRuntimeError("timed out while cancelling request") from error

    def metrics_snapshot(self) -> dict[str, Any]:
        """Return a driver-consistent metrics snapshot without concurrent Engine access."""
        if not self.ready:
            return self._local_metrics_snapshot()
        response: Future[dict[str, Any]] = Future()
        self._commands.put(_MetricsCommand(response))
        self._wake.set()
        try:
            return response.result(timeout=self._submit_timeout_seconds)
        except TimeoutError as error:
            raise ServiceRuntimeError("timed out while collecting service metrics") from error

    def close(self, *, timeout: float = 30.0):
        """Stop accepting work, terminate streams, and release Engine resources."""
        self._stop_requested.set()
        self._wake.set()
        thread = self._thread
        if thread is not None and thread is not current_thread():
            thread.join(timeout)
            if thread.is_alive():
                raise ServiceRuntimeError("timed out while stopping serving runtime")

    def _run(self):
        engine = None
        try:
            engine = self._engine_factory()
            if self._tokenizer is None:
                self._tokenizer = getattr(engine, "tokenizer", None)
            if self._tokenizer is None:
                raise ServiceRuntimeError(
                    "serving runtime requires an explicit tokenizer for this engine"
                )
            self._engine = engine
            with self._state_lock:
                self._ready = True
        except BaseException as error:
            with self._state_lock:
                self._startup_error = error
            self._started.set()
            self._stopped.set()
            return

        self._started.set()
        try:
            while not self._stop_requested.is_set():
                self._drain_commands(engine)
                self._expire_requests(engine)
                if self._active:
                    try:
                        outputs, _ = engine.step()
                    except BaseException as error:
                        self._fail_runtime(error)
                        break
                    self._publish_step(engine, outputs)
                    continue
                self._wake.wait(self._idle_wait_seconds)
                self._wake.clear()
        finally:
            self._reject_queued_submissions()
            self._cancel_active_for_shutdown()
            if engine is not None:
                try:
                    engine.exit()
                except BaseException as error:
                    self._record_fatal_error(error)
            with self._state_lock:
                self._ready = False
            self._stopped.set()

    def _drain_commands(self, engine: Any):
        while True:
            try:
                command = self._commands.get_nowait()
            except Empty:
                return
            if isinstance(command, _SubmitCommand):
                self._handle_submit(engine, command)
            elif isinstance(command, _CancelCommand):
                self._handle_cancel(engine, command)
            else:
                self._handle_metrics(engine, command)

    def _handle_submit(self, engine: Any, command: _SubmitCommand):
        request = command.request
        try:
            request_id = int(engine.add_request(command.prompt, command.sampling_params))
            request._engine_request_id = request_id
            self._active[request_id] = request
            self._accepted_requests += 1
        except BaseException as error:
            request._assignment_error = error
            self._release_admission()
        finally:
            request._assigned.set()

    def _handle_cancel(self, engine: Any, command: _CancelCommand):
        request = command.request
        request_id = request._engine_request_id
        if request._terminal or request_id is None or request_id not in self._active:
            command.acknowledged.set_result(False)
            return
        try:
            cancelled = bool(engine.abort_request(request_id))
        except BaseException as error:
            command.acknowledged.set_exception(error)
            return
        if cancelled:
            self._active.pop(request_id, None)
            self._cancelled_requests += 1
            self._release_admission()
            self._publish_terminal(request, "cancelled")
        command.acknowledged.set_result(cancelled)

    def _handle_metrics(self, engine: Any, command: _MetricsCommand):
        try:
            engine_metrics = engine.get_metrics()
            summary = dict(engine_metrics.get("summary") or {})
            last_step = dict(getattr(engine, "last_step_events", {}) or {})
            with self._admission_lock:
                admission_reserved = self._admission_reserved
                rejected_requests = self._rejected_requests
                high_watermark = self._inflight_high_watermark
            command.response.set_result({
                "ready": self.ready,
                "accepted_requests": self._accepted_requests,
                "rejected_requests": rejected_requests,
                "completed_requests": self._completed_requests,
                "cancelled_requests": self._cancelled_requests,
                "failed_requests": self._failed_requests,
                "timed_out_requests": self._timed_out_requests,
                "inflight_requests": len(self._active),
                "admission_reserved": admission_reserved,
                "inflight_limit": self._max_inflight_requests,
                "inflight_high_watermark": high_watermark,
                "queue_waiting": int(last_step.get("waiting_queue_size", 0)),
                "queue_running": int(last_step.get("running_queue_size", 0)),
                "engine_summary": summary,
                "fatal_error": self._error_snapshot(self.fatal_error),
            })
        except BaseException as error:
            command.response.set_exception(error)

    def _publish_step(self, engine: Any, outputs: Any):
        events = dict(getattr(engine, "last_step_events", {}) or {})
        emitted = events.get("emitted_token_ids_by_seq", {}) or {}
        for raw_request_id, raw_token_ids in emitted.items():
            request_id = int(raw_request_id)
            request = self._active.get(request_id)
            if request is None:
                continue
            self._publish_tokens(engine, request, raw_token_ids)

        final_token_ids = {
            int(request_id): tuple(token_ids)
            for request_id, token_ids in (outputs or ())
        }
        finished_ids = {
            int(request_id)
            for request_id in events.get("finished_seq_ids", ())
        }
        finished_ids.update(final_token_ids)
        for request_id in finished_ids:
            request = self._active.pop(request_id, None)
            if request is None:
                continue
            complete_tokens = final_token_ids.get(request_id)
            if complete_tokens is not None:
                unseen = complete_tokens[len(request._emitted_token_ids):]
                if unseen:
                    self._publish_tokens(engine, request, unseen)
            self._completed_requests += 1
            self._release_admission()
            self._publish_terminal(request, "completed")

    def _reserve_admission(self, request: GenerationRequest):
        with self._admission_lock:
            if (
                self._max_inflight_requests is not None
                and self._admission_reserved >= self._max_inflight_requests
            ):
                self._rejected_requests += 1
                raise ServiceOverloadedError(
                    "service in-flight limit has been reached"
                )
            self._admission_reserved += 1
            self._inflight_high_watermark = max(
                self._inflight_high_watermark,
                self._admission_reserved,
            )
        if self._request_timeout_seconds is not None:
            request._deadline = monotonic() + self._request_timeout_seconds

    def _release_admission(self):
        with self._admission_lock:
            if self._admission_reserved <= 0:
                raise RuntimeError("service admission reservation underflow")
            self._admission_reserved -= 1

    def _expire_requests(self, engine: Any):
        now = monotonic()
        for request_id, request in list(self._active.items()):
            if request._deadline is None or now < request._deadline:
                continue
            try:
                cancelled = bool(engine.abort_request(request_id))
            except BaseException as error:
                self._fail_runtime(error)
                return
            if not cancelled:
                continue
            self._active.pop(request_id, None)
            self._timed_out_requests += 1
            self._release_admission()
            self._publish_terminal(
                request,
                "timed_out",
                "request deadline exceeded",
            )

    def _publish_tokens(
        self,
        engine: Any,
        request: GenerationRequest,
        token_ids: Any,
    ):
        normalized = tuple(int(token_id) for token_id in token_ids)
        if not normalized:
            return
        request._emitted_token_ids.extend(normalized)
        decoded = self.tokenizer.decode(
            request._emitted_token_ids,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )
        if decoded.startswith(request._emitted_text):
            text = decoded[len(request._emitted_text):]
        else:
            text = self.tokenizer.decode(
                normalized,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False,
            )
        request._emitted_text = decoded
        request._events.put(GenerationEvent(
            kind="token",
            request_id=request.engine_request_id,
            text=text,
            token_ids=normalized,
        ))

    def _publish_terminal(self, request: GenerationRequest, kind: str, message: str | None = None):
        if request._terminal:
            return
        request._terminal = True
        request._events.put(GenerationEvent(
            kind=kind,
            request_id=request.engine_request_id,
            token_ids=tuple(request._emitted_token_ids),
            message=message,
        ))

    def _fail_runtime(self, error: BaseException):
        self._record_fatal_error(error)
        with self._state_lock:
            self._ready = False
        for request in list(self._active.values()):
            self._failed_requests += 1
            self._release_admission()
            self._publish_terminal(request, "failed", str(error))
        self._active.clear()

    def _cancel_active_for_shutdown(self):
        for request in list(self._active.values()):
            self._cancelled_requests += 1
            self._release_admission()
            self._publish_terminal(request, "cancelled", "service is shutting down")
        self._active.clear()

    def _reject_queued_submissions(self):
        unavailable_error = self._unavailable_error()
        while True:
            try:
                command = self._commands.get_nowait()
            except Empty:
                return
            if isinstance(command, _SubmitCommand):
                command.request._assignment_error = unavailable_error
                self._release_admission()
                command.request._assigned.set()
            elif isinstance(command, _CancelCommand):
                command.acknowledged.set_result(False)
            else:
                command.response.set_exception(unavailable_error)

    def _record_fatal_error(self, error: BaseException):
        with self._state_lock:
            if self._fatal_error is None:
                self._fatal_error = error

    @staticmethod
    def _error_snapshot(error: BaseException | None) -> dict[str, str] | None:
        if error is None:
            return None
        return {
            "type": type(error).__name__,
            "message": str(error),
        }

    def _unavailable_error(self) -> ServiceRuntimeError:
        with self._state_lock:
            error = self._fatal_error or self._startup_error
        if error is not None:
            return ServiceRuntimeError(f"serving runtime failed: {error}")
        return ServiceRuntimeError("serving runtime is not ready")

    def _local_metrics_snapshot(self) -> dict[str, Any]:
        """Report final counters after the driver is no longer available."""
        with self._admission_lock:
            admission_reserved = self._admission_reserved
            rejected_requests = self._rejected_requests
            high_watermark = self._inflight_high_watermark
        return {
            "ready": self.ready,
            "accepted_requests": self._accepted_requests,
            "rejected_requests": rejected_requests,
            "completed_requests": self._completed_requests,
            "cancelled_requests": self._cancelled_requests,
            "failed_requests": self._failed_requests,
            "timed_out_requests": self._timed_out_requests,
            "inflight_requests": len(self._active),
            "admission_reserved": admission_reserved,
            "inflight_limit": self._max_inflight_requests,
            "inflight_high_watermark": high_watermark,
            "queue_waiting": 0,
            "queue_running": 0,
            "engine_summary": {},
            "fatal_error": self._error_snapshot(self.fatal_error),
        }
