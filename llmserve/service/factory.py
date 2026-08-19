"""Factories for single-host Collocated and PD+Shared service deployments."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from llmserve import LLM
from llmserve.pd import PDConfig, PDCoordinator, PDServingEngine
from llmserve.service.runtime import EngineServiceRuntime


@dataclass(frozen=True, slots=True)
class ServiceLaunchConfig:
    """Explicit deployment settings shared by the HTTP service modes."""

    model: str
    mode: Literal["collocated", "pd-shared"] = "collocated"
    max_model_len: int = 4096
    max_num_batched_tokens: int = 1024
    max_num_seqs: int = 128
    gpu_memory_utilization: float = 0.9
    enable_chunked_prefill: bool = True
    enable_kv_capacity_admission: bool = True
    enforce_eager: bool = False
    speculative_model: str | None = None
    speculative_gamma: int = 3
    max_inflight_requests: int | None = None
    request_timeout_seconds: float | None = None
    prefill_gpu: int = 0
    decode_gpus: tuple[int, ...] = (1,)
    prefill_batch_size: int = 4
    kv_slot_count: int = 2
    kv_slot_capacity_tokens: int = 8192
    prefill_init_method: str = "tcp://127.0.0.1:24431"
    decode_init_methods: tuple[str, ...] = ()
    startup_timeout_seconds: float = 300.0

    def __post_init__(self):
        if self.mode not in {"collocated", "pd-shared"}:
            raise ValueError(f"unsupported service mode: {self.mode}")
        if self.mode == "pd-shared" and self.speculative_model is not None:
            raise ValueError("EAGLE is not available in PD+Shared service mode")
        if not self.model:
            raise ValueError("model must not be empty")
        if self.max_model_len <= 0 or self.max_num_batched_tokens <= 0:
            raise ValueError("model length and token budget must be positive")
        if self.max_num_seqs <= 0:
            raise ValueError("max_num_seqs must be positive")
        if self.max_inflight_requests is not None and self.max_inflight_requests <= 0:
            raise ValueError("max_inflight_requests must be positive when set")
        if self.request_timeout_seconds is not None and self.request_timeout_seconds <= 0:
            raise ValueError("request_timeout_seconds must be positive when set")
        if not 0 < self.gpu_memory_utilization <= 1:
            raise ValueError("gpu_memory_utilization must be in (0, 1]")
        if self.prefill_batch_size <= 0:
            raise ValueError("prefill_batch_size must be positive")
        if self.mode == "pd-shared":
            decode_gpus = tuple(self.decode_gpus)
            if not decode_gpus:
                raise ValueError("PD+Shared requires at least one Decode GPU")
            if len(set(decode_gpus)) != len(decode_gpus):
                raise ValueError("PD+Shared Decode GPU ids must be unique")
            if self.prefill_gpu in decode_gpus:
                raise ValueError("PD+Shared requires Prefill and Decode on different GPUs")
            if self.kv_slot_count < 2:
                raise ValueError("PD+Shared requires at least two KV slots")

    def engine_kwargs(self) -> dict:
        return {
            "max_model_len": self.max_model_len,
            "max_num_batched_tokens": self.max_num_batched_tokens,
            "max_num_seqs": self.max_num_seqs,
            "gpu_memory_utilization": self.gpu_memory_utilization,
            "enable_chunked_prefill": self.enable_chunked_prefill,
            "enable_kv_capacity_admission": self.enable_kv_capacity_admission,
            "enforce_eager": self.enforce_eager,
            "speculative_model": self.speculative_model,
            "speculative_gamma": self.speculative_gamma,
        }

    @property
    def resolved_max_inflight_requests(self) -> int:
        if self.max_inflight_requests is not None:
            return self.max_inflight_requests
        if self.mode == "pd-shared":
            return self.max_num_seqs * len(self.decode_gpus)
        return self.max_num_seqs


def build_engine_factory(config: ServiceLaunchConfig):
    """Return an Engine factory without constructing GPU state in the caller."""
    if config.mode == "collocated":
        kwargs = config.engine_kwargs()
        return lambda: LLM(config.model, **kwargs)

    engine_kwargs = config.engine_kwargs()
    decode_gpus = tuple(config.decode_gpus)
    decode_init_methods = config.decode_init_methods or _decode_init_methods(
        config.prefill_init_method,
        len(decode_gpus),
    )
    pd_config = PDConfig(
        model=config.model,
        prefill_gpu=config.prefill_gpu,
        decode_gpu=decode_gpus[0],
        decode_gpus=decode_gpus,
        prefill_enforce_eager=True,
        decode_enforce_eager=config.enforce_eager,
        prefill_init_method=config.prefill_init_method,
        decode_init_methods=decode_init_methods,
        kv_slot_count=config.kv_slot_count,
        kv_slot_capacity_tokens=config.kv_slot_capacity_tokens,
        startup_timeout_seconds=config.startup_timeout_seconds,
        engine_kwargs=engine_kwargs,
    )

    def build_pd_engine():
        coordinator = PDCoordinator(pd_config)
        return PDServingEngine(
            coordinator,
            prefill_batch_size=config.prefill_batch_size,
            enable_transport_overlap=True,
        )

    return build_pd_engine


def build_service_runtime(config: ServiceLaunchConfig) -> EngineServiceRuntime:
    """Load the CPU tokenizer once and construct a not-yet-started runtime."""
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(config.model)
    return EngineServiceRuntime(
        build_engine_factory(config),
        tokenizer=tokenizer,
        max_inflight_requests=config.resolved_max_inflight_requests,
        request_timeout_seconds=config.request_timeout_seconds,
    )


def _decode_init_methods(prefill_endpoint: str, count: int) -> tuple[str, ...]:
    if count <= 0:
        raise ValueError("PD+Shared requires at least one Decode GPU")
    prefix, separator, port_text = prefill_endpoint.rpartition(":")
    if not separator:
        raise ValueError("prefill_init_method must end in a TCP port")
    try:
        first_port = int(port_text) + 1
    except ValueError as error:
        raise ValueError("prefill_init_method must end in a TCP port") from error
    return tuple(f"{prefix}:{first_port + index}" for index in range(count))
