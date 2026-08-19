import os
from dataclasses import dataclass
from transformers import AutoConfig


@dataclass(slots=True)
class Config:
    model: str
    max_num_batched_tokens: int = 16384
    max_num_seqs: int = 512
    max_model_len: int = 4096
    gpu_memory_utilization: float = 0.9
    tensor_parallel_size: int = 1
    distributed_init_method: str = "tcp://localhost:2333"
    enforce_eager: bool = False
    # ===== 2026-06-07 chunked prefill =====
    # Stage 2 的实验调度开关；默认关闭，保留原始 baseline 行为。
    enable_chunked_prefill: bool = False
    enable_kv_capacity_admission: bool = False
    # ===== 2026-06-07 chunked prefill =====
    speculative_model: str | None = None
    speculative_gamma: int = 3
    speculative_accept_mode: str = "greedy"
    speculative_trace: bool = False
    enable_speculative_cuda_graph: bool = False
    # Benchmark-only observability. Disabled for normal serving paths.
    enable_latency_telemetry: bool = False
    random_seed: int | None = None
    hf_config: AutoConfig | None = None
    eos: int = -1
    kvcache_block_size: int = 256
    num_kvcache_blocks: int = -1

    def __post_init__(self):
        assert os.path.isdir(self.model)
        assert self.kvcache_block_size % 256 == 0
        assert 1 <= self.tensor_parallel_size <= 8
        assert self.speculative_gamma > 0
        if self.enable_speculative_cuda_graph and self.speculative_model is None:
            raise ValueError(
                "speculative CUDA Graph requires a speculative model"
            )
        if self.enable_speculative_cuda_graph and self.speculative_accept_mode != "greedy":
            raise ValueError(
                "speculative CUDA Graph requires greedy acceptance"
            )
        assert self.speculative_accept_mode in {"greedy", "rejection"}
        if self.speculative_model is not None:
            assert os.path.isdir(self.speculative_model)
        self.hf_config = AutoConfig.from_pretrained(self.model)
        if getattr(self.hf_config, "quantization_config", None) is not None:
            raise ValueError("quantized checkpoints are not supported")
        self.max_model_len = min(self.max_model_len, self.hf_config.max_position_embeddings)
