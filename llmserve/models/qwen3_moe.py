"""Qwen3-MoE model backed by GPTQ W4A16 CUDA linears."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from llmserve.layers.attention import Attention
from llmserve.layers.embed_head import ParallelLMHead, VocabParallelEmbedding
from llmserve.layers.layernorm import RMSNorm
from llmserve.layers.quantized import GPTQLinear
from llmserve.layers.rotary_embedding import get_rope


class ExpertRouter:
    """Select and normalize the sparse expert assignments for each token."""

    def __init__(self, *, num_experts: int, top_k: int, normalize_topk: bool):
        if not 0 < top_k <= num_experts:
            raise ValueError("top_k must be in (0, num_experts]")
        self.num_experts = num_experts
        self.top_k = top_k
        self.normalize_topk = normalize_topk

    def __call__(
        self,
        hidden_states: torch.Tensor,
        router_weight: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        router_logits = F.linear(hidden_states, router_weight)
        routing_probabilities = F.softmax(router_logits, dim=-1, dtype=torch.float32)
        routing_weights, expert_ids = torch.topk(
            routing_probabilities,
            self.top_k,
            dim=-1,
        )
        if self.normalize_topk:
            routing_weights = routing_weights / routing_weights.sum(
                dim=-1,
                keepdim=True,
            )
        return routing_weights.to(hidden_states.dtype), expert_ids

class Qwen3MoeAttention(nn.Module):
    """Qwen3 attention with unfused GPTQ projections for one GPU."""

    def __init__(self, config) -> None:
        super().__init__()
        self.num_heads = config.num_attention_heads
        self.num_kv_heads = config.num_key_value_heads
        self.head_dim = getattr(config, "head_dim", config.hidden_size // self.num_heads)
        self.q_size = self.num_heads * self.head_dim
        self.kv_size = self.num_kv_heads * self.head_dim
        group_size = config.quantization_config["group_size"]

        self.q_proj = GPTQLinear(config.hidden_size, self.q_size, group_size=group_size)
        self.k_proj = GPTQLinear(config.hidden_size, self.kv_size, group_size=group_size)
        self.v_proj = GPTQLinear(config.hidden_size, self.kv_size, group_size=group_size)
        self.o_proj = GPTQLinear(self.q_size, config.hidden_size, group_size=group_size)
        self.q_norm = RMSNorm(self.head_dim, eps=config.rms_norm_eps)
        self.k_norm = RMSNorm(self.head_dim, eps=config.rms_norm_eps)
        self.rotary_emb = get_rope(
            self.head_dim,
            rotary_dim=self.head_dim,
            max_position=config.max_position_embeddings,
            base=getattr(config, "rope_theta", 1000000),
        )
        self.attn = Attention(
            self.num_heads,
            self.head_dim,
            self.head_dim ** -0.5,
            self.num_kv_heads,
        )

    def forward(self, positions: torch.Tensor, hidden_states: torch.Tensor) -> torch.Tensor:
        q = self.q_proj(hidden_states).view(-1, self.num_heads, self.head_dim)
        k = self.k_proj(hidden_states).view(-1, self.num_kv_heads, self.head_dim)
        v = self.v_proj(hidden_states).view(-1, self.num_kv_heads, self.head_dim)
        q = self.q_norm(q)
        k = self.k_norm(k)
        q, k = self.rotary_emb(positions, q, k)
        output = self.attn(q, k, v)
        return self.o_proj(output.flatten(1, -1))


class Qwen3MoeExpert(nn.Module):

    def __init__(self, hidden_size: int, intermediate_size: int, group_size: int) -> None:
        super().__init__()
        self.gate_proj = GPTQLinear(hidden_size, intermediate_size, group_size=group_size)
        self.up_proj = GPTQLinear(hidden_size, intermediate_size, group_size=group_size)
        self.down_proj = GPTQLinear(intermediate_size, hidden_size, group_size=group_size)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.down_proj(F.silu(self.gate_proj(hidden_states)) * self.up_proj(hidden_states))


class Qwen3MoeSparseMoeBlock(nn.Module):
    """Top-k dispatch with stable token/expert grouping.

    Each active expert is called once per batch. This keeps the execution
    semantics explicit while quantized GEMMs use the native CUDA provider.
    """

    def __init__(self, config) -> None:
        super().__init__()
        group_size = config.quantization_config["group_size"]
        self.gate = nn.Linear(config.hidden_size, config.num_experts, bias=False)
        self.router = ExpertRouter(
            num_experts=config.num_experts,
            top_k=config.num_experts_per_tok,
            normalize_topk=config.norm_topk_prob,
        )
        self.experts = nn.ModuleList([
            Qwen3MoeExpert(config.hidden_size, config.moe_intermediate_size, group_size)
            for _ in range(config.num_experts)
        ])

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        routing_weights, expert_ids = self.router(hidden_states, self.gate.weight)
        num_tokens, top_k = expert_ids.shape
        flat_experts = expert_ids.reshape(-1)
        flat_tokens = torch.arange(num_tokens, device=hidden_states.device).repeat_interleave(top_k)
        flat_weights = routing_weights.reshape(-1)
        order = flat_experts.argsort(stable=True)
        sorted_experts = flat_experts.index_select(0, order)
        sorted_tokens = flat_tokens.index_select(0, order)
        sorted_weights = flat_weights.index_select(0, order)
        active_experts, counts = torch.unique_consecutive(sorted_experts, return_counts=True)
        # The current one-GPU implementation dispatches each active expert from
        # Python. Move both grouping vectors together to avoid two device syncs.
        groups = torch.stack((active_experts, counts), dim=1).tolist()

        output = torch.zeros_like(hidden_states)
        offset = 0
        for expert_id, count in groups:
            next_offset = offset + count
            token_ids = sorted_tokens[offset:next_offset]
            expert_output = self.experts[expert_id](hidden_states.index_select(0, token_ids))
            output.index_add_(
                0,
                token_ids,
                expert_output * sorted_weights[offset:next_offset].unsqueeze(-1),
            )
            offset = next_offset
        return output


class Qwen3MoeDecoderLayer(nn.Module):

    def __init__(self, config) -> None:
        super().__init__()
        self.self_attn = Qwen3MoeAttention(config)
        self.mlp = Qwen3MoeSparseMoeBlock(config)
        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        residual: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if residual is None:
            hidden_states, residual = self.input_layernorm(hidden_states), hidden_states
        else:
            hidden_states, residual = self.input_layernorm(hidden_states, residual)
        hidden_states = self.self_attn(positions, hidden_states)
        hidden_states, residual = self.post_attention_layernorm(hidden_states, residual)
        return self.mlp(hidden_states), residual


class Qwen3MoeModel(nn.Module):

    def __init__(self, config) -> None:
        super().__init__()
        self.embed_tokens = VocabParallelEmbedding(config.vocab_size, config.hidden_size)
        self.layers = nn.ModuleList([
            Qwen3MoeDecoderLayer(config)
            for _ in range(config.num_hidden_layers)
        ])
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(self, input_ids: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
        hidden_states = self.embed_tokens(input_ids)
        residual = None
        for layer in self.layers:
            hidden_states, residual = layer(positions, hidden_states, residual)
        hidden_states, _ = self.norm(hidden_states, residual)
        return hidden_states


class Qwen3MoeForCausalLM(nn.Module):

    def __init__(
        self,
        config,
        *,
        gptq_backend: str = "tinygemm",
        marlin_library: str | None = None,
    ) -> None:
        super().__init__()
        self.model = Qwen3MoeModel(config)
        self.lm_head = ParallelLMHead(config.vocab_size, config.hidden_size)
        for module in self.modules():
            if isinstance(module, GPTQLinear):
                module.configure_backend(gptq_backend, marlin_library)

    def forward(self, input_ids: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
        return self.model(input_ids, positions)

    @torch.no_grad()
    def prepare_for_runtime(self) -> None:
        for module in self.modules():
            if isinstance(module, GPTQLinear):
                module.prepare_for_runtime()

    def compute_logits(self, hidden_states: torch.Tensor, all_tokens: bool = False) -> torch.Tensor:
        if all_tokens:
            return F.linear(hidden_states, self.lm_head.weight)
        return self.lm_head(hidden_states)
