import json
import os
import torch
import torch.nn.functional as F
from safetensors import safe_open
from torch import nn
from thrustlm.speculative.draft import (
    _compact_eagle3_draft_kv,
    _pack_eagle3_draft_kv,
    generate_eagle3_draft_tokens,
    generate_eagle3_draft_tokens_batched,
    run_eagle3_offline_step,
    run_eagle3_speculative_cycle,
    run_eagle3_target_verify,
)
from thrustlm.speculative.sampling import (
    correction_distribution,
    sample_from_logits,
    sample_from_probs,
    speculative_accept_greedy_from_logits,
    speculative_accept_reject,
    speculative_accept_reject_from_logits,
)
from thrustlm.speculative.types import (
    Eagle3CycleResult,
    Eagle3DraftOutput,
    Eagle3DraftSequence,
    Eagle3OfflineStepResult,
    Eagle3TargetVerifyOutput,
    SpeculativeSampleResult,
)


def normalize_eagle3_config(config: dict) -> dict:
    layer_config = config.get("transformer_layer_config")
    if layer_config is None:
        return dict(config)

    for name, value in layer_config.items():
        if name in config and config[name] != value:
            raise ValueError(f"conflicting EAGLE3 config field: {name}")
    normalized = dict(layer_config)
    for name in ("draft_vocab_size", "target_hidden_size", "norm_before_residual"):
        if name in config:
            normalized[name] = config[name]
    return normalized


def normalize_eagle3_weight_name(name: str) -> str:
    if name.startswith("layers.0."):
        return "midlayer." + name[len("layers.0."):]
    return name


def validate_eagle3_target_config(draft_config: dict, target_config: dict):
    draft_config = normalize_eagle3_config(draft_config)
    target_hidden_size = target_config["hidden_size"]
    expected_hidden_size = draft_config.get("target_hidden_size") or draft_config["hidden_size"]
    if target_hidden_size != expected_hidden_size:
        raise ValueError(
            f"target hidden_size {target_hidden_size} does not match "
            f"draft target_hidden_size {expected_hidden_size}"
        )
    target_vocab_size = target_config["vocab_size"]
    if target_vocab_size != draft_config["vocab_size"]:
        raise ValueError(
            f"target vocab_size {target_vocab_size} does not match "
            f"draft vocab_size {draft_config['vocab_size']}"
        )
    if target_config.get("num_hidden_layers", 0) < 6:
        raise ValueError("EAGLE3 target must have at least 6 hidden layers")



class Eagle3RMSNorm(nn.Module):

    def __init__(self, hidden_size: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(hidden_size))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        orig_dtype = x.dtype
        x = x.float()
        var = x.pow(2).mean(dim=-1, keepdim=True)
        x = x * torch.rsqrt(var + self.eps)
        return x.to(orig_dtype) * self.weight


class Eagle3Attention(nn.Module):

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        num_kv_heads: int,
        head_dim: int,
        max_position: int,
        rope_theta: float,
    ):
        super().__init__()
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.num_kv_groups = num_heads // num_kv_heads

        # EAGLE3 的 draft layer 输入是 token embedding 和辅助 hidden 拼接后的 2H。
        input_size = 2 * hidden_size
        self.q_proj = nn.Linear(input_size, num_heads * head_dim, bias=False)
        self.k_proj = nn.Linear(input_size, num_kv_heads * head_dim, bias=False)
        self.v_proj = nn.Linear(input_size, num_kv_heads * head_dim, bias=False)
        self.o_proj = nn.Linear(num_heads * head_dim, hidden_size, bias=False)

        inv_freq = 1.0 / (rope_theta ** (torch.arange(0, head_dim, 2, dtype=torch.float) / head_dim))
        positions = torch.arange(max_position, dtype=torch.float)
        freqs = torch.einsum("i,j->ij", positions, inv_freq)
        self.register_buffer("rope_cos", freqs.cos(), persistent=False)
        self.register_buffer("rope_sin", freqs.sin(), persistent=False)

    def apply_rope(self, positions: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        cos = self.rope_cos[positions].unsqueeze(2)
        sin = self.rope_sin[positions].unsqueeze(2)
        x1, x2 = x.float().chunk(2, dim=-1)
        y1 = x1 * cos - x2 * sin
        y2 = x2 * cos + x1 * sin
        return torch.cat([y1, y2], dim=-1).to(x.dtype)

    def forward(
        self,
        hidden_states: torch.Tensor,
        positions: torch.Tensor,
        past_kv: tuple[torch.Tensor, torch.Tensor] | None = None,
        kv_valid_lens: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        bsz, seq_len, _ = hidden_states.shape

        q = self.q_proj(hidden_states).view(bsz, seq_len, self.num_heads, self.head_dim)
        k = self.k_proj(hidden_states).view(bsz, seq_len, self.num_kv_heads, self.head_dim)
        v = self.v_proj(hidden_states).view(bsz, seq_len, self.num_kv_heads, self.head_dim)
        q = self.apply_rope(positions, q).transpose(1, 2)
        k = self.apply_rope(positions, k).transpose(1, 2)
        v = v.transpose(1, 2)

        past_len = 0
        if past_kv is not None:
            past_k, past_v = past_kv
            past_len = past_k.size(2)
            k = torch.cat([past_k, k], dim=2)
            v = torch.cat([past_v, v], dim=2)
        new_past_kv = (k, v)

        if self.num_kv_groups > 1:
            k = k.repeat_interleave(self.num_kv_groups, dim=1)
            v = v.repeat_interleave(self.num_kv_groups, dim=1)

        attn_mask = None
        is_causal = past_kv is None and seq_len > 1
        if past_kv is not None and (seq_len > 1 or kv_valid_lens is not None):
            total_len = past_len + seq_len
            key_pos = torch.arange(total_len, device=hidden_states.device)
            query_pos = past_len + torch.arange(seq_len, device=hidden_states.device)
            attn_mask = key_pos.unsqueeze(0) <= query_pos.unsqueeze(1)
            attn_mask = attn_mask.unsqueeze(0).unsqueeze(0)
            if kv_valid_lens is not None:
                valid_past = key_pos.unsqueeze(0) < kv_valid_lens.unsqueeze(1)
                valid_new = key_pos.unsqueeze(0) >= past_len
                valid = (valid_past | valid_new).unsqueeze(1).unsqueeze(2)
                attn_mask = attn_mask & valid

        output = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask, is_causal=is_causal)
        output = output.transpose(1, 2).contiguous().view(bsz, seq_len, -1)
        return self.o_proj(output), new_past_kv


class Eagle3MLP(nn.Module):

    def __init__(self, hidden_size: int, intermediate_size: int):
        super().__init__()
        self.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.up_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


class Eagle3DecoderLayer(nn.Module):

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        num_kv_heads: int,
        head_dim: int,
        intermediate_size: int,
        max_position: int,
        rope_theta: float,
        rms_norm_eps: float,
        norm_before_residual: bool = False,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.norm_before_residual = norm_before_residual
        self.input_layernorm = Eagle3RMSNorm(hidden_size, eps=rms_norm_eps)
        self.hidden_norm = Eagle3RMSNorm(hidden_size, eps=rms_norm_eps)
        self.post_attention_layernorm = Eagle3RMSNorm(hidden_size, eps=rms_norm_eps)
        self.self_attn = Eagle3Attention(hidden_size, num_heads, num_kv_heads, head_dim, max_position, rope_theta)
        self.mlp = Eagle3MLP(hidden_size, intermediate_size)

    def forward(
        self,
        hidden_states: torch.Tensor,
        positions: torch.Tensor,
        past_kv: tuple[torch.Tensor, torch.Tensor] | None = None,
        kv_valid_lens: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        embeds = hidden_states[:, :, :self.hidden_size]
        hidden = hidden_states[:, :, self.hidden_size:]

        if self.norm_before_residual:
            hidden = self.hidden_norm(hidden)
            residual = hidden
        else:
            residual = hidden
            hidden = self.hidden_norm(hidden)
        embeds = self.input_layernorm(embeds)
        attn_input = torch.cat([embeds, hidden], dim=-1)

        attn_output, new_past_kv = self.self_attn(attn_input, positions, past_kv, kv_valid_lens)
        residual = residual + attn_output
        hidden_states = self.post_attention_layernorm(residual)
        hidden_states = self.mlp(hidden_states)
        return residual + hidden_states, new_past_kv


class Eagle3Speculator(nn.Module):

    def __init__(self, config: dict):
        super().__init__()
        self.hidden_size = config["hidden_size"]
        self.target_hidden_size = config.get("target_hidden_size") or self.hidden_size
        self.draft_vocab_size = config.get("draft_vocab_size", 32000)
        self.target_vocab_size = config["vocab_size"]

        self.embed_tokens = nn.Embedding(self.target_vocab_size, self.hidden_size)
        self.fc = nn.Linear(3 * self.target_hidden_size, self.hidden_size, bias=False)
        self.midlayer = Eagle3DecoderLayer(
            hidden_size=self.hidden_size,
            num_heads=config["num_attention_heads"],
            num_kv_heads=config["num_key_value_heads"],
            head_dim=config.get("head_dim", self.hidden_size // config["num_attention_heads"]),
            intermediate_size=config["intermediate_size"],
            max_position=config.get("max_position_embeddings", 40960),
            rope_theta=config.get("rope_theta", 1000000),
            rms_norm_eps=config.get("rms_norm_eps", 1e-6),
            norm_before_residual=config.get("norm_before_residual", False),
        )
        self.norm = Eagle3RMSNorm(self.hidden_size, eps=config.get("rms_norm_eps", 1e-6))
        self.lm_head = nn.Linear(self.hidden_size, self.draft_vocab_size, bias=False)

        # d2t/t2d 是 draft vocab 和 target vocab 的桥；概率拒绝采样需要在 target vocab 上比较 p/q。
        self.register_buffer("d2t", torch.zeros(self.draft_vocab_size, dtype=torch.long))
        self.register_buffer("t2d", torch.zeros(self.target_vocab_size, dtype=torch.bool))

    @classmethod
    def from_pretrained(cls, path: str, target_model_path: str | None = None) -> "Eagle3Speculator":
        with open(os.path.join(path, "config.json")) as f:
            raw_config = json.load(f)
        config = normalize_eagle3_config(raw_config)
        if target_model_path is not None:
            with open(os.path.join(target_model_path, "config.json")) as f:
                target_config = json.load(f)
            validate_eagle3_target_config(config, target_config)
        model = cls(config)
        state = {}
        for file_name in os.listdir(path):
            if not file_name.endswith(".safetensors"):
                continue
            with safe_open(os.path.join(path, file_name), framework="pt", device="cpu") as f:
                for key in f.keys():
                    normalized_key = normalize_eagle3_weight_name(key)
                    if normalized_key in state:
                        raise RuntimeError(f"duplicate EAGLE3 weight after normalization: {normalized_key}")
                    state[normalized_key] = f.get_tensor(key)
        missing, unexpected = model.load_state_dict(state, strict=False)
        if unexpected:
            raise RuntimeError(f"unexpected EAGLE3 weights: {unexpected}")
        missing = [name for name in missing if name != "embed_tokens.weight"]
        if missing:
            raise RuntimeError(f"missing EAGLE3 weights: {missing}")
        if "embed_tokens.weight" not in state:
            if target_model_path is None:
                raise ValueError("target_model_path is required when draft weights do not include embed_tokens.weight")
            model.load_target_embedding(target_model_path)
        return model

    def load_target_embedding(self, target_model_path: str):
        target_weight_name = "model.embed_tokens.weight"
        for file_name in os.listdir(target_model_path):
            if not file_name.endswith(".safetensors"):
                continue
            with safe_open(os.path.join(target_model_path, file_name), framework="pt", device="cpu") as f:
                if target_weight_name in f.keys():
                    self.embed_tokens.weight.data.copy_(f.get_tensor(target_weight_name))
                    return
        raise RuntimeError(f"{target_weight_name} not found in {target_model_path}")

    def forward(
        self,
        input_ids: torch.Tensor,
        aux_hidden_states: torch.Tensor,
        positions: torch.Tensor,
        past_kv: tuple[torch.Tensor, torch.Tensor] | None = None,
        kv_valid_lens: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        if input_ids.dim() == 1:
            input_ids = input_ids.unsqueeze(1)
        if positions.dim() == 1:
            positions = positions.unsqueeze(1)
        if aux_hidden_states.dim() == 2:
            aux_hidden_states = aux_hidden_states.unsqueeze(1)

        embeds = self.embed_tokens(input_ids)
        fused_hidden = aux_hidden_states if aux_hidden_states.size(-1) == self.hidden_size else self.fc(aux_hidden_states)
        layer_input = torch.cat([embeds, fused_hidden], dim=-1)
        hidden_states, new_past_kv = self.midlayer(layer_input, positions, past_kv, kv_valid_lens)
        draft_logits = self.lm_head(self.norm(hidden_states))
        return hidden_states, draft_logits, new_past_kv

    def compute_logits(self, hidden_states: torch.Tensor) -> torch.Tensor:
        draft_logits = self.lm_head(self.norm(hidden_states))
        return self.map_draft_logits(draft_logits)

    def map_draft_logits(self, draft_logits: torch.Tensor) -> torch.Tensor:
        bsz, seq_len, _ = draft_logits.shape
        target_indices = torch.arange(self.draft_vocab_size, device=draft_logits.device) + self.d2t
        mapped = draft_logits.new_full((bsz, seq_len, self.target_vocab_size), float("-inf"))
        mapped[:, :, target_indices] = draft_logits
        return mapped

    def greedy_sample(self, draft_logits: torch.Tensor) -> torch.Tensor:
        draft_ids = draft_logits.argmax(dim=-1)
        return draft_ids + self.d2t[draft_ids]

    def propose(
        self,
        input_ids: torch.Tensor,
        aux_hidden_states: torch.Tensor,
        positions: torch.Tensor,
        temperature: float,
        past_kv: tuple[torch.Tensor, torch.Tensor] | None = None,
        kv_valid_lens: torch.Tensor | None = None,
    ) -> Eagle3DraftOutput:
        hidden_states, draft_logits, new_past_kv = self(
            input_ids,
            aux_hidden_states,
            positions,
            past_kv=past_kv,
            kv_valid_lens=kv_valid_lens,
        )
        # 第一版概率拒绝采样要在 target vocab 空间比较 p/q，所以 draft logits 先映射成 target logits。
        target_logits = self.map_draft_logits(draft_logits)
        token_ids = sample_from_logits(target_logits, temperature)
        return Eagle3DraftOutput(hidden_states, draft_logits, target_logits, token_ids, new_past_kv)
