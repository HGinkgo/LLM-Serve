import json
import os
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from safetensors import safe_open
from torch import nn


@dataclass(slots=True)
class Eagle3DraftOutput:
    hidden_states: torch.Tensor
    draft_logits: torch.Tensor
    target_logits: torch.Tensor
    token_ids: torch.Tensor
    past_kv: tuple[torch.Tensor, torch.Tensor]


@dataclass(slots=True)
class SpeculativeSampleResult:
    token_ids: list[int]
    accepted_token_ids: list[int]
    final_token_id: int
    num_accepted: int
    accepted_all: bool


@dataclass(slots=True)
class Eagle3CycleResult:
    draft_token_ids: list[int]
    draft_target_logits: torch.Tensor
    sample_result: SpeculativeSampleResult
    past_kv: tuple[torch.Tensor, torch.Tensor]


@dataclass(slots=True)
class Eagle3DraftSequence:
    draft_token_ids: list[int]
    draft_target_logits: torch.Tensor
    past_kv: tuple[torch.Tensor, torch.Tensor]


@dataclass(slots=True)
class Eagle3TargetVerifyOutput:
    target_logits: torch.Tensor
    target_aux_hidden: torch.Tensor


@dataclass(slots=True)
class Eagle3OfflineStepResult:
    draft_token_ids: list[int]
    draft_target_logits: torch.Tensor
    verify_output: Eagle3TargetVerifyOutput
    sample_result: SpeculativeSampleResult
    past_kv: tuple[torch.Tensor, torch.Tensor]


def sample_from_logits(logits: torch.Tensor, temperature: float) -> torch.Tensor:
    assert temperature > 1e-10
    probs = torch.softmax(logits.float() / temperature, dim=-1)
    flat_probs = probs.reshape(-1, probs.size(-1))
    sampled = torch.multinomial(flat_probs, num_samples=1)
    return sampled.view(*logits.shape[:-1])


def sample_from_probs(probs: torch.Tensor) -> int:
    return int(torch.multinomial(probs.float(), num_samples=1).item())


def correction_distribution(target_probs: torch.Tensor, draft_probs: torch.Tensor) -> torch.Tensor:
    diff = torch.clamp(target_probs.float() - draft_probs.float(), min=0)
    total = diff.sum()
    if total <= 0:
        return target_probs.float() / target_probs.float().sum()
    return diff / total


def speculative_accept_reject(
    target_probs: torch.Tensor,
    draft_probs: torch.Tensor,
    draft_token_ids: torch.Tensor,
    random_values: torch.Tensor | None = None,
) -> SpeculativeSampleResult:
    num_draft_tokens = int(draft_token_ids.numel())
    assert target_probs.ndim == 2
    assert draft_probs.ndim == 2
    assert target_probs.size(0) == num_draft_tokens + 1
    assert draft_probs.size(0) == num_draft_tokens

    accepted: list[int] = []
    for i in range(num_draft_tokens):
        token_id = int(draft_token_ids[i].item())
        p = float(target_probs[i, token_id].item())
        q = float(draft_probs[i, token_id].item())
        accept_prob = 1.0 if q <= 0 else min(1.0, p / q)
        u = float(random_values[i].item()) if random_values is not None else float(torch.rand(()).item())
        if u <= accept_prob:
            accepted.append(token_id)
            continue
        correction_probs = correction_distribution(target_probs[i], draft_probs[i])
        correction_token = sample_from_probs(correction_probs)
        return SpeculativeSampleResult(
            token_ids=accepted + [correction_token],
            accepted_token_ids=accepted,
            final_token_id=correction_token,
            num_accepted=len(accepted),
            accepted_all=False,
        )

    bonus_token = sample_from_probs(target_probs[num_draft_tokens])
    return SpeculativeSampleResult(
        token_ids=accepted + [bonus_token],
        accepted_token_ids=accepted,
        final_token_id=bonus_token,
        num_accepted=len(accepted),
        accepted_all=True,
    )


def speculative_accept_reject_from_logits(
    target_logits: torch.Tensor,
    draft_logits: torch.Tensor,
    draft_token_ids: torch.Tensor,
    temperature: float,
    random_values: torch.Tensor | None = None,
) -> SpeculativeSampleResult:
    # draft_logits 这里必须已经映射到 target vocab；不能直接传 32000 维 raw draft logits。
    assert temperature > 1e-10
    target_probs = torch.softmax(target_logits.float() / temperature, dim=-1)
    draft_probs = torch.softmax(draft_logits.float() / temperature, dim=-1)
    return speculative_accept_reject(target_probs, draft_probs, draft_token_ids, random_values=random_values)


def speculative_accept_greedy_from_logits(
    target_logits: torch.Tensor,
    draft_token_ids: torch.Tensor,
) -> SpeculativeSampleResult:
    num_draft_tokens = int(draft_token_ids.numel())
    assert target_logits.ndim == 2
    assert target_logits.size(0) == num_draft_tokens + 1

    target_token_ids = target_logits.argmax(dim=-1)
    accepted: list[int] = []
    for i in range(num_draft_tokens):
        draft_token_id = int(draft_token_ids[i].item())
        target_token_id = int(target_token_ids[i].item())
        if target_token_id != draft_token_id:
            return SpeculativeSampleResult(
                token_ids=accepted + [target_token_id],
                accepted_token_ids=accepted,
                final_token_id=target_token_id,
                num_accepted=len(accepted),
                accepted_all=False,
            )
        accepted.append(draft_token_id)

    bonus_token = int(target_token_ids[num_draft_tokens].item())
    return SpeculativeSampleResult(
        token_ids=accepted + [bonus_token],
        accepted_token_ids=accepted,
        final_token_id=bonus_token,
        num_accepted=len(accepted),
        accepted_all=True,
    )


def generate_eagle3_draft_tokens(
    draft_model,
    start_token_id: int,
    start_aux_hidden: torch.Tensor,
    start_position: int,
    gamma: int,
    temperature: float,
    past_kv: tuple[torch.Tensor, torch.Tensor] | None = None,
    kv_valid_len: int | None = None,
    draft_sampling_mode: str = "sample",
):
    assert gamma > 0
    assert draft_sampling_mode in {"sample", "greedy"}

    device = start_aux_hidden.device
    input_ids = torch.tensor([[start_token_id]], dtype=torch.long, device=device)
    positions = torch.tensor([[start_position]], dtype=torch.long, device=device)
    aux_hidden = start_aux_hidden
    draft_token_ids: list[int] = []
    draft_target_logits = []

    for step in range(gamma):
        kv_valid_lens = None
        if kv_valid_len is not None:
            kv_valid_lens = torch.tensor([kv_valid_len + step], dtype=torch.long, device=device)
        output = draft_model.propose(
            input_ids,
            aux_hidden,
            positions,
            temperature=temperature,
            past_kv=past_kv,
            kv_valid_lens=kv_valid_lens,
        )
        if draft_sampling_mode == "greedy":
            next_token_ids = draft_model.greedy_sample(output.draft_logits[:, -1:, :])
        else:
            next_token_ids = output.token_ids
        token_id = int(next_token_ids.item())
        draft_token_ids.append(token_id)
        draft_target_logits.append(output.target_logits[0, -1])

        input_ids = next_token_ids[:, -1:].to(device)
        # 第一步使用 target 的 3H aux hidden；后续 draft 自回归用上一轮 draft hidden 的 H 表示。
        aux_hidden = output.hidden_states[:, -1:, :]
        positions = positions + 1
        past_kv = output.past_kv

    draft_logits = torch.stack(draft_target_logits, dim=0)
    return Eagle3DraftSequence(draft_token_ids, draft_logits, past_kv)


def run_eagle3_speculative_cycle(
    draft_model,
    start_token_id: int,
    start_aux_hidden: torch.Tensor,
    start_position: int,
    target_verify_logits: torch.Tensor,
    gamma: int,
    temperature: float,
    random_values: torch.Tensor | None = None,
) -> Eagle3CycleResult:
    assert gamma > 0
    assert target_verify_logits.ndim == 2
    assert target_verify_logits.size(0) == gamma + 1

    draft_sequence = generate_eagle3_draft_tokens(
        draft_model,
        start_token_id=start_token_id,
        start_aux_hidden=start_aux_hidden,
        start_position=start_position,
        gamma=gamma,
        temperature=temperature,
    )
    sample_result = speculative_accept_reject_from_logits(
        target_verify_logits,
        draft_sequence.draft_target_logits,
        torch.tensor(draft_sequence.draft_token_ids, dtype=torch.long, device=target_verify_logits.device),
        temperature=temperature,
        random_values=random_values,
    )
    return Eagle3CycleResult(
        draft_sequence.draft_token_ids,
        draft_sequence.draft_target_logits,
        sample_result,
        draft_sequence.past_kv,
    )


def run_eagle3_target_verify(
    target_model,
    start_token_id: int,
    draft_token_ids: list[int],
    start_position: int,
) -> Eagle3TargetVerifyOutput:
    token_ids = [start_token_id] + list(draft_token_ids)
    device = next(target_model.parameters()).device if isinstance(target_model, nn.Module) else torch.device("cpu")
    input_ids = torch.tensor(token_ids, dtype=torch.long, device=device)
    positions = torch.arange(start_position, start_position + len(token_ids), dtype=torch.long, device=device)
    hidden_states, aux_hidden = target_model.forward_with_eagle3_aux(input_ids, positions)
    target_logits = target_model.compute_logits(hidden_states)
    return Eagle3TargetVerifyOutput(target_logits, aux_hidden)


def run_eagle3_offline_step(
    target_model,
    draft_model,
    start_token_id: int,
    start_aux_hidden: torch.Tensor,
    start_position: int,
    gamma: int,
    temperature: float,
    random_values: torch.Tensor | None = None,
) -> Eagle3OfflineStepResult:
    draft_sequence = generate_eagle3_draft_tokens(
        draft_model,
        start_token_id=start_token_id,
        start_aux_hidden=start_aux_hidden,
        start_position=start_position,
        gamma=gamma,
        temperature=temperature,
    )
    verify_output = run_eagle3_target_verify(
        target_model,
        start_token_id=start_token_id,
        draft_token_ids=draft_sequence.draft_token_ids,
        start_position=start_position,
    )
    sample_result = speculative_accept_reject_from_logits(
        verify_output.target_logits,
        draft_sequence.draft_target_logits,
        torch.tensor(draft_sequence.draft_token_ids, dtype=torch.long, device=verify_output.target_logits.device),
        temperature=temperature,
        random_values=random_values,
    )
    return Eagle3OfflineStepResult(
        draft_token_ids=draft_sequence.draft_token_ids,
        draft_target_logits=draft_sequence.draft_target_logits,
        verify_output=verify_output,
        sample_result=sample_result,
        past_kv=draft_sequence.past_kv,
    )


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
    ):
        super().__init__()
        self.hidden_size = hidden_size
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
        self.draft_vocab_size = config.get("draft_vocab_size", 32000)
        self.target_vocab_size = config["vocab_size"]

        self.embed_tokens = nn.Embedding(self.target_vocab_size, self.hidden_size)
        self.fc = nn.Linear(3 * self.hidden_size, self.hidden_size, bias=False)
        self.midlayer = Eagle3DecoderLayer(
            hidden_size=self.hidden_size,
            num_heads=config["num_attention_heads"],
            num_kv_heads=config["num_key_value_heads"],
            head_dim=config.get("head_dim", self.hidden_size // config["num_attention_heads"]),
            intermediate_size=config["intermediate_size"],
            max_position=config.get("max_position_embeddings", 40960),
            rope_theta=config.get("rope_theta", 1000000),
            rms_norm_eps=config.get("rms_norm_eps", 1e-6),
        )
        self.norm = Eagle3RMSNorm(self.hidden_size, eps=config.get("rms_norm_eps", 1e-6))
        self.lm_head = nn.Linear(self.hidden_size, self.draft_vocab_size, bias=False)

        # d2t/t2d 是 draft vocab 和 target vocab 的桥；概率拒绝采样需要在 target vocab 上比较 p/q。
        self.register_buffer("d2t", torch.zeros(self.draft_vocab_size, dtype=torch.long))
        self.register_buffer("t2d", torch.zeros(self.target_vocab_size, dtype=torch.bool))

    @classmethod
    def from_pretrained(cls, path: str, target_model_path: str | None = None) -> "Eagle3Speculator":
        with open(os.path.join(path, "config.json")) as f:
            config = json.load(f)
        model = cls(config)
        state = {}
        for file_name in os.listdir(path):
            if not file_name.endswith(".safetensors"):
                continue
            with safe_open(os.path.join(path, file_name), framework="pt", device="cpu") as f:
                for key in f.keys():
                    state[key] = f.get_tensor(key)
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
