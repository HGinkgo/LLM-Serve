import torch
import torch.nn.functional as F
from torch import nn
import triton
import triton.language as tl

from flash_attn import flash_attn_varlen_func, flash_attn_with_kvcache
from llmserve.utils.context import get_context


@triton.jit
def store_kvcache_kernel(
    key_ptr,
    key_stride,
    value_ptr,
    value_stride,
    k_cache_ptr,
    v_cache_ptr,
    slot_mapping_ptr,
    D: tl.constexpr,
):
    idx = tl.program_id(0)
    slot = tl.load(slot_mapping_ptr + idx)
    if slot == -1: return
    key_offsets = idx * key_stride + tl.arange(0, D)
    value_offsets = idx * value_stride + tl.arange(0, D)
    key = tl.load(key_ptr + key_offsets)
    value = tl.load(value_ptr + value_offsets)
    cache_offsets = slot * D + tl.arange(0, D)
    tl.store(k_cache_ptr + cache_offsets, key)
    tl.store(v_cache_ptr + cache_offsets, value)


def store_kvcache(key: torch.Tensor, value: torch.Tensor, k_cache: torch.Tensor, v_cache: torch.Tensor, slot_mapping: torch.Tensor):
    N, num_heads, head_dim = key.shape
    D = num_heads * head_dim
    assert key.stride(-1) == 1 and value.stride(-1) == 1
    assert key.stride(1) == head_dim and value.stride(1) == head_dim
    assert k_cache.stride(1) == D and v_cache.stride(1) == D
    assert slot_mapping.numel() == N
    store_kvcache_kernel[(N,)](key, key.stride(0), value, value.stride(0), k_cache, v_cache, slot_mapping, D)


class Attention(nn.Module):

    def __init__(
        self,
        num_heads,
        head_dim,
        scale,
        num_kv_heads,
    ):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.scale = scale
        self.num_kv_heads = num_kv_heads
        self.k_cache = self.v_cache = torch.tensor([])
        self.tree_k = self.tree_v = torch.tensor([])

    def _forward_tree(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, context):
        self.tree_k = k
        self.tree_v = v
        prefix_slots = context.tree_prefix_slots.to(device=k.device, dtype=torch.long)
        flat_k_cache = self.k_cache.view(-1, self.num_kv_heads, self.head_dim)
        flat_v_cache = self.v_cache.view(-1, self.num_kv_heads, self.head_dim)
        prefix_k = flat_k_cache.index_select(0, prefix_slots)
        prefix_v = flat_v_cache.index_select(0, prefix_slots)
        all_k = torch.cat([prefix_k, k], dim=0)
        all_v = torch.cat([prefix_v, v], dim=0)
        if self.num_heads != self.num_kv_heads:
            groups = self.num_heads // self.num_kv_heads
            all_k = all_k.repeat_interleave(groups, dim=1)
            all_v = all_v.repeat_interleave(groups, dim=1)

        num_prefix = prefix_slots.numel()
        prefix_mask = torch.ones(
            (q.size(0), num_prefix),
            dtype=torch.bool,
            device=q.device,
        )
        tree_mask = context.tree_attention_mask.to(device=q.device, dtype=torch.bool)
        attention_mask = torch.cat([prefix_mask, tree_mask], dim=1)[None, None]
        output = F.scaled_dot_product_attention(
            q.transpose(0, 1).unsqueeze(0),
            all_k.transpose(0, 1).unsqueeze(0),
            all_v.transpose(0, 1).unsqueeze(0),
            attn_mask=attention_mask,
            dropout_p=0.0,
            is_causal=False,
            scale=self.scale,
        )
        return output.squeeze(0).transpose(0, 1).contiguous()

    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor):
        context = get_context()
        k_cache, v_cache = self.k_cache, self.v_cache
        if context.tree_attention_mask is not None:
            return self._forward_tree(q, k, v, context)
        if k_cache.numel() and v_cache.numel():
            store_kvcache(k, v, k_cache, v_cache, context.slot_mapping)
        if context.is_prefill:
            if context.block_tables is not None:    # prefix cache
                k, v = k_cache, v_cache
            o = flash_attn_varlen_func(q, k, v,
                                       max_seqlen_q=context.max_seqlen_q, cu_seqlens_q=context.cu_seqlens_q,
                                       max_seqlen_k=context.max_seqlen_k, cu_seqlens_k=context.cu_seqlens_k,
                                       softmax_scale=self.scale, causal=True, block_table=context.block_tables)
        else:    # decode
            o = flash_attn_with_kvcache(q.unsqueeze(1), k_cache, v_cache,
                                        cache_seqlens=context.context_lens, block_table=context.block_tables, 
                                        softmax_scale=self.scale, causal=True)
        return o
