from collections.abc import Sequence

import torch

try:
    import triton
    import triton.language as tl
except ImportError:
    triton = None
    tl = None


if triton is not None:
    @triton.jit
    def _fused_commit_tree_kv_kernel(
        tree_k_ptr,
        tree_v_ptr,
        k_cache_ptr,
        v_cache_ptr,
        node_indices_ptr,
        slot_mapping_ptr,
        tree_layer_stride: tl.constexpr,
        cache_layer_stride: tl.constexpr,
        KV_DIM: tl.constexpr,
        BLOCK_SIZE: tl.constexpr,
    ):
        layer_index = tl.program_id(0)
        commit_index = tl.program_id(1)
        node_index = tl.load(node_indices_ptr + commit_index)
        slot = tl.load(slot_mapping_ptr + commit_index)
        offsets = tl.arange(0, BLOCK_SIZE)
        mask = offsets < KV_DIM
        source_offsets = layer_index * tree_layer_stride + node_index * KV_DIM + offsets
        target_offsets = layer_index * cache_layer_stride + slot * KV_DIM + offsets
        key = tl.load(tree_k_ptr + source_offsets, mask=mask)
        value = tl.load(tree_v_ptr + source_offsets, mask=mask)
        tl.store(k_cache_ptr + target_offsets, key, mask=mask)
        tl.store(v_cache_ptr + target_offsets, value, mask=mask)
else:
    _fused_commit_tree_kv_kernel = None


def fused_commit_tree_kv(
    tree_k: torch.Tensor,
    tree_v: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    node_indices: torch.Tensor,
    slot_mapping: torch.Tensor,
):
    if tree_k.ndim != 4 or tree_k.shape != tree_v.shape:
        raise ValueError("fused Tree K/V must have matching four-dimensional shapes")
    if k_cache.ndim != 5 or k_cache.shape != v_cache.shape:
        raise ValueError("target K/V cache must have matching five-dimensional shapes")
    if tree_k.size(0) != k_cache.size(0) or tree_k.shape[2:] != k_cache.shape[3:]:
        raise ValueError("Tree KV and target cache layer/head shapes must match")
    if not tree_k.is_contiguous() or not tree_v.is_contiguous():
        raise ValueError("fused Tree K/V buffers must be contiguous")
    if node_indices.ndim != 1 or node_indices.numel() != slot_mapping.numel():
        raise ValueError("selected nodes and slots must have matching one-dimensional shapes")
    if _fused_commit_tree_kv_kernel is None:
        raise RuntimeError("Triton is required for fused Tree KV commit")
    kv_dim = tree_k.size(2) * tree_k.size(3)
    block_size = triton.next_power_of_2(kv_dim)
    _fused_commit_tree_kv_kernel[(tree_k.size(0), node_indices.numel())](
        tree_k,
        tree_v,
        k_cache,
        v_cache,
        node_indices,
        slot_mapping,
        tree_k.stride(0),
        k_cache.stride(0),
        kv_dim,
        block_size,
    )


class TreeKVCacheManager:

    def __init__(
        self,
        attention_layers: Sequence,
        *,
        k_cache: torch.Tensor | None = None,
        v_cache: torch.Tensor | None = None,
    ):
        if not attention_layers:
            raise ValueError("Tree KV manager requires at least one attention layer")
        self.attention_layers = list(attention_layers)
        self.k_cache = k_cache
        self.v_cache = v_cache

    @property
    def device(self) -> torch.device:
        for layer in self.attention_layers:
            if layer.tree_k.numel():
                return layer.tree_k.device
        if self.k_cache is not None:
            return self.k_cache.device
        return torch.device("cpu")

    def _clear_captures(self):
        for layer in self.attention_layers:
            layer.tree_k = layer.tree_k.new_empty(0)
            layer.tree_v = layer.tree_v.new_empty(0)

    def _validate_commit(self, node_indices: list[int], slot_mapping: torch.Tensor):
        if slot_mapping.ndim != 1 or slot_mapping.numel() != len(node_indices):
            raise ValueError("Tree KV slot count must match selected node count")
        shapes = []
        for layer in self.attention_layers:
            if layer.tree_k.numel() == 0 or layer.tree_v.numel() == 0:
                raise ValueError("every target layer must capture Tree KV before commit")
            if layer.tree_k.shape != layer.tree_v.shape:
                raise ValueError("Tree K and V must have the same shape")
            shapes.append(layer.tree_k.shape)
        if any(shape != shapes[0] for shape in shapes[1:]):
            raise ValueError("all target layers must capture Tree KV with the same shape")
        num_nodes = shapes[0][0]
        if not node_indices or any(
            not isinstance(index, int) or index < 0 or index >= num_nodes
            for index in node_indices
        ):
            raise ValueError("invalid Tree KV node index")

    def commit(self, node_indices: list[int], slot_mapping: torch.Tensor):
        try:
            self._validate_commit(node_indices, slot_mapping)
            if self.k_cache is None or self.v_cache is None:
                raise ValueError("fused Tree KV commit requires the global target cache")
            tree_k = torch.stack(
                [layer.tree_k for layer in self.attention_layers],
                dim=0,
            )
            tree_v = torch.stack(
                [layer.tree_v for layer in self.attention_layers],
                dim=0,
            )
            node_indices_tensor = torch.tensor(
                node_indices,
                dtype=torch.int32,
                device=tree_k.device,
            )
            fused_commit_tree_kv(
                tree_k,
                tree_v,
                self.k_cache,
                self.v_cache,
                node_indices_tensor,
                slot_mapping,
            )
        finally:
            self._clear_captures()
