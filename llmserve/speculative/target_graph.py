from dataclasses import dataclass
from typing import Sequence

import torch

from llmserve.utils.context import reset_context, set_context


@dataclass(frozen=True, slots=True)
class TargetVerifyGraphKey:
    batch_size: int
    context_frontier: int


class TargetVerifyGraphPlan:
    DEFAULT_BATCH_BUCKETS = (1, 4, 8)
    DEFAULT_CONTEXT_FRONTIERS = (256, 1024)
    MAX_CAPTURE_CONTEXT = max(DEFAULT_CONTEXT_FRONTIERS)

    def __init__(
        self,
        *,
        gamma: int,
        max_batch_size: int,
        max_model_len: int,
    ):
        if gamma <= 0 or max_batch_size <= 0 or max_model_len <= 0:
            raise ValueError("target verify graph limits must be positive")
        self.verify_width = gamma + 1
        self.batch_buckets = self._bounded_frontiers(
            self.DEFAULT_BATCH_BUCKETS,
            max_batch_size,
            include_limit=False,
        )
        capture_context_limit = min(max_model_len, self.MAX_CAPTURE_CONTEXT)
        self.context_frontiers = self._bounded_frontiers(
            self.DEFAULT_CONTEXT_FRONTIERS,
            capture_context_limit,
        )

    @staticmethod
    def _bounded_frontiers(
        frontiers: Sequence[int],
        limit: int,
        *,
        include_limit: bool = True,
    ) -> tuple[int, ...]:
        bounded = {value for value in frontiers if value <= limit}
        if include_limit:
            bounded.add(limit)
        return tuple(sorted(bounded))

    def select(
        self,
        verify_lengths: Sequence[int],
        max_seqlen_k: int,
    ) -> TargetVerifyGraphKey | None:
        if not verify_lengths or any(
            length != self.verify_width for length in verify_lengths
        ):
            return None
        batch_size = (
            len(verify_lengths)
            if len(verify_lengths) in self.batch_buckets
            else None
        )
        context_frontier = next(
            (
                frontier
                for frontier in self.context_frontiers
                if frontier >= max_seqlen_k
            ),
            None,
        )
        if batch_size is None or context_frontier is None:
            return None
        return TargetVerifyGraphKey(batch_size, context_frontier)


class TargetVerifyGraphWorkspace:
    def __init__(
        self,
        key: TargetVerifyGraphKey,
        *,
        verify_width: int,
        block_size: int,
        device: torch.device | str,
    ):
        self.key = key
        self.verify_width = verify_width
        self.block_size = block_size
        self.device = torch.device(device)
        num_tokens = key.batch_size * verify_width
        max_blocks = (key.context_frontier + block_size - 1) // block_size
        self.input_ids = torch.empty(num_tokens, dtype=torch.int64, device=self.device)
        self.positions = torch.empty(num_tokens, dtype=torch.int64, device=self.device)
        self.slot_mapping = torch.empty(num_tokens, dtype=torch.int32, device=self.device)
        self.cu_seqlens_q = torch.empty(
            key.batch_size + 1,
            dtype=torch.int32,
            device=self.device,
        )
        self.cu_seqlens_k = torch.empty_like(self.cu_seqlens_q)
        self.block_tables = torch.full(
            (key.batch_size, max_blocks),
            -1,
            dtype=torch.int32,
            device=self.device,
        )

    def _copy_sequence(self, destination: torch.Tensor, values: Sequence[int]):
        if len(values) != destination.numel():
            raise ValueError(
                f"target verify graph token count mismatch: "
                f"expected {destination.numel()}, got {len(values)}"
            )
        destination.copy_(
            torch.as_tensor(values, dtype=destination.dtype, device=self.device)
        )

    def load(self, metadata: dict, block_tables: torch.Tensor):
        self._copy_sequence(self.input_ids, metadata["input_ids"])
        self._copy_sequence(self.positions, metadata["positions"])
        self._copy_sequence(self.slot_mapping, metadata["slot_mapping"])
        self._copy_sequence(self.cu_seqlens_q, metadata["cu_seqlens_q"])
        self._copy_sequence(self.cu_seqlens_k, metadata["cu_seqlens_k"])

        if block_tables.ndim != 2 or block_tables.size(0) != self.key.batch_size:
            raise ValueError("target verify graph block table batch mismatch")
        if block_tables.size(1) > self.block_tables.size(1):
            raise ValueError("target verify graph block table exceeds context frontier")
        self.block_tables.fill_(-1)
        self.block_tables[:, :block_tables.size(1)].copy_(block_tables)


class TargetVerifyGraphBackend:
    def __init__(
        self,
        model,
        *,
        gamma: int,
        max_batch_size: int,
        max_model_len: int,
        block_size: int,
        device: torch.device | str,
    ):
        self.model = model
        self.device = torch.device(device)
        self.plan = TargetVerifyGraphPlan(
            gamma=gamma,
            max_batch_size=max_batch_size,
            max_model_len=max_model_len,
        )
        self.block_size = block_size
        self.workspaces: dict[TargetVerifyGraphKey, TargetVerifyGraphWorkspace] = {}
        self.graphs = {}
        self.outputs = {}
        self.graph_pool = None
        self.graph_replays = 0
        self.eager_fallbacks = 0
        self.graph_replays_by_key = {}
        self.fallback_reasons = {}
        self.last_key = None

    @staticmethod
    def _key_label(key: TargetVerifyGraphKey | None):
        return (
            f"batch{key.batch_size}-context{key.context_frontier}"
            if key is not None else "unsupported"
        )

    def _record_fallback(self, reason: str):
        self.eager_fallbacks += 1
        self.fallback_reasons[reason] = self.fallback_reasons.get(reason, 0) + 1

    def _capture_metadata(self, key: TargetVerifyGraphKey):
        width = self.plan.verify_width
        max_blocks = (key.context_frontier + self.block_size - 1) // self.block_size
        input_ids = list(range(key.batch_size * width))
        positions = []
        slot_mapping = []
        cu_q = [0]
        cu_k = [0]
        for batch_index in range(key.batch_size):
            base_pos = key.context_frontier - width
            positions.extend(range(base_pos, base_pos + width))
            for position in range(base_pos, base_pos + width):
                slot_mapping.append(
                    (position // self.block_size) * self.block_size
                    + position % self.block_size
                )
            cu_q.append(cu_q[-1] + width)
            cu_k.append(cu_k[-1] + key.context_frontier)
        block_tables = torch.arange(
            max_blocks,
            dtype=torch.int32,
            device=self.device,
        ).expand(key.batch_size, max_blocks).contiguous()
        return {
            "input_ids": input_ids,
            "positions": positions,
            "slot_mapping": slot_mapping,
            "cu_seqlens_q": cu_q,
            "cu_seqlens_k": cu_k,
        }, block_tables

    @torch.inference_mode()
    def capture(self):
        if self.device.type != "cuda":
            raise RuntimeError("target verify CUDA Graph requires a CUDA device")
        capture_stream = torch.cuda.Stream(device=self.device)
        attention_layers = [
            module for module in self.model.modules()
            if hasattr(module, "k_cache") and hasattr(module, "v_cache")
        ]
        if not attention_layers:
            raise RuntimeError("target verify CUDA Graph requires attention KV caches")
        num_kv_blocks = attention_layers[0].k_cache.size(0)
        torch.cuda.synchronize(self.device)
        for batch_size in self.plan.batch_buckets:
            for context_frontier in self.plan.context_frontiers:
                key = TargetVerifyGraphKey(batch_size, context_frontier)
                workspace = TargetVerifyGraphWorkspace(
                    key,
                    verify_width=self.plan.verify_width,
                    block_size=self.block_size,
                    device=self.device,
                )
                metadata, block_tables = self._capture_metadata(key)
                if block_tables.max().item() >= num_kv_blocks:
                    raise RuntimeError(
                        f"target verify graph needs block {int(block_tables.max().item())}, "
                        f"but KV cache has {num_kv_blocks} blocks"
                    )
                workspace.load(metadata, block_tables)
                graph = torch.cuda.CUDAGraph()
                try:
                    with torch.cuda.stream(capture_stream):
                        try:
                            set_context(
                                True,
                                cu_seqlens_q=workspace.cu_seqlens_q,
                                cu_seqlens_k=workspace.cu_seqlens_k,
                                max_seqlen_q=self.plan.verify_width,
                                max_seqlen_k=context_frontier,
                                slot_mapping=workspace.slot_mapping,
                                block_tables=workspace.block_tables,
                            )
                            self.model.forward_with_eagle3_aux(
                                workspace.input_ids,
                                workspace.positions,
                            )
                            capture_stream.synchronize()
                            with torch.cuda.graph(
                                graph,
                                self.graph_pool,
                                stream=capture_stream,
                            ):
                                hidden_states, aux_hidden = self.model.forward_with_eagle3_aux(
                                    workspace.input_ids,
                                    workspace.positions,
                                )
                                logits = self.model.compute_logits(
                                    hidden_states,
                                    all_tokens=True,
                                )
                        finally:
                            reset_context()
                except Exception as error:
                    self.workspaces.clear()
                    self.graphs.clear()
                    self.outputs.clear()
                    self.graph_pool = None
                    raise RuntimeError(
                        f"target verify graph capture failed for batch={batch_size}, "
                        f"context={context_frontier}, kv_blocks={num_kv_blocks}"
                    ) from error
                if self.graph_pool is None:
                    self.graph_pool = graph.pool()
                self.workspaces[key] = workspace
                self.graphs[key] = graph
                self.outputs[key] = (logits, aux_hidden)
        torch.cuda.synchronize(self.device)

    def run_if_supported(self, metadata: dict, block_tables: torch.Tensor):
        key = self.plan.select(
            metadata["verify_lengths"],
            metadata["max_seqlen_k"],
        )
        if key is None:
            self._record_fallback("unsupported_shape")
            return None
        if key not in self.graphs:
            self._record_fallback("graph_not_captured")
            return None
        workspace = self.workspaces[key]
        if block_tables.ndim != 2 or block_tables.size(0) != key.batch_size:
            self._record_fallback("block_table_batch_mismatch")
            return None
        if block_tables.size(1) > workspace.block_tables.size(1):
            self._record_fallback("block_table_capacity")
            return None
        try:
            workspace.load(metadata, block_tables)
        except ValueError:
            self._record_fallback("workspace_input_mismatch")
            return None
        set_context(
            True,
            cu_seqlens_q=workspace.cu_seqlens_q,
            cu_seqlens_k=workspace.cu_seqlens_k,
            max_seqlen_q=self.plan.verify_width,
            max_seqlen_k=key.context_frontier,
            slot_mapping=workspace.slot_mapping,
            block_tables=workspace.block_tables,
        )
        try:
            self.graphs[key].replay()
        finally:
            reset_context()
        self.graph_replays += 1
        label = self._key_label(key)
        self.graph_replays_by_key[label] = self.graph_replays_by_key.get(label, 0) + 1
        self.last_key = key
        logits, aux_hidden = self.outputs[key]
        return logits, aux_hidden

    def metrics(self):
        return {
            "enabled": True,
            "captured_graphs": len(self.graphs),
            "graph_replays": self.graph_replays,
            "eager_fallbacks": self.eager_fallbacks,
            "graph_replays_by_key": dict(sorted(self.graph_replays_by_key.items())),
            "fallback_reasons": dict(sorted(self.fallback_reasons.items())),
            "last_key": (
                {
                    "batch_size": self.last_key.batch_size,
                    "context_frontier": self.last_key.context_frontier,
                }
                if self.last_key is not None else None
            ),
        }
