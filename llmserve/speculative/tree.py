from dataclasses import dataclass

import torch

from llmserve.speculative.draft import (
    _compact_eagle3_draft_kv,
    _pack_eagle3_draft_kv,
)


@dataclass(frozen=True, slots=True)
class Eagle3TreeTopology:
    parents: tuple[int, ...]

    def __post_init__(self):
        if not self.parents or self.parents[0] != -1:
            raise ValueError("tree root must have parent -1")
        for node_index, parent_index in enumerate(self.parents[1:], start=1):
            if parent_index < 0 or parent_index >= node_index:
                raise ValueError("tree parents must precede their children")

    @property
    def num_draft_nodes(self) -> int:
        return len(self.parents) - 1

    @property
    def depths(self) -> tuple[int, ...]:
        depths = [0]
        for parent_index in self.parents[1:]:
            depths.append(depths[parent_index] + 1)
        return tuple(depths)

    def children(self, parent_index: int) -> tuple[int, ...]:
        return tuple(
            node_index
            for node_index, candidate_parent in enumerate(self.parents)
            if candidate_parent == parent_index
        )

    def attention_mask(self, device: torch.device | None = None) -> torch.Tensor:
        num_nodes = len(self.parents)
        mask = torch.zeros((num_nodes, num_nodes), dtype=torch.bool, device=device)
        for node_index in range(num_nodes):
            ancestor = node_index
            while ancestor >= 0:
                mask[node_index, ancestor] = True
                ancestor = self.parents[ancestor]
        return mask


@dataclass(slots=True)
class Eagle3TreeAcceptResult:
    token_ids: list[int]
    accepted_node_indices: list[int]
    commit_node_indices: list[int]
    final_token_id: int
    final_node_index: int
    num_accepted: int
    accepted_all: bool


@dataclass(slots=True)
class Eagle3DraftTree:
    topology: Eagle3TreeTopology
    draft_token_ids: list[int]
    processed_past_kv: list[tuple[torch.Tensor, torch.Tensor] | None]

    def past_kv_for_path(
        self,
        accepted_node_indices: list[int],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        current_node = 0
        for node_index in accepted_node_indices:
            if self.topology.parents[node_index] != current_node:
                raise ValueError("accepted tree nodes do not form a path")
            current_node = node_index
        while self.processed_past_kv[current_node] is None:
            current_node = self.topology.parents[current_node]
        return self.processed_past_kv[current_node]


def build_fixed_tree_topology(num_draft_nodes: int) -> Eagle3TreeTopology:
    if num_draft_nodes == 6:
        parents = (-1, 0, 0, 1, 2, 3, 4)
    elif num_draft_nodes == 10:
        parents = (-1, 0, 0, 1, 1, 2, 2, 3, 4, 5, 6)
    else:
        raise ValueError("fixed EAGLE3 tree supports 6 or 10 draft nodes")
    return Eagle3TreeTopology(parents)


def select_greedy_tree_path(
    topology: Eagle3TreeTopology,
    draft_token_ids: list[int],
    target_logits: torch.Tensor,
) -> Eagle3TreeAcceptResult:
    if len(draft_token_ids) != topology.num_draft_nodes:
        raise ValueError("draft token count does not match tree topology")
    if target_logits.ndim != 2 or target_logits.size(0) != len(topology.parents):
        raise ValueError("target logits do not match tree topology")

    current_node = 0
    accepted_nodes = []
    accepted_tokens = []
    while True:
        target_token_id = int(target_logits[current_node].argmax().item())
        matching_child = next(
            (
                child_index
                for child_index in topology.children(current_node)
                if int(draft_token_ids[child_index - 1]) == target_token_id
            ),
            None,
        )
        if matching_child is None:
            final_token_id = target_token_id
            break
        accepted_nodes.append(matching_child)
        accepted_tokens.append(target_token_id)
        current_node = matching_child

    accepted_all = not topology.children(current_node)
    return Eagle3TreeAcceptResult(
        token_ids=accepted_tokens + [final_token_id],
        accepted_node_indices=accepted_nodes,
        commit_node_indices=[0] + accepted_nodes,
        final_token_id=final_token_id,
        final_node_index=current_node,
        num_accepted=len(accepted_nodes),
        accepted_all=accepted_all,
    )


def _greedy_topk_tokens(draft_model, draft_logits: torch.Tensor, k: int) -> torch.Tensor:
    if hasattr(draft_model, "greedy_topk"):
        return draft_model.greedy_topk(draft_logits, k)
    draft_ids = torch.topk(draft_logits, k=k, dim=-1).indices
    return draft_ids + draft_model.d2t[draft_ids]


def generate_eagle3_draft_tree(
    draft_model,
    *,
    topology: Eagle3TreeTopology,
    start_token_id: int,
    start_aux_hidden: torch.Tensor,
    start_position: int,
    temperature: float,
    past_kv: tuple[torch.Tensor, torch.Tensor] | None = None,
) -> Eagle3DraftTree:
    num_nodes = len(topology.parents)
    device = start_aux_hidden.device
    token_ids: list[int | None] = [None] * num_nodes
    token_ids[0] = int(start_token_id)
    aux_hidden: list[torch.Tensor | None] = [None] * num_nodes
    aux_hidden[0] = start_aux_hidden
    past_before: list[tuple[torch.Tensor, torch.Tensor] | None] = [None] * num_nodes
    past_before[0] = past_kv
    processed_past: list[tuple[torch.Tensor, torch.Tensor] | None] = [None] * num_nodes

    max_depth = max(topology.depths)
    for depth in range(max_depth):
        active_nodes = [
            node_index
            for node_index, node_depth in enumerate(topology.depths)
            if node_depth == depth and topology.children(node_index)
        ]
        active_past = [past_before[node_index] for node_index in active_nodes]
        packed_past, valid_lengths = _pack_eagle3_draft_kv(active_past)
        kv_valid_lens = None
        if packed_past is not None:
            kv_valid_lens = torch.tensor(valid_lengths, dtype=torch.long, device=device)
        input_ids = torch.tensor(
            [token_ids[node_index] for node_index in active_nodes],
            dtype=torch.long,
            device=device,
        ).view(len(active_nodes), 1)
        positions = torch.tensor(
            [start_position + depth] * len(active_nodes),
            dtype=torch.long,
            device=device,
        ).view(len(active_nodes), 1)
        active_aux = torch.cat([aux_hidden[node_index] for node_index in active_nodes], dim=0)
        output = draft_model.propose(
            input_ids,
            active_aux,
            positions,
            temperature=temperature,
            past_kv=packed_past,
            kv_valid_lens=kv_valid_lens,
        )
        compact_past = _compact_eagle3_draft_kv(
            packed_past,
            output.past_kv,
            valid_lengths,
        )

        for active_index, parent_index in enumerate(active_nodes):
            processed_past[parent_index] = compact_past[active_index]
            children = topology.children(parent_index)
            child_tokens = _greedy_topk_tokens(
                draft_model,
                output.draft_logits[active_index:active_index + 1, -1:, :],
                len(children),
            ).flatten().tolist()
            for child_index, child_token in zip(children, child_tokens):
                token_ids[child_index] = int(child_token)
                aux_hidden[child_index] = output.hidden_states[
                    active_index:active_index + 1, -1:, :
                ]
                past_before[child_index] = compact_past[active_index]

    if any(token_id is None for token_id in token_ids):
        raise RuntimeError("draft tree generation left uninitialized nodes")
    return Eagle3DraftTree(
        topology=topology,
        draft_token_ids=[int(token_id) for token_id in token_ids[1:]],
        processed_past_kv=processed_past,
    )
