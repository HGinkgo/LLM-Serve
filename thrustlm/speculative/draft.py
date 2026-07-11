from contextlib import contextmanager
from time import perf_counter

import torch
import torch.nn.functional as F
from torch import nn

from thrustlm.speculative.sampling import speculative_accept_reject_from_logits
from thrustlm.speculative.types import (
    Eagle3CycleResult,
    Eagle3DraftSequence,
    Eagle3OfflineStepResult,
    Eagle3TargetVerifyOutput,
)


class _DraftStageProfiler:
    stage_names = (
        "draft_pack_time",
        "draft_forward_time",
        "draft_sample_time",
        "draft_compact_time",
    )

    def __init__(self, device: torch.device):
        self.device = device
        self.timings = {name: 0.0 for name in self.stage_names}
        self.cuda_events = []

    @contextmanager
    def stage(self, name: str):
        if self.device.type == "cuda":
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            yield
            end.record()
            self.cuda_events.append((name, start, end))
            return
        start = perf_counter()
        yield
        self.timings[name] += perf_counter() - start

    def finish(self) -> dict[str, float]:
        if self.cuda_events:
            torch.cuda.synchronize(self.device)
            for name, start, end in self.cuda_events:
                self.timings[name] += start.elapsed_time(end) / 1000
        return dict(self.timings)


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
    profiler = _DraftStageProfiler(device)
    input_ids = torch.tensor([[start_token_id]], dtype=torch.long, device=device)
    positions = torch.tensor([[start_position]], dtype=torch.long, device=device)
    aux_hidden = start_aux_hidden
    draft_token_ids = []
    draft_target_logits = []

    for step in range(gamma):
        with profiler.stage("draft_pack_time"):
            kv_valid_lens = None
            if kv_valid_len is not None:
                kv_valid_lens = torch.tensor([kv_valid_len + step], dtype=torch.long, device=device)
        with profiler.stage("draft_forward_time"):
            output = draft_model.propose(
                input_ids,
                aux_hidden,
                positions,
                temperature=temperature,
                past_kv=past_kv,
                kv_valid_lens=kv_valid_lens,
            )
        with profiler.stage("draft_sample_time"):
            next_token_ids = (
                draft_model.greedy_sample(output.draft_logits[:, -1:, :])
                if draft_sampling_mode == "greedy"
                else output.token_ids
            )
            token_id = int(next_token_ids.item())
        with profiler.stage("draft_compact_time"):
            draft_token_ids.append(token_id)
            draft_target_logits.append(output.target_logits[0, -1])
            input_ids = next_token_ids[:, -1:].to(device)
            aux_hidden = output.hidden_states[:, -1:, :]
            positions = positions + 1
            past_kv = output.past_kv

    return Eagle3DraftSequence(
        draft_token_ids,
        torch.stack(draft_target_logits, dim=0),
        past_kv,
        profiler.finish(),
    )


def _pack_eagle3_draft_kv(
    past_kv: list[tuple[torch.Tensor, torch.Tensor] | None],
) -> tuple[tuple[torch.Tensor, torch.Tensor] | None, list[int]]:
    valid_lengths = [0 if item is None else item[0].size(2) for item in past_kv]
    max_length = max(valid_lengths, default=0)
    if max_length == 0:
        return None, valid_lengths

    template = next(item for item in past_kv if item is not None)
    packed_k, packed_v = [], []
    for item, valid_length in zip(past_kv, valid_lengths):
        if item is None:
            shape = (1, template[0].size(1), max_length, template[0].size(3))
            packed_k.append(template[0].new_zeros(shape))
            packed_v.append(template[1].new_zeros(shape))
            continue
        if item[0].size(0) != 1 or item[1].size(0) != 1:
            raise ValueError("per-request draft KV must have batch size 1")
        padding = max_length - valid_length
        packed_k.append(F.pad(item[0], (0, 0, 0, padding)))
        packed_v.append(F.pad(item[1], (0, 0, 0, padding)))
    return (torch.cat(packed_k, dim=0), torch.cat(packed_v, dim=0)), valid_lengths


def _compact_eagle3_draft_kv(
    packed_past: tuple[torch.Tensor, torch.Tensor] | None,
    proposed_past: tuple[torch.Tensor, torch.Tensor],
    valid_lengths: list[int],
) -> list[tuple[torch.Tensor, torch.Tensor]]:
    compact = []
    for index, valid_length in enumerate(valid_lengths):
        if packed_past is None:
            old_k = proposed_past[0][index:index + 1, :, :0, :]
            old_v = proposed_past[1][index:index + 1, :, :0, :]
        else:
            old_k = packed_past[0][index:index + 1, :, :valid_length, :]
            old_v = packed_past[1][index:index + 1, :, :valid_length, :]
        new_k = proposed_past[0][index:index + 1, :, -1:, :]
        new_v = proposed_past[1][index:index + 1, :, -1:, :]
        compact.append((
            torch.cat([old_k, new_k], dim=2).contiguous(),
            torch.cat([old_v, new_v], dim=2).contiguous(),
        ))
    return compact


def generate_eagle3_draft_tokens_batched(
    draft_model,
    start_token_ids: list[int],
    start_aux_hidden: torch.Tensor,
    start_positions: list[int],
    gamma: int,
    temperature: float,
    past_kv: list[tuple[torch.Tensor, torch.Tensor] | None] | None = None,
    draft_sampling_mode: str = "sample",
    gammas: list[int] | None = None,
) -> list[Eagle3DraftSequence]:
    assert gamma > 0
    assert draft_sampling_mode in {"sample", "greedy"}
    batch_size = len(start_token_ids)
    if batch_size == 0:
        return []
    if len(start_positions) != batch_size or start_aux_hidden.size(0) != batch_size:
        raise ValueError("batched draft inputs have inconsistent sizes")
    request_gammas = list(gammas) if gammas is not None else [gamma] * batch_size
    if len(request_gammas) != batch_size or any(value <= 0 for value in request_gammas):
        raise ValueError("batched draft gammas must be positive and match batch size")

    device = start_aux_hidden.device
    profiler = _DraftStageProfiler(device)
    current_token_ids = list(start_token_ids)
    current_positions = list(start_positions)
    current_aux_hidden = [start_aux_hidden[index:index + 1] for index in range(batch_size)]
    compact_past = list(past_kv) if past_kv is not None else [None] * batch_size
    if len(compact_past) != batch_size:
        raise ValueError("batched draft KV count does not match batch size")
    draft_token_ids = [[] for _ in range(batch_size)]
    draft_target_logits = [[] for _ in range(batch_size)]

    for depth in range(max(request_gammas)):
        with profiler.stage("draft_pack_time"):
            active = [index for index, value in enumerate(request_gammas) if depth < value]
            packed_past, valid_lengths = _pack_eagle3_draft_kv([compact_past[index] for index in active])
            kv_valid_lens = (
                torch.tensor(valid_lengths, dtype=torch.long, device=device)
                if packed_past is not None else None
            )
            input_ids = torch.tensor(
                [current_token_ids[index] for index in active], dtype=torch.long, device=device
            ).view(len(active), 1)
            positions = torch.tensor(
                [current_positions[index] for index in active], dtype=torch.long, device=device
            ).view(len(active), 1)
            aux_hidden = torch.cat([current_aux_hidden[index] for index in active], dim=0)
        with profiler.stage("draft_forward_time"):
            output = draft_model.propose(
                input_ids,
                aux_hidden,
                positions,
                temperature=temperature,
                past_kv=packed_past,
                kv_valid_lens=kv_valid_lens,
            )
        with profiler.stage("draft_sample_time"):
            next_token_ids = (
                draft_model.greedy_sample(output.draft_logits[:, -1:, :])
                if draft_sampling_mode == "greedy" else output.token_ids
            )
        with profiler.stage("draft_compact_time"):
            proposed_past = _compact_eagle3_draft_kv(packed_past, output.past_kv, valid_lengths)
            for active_index, request_index in enumerate(active):
                token_id = int(next_token_ids[active_index, -1].item())
                draft_token_ids[request_index].append(token_id)
                draft_target_logits[request_index].append(output.target_logits[active_index, -1])
                compact_past[request_index] = proposed_past[active_index]
                current_token_ids[request_index] = token_id
                current_aux_hidden[request_index] = output.hidden_states[active_index:active_index + 1, -1:, :]
                current_positions[request_index] += 1

    timing = {name: value / batch_size for name, value in profiler.finish().items()}
    return [
        Eagle3DraftSequence(
            draft_token_ids[index],
            torch.stack(draft_target_logits[index], dim=0),
            compact_past[index],
            dict(timing),
        )
        for index in range(batch_size)
    ]


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
    assert gamma > 0 and target_verify_logits.ndim == 2
    assert target_verify_logits.size(0) == gamma + 1
    draft = generate_eagle3_draft_tokens(
        draft_model, start_token_id, start_aux_hidden, start_position, gamma, temperature
    )
    sample = speculative_accept_reject_from_logits(
        target_verify_logits,
        draft.draft_target_logits,
        torch.tensor(draft.draft_token_ids, dtype=torch.long, device=target_verify_logits.device),
        temperature=temperature,
        random_values=random_values,
    )
    return Eagle3CycleResult(draft.draft_token_ids, draft.draft_target_logits, sample, draft.past_kv)


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
    return Eagle3TargetVerifyOutput(target_model.compute_logits(hidden_states), aux_hidden)


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
    draft = generate_eagle3_draft_tokens(
        draft_model, start_token_id, start_aux_hidden, start_position, gamma, temperature
    )
    verify = run_eagle3_target_verify(target_model, start_token_id, draft.draft_token_ids, start_position)
    sample = speculative_accept_reject_from_logits(
        verify.target_logits,
        draft.draft_target_logits,
        torch.tensor(draft.draft_token_ids, dtype=torch.long, device=verify.target_logits.device),
        temperature=temperature,
        random_values=random_values,
    )
    return Eagle3OfflineStepResult(
        draft.draft_token_ids,
        draft.draft_target_logits,
        verify,
        sample,
        draft.past_kv,
    )
