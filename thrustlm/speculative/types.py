from dataclasses import dataclass

import torch


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
    proposal_timing: dict[str, float] | None = None


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


@dataclass(slots=True)
class TargetDecodeAuxOutput:
    token_ids: list[int]
    logits: torch.Tensor
    aux_hidden: torch.Tensor
    positions: torch.Tensor


@dataclass(slots=True)
class SpeculativeDecodeOutput:
    token_ids: list[int]
    num_draft_tokens: int
    num_accepted: int
    accepted_all: bool
    emitted_tokens: int
    timing: dict | None = None
    debug: dict | None = None
