import torch

from thrustlm.speculative.types import SpeculativeSampleResult


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
        correction_token = sample_from_probs(correction_distribution(target_probs[i], draft_probs[i]))
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
