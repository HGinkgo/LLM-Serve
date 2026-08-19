"""The GPTQ checkpoint contract used by the Qwen3-MoE runtime."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True, slots=True)
class GPTQConfig:
    """Official Qwen GPTQ-Int4, group-wise symmetric, no act-order."""

    bits: int
    group_size: int
    sym: bool
    desc_act: bool
    checkpoint_format: str

    @classmethod
    def from_dict(cls, config: dict[str, Any]) -> "GPTQConfig":
        if (
            config.get("quant_method") != "gptq"
            or config.get("bits") != 4
            or config.get("group_size") != 128
            or config.get("sym") is not True
            or config.get("desc_act") is not False
            or config.get("checkpoint_format") != "gptq"
        ):
            raise ValueError(
                "only GPTQ-Int4 with group_size=128, symmetric weights, "
                "desc_act=False and checkpoint_format=gptq is supported"
            )
        return cls(
            bits=4,
            group_size=128,
            sym=True,
            desc_act=False,
            checkpoint_format="gptq",
        )
