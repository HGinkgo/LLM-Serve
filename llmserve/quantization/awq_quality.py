import argparse
import json
import math
from pathlib import Path
import time

import torch
import torch.nn.functional as F
from torch import nn

from llmserve.layers.quantization.awq import awq_reference_linear
from llmserve.quantization.qwen3_awq import QWEN3_QUANTIZED_LINEAR_NAMES


class ReferenceAWQLinear(nn.Module):
    def __init__(
        self,
        input_size: int,
        output_size: int,
        *,
        group_size: int = 128,
        dtype: torch.dtype = torch.bfloat16,
        device=None,
    ):
        super().__init__()
        if input_size % group_size != 0 or output_size % 8 != 0:
            raise ValueError("AWQ linear shape is incompatible with packing")
        self.group_size = group_size
        self.in_features = input_size
        self.out_features = output_size
        self.register_buffer(
            "qweight",
            torch.empty(input_size, output_size // 8, dtype=torch.int32, device=device),
        )
        self.register_buffer(
            "qzeros",
            torch.empty(
                input_size // group_size,
                output_size // 8,
                dtype=torch.int32,
                device=device,
            ),
        )
        self.register_buffer(
            "scales",
            torch.empty(
                input_size // group_size,
                output_size,
                dtype=dtype,
                device=device,
            ),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return awq_reference_linear(
            inputs,
            self.qweight,
            self.qzeros,
            self.scales,
            group_size=self.group_size,
        )


def _reset_cuda_peak_memory_stats(device: torch.device) -> None:
    torch.cuda.set_device(device)
    torch.cuda.reset_peak_memory_stats()


def _max_cuda_memory_allocated(device: torch.device) -> int:
    torch.cuda.set_device(device)
    return torch.cuda.max_memory_allocated()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Evaluate BF16 or packed AWQ Qwen3 quality on fixed local text."
    )
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--eval-file", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--mode", choices=("auto", "bf16", "awq-reference"), default="auto")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--max-samples", type=int, default=32)
    parser.add_argument("--sequence-length", type=int, default=128)
    parser.add_argument("--max-eval-tokens", type=int, default=2048)
    parser.add_argument("--generation-prompts", type=int, default=2)
    parser.add_argument("--max-new-tokens", type=int, default=16)
    return parser


def _replace_qwen3_linears(model: nn.Module, group_size: int, dtype: torch.dtype) -> None:
    target_suffixes = tuple(f".{name}" for name in QWEN3_QUANTIZED_LINEAR_NAMES)
    for name, module in list(model.named_modules()):
        if not isinstance(module, nn.Linear):
            continue
        if not any(name.endswith(suffix) for suffix in target_suffixes):
            continue
        if module.bias is not None:
            raise ValueError(f"AWQ reference loader does not support bias: {name}")
        parent_name, child_name = name.rsplit(".", 1)
        parent = model.get_submodule(parent_name)
        replacement = ReferenceAWQLinear(
            module.in_features,
            module.out_features,
            group_size=group_size,
            dtype=dtype,
            device=module.weight.device,
        )
        setattr(parent, child_name, replacement)


def _load_awq_reference_model(model_path: Path, device: torch.device):
    from accelerate import init_empty_weights, load_checkpoint_in_model
    from accelerate.utils import set_module_tensor_to_device
    from transformers import AutoConfig
    from transformers.models.qwen3.modeling_qwen3 import (
        Qwen3ForCausalLM,
        Qwen3RotaryEmbedding,
    )

    config = AutoConfig.from_pretrained(model_path)
    quantization_config = getattr(config, "quantization_config", None)
    if not isinstance(quantization_config, dict):
        raise ValueError("AWQ reference mode requires quantization_config")
    group_size = int(quantization_config.get("group_size", 0))
    if quantization_config.get("quant_method") != "awq" or group_size <= 0:
        raise ValueError("unsupported quantization_config")
    dtype = getattr(config, "dtype", torch.bfloat16)
    with init_empty_weights(include_buffers=True):
        model = Qwen3ForCausalLM(config)
        _replace_qwen3_linears(model, group_size, dtype)
    load_checkpoint_in_model(
        model,
        checkpoint=str(model_path),
        device_map={"": str(device)},
        dtype=dtype,
        strict=False,
    )
    rotary = Qwen3RotaryEmbedding(config, device=device)
    for name, buffer in list(model.named_buffers()):
        if buffer.device.type != "meta":
            continue
        if name == "model.rotary_emb.inv_freq":
            set_module_tensor_to_device(
                model,
                name,
                device,
                value=rotary.inv_freq,
            )
            continue
        raise RuntimeError(f"checkpoint left an uninitialized buffer: {name}")
    return model.eval()


def _load_model(model_path: Path, mode: str, device: torch.device):
    from transformers import AutoConfig, AutoModelForCausalLM

    config = AutoConfig.from_pretrained(model_path)
    is_awq = isinstance(getattr(config, "quantization_config", None), dict)
    if mode == "auto":
        mode = "awq-reference" if is_awq else "bf16"
    if mode == "awq-reference":
        return _load_awq_reference_model(model_path, device), mode
    if is_awq:
        raise ValueError("BF16 mode cannot load a packed AWQ checkpoint")
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
    ).to(device).eval()
    return model, mode


def _read_texts(path: Path, max_samples: int) -> list[str]:
    if max_samples <= 0:
        raise ValueError("max_samples must be positive")
    texts = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            text = line.strip()
            if text:
                texts.append(text)
            if len(texts) == max_samples:
                break
    if not texts:
        raise ValueError("evaluation file contains no nonempty samples")
    return texts


@torch.inference_mode()
def _evaluate_perplexity(
    model,
    tokenizer,
    texts: list[str],
    *,
    device: torch.device,
    sequence_length: int,
    max_eval_tokens: int,
) -> dict:
    token_ids = []
    eos = tokenizer.eos_token_id
    for text in texts:
        token_ids.extend(tokenizer.encode(text, add_special_tokens=False))
        if eos is not None:
            token_ids.append(eos)
        if len(token_ids) >= max_eval_tokens + 1:
            break
    token_ids = token_ids[: max_eval_tokens + 1]
    if len(token_ids) < 2:
        raise ValueError("evaluation text produced fewer than two tokens")

    total_nll = 0.0
    evaluated_tokens = 0
    first_topk = None
    for start in range(0, len(token_ids) - 1, sequence_length):
        chunk = token_ids[start : start + sequence_length + 1]
        if len(chunk) < 2:
            continue
        inputs = torch.tensor([chunk[:-1]], dtype=torch.long, device=device)
        targets = torch.tensor(chunk[1:], dtype=torch.long, device=device)
        logits = model(input_ids=inputs, use_cache=False).logits[0]
        total_nll += F.cross_entropy(
            logits.float(),
            targets,
            reduction="sum",
        ).item()
        evaluated_tokens += targets.numel()
        if first_topk is None:
            values, indices = logits[-1].float().topk(5)
            first_topk = {
                "token_ids": indices.cpu().tolist(),
                "logits": values.cpu().tolist(),
            }
    mean_nll = total_nll / evaluated_tokens
    return {
        "definition": "non-overlapping fixed-block causal perplexity",
        "tokens": evaluated_tokens,
        "mean_nll": mean_nll,
        "perplexity": math.exp(mean_nll),
        "first_block_last_token_top5": first_topk,
    }


@torch.inference_mode()
def _generate(model, tokenizer, texts: list[str], device: torch.device, max_new_tokens: int):
    if not texts:
        return []
    tokenizer.padding_side = "left"
    encoded = tokenizer(texts, return_tensors="pt", padding=True)
    encoded = {name: tensor.to(device) for name, tensor in encoded.items()}
    output_ids = model.generate(
        **encoded,
        do_sample=False,
        max_new_tokens=max_new_tokens,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )
    return [
        {
            "prompt": prompt,
            "output": tokenizer.decode(output, skip_special_tokens=True),
        }
        for prompt, output in zip(texts, output_ids)
    ]


def run(args: argparse.Namespace) -> dict:
    from transformers import AutoTokenizer

    for name in (
        "max_samples",
        "sequence_length",
        "max_eval_tokens",
        "max_new_tokens",
    ):
        if getattr(args, name) <= 0:
            raise ValueError(f"{name} must be positive")
    if args.generation_prompts < 0:
        raise ValueError("generation_prompts cannot be negative")
    device = torch.device(args.device)
    texts = _read_texts(args.eval_file, args.max_samples)
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    started_at = time.time()
    if device.type == "cuda":
        _reset_cuda_peak_memory_stats(device)
    model, resolved_mode = _load_model(args.model, args.mode, device)
    perplexity = _evaluate_perplexity(
        model,
        tokenizer,
        texts,
        device=device,
        sequence_length=args.sequence_length,
        max_eval_tokens=args.max_eval_tokens,
    )
    generations = _generate(
        model,
        tokenizer,
        texts[: args.generation_prompts],
        device,
        args.max_new_tokens,
    )
    result = {
        "schema_version": 1,
        "model": args.model.name,
        "mode": resolved_mode,
        "evaluation_file": args.eval_file.name,
        "samples": len(texts),
        "sequence_length": args.sequence_length,
        "perplexity": perplexity,
        "generations": generations,
        "elapsed_seconds": time.time() - started_at,
        "peak_cuda_memory_bytes": (
            _max_cuda_memory_allocated(device) if device.type == "cuda" else None
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2, ensure_ascii=False, sort_keys=True)
        handle.write("\n")
    temporary.replace(args.output)
    return result


def main() -> None:
    result = run(build_parser().parse_args())
    print(json.dumps(result, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
