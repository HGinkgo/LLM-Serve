import argparse
import hashlib
import json
from pathlib import Path
import time

import torch
from torch import nn

from llmserve.quantization.qwen3_awq import (
    apply_qwen3_layer_cache,
    calibrate_qwen3_layer,
    export_qwen3_awq_checkpoint,
    load_qwen3_layer_cache,
    save_qwen3_layer_cache,
)


class _CaptureComplete(Exception):
    pass


class _FirstLayerCatcher(nn.Module):
    def __init__(self, layer: nn.Module):
        super().__init__()
        self.layer = layer
        self.attention_type = layer.attention_type
        self.hidden_states = None
        self.forward_kwargs = None

    def forward(self, hidden_states: torch.Tensor, **kwargs):
        self.hidden_states = hidden_states.detach()
        self.forward_kwargs = kwargs
        raise _CaptureComplete


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Calibrate and export a Qwen3 checkpoint as runtime-native AWQ W4A16."
    )
    parser.add_argument("--model", type=Path, required=True, help="BF16 Qwen3 model")
    parser.add_argument("--output", type=Path, required=True, help="AWQ checkpoint directory")
    parser.add_argument("--calib-file", type=Path, required=True, help="One sample per line")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--max-calib-samples", type=int, default=32)
    parser.add_argument("--max-seq-len", type=int, default=512)
    parser.add_argument("--search-tokens", type=int, default=512)
    parser.add_argument("--n-grid", type=int, default=20)
    parser.add_argument("--group-size", type=int, default=128)
    parser.add_argument("--no-clip", action="store_true")
    parser.add_argument("--cache-dir", type=Path)
    parser.add_argument("--max-shard-size", default="4GB")
    return parser


def load_calibration_texts(path: Path, *, max_samples: int) -> list[str]:
    if max_samples <= 0:
        raise ValueError("max_samples must be positive")
    if not path.is_file():
        raise FileNotFoundError(f"calibration file does not exist: {path}")
    texts = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            text = line.strip()
            if text:
                texts.append(text)
            if len(texts) == max_samples:
                break
    if not texts:
        raise ValueError("calibration file contains no nonempty samples")
    return texts


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False, sort_keys=True)
        handle.write("\n")
    temporary.replace(path)


def _move_nested(value, device: torch.device):
    if isinstance(value, torch.Tensor):
        return value.to(device)
    if isinstance(value, tuple):
        return tuple(_move_nested(item, device) for item in value)
    if isinstance(value, list):
        return [_move_nested(item, device) for item in value]
    if isinstance(value, dict):
        return {key: _move_nested(item, device) for key, item in value.items()}
    return value


def _reset_cuda_peak_memory_stats(device: torch.device) -> None:
    torch.cuda.set_device(device)
    torch.cuda.reset_peak_memory_stats()


def _max_cuda_memory_allocated(device: torch.device) -> int:
    torch.cuda.set_device(device)
    return torch.cuda.max_memory_allocated()


@torch.no_grad()
def _capture_first_layer_input(model, encoded: dict, device: torch.device):
    base_model = model.model
    first_layer = base_model.layers[0].to(device)
    base_model.embed_tokens.to(device)
    base_model.rotary_emb.to(device)
    catcher = _FirstLayerCatcher(first_layer)
    base_model.layers[0] = catcher
    try:
        base_model(
            input_ids=encoded["input_ids"].to(device),
            attention_mask=encoded.get("attention_mask", None).to(device)
            if encoded.get("attention_mask", None) is not None
            else None,
            use_cache=False,
        )
    except _CaptureComplete:
        pass
    finally:
        base_model.layers[0] = first_layer
        base_model.embed_tokens.to("cpu")
        base_model.rotary_emb.to("cpu")
    if catcher.hidden_states is None or catcher.forward_kwargs is None:
        raise RuntimeError("failed to capture the first Qwen3 decoder layer input")
    return catcher.hidden_states, catcher.forward_kwargs


@torch.no_grad()
def _layer_forward(layer: nn.Module, hidden_states: torch.Tensor, kwargs: dict):
    output = layer(hidden_states, **kwargs)
    return output[0] if isinstance(output, tuple) else output


def _validate_args(args: argparse.Namespace) -> None:
    if args.group_size != 128:
        raise ValueError("LLM-Serve AWQ runtime currently requires group_size=128")
    for name in ("max_calib_samples", "max_seq_len", "search_tokens", "n_grid"):
        if getattr(args, name) <= 0:
            raise ValueError(f"{name} must be positive")
    if args.model.resolve() == args.output.resolve():
        raise ValueError("output directory must differ from the BF16 source model")


def run(args: argparse.Namespace) -> dict:
    from transformers import AutoModelForCausalLM, AutoTokenizer

    _validate_args(args)
    started_at = time.time()
    device = torch.device(args.device)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("Qwen3 AWQ calibration requires a CUDA device")

    texts = load_calibration_texts(
        args.calib_file,
        max_samples=args.max_calib_samples,
    )
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=False)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    encoded = tokenizer(
        texts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=args.max_seq_len,
    )
    token_count = int(encoded["attention_mask"].sum().item())

    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        attn_implementation="sdpa",
    ).eval()
    if getattr(model.config, "model_type", None) != "qwen3":
        raise ValueError("only Qwen3 checkpoints are supported")

    cache_dir = args.cache_dir or (args.output / ".calibration-cache")
    cache_dir.mkdir(parents=True, exist_ok=True)
    progress_path = cache_dir / "calibration_progress.json"
    signature = {
        "source_model": str(args.model.resolve()),
        "calibration_file": str(args.calib_file.resolve()),
        "calibration_sha256": _sha256(args.calib_file),
        "samples": len(texts),
        "tokens": token_count,
        "max_seq_len": args.max_seq_len,
        "search_tokens": args.search_tokens,
        "n_grid": args.n_grid,
        "group_size": args.group_size,
        "apply_clip": not args.no_clip,
    }
    progress = {"signature": signature, "layers": {}, "status": "calibrating"}
    if progress_path.exists():
        with progress_path.open("r", encoding="utf-8") as handle:
            existing = json.load(handle)
        if existing.get("signature") != signature:
            raise RuntimeError(
                "calibration settings differ from the existing cache; use a new cache directory"
            )
        progress = existing
        progress["status"] = "calibrating"

    _reset_cuda_peak_memory_stats(device)
    hidden_states, forward_kwargs = _capture_first_layer_input(model, encoded, device)
    forward_kwargs = _move_nested(forward_kwargs, device)
    cache_paths = []

    for layer_index, layer in enumerate(model.model.layers):
        layer_started_at = time.time()
        layer.to(device)
        cache_path = cache_dir / f"layer-{layer_index:02d}.safetensors"
        cache_paths.append(cache_path)
        if cache_path.exists():
            cached = load_qwen3_layer_cache(cache_path)
            apply_qwen3_layer_cache(layer, cached, group_size=args.group_size)
            hidden_states = _layer_forward(layer, hidden_states, forward_kwargs)
            status = "resumed"
            clipped_groups = progress.get("layers", {}).get(
                str(layer_index), {}
            ).get("clipped_groups")
            evaluated_groups = progress.get("layers", {}).get(
                str(layer_index), {}
            ).get("evaluated_clip_groups")
            del cached
        else:
            result = calibrate_qwen3_layer(
                layer,
                hidden_states,
                forward_kwargs=forward_kwargs,
                group_size=args.group_size,
                n_grid=args.n_grid,
                max_tokens=args.search_tokens,
                apply_clip=not args.no_clip,
            )
            hidden_states = result.hidden_states
            save_qwen3_layer_cache(layer, result.packed_weights, cache_path)
            status = "calibrated"
            clipped_groups = result.clipped_group_count
            evaluated_groups = result.evaluated_clip_group_count
            del result

        layer.to("cpu")
        torch.cuda.empty_cache()
        layer_record = {
            "status": status,
            "cache": str(cache_path),
            "elapsed_seconds": time.time() - layer_started_at,
            "clipped_groups": clipped_groups,
            "evaluated_clip_groups": evaluated_groups,
        }
        progress.setdefault("layers", {})[str(layer_index)] = layer_record
        progress["completed_layers"] = layer_index + 1
        _atomic_write_json(progress_path, progress)
        print(
            json.dumps(
                {"layer": layer_index, **layer_record},
                ensure_ascii=False,
            ),
            flush=True,
        )

    del hidden_states, forward_kwargs, encoded
    torch.cuda.empty_cache()
    calibration_metadata = {
        **signature,
        "layers": progress["layers"],
        "elapsed_seconds": time.time() - started_at,
        "peak_cuda_memory_bytes": _max_cuda_memory_allocated(device),
    }
    export_qwen3_awq_checkpoint(
        model,
        cache_paths,
        args.model,
        args.output,
        calibration_metadata=calibration_metadata,
        group_size=args.group_size,
        max_shard_size=args.max_shard_size,
    )
    progress["status"] = "complete"
    progress["output"] = str(args.output.resolve())
    progress["elapsed_seconds"] = time.time() - started_at
    _atomic_write_json(progress_path, progress)
    return progress


def main() -> None:
    args = build_parser().parse_args()
    result = run(args)
    print(json.dumps(result, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
