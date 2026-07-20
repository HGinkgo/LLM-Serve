import argparse
import gc
import json
import os
from dataclasses import asdict, dataclass

import torch

from llmserve import LLM, SamplingParams


@dataclass(slots=True)
class RunResult:
    token_ids: list[int]
    metrics: dict
    step_events: list[dict]


@dataclass(slots=True)
class BatchRunResult:
    token_ids: list[list[int]]
    metrics: dict
    step_events: list[dict]


DEFAULT_BATCH_PROMPTS = [
    "Explain speculative decoding in one concise paragraph.",
    "Summarize why paged KV cache improves LLM serving.",
    "Describe the difference between prefill and decode.",
    "Explain continuous batching to a systems engineer.",
]


class ArgmaxSampler:
    def __call__(self, logits: torch.Tensor, temperatures: torch.Tensor):
        return logits.argmax(dim=-1)


def parse_token_ids(value: str) -> list[int]:
    token_ids = [item.strip() for item in value.split(",")]
    token_ids = [item for item in token_ids if item]
    if not token_ids:
        raise ValueError("--prompt-token-ids must contain at least one token id")
    return [int(item) for item in token_ids]


def parse_batch_sizes(value: str) -> list[int]:
    sizes = [item.strip() for item in value.split(",") if item.strip()]
    if not sizes:
        raise ValueError("--batch-sizes must contain at least one positive integer")
    parsed = [int(item) for item in sizes]
    if any(size <= 0 for size in parsed):
        raise ValueError("--batch-sizes values must be positive")
    return sorted(set(parsed))


def parse_args():
    parser = argparse.ArgumentParser(description="Small speculative decoding correctness check for LLM-Serve")
    parser.add_argument("--model", default=os.environ.get("MODEL_PATH", "~/models/Qwen3-8B/"))
    parser.add_argument("--speculative-model", default=os.environ.get("SPECULATIVE_MODEL"), required=False)
    parser.add_argument("--prompt", default=None)
    parser.add_argument("--prompt-token-ids", default=None, type=parse_token_ids)
    parser.add_argument("--output-len", type=int, default=4)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--speculative-gamma", type=int, default=3)
    parser.add_argument("--speculative-tree-nodes", type=int, choices=[0, 6, 10], default=0)
    parser.add_argument("--speculative-accept-mode", choices=["greedy", "rejection"], default="greedy")
    parser.add_argument("--speculative-trace", action="store_true")
    parser.add_argument("--argmax-sampler", action="store_true")
    parser.add_argument("--max-model-len", type=int, default=64)
    parser.add_argument("--max-num-batched-tokens", type=int, default=64)
    parser.add_argument("--max-steps", type=int, default=64)
    parser.add_argument("--require-token-match", action="store_true")
    parser.add_argument("--batch-consistency", action="store_true")
    parser.add_argument("--batch-sizes", type=parse_batch_sizes, default=[2, 4])
    parser.add_argument("--output-json", default=None)
    return parser.parse_args()


def set_seed(seed: int):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def cleanup_cuda_memory():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()


def run_single_request(
    *,
    model: str,
    prompt: str | list[int],
    output_len: int,
    temperature: float,
    seed: int,
    speculative_model: str | None,
    speculative_gamma: int,
    speculative_tree_nodes: int,
    speculative_accept_mode: str,
    speculative_trace: bool,
    argmax_sampler: bool,
    max_model_len: int,
    max_num_batched_tokens: int,
    max_steps: int,
) -> RunResult:
    set_seed(seed)
    engine = None
    try:
        engine = LLM(
            model,
            enforce_eager=True,
            tensor_parallel_size=1,
            max_model_len=max_model_len,
            max_num_batched_tokens=max_num_batched_tokens,
            speculative_model=speculative_model,
            speculative_gamma=speculative_gamma,
            speculative_tree_nodes=speculative_tree_nodes,
            speculative_accept_mode=speculative_accept_mode,
            speculative_trace=speculative_trace,
        )
        if argmax_sampler:
            engine.model_runner.sampler = ArgmaxSampler()
        seq_id = engine.add_request(
            prompt,
            SamplingParams(
                temperature=temperature,
                max_tokens=output_len,
                ignore_eos=True,
            ),
        )
        outputs = {}
        step_events = []
        steps = 0
        while not engine.is_finished():
            step_outputs, _ = engine.step()
            steps += 1
            step_events.append(dict(getattr(engine, "last_step_events", {})))
            for output_seq_id, token_ids in step_outputs:
                outputs[output_seq_id] = token_ids
            if steps > max_steps:
                raise RuntimeError(f"request did not finish within {max_steps} engine steps")
        metrics = engine.get_metrics()
        return RunResult(
            token_ids=outputs.get(seq_id, []),
            metrics=metrics,
            step_events=step_events,
        )
    finally:
        if engine is not None:
            engine.exit()
            del engine
        cleanup_cuda_memory()


def run_request_group(
    *,
    model: str,
    prompts: list[str],
    output_len: int,
    seed: int,
    speculative_model: str,
    speculative_gamma: int,
    max_model_len: int,
    max_num_batched_tokens: int,
    max_steps: int,
    submit_all: bool,
) -> BatchRunResult:
    set_seed(seed)
    engine = None
    try:
        engine = LLM(
            model,
            enforce_eager=True,
            tensor_parallel_size=1,
            max_model_len=max_model_len,
            max_num_batched_tokens=max_num_batched_tokens,
            speculative_model=speculative_model,
            speculative_gamma=speculative_gamma,
            speculative_accept_mode="greedy",
            speculative_trace=False,
        )
        engine.model_runner.sampler = ArgmaxSampler()
        outputs = {}
        seq_ids = []
        step_events = []

        def add(prompt: str):
            seq_id = engine.add_request(
                prompt,
                SamplingParams(
                    temperature=1.0,
                    max_tokens=output_len,
                    ignore_eos=True,
                ),
            )
            seq_ids.append(seq_id)

        def drain(step_limit: int):
            steps = 0
            while not engine.is_finished():
                step_outputs, _ = engine.step()
                steps += 1
                step_events.append(dict(getattr(engine, "last_step_events", {})))
                for output_seq_id, token_ids in step_outputs:
                    outputs[output_seq_id] = token_ids
                if steps > step_limit:
                    raise RuntimeError(f"request group did not finish within {step_limit} engine steps")

        if submit_all:
            for prompt in prompts:
                add(prompt)
            drain(max_steps * len(prompts))
        else:
            for prompt in prompts:
                add(prompt)
                drain(max_steps)

        return BatchRunResult(
            token_ids=[outputs.get(seq_id, []) for seq_id in seq_ids],
            metrics=engine.get_metrics(),
            step_events=step_events,
        )
    finally:
        if engine is not None:
            engine.exit()
            del engine
        cleanup_cuda_memory()


def speculative_summary(result: RunResult) -> dict:
    return result.metrics["summary"].get("speculative", {})


def build_checks(baseline: RunResult, speculative: RunResult, max_tokens: int) -> dict:
    baseline_summary = baseline.metrics["summary"]
    speculative_metrics = speculative.metrics["summary"]
    spec = speculative_summary(speculative)
    acceptance_rate = spec.get("acceptance_rate")

    baseline_output_tokens = baseline_summary["total_output_tokens"]
    speculative_output_tokens = speculative_metrics["total_output_tokens"]
    baseline_finished = baseline_summary["num_finished"] == 1 and baseline_summary["num_failed"] == 0
    speculative_finished = speculative_metrics["num_finished"] == 1 and speculative_metrics["num_failed"] == 0
    speculative_metrics_present = spec.get("steps", 0) > 0
    speculative_metrics_consistent = (
        spec.get("draft_tokens", 0) >= spec.get("accepted_tokens", 0) >= 0
        and spec.get("emitted_tokens", 0) >= 0
        and spec.get("emitted_tokens", 0) <= speculative_output_tokens
        and (acceptance_rate is None or 0.0 <= acceptance_rate <= 1.0)
    )

    checks = {
        "baseline_finished": baseline_finished,
        "speculative_finished": speculative_finished,
        "baseline_output_len_matches_metrics": len(baseline.token_ids) == baseline_output_tokens,
        "speculative_output_len_matches_metrics": len(speculative.token_ids) == speculative_output_tokens,
        "baseline_output_len_within_max": len(baseline.token_ids) <= max_tokens,
        "speculative_output_len_within_max": len(speculative.token_ids) <= max_tokens,
        "speculative_metrics_present": speculative_metrics_present,
        "speculative_metrics_consistent": speculative_metrics_consistent,
        "token_ids_match": baseline.token_ids == speculative.token_ids,
    }
    checks["required_checks_pass"] = all(
        checks[name]
        for name in [
            "baseline_finished",
            "speculative_finished",
            "baseline_output_len_matches_metrics",
            "speculative_output_len_matches_metrics",
            "baseline_output_len_within_max",
            "speculative_output_len_within_max",
            "speculative_metrics_present",
            "speculative_metrics_consistent",
        ]
    )
    return checks


def build_batch_checks(
    serial_token_ids: list[list[int]],
    batched: BatchRunResult,
    expected_batch_size: int,
    output_len: int,
) -> dict:
    summary = batched.metrics["summary"]
    speculative = summary.get("speculative", {})
    per_request_matches = [
        serial_tokens == batched_tokens
        for serial_tokens, batched_tokens in zip(serial_token_ids, batched.token_ids)
    ]
    checks = {
        "request_count_matches": len(batched.token_ids) == expected_batch_size,
        "all_requests_finished": (
            summary.get("num_finished") == expected_batch_size
            and summary.get("num_failed") == 0
        ),
        "all_output_lengths_match": all(
            len(token_ids) == output_len for token_ids in batched.token_ids
        ),
        "speculative_metrics_present": speculative.get("steps", 0) > 0,
        "multi_request_batch_used": speculative.get("max_batch_size", 0) >= expected_batch_size,
        "token_ids_match": (
            len(per_request_matches) == expected_batch_size
            and all(per_request_matches)
        ),
    }
    checks["required_checks_pass"] = all(checks.values())
    checks["per_request_matches"] = per_request_matches
    return checks


def run_batch_consistency(args) -> dict:
    model = os.path.expanduser(args.model)
    speculative_model = os.path.expanduser(args.speculative_model) if args.speculative_model else None
    if speculative_model is None:
        raise ValueError("--speculative-model or SPECULATIVE_MODEL is required")

    batch_sizes = args.batch_sizes
    max_batch_size = max(batch_sizes)
    prompts = [DEFAULT_BATCH_PROMPTS[i % len(DEFAULT_BATCH_PROMPTS)] for i in range(max_batch_size)]
    serial = run_request_group(
        model=model,
        prompts=prompts,
        output_len=args.output_len,
        seed=args.seed,
        speculative_model=speculative_model,
        speculative_gamma=args.speculative_gamma,
        max_model_len=args.max_model_len,
        max_num_batched_tokens=args.max_num_batched_tokens,
        max_steps=args.max_steps,
        submit_all=False,
    )

    batches = []
    for batch_size in batch_sizes:
        batched = run_request_group(
            model=model,
            prompts=prompts[:batch_size],
            output_len=args.output_len,
            seed=args.seed,
            speculative_model=speculative_model,
            speculative_gamma=args.speculative_gamma,
            max_model_len=args.max_model_len,
            max_num_batched_tokens=args.max_num_batched_tokens,
            max_steps=args.max_steps,
            submit_all=True,
        )
        checks = build_batch_checks(
            serial.token_ids[:batch_size],
            batched,
            batch_size,
            args.output_len,
        )
        batches.append({
            "batch_size": batch_size,
            "prompts": prompts[:batch_size],
            "serial_token_ids": serial.token_ids[:batch_size],
            "batched": asdict(batched),
            "checks": checks,
        })

    return {
        "config": {
            "model": model,
            "speculative_model": speculative_model,
            "batch_sizes": batch_sizes,
            "output_len": args.output_len,
            "seed": args.seed,
            "speculative_gamma": args.speculative_gamma,
            "argmax_sampler": True,
            "max_model_len": args.max_model_len,
            "max_num_batched_tokens": args.max_num_batched_tokens,
        },
        "serial": asdict(serial),
        "batches": batches,
        "required_checks_pass": all(batch["checks"]["required_checks_pass"] for batch in batches),
    }


def run_comparison(args) -> dict:
    model = os.path.expanduser(args.model)
    speculative_model = os.path.expanduser(args.speculative_model) if args.speculative_model else None
    speculative_tree_nodes = getattr(args, "speculative_tree_nodes", 0)
    if speculative_model is None:
        raise ValueError("--speculative-model or SPECULATIVE_MODEL is required")
    prompt = args.prompt if args.prompt is not None else args.prompt_token_ids
    if prompt is None:
        prompt = [1, 2, 3, 4]

    baseline = run_single_request(
        model=model,
        prompt=prompt,
        output_len=args.output_len,
        temperature=args.temperature,
        seed=args.seed,
        speculative_model=None,
        speculative_gamma=args.speculative_gamma,
        speculative_tree_nodes=0,
        speculative_accept_mode=args.speculative_accept_mode,
        speculative_trace=args.speculative_trace,
        argmax_sampler=args.argmax_sampler,
        max_model_len=args.max_model_len,
        max_num_batched_tokens=args.max_num_batched_tokens,
        max_steps=args.max_steps,
    )
    speculative = run_single_request(
        model=model,
        prompt=prompt,
        output_len=args.output_len,
        temperature=args.temperature,
        seed=args.seed,
        speculative_model=speculative_model,
        speculative_gamma=args.speculative_gamma,
        speculative_tree_nodes=speculative_tree_nodes,
        speculative_accept_mode=args.speculative_accept_mode,
        speculative_trace=args.speculative_trace,
        argmax_sampler=args.argmax_sampler,
        max_model_len=args.max_model_len,
        max_num_batched_tokens=args.max_num_batched_tokens,
        max_steps=args.max_steps,
    )
    checks = build_checks(baseline, speculative, args.output_len)
    return {
        "config": {
            "model": model,
            "speculative_model": speculative_model,
            "prompt": args.prompt,
            "prompt_token_ids": None if args.prompt is not None else prompt,
            "output_len": args.output_len,
            "temperature": args.temperature,
            "seed": args.seed,
            "speculative_gamma": args.speculative_gamma,
            "speculative_tree_nodes": speculative_tree_nodes,
            "speculative_accept_mode": args.speculative_accept_mode,
            "speculative_trace": args.speculative_trace,
            "argmax_sampler": args.argmax_sampler,
            "max_model_len": args.max_model_len,
            "max_num_batched_tokens": args.max_num_batched_tokens,
        },
        "baseline": asdict(baseline),
        "speculative": asdict(speculative),
        "checks": checks,
    }


def status(value: bool) -> str:
    return "PASS" if value else "FAIL"


def print_report(report: dict):
    checks = report["checks"]
    spec = report["speculative"]["metrics"]["summary"].get("speculative", {})
    print()
    print("Speculative Correctness Check")
    print("=============================")
    print(f"baseline tokens:     {report['baseline']['token_ids']}")
    print(f"speculative tokens:  {report['speculative']['token_ids']}")
    print(f"token ids match:     {checks['token_ids_match']}  (observation, not a distribution proof)")
    print()
    print("Required Checks")
    print("---------------")
    for name, value in checks.items():
        if name in {"token_ids_match", "required_checks_pass"}:
            continue
        print(f"{name:<40} {status(value)}")
    print(f"{'required_checks_pass':<40} {status(checks['required_checks_pass'])}")
    print()
    print("Speculative Metrics")
    print("-------------------")
    print(f"steps:          {spec.get('steps')}")
    print(f"draft tokens:   {spec.get('draft_tokens')}")
    print(f"accepted:       {spec.get('accepted_tokens')}")
    print(f"emitted:        {spec.get('emitted_tokens')}")
    print(f"acceptance:     {spec.get('acceptance_rate')}")
    print(f"accept-all:     {spec.get('accept_all_count')}")
    print()


def print_batch_report(report: dict):
    print()
    print("Batched Speculative Consistency Check")
    print("=====================================")
    for batch in report["batches"]:
        checks = batch["checks"]
        speculative = batch["batched"]["metrics"]["summary"]["speculative"]
        print(f"batch size:       {batch['batch_size']}")
        print(f"token ids match:  {status(checks['token_ids_match'])}")
        print(f"finished:         {status(checks['all_requests_finished'])}")
        print(f"max batch seen:   {speculative.get('max_batch_size')}")
        print(f"per request:      {checks['per_request_matches']}")
        print()
    print(f"required checks:  {status(report['required_checks_pass'])}")
    print()


def main():
    args = parse_args()
    if args.batch_consistency:
        report = run_batch_consistency(args)
        print_batch_report(report)
        ok = report["required_checks_pass"]
    else:
        report = run_comparison(args)
        print_report(report)
        ok = report["checks"]["required_checks_pass"]
        if args.require_token_match:
            ok = ok and report["checks"]["token_ids_match"]
    if args.output_json:
        output_path = os.path.expanduser(args.output_json)
        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(report, f, indent=2)
        print(f"Wrote JSON results to {output_path}")

    raise SystemExit(0 if ok else 1)


if __name__ == "__main__":
    main()
