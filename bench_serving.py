"""
Serving benchmark for ThrustLM.

This benchmark records request-level serving metrics. It preserves baseline
runtime behavior by default and can enable experimental features via flags.
"""

import argparse
import json
import os
import time
from collections import deque
from random import Random

import torch

from thrustlm import LLM, SamplingParams


DEFAULT_NATURAL_PROMPTS = [
    "Explain speculative decoding in one concise paragraph.",
    "Summarize why paged KV cache improves memory management for LLM serving.",
    "Write a short Python function that computes the mean of a list.",
    "Describe the difference between prefill and decode in transformer inference.",
    "Give three practical tips for debugging CUDA out-of-memory errors.",
    "Translate this sentence into Chinese: high-throughput inference requires careful scheduling.",
    "What are the trade-offs of using a smaller draft model for speculative decoding?",
    "Explain continuous batching to a systems engineering interviewer.",
]


class ArgmaxSampler:
    def __call__(self, logits: torch.Tensor, temperatures: torch.Tensor):
        return logits.argmax(dim=-1)


def parse_args():
    parser = argparse.ArgumentParser(description="Request-level serving benchmark for ThrustLM")
    parser.add_argument("--model", default=os.environ.get("MODEL_PATH", "~/models/Qwen3-8B/"))
    parser.add_argument("--num-requests", type=int, default=8)
    parser.add_argument("--input-len", type=int, default=256)
    parser.add_argument("--output-len", type=int, default=128)
    parser.add_argument("--prompt-mode", choices=["random-token", "natural"], default="random-token")
    parser.add_argument("--prompt-file", default=None, help="Plain text file with one prompt per non-empty line")
    parser.add_argument("--arrival", choices=["all", "poisson", "closed-loop"], default="all")
    parser.add_argument("--request-rate", type=float, default=4.0, help="Requests per second for poisson arrival")
    parser.add_argument("--max-concurrency", type=int, default=8)
    parser.add_argument("--warmup-seconds", type=float, default=5.0)
    parser.add_argument("--measurement-seconds", type=float, default=15.0)
    parser.add_argument("--enforce-eager", action="store_true")
    # ===== 2026-06-07 chunked prefill =====
    # 默认关闭，方便同一个脚本做 baseline / chunked prefill A/B。
    parser.add_argument("--enable-chunked-prefill", action="store_true")
    # ===== 2026-06-07 chunked prefill =====
    parser.add_argument("--max-model-len", type=int, default=4096)
    parser.add_argument("--max-num-batched-tokens", type=int, default=16384)
    parser.add_argument("--temperature", type=float, default=0.6)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--speculative-model", default=os.environ.get("SPECULATIVE_MODEL"))
    parser.add_argument("--speculative-gamma", type=int, default=3)
    parser.add_argument("--speculative-tree-nodes", type=int, choices=[0, 6, 10], default=0)
    parser.add_argument("--speculative-accept-mode", choices=["greedy", "rejection"], default="greedy")
    parser.add_argument("--argmax-sampler", action="store_true")
    parser.add_argument("--speculative-trace", action="store_true")
    parser.add_argument("--output-json", default=None)
    return parser.parse_args()


def build_arrivals(num_requests: int, arrival: str, request_rate: float, rng: Random):
    if arrival == "all":
        return [0.0] * num_requests
    if request_rate <= 0:
        raise ValueError("--request-rate must be positive for poisson arrival")
    arrivals = []
    current = 0.0
    for _ in range(num_requests):
        current += rng.expovariate(request_rate)
        arrivals.append(current)
    return arrivals


def load_prompt_file(path: str):
    with open(os.path.expanduser(path), "r") as f:
        prompts = [line.strip() for line in f if line.strip()]
    if not prompts:
        raise ValueError("--prompt-file must contain at least one non-empty prompt")
    return prompts


def repeat_to_length(items: list, length: int):
    return [items[i % len(items)] for i in range(length)]


def build_prompts(args, rng: Random):
    if args.prompt_file:
        return repeat_to_length(load_prompt_file(args.prompt_file), args.num_requests)
    if args.prompt_mode == "natural":
        return repeat_to_length(DEFAULT_NATURAL_PROMPTS, args.num_requests)
    return [
        [rng.randint(0, 10000) for _ in range(args.input_len)]
        for _ in range(args.num_requests)
    ]


def build_workload(args):
    rng = Random(args.seed)
    arrivals = build_arrivals(args.num_requests, args.arrival, args.request_rate, rng)
    prompts = build_prompts(args, rng)
    sampling_params = [
        SamplingParams(
            temperature=args.temperature,
            max_tokens=args.output_len,
            ignore_eos=True,
        )
        for _ in range(args.num_requests)
    ]
    return deque(zip(arrivals, prompts, sampling_params))


def build_request_stream(args):
    rng = Random(args.seed)
    if args.prompt_file:
        prompt_pool = load_prompt_file(args.prompt_file)
    elif args.prompt_mode == "natural":
        prompt_pool = DEFAULT_NATURAL_PROMPTS
    else:
        prompt_pool = None

    index = 0
    while True:
        if prompt_pool is None:
            prompt = [rng.randint(0, 10000) for _ in range(args.input_len)]
        else:
            prompt = prompt_pool[index % len(prompt_pool)]
        yield prompt, SamplingParams(
            temperature=args.temperature,
            max_tokens=args.output_len,
            ignore_eos=True,
        )
        index += 1


def percentile(values: list[float], percentile_value: float):
    if not values:
        return None
    values = sorted(values)
    if len(values) == 1:
        return values[0]
    rank = (len(values) - 1) * percentile_value / 100
    low = int(rank)
    high = min(low + 1, len(values) - 1)
    weight = rank - low
    return values[low] * (1 - weight) + values[high] * weight


def summarize_values(values: list[float]):
    if not values:
        return {"mean": None, "p50": None, "p99": None, "max": None}
    return {
        "mean": sum(values) / len(values),
        "p50": percentile(values, 50),
        "p99": percentile(values, 99),
        "max": max(values),
    }


def summarize_speculative_requests(requests: list[dict]):
    steps = sum(request.get("speculative_steps", 0) for request in requests)
    draft_tokens = sum(request.get("speculative_draft_tokens", 0) for request in requests)
    accepted_tokens = sum(request.get("speculative_accepted_tokens", 0) for request in requests)
    emitted_tokens = sum(request.get("speculative_emitted_tokens", 0) for request in requests)
    accept_all_count = sum(request.get("speculative_accept_all_count", 0) for request in requests)
    gamma_counts = {}
    timing_totals = {}
    for request in requests:
        for gamma, count in request.get("speculative_gamma_counts", {}).items():
            gamma_counts[gamma] = gamma_counts.get(gamma, 0) + count
        for name, value in request.get("speculative_timing", {}).items():
            timing_totals[name] = timing_totals.get(name, 0.0) + value
    timing = {
        name: {
            "total": value,
            "mean": value / steps if steps > 0 else None,
        }
        for name, value in sorted(timing_totals.items())
    }
    return {
        "steps": steps,
        "draft_tokens": draft_tokens,
        "accepted_tokens": accepted_tokens,
        "emitted_tokens": emitted_tokens,
        "acceptance_rate": accepted_tokens / draft_tokens if draft_tokens > 0 else None,
        "acceptance_length": emitted_tokens / steps if steps > 0 else None,
        "accepted_length": accepted_tokens / steps if steps > 0 else None,
        "draft_tokens_per_step": draft_tokens / steps if steps > 0 else None,
        "accept_all_count": accept_all_count,
        "gamma_counts": dict(sorted(gamma_counts.items(), key=lambda item: int(item[0]))),
        "timing": timing,
    }


def build_steady_state_summary(
    metrics: dict,
    measurement_start: float,
    measurement_end: float,
    scheduled_batch_sizes: list[int],
    num_admitted: int,
):
    duration = measurement_end - measurement_start
    if duration <= 0:
        raise ValueError("measurement window must have positive duration")

    requests = metrics["requests"]
    output_tokens = sum(
        measurement_start <= token_time < measurement_end
        for request in requests
        for token_time in request["token_times"]
    )
    finished_in_window = [
        request
        for request in requests
        if request["finish_time"] is not None
        and measurement_start <= request["finish_time"] < measurement_end
    ]
    fully_measured_requests = [
        request
        for request in requests
        if request["arrival_time"] >= measurement_start
        and request["finish_time"] is not None
        and request["finish_time"] < measurement_end
    ]

    ttfts = []
    itls = []
    tpots = []
    latencies = []
    for request in fully_measured_requests:
        first_token_time = request["first_token_time"]
        finish_time = request["finish_time"]
        output_count = request["output_tokens"]
        if first_token_time is not None:
            ttfts.append(first_token_time - request["arrival_time"])
        request_itls = request.get("itl") or [
            request["token_times"][i] - request["token_times"][i - 1]
            for i in range(1, len(request["token_times"]))
        ]
        itls.extend(request_itls)
        latencies.append(finish_time - request["arrival_time"])
        if output_count > 1 and first_token_time is not None:
            tpots.append((finish_time - first_token_time) / (output_count - 1))
        elif output_count == 1:
            tpots.append(0.0)

    return {
        "duration": duration,
        "output_tokens": output_tokens,
        "throughput": output_tokens / duration,
        "num_requests_admitted": num_admitted,
        "num_requests_completed": len(finished_in_window),
        "num_requests_fully_measured": len(fully_measured_requests),
        "num_step_samples": len(scheduled_batch_sizes),
        "mean_scheduled_batch_size": (
            sum(scheduled_batch_sizes) / len(scheduled_batch_sizes)
            if scheduled_batch_sizes else None
        ),
        "max_scheduled_batch_size": max(scheduled_batch_sizes, default=0),
        "ttft": summarize_values(ttfts),
        "itl": summarize_values(itls),
        "tpot": summarize_values(tpots),
        "request_latency": summarize_values(latencies),
        "speculative": summarize_speculative_requests(fully_measured_requests),
    }


def validate_benchmark_args(args):
    tree_nodes = getattr(args, "speculative_tree_nodes", 0)
    if tree_nodes:
        if not args.speculative_model:
            raise ValueError("--speculative-tree-nodes requires --speculative-model")
        if args.speculative_gamma != 3:
            raise ValueError("tree speculation requires --speculative-gamma 3")
        if args.speculative_accept_mode != "greedy":
            raise ValueError("tree speculation requires greedy accept mode")
        if not getattr(args, "argmax_sampler", False):
            raise ValueError("tree speculation requires --argmax-sampler")
        if args.arrival != "all" or args.num_requests != 1:
            raise ValueError("tree speculation benchmark currently requires one all-at-once request")
    if args.arrival == "closed-loop":
        if args.max_concurrency <= 0:
            raise ValueError("--max-concurrency must be positive")
        if args.warmup_seconds < 0:
            raise ValueError("--warmup-seconds must be non-negative")
        if args.measurement_seconds <= 0:
            raise ValueError("--measurement-seconds must be positive")


def run_closed_loop(engine, args):
    requests = build_request_stream(args)
    active_requests = 0
    num_admitted = 0

    def refill():
        nonlocal active_requests, num_admitted
        while active_requests < args.max_concurrency:
            prompt, sampling_params = next(requests)
            engine.add_request(prompt, sampling_params)
            active_requests += 1
            num_admitted += 1

    refill()
    warmup_start = time.perf_counter()
    measurement_start = warmup_start + args.warmup_seconds
    measurement_end = measurement_start + args.measurement_seconds
    scheduled_batch_sizes = []

    while time.perf_counter() < measurement_end:
        outputs, _ = engine.step()
        active_requests -= len(outputs)
        events = engine.last_step_events
        step_end = events.get("step_end", time.perf_counter())
        if measurement_start <= step_end < measurement_end:
            scheduled_batch_sizes.append(len(events.get("scheduled_seq_ids", [])))
        if step_end < measurement_end:
            refill()

    while not engine.is_finished():
        engine.step()

    return {
        "measurement_start": measurement_start,
        "measurement_end": measurement_end,
        "scheduled_batch_sizes": scheduled_batch_sizes,
        "num_admitted": num_admitted,
    }


def format_ms(value):
    if value is None:
        return "N/A"
    return f"{value * 1000:.2f}"


def format_percent(value):
    if value is None:
        return "N/A"
    return f"{value * 100:.2f}%"


def format_float(value):
    if value is None:
        return "N/A"
    return f"{value:.2f}"


def format_gamma_counts(gamma_counts):
    return ", ".join(f"{gamma}:{count}" for gamma, count in gamma_counts.items()) or "N/A"


def print_summary(result):
    summary = result["metrics"]["summary"]
    print()
    print("Serving Benchmark")
    print("=================")
    print(f"requests:       {summary['num_requests']}")
    print(f"finished:       {summary['num_finished']}")
    print(f"failed:         {summary['num_failed']}")
    print(f"output tokens:  {summary['total_output_tokens']}")
    print(f"wall time:      {summary['wall_time']:.3f}s")
    print(f"throughput:     {summary['throughput']:.2f} tok/s")
    print()
    print("Latency Metrics (ms)")
    print("--------------------")
    for name, label in [
        ("ttft", "TTFT"),
        ("itl", "ITL"),
        ("tpot", "TPOT"),
        ("request_latency", "Request latency"),
    ]:
        stats = summary[name]
        print(
            f"{label:<18} "
            f"mean={format_ms(stats['mean']):>10} "
            f"p50={format_ms(stats['p50']):>10} "
            f"p99={format_ms(stats['p99']):>10} "
            f"max={format_ms(stats['max']):>10}"
        )
    print()
    steady_state = result["metrics"].get("steady_state")
    if steady_state:
        print("Steady-State Metrics")
        print("--------------------")
        print(f"duration:       {steady_state['duration']:.3f}s")
        print(f"output tokens:  {steady_state['output_tokens']}")
        print(f"throughput:     {steady_state['throughput']:.2f} tok/s")
        print(f"admitted:       {steady_state['num_requests_admitted']}")
        print(f"completed:      {steady_state['num_requests_completed']}")
        print(f"fully measured: {steady_state['num_requests_fully_measured']}")
        print(f"mean batch:     {format_float(steady_state['mean_scheduled_batch_size'])}")
        print(f"max batch:      {steady_state['max_scheduled_batch_size']}")
        print()
        print("Steady-State Latency Metrics (ms)")
        print("---------------------------------")
        for name, label in [
            ("ttft", "TTFT"),
            ("itl", "ITL"),
            ("tpot", "TPOT"),
            ("request_latency", "Request latency"),
        ]:
            stats = steady_state[name]
            print(
                f"{label:<18} "
                f"mean={format_ms(stats['mean']):>10} "
                f"p50={format_ms(stats['p50']):>10} "
                f"p99={format_ms(stats['p99']):>10} "
                f"max={format_ms(stats['max']):>10}"
            )
        print()
        steady_speculative = steady_state["speculative"]
        if steady_speculative["steps"] > 0:
            print("Steady-State Speculative Metrics")
            print("--------------------------------")
            print(f"steps:          {steady_speculative['steps']}")
            print(f"draft tokens:   {steady_speculative['draft_tokens']}")
            print(f"accepted:       {steady_speculative['accepted_tokens']}")
            print(f"emitted:        {steady_speculative['emitted_tokens']}")
            print(f"acceptance:     {format_percent(steady_speculative['acceptance_rate'])}")
            print(f"accept length:  {format_float(steady_speculative['acceptance_length'])}")
            print(f"gamma counts:   {format_gamma_counts(steady_speculative.get('gamma_counts', {}))}")
            print()
    speculative = summary.get("speculative")
    if speculative and speculative["steps"] > 0:
        print("Speculative Metrics")
        print("-------------------")
        print(f"steps:          {speculative['steps']}")
        print(f"batch calls:    {speculative.get('batch_calls', 0)}")
        print(f"mean batch:     {format_float(speculative.get('mean_batch_size'))}")
        print(f"max batch:      {speculative.get('max_batch_size', 0)}")
        print(f"draft tokens:   {speculative['draft_tokens']}")
        print(f"accepted:       {speculative['accepted_tokens']}")
        print(f"emitted:        {speculative['emitted_tokens']}")
        print(f"acceptance:     {format_percent(speculative['acceptance_rate'])}")
        print(f"accept length:  {format_float(speculative.get('acceptance_length'))}")
        print(f"accepted/step:  {format_float(speculative.get('accepted_length'))}")
        print(f"draft/step:     {format_float(speculative.get('draft_tokens_per_step'))}")
        print(f"gamma counts:   {format_gamma_counts(speculative.get('gamma_counts', {}))}")
        print(f"accept-all:     {speculative['accept_all_count']}")
        print()
        timing = speculative.get("timing") or {}
        if timing:
            print("Speculative Timing (ms)")
            print("-----------------------")
            for name, label in [
                ("target_decode_time", "target decode"),
                ("draft_proposal_time", "draft proposal"),
                ("draft_pack_time", "  draft pack"),
                ("draft_forward_time", "  draft forward"),
                ("draft_sample_time", "  draft sample"),
                ("draft_compact_time", "  draft compact"),
                ("target_verify_time", "target verify"),
                ("accept_time", "accept/reject"),
                ("kv_update_time", "kv update"),
                ("trace_time", "trace"),
                ("total_time", "total"),
            ]:
                stats = timing.get(name)
                if not stats:
                    continue
                print(
                    f"{label:<16} "
                    f"total={format_ms(stats['total']):>10} "
                    f"mean={format_ms(stats['mean']):>10}"
                )
            print()


def run_benchmark(args):
    validate_benchmark_args(args)
    model_path = os.path.expanduser(args.model)
    speculative_model = os.path.expanduser(args.speculative_model) if args.speculative_model else None
    workload = build_workload(args) if args.arrival != "closed-loop" else None
    engine = LLM(
        model_path,
        enforce_eager=args.enforce_eager,
        # ===== 2026-06-07 chunked prefill =====
        enable_chunked_prefill=args.enable_chunked_prefill,
        # ===== 2026-06-07 chunked prefill =====
        max_model_len=args.max_model_len,
        max_num_batched_tokens=args.max_num_batched_tokens,
        speculative_model=speculative_model,
        speculative_gamma=args.speculative_gamma,
        speculative_tree_nodes=getattr(args, "speculative_tree_nodes", 0),
        speculative_accept_mode=args.speculative_accept_mode,
        speculative_trace=args.speculative_trace,
    )
    if args.argmax_sampler:
        engine.model_runner.sampler = ArgmaxSampler()

    if args.arrival == "closed-loop":
        closed_loop = run_closed_loop(engine, args)
    else:
        closed_loop = None
        start = time.perf_counter()
        while workload or not engine.is_finished():
            now = time.perf_counter()
            elapsed = now - start
            while workload and workload[0][0] <= elapsed:
                _, prompt, sampling_params = workload.popleft()
                engine.add_request(prompt, sampling_params)

            if not engine.is_finished():
                engine.step()
                continue

            if workload:
                sleep_time = max(0.0, workload[0][0] - (time.perf_counter() - start))
                time.sleep(min(sleep_time, 0.01))

    metrics = engine.get_metrics()
    if closed_loop is not None:
        metrics["steady_state"] = build_steady_state_summary(
            metrics,
            measurement_start=closed_loop["measurement_start"],
            measurement_end=closed_loop["measurement_end"],
            scheduled_batch_sizes=closed_loop["scheduled_batch_sizes"],
            num_admitted=closed_loop["num_admitted"],
        )
    engine.exit()
    return {
        "config": {
            "model": model_path,
            "num_requests": args.num_requests if args.arrival != "closed-loop" else None,
            "input_len": args.input_len,
            "output_len": args.output_len,
            "prompt_mode": args.prompt_mode,
            "prompt_file": os.path.expanduser(args.prompt_file) if args.prompt_file else None,
            "arrival": args.arrival,
            "request_rate": args.request_rate if args.arrival == "poisson" else None,
            "max_concurrency": args.max_concurrency if args.arrival == "closed-loop" else None,
            "warmup_seconds": args.warmup_seconds if args.arrival == "closed-loop" else None,
            "measurement_seconds": args.measurement_seconds if args.arrival == "closed-loop" else None,
            "enforce_eager": args.enforce_eager,
            # ===== 2026-06-07 chunked prefill =====
            "enable_chunked_prefill": args.enable_chunked_prefill,
            # ===== 2026-06-07 chunked prefill =====
            "max_model_len": args.max_model_len,
            "max_num_batched_tokens": args.max_num_batched_tokens,
            "temperature": args.temperature,
            "seed": args.seed,
            "speculative_model": speculative_model,
            "speculative_gamma": args.speculative_gamma,
            "speculative_tree_nodes": getattr(args, "speculative_tree_nodes", 0),
            "speculative_accept_mode": args.speculative_accept_mode,
            "argmax_sampler": args.argmax_sampler,
            "speculative_trace": args.speculative_trace,
        },
        "metrics": metrics,
    }


def main():
    args = parse_args()
    result = run_benchmark(args)
    print_summary(result)
    if args.output_json:
        output_path = os.path.expanduser(args.output_json)
        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(result, f, indent=2)
        print(f"Wrote JSON results to {output_path}")


if __name__ == "__main__":
    main()
