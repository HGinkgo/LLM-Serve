"""
Serving benchmark for nano-vllm-runtime.

This benchmark records request-level serving metrics. It does not change
scheduler, model runner, attention, or kernel behavior.
"""

import argparse
import json
import os
import time
from collections import deque
from random import Random

from nanovllm import LLM, SamplingParams


def parse_args():
    parser = argparse.ArgumentParser(description="Request-level serving benchmark for nano-vllm-runtime")
    parser.add_argument("--model", default=os.environ.get("MODEL_PATH", "~/models/Qwen3-0.6B/"))
    parser.add_argument("--num-requests", type=int, default=8)
    parser.add_argument("--input-len", type=int, default=256)
    parser.add_argument("--output-len", type=int, default=128)
    parser.add_argument("--arrival", choices=["all", "poisson"], default="all")
    parser.add_argument("--request-rate", type=float, default=4.0, help="Requests per second for poisson arrival")
    parser.add_argument("--enforce-eager", action="store_true")
    parser.add_argument("--max-model-len", type=int, default=4096)
    parser.add_argument("--max-num-batched-tokens", type=int, default=16384)
    parser.add_argument("--temperature", type=float, default=0.6)
    parser.add_argument("--seed", type=int, default=0)
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


def build_workload(args):
    rng = Random(args.seed)
    arrivals = build_arrivals(args.num_requests, args.arrival, args.request_rate, rng)
    prompts = [
        [rng.randint(0, 10000) for _ in range(args.input_len)]
        for _ in range(args.num_requests)
    ]
    sampling_params = [
        SamplingParams(
            temperature=args.temperature,
            max_tokens=args.output_len,
            ignore_eos=True,
        )
        for _ in range(args.num_requests)
    ]
    return deque(zip(arrivals, prompts, sampling_params))


def format_ms(value):
    if value is None:
        return "N/A"
    return f"{value * 1000:.2f}"


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


def run_benchmark(args):
    model_path = os.path.expanduser(args.model)
    workload = build_workload(args)
    engine = LLM(
        model_path,
        enforce_eager=args.enforce_eager,
        max_model_len=args.max_model_len,
        max_num_batched_tokens=args.max_num_batched_tokens,
    )

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
    engine.exit()
    return {
        "config": {
            "model": model_path,
            "num_requests": args.num_requests,
            "input_len": args.input_len,
            "output_len": args.output_len,
            "arrival": args.arrival,
            "request_rate": args.request_rate if args.arrival == "poisson" else None,
            "enforce_eager": args.enforce_eager,
            "max_model_len": args.max_model_len,
            "max_num_batched_tokens": args.max_num_batched_tokens,
            "temperature": args.temperature,
            "seed": args.seed,
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
