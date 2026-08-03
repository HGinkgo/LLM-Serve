"""Run short requests through the dual-GPU PD serving engine."""

import argparse


def build_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--num-requests", type=int, default=4)
    parser.add_argument("--prefill-batch-size", type=int, default=2)
    parser.add_argument("--max-tokens", type=int, default=4)
    parser.add_argument("--prefill-gpu", type=int, default=0)
    parser.add_argument("--decode-gpu", type=int, default=1)
    parser.add_argument("--max-model-len", type=int, default=256)
    parser.add_argument("--max-num-batched-tokens", type=int, default=128)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.95)
    parser.add_argument("--enable-chunked-prefill", action="store_true")
    parser.add_argument("--prompt-repeat", type=int, default=1)
    parser.add_argument("--print-metrics", action="store_true")
    return parser


def main():
    from transformers import AutoTokenizer

    from llmserve import SamplingParams
    from llmserve.pd import PDConfig, PDCoordinator, PDServingEngine

    args = build_parser().parse_args()
    if args.num_requests <= 0:
        raise ValueError("num_requests must be positive")
    if args.prompt_repeat <= 0:
        raise ValueError("prompt_repeat must be positive")

    tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=True)
    prompts = []
    for request_id in range(args.num_requests):
        prompt = (
            "Explain paged KV cache briefly. " * args.prompt_repeat
            + f"Request {request_id}."
        )
        prompts.append(tokenizer.encode(prompt))

    config = PDConfig(
        model=args.model,
        prefill_gpu=args.prefill_gpu,
        decode_gpu=args.decode_gpu,
        engine_kwargs={
            "max_model_len": args.max_model_len,
            "max_num_batched_tokens": args.max_num_batched_tokens,
            "max_num_seqs": max(4, args.num_requests),
            "gpu_memory_utilization": args.gpu_memory_utilization,
            "enable_chunked_prefill": args.enable_chunked_prefill,
        },
    )
    engine = PDServingEngine(
        PDCoordinator(config),
        prefill_batch_size=args.prefill_batch_size,
    )
    outputs = {}
    try:
        for prompt_token_ids in prompts:
            engine.add_request(
                prompt_token_ids,
                SamplingParams(
                    max_tokens=args.max_tokens,
                    temperature=0.01,
                    ignore_eos=True,
                ),
            )
        while not engine.is_finished():
            for request_id, token_ids in engine.step()[0]:
                outputs[request_id] = token_ids
        metrics = engine.get_metrics()
    finally:
        engine.exit()
    for request_id, prompt_token_ids in enumerate(prompts):
        token_ids = outputs[request_id]
        print(
            {
                "request_id": request_id,
                "prompt_tokens": len(prompt_token_ids),
                "output_tokens": len(token_ids),
                "token_ids": token_ids,
                "text": tokenizer.decode(token_ids),
            }
        )
    if args.print_metrics:
        print({"pd_metrics": metrics})


if __name__ == "__main__":
    main()
