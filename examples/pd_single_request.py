"""Run one request through the dual-GPU Prefill/Decode serving engine."""

import argparse


def main():
    from transformers import AutoTokenizer
    from llmserve import SamplingParams
    from llmserve.pd import PDConfig, PDCoordinator, PDServingEngine

    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--prompt", default="Explain paged KV cache in one paragraph.")
    parser.add_argument("--max-tokens", type=int, default=16)
    parser.add_argument("--prefill-gpu", type=int, default=0)
    parser.add_argument("--decode-gpu", type=int, default=1)
    parser.add_argument("--max-model-len", type=int, default=1024)
    parser.add_argument("--max-num-batched-tokens", type=int, default=1024)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.85)
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=True)
    prompt_token_ids = tokenizer.encode(args.prompt)
    config = PDConfig(
        model=args.model,
        prefill_gpu=args.prefill_gpu,
        decode_gpu=args.decode_gpu,
        engine_kwargs={
            "max_model_len": args.max_model_len,
            "max_num_batched_tokens": args.max_num_batched_tokens,
            "max_num_seqs": 4,
            "gpu_memory_utilization": args.gpu_memory_utilization,
            "enable_chunked_prefill": False,
        },
    )
    engine = PDServingEngine(PDCoordinator(config), prefill_batch_size=1)
    token_ids = []
    try:
        request_id = engine.add_request(
            prompt_token_ids,
            SamplingParams(
                max_tokens=args.max_tokens,
                temperature=0.01,
                ignore_eos=True,
            ),
        )
        while not engine.is_finished():
            for finished_request_id, output_ids in engine.step()[0]:
                if finished_request_id == request_id:
                    token_ids = output_ids
    finally:
        engine.exit()
    print(tokenizer.decode(token_ids, skip_special_tokens=False))
    print(
        {
            "prompt_tokens": len(prompt_token_ids),
            "output_tokens": len(token_ids),
            "token_ids": token_ids,
        }
    )


if __name__ == "__main__":
    main()
