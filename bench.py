"""
    这是一个面向推理引擎的 benchmark 脚本，不是模型效果评测脚本。

    这个文件的主要用途是给推理框架做压力测试和吞吐测试。它不会使用
    真实用户 prompt，而是构造随机 token id 序列作为请求，因此更关注
    调度行为、KV cache 使用、prefill/decode 性能，以及整体 tokens/s。

    需要重点关注的参数：
    - num_seqs：本次运行的总请求数，也就是并发压力规模。
    - max_input_len：随机输入长度的上限，主要影响 prefill 开销。
    - max_ouput_len：随机输出长度的上限，主要影响 decode 开销。
    - enforce_eager：False 表示更偏性能优化路径；True 表示更保守、
      更便于调试。
    - max_model_len：引擎预留的最大上下文窗口，会影响 KV cache 大小、
      显存占用和调度空间。
"""

import os
import time
from random import randint, seed
from nanovllm import LLM, SamplingParams
# from vllm import LLM, SamplingParams


def main():
    seed(0)                                 # 随机种子，保证输入一致
    num_seqs = 256                          # 这次总共有 256 条请求一起进系统，即并发规模，真实情况要根据用户的 prompt
    max_input_len = 1024                    # 输入最长长度，后面会随机采样，prefill
    max_ouput_len = 1024                    # 输出最长长度，后面会随机采样，decode

    path = os.path.expanduser(os.environ.get("MODEL_PATH", "~/models/Qwen3-0.6B/"))
    llm = LLM(path, enforce_eager=False, max_model_len=4096)    # 初始化推理引擎，enforce_eager：允许走更偏性能优化的执行路径，max_model_len：允许的最大上下文窗口
                                                                # max_model_len 比较重要，会影响 KV cache，显存调用，调度空间，顾此失彼

    # 
    prompt_token_ids = [[randint(0, 10000) for _ in range(randint(100, max_input_len))] for _ in range(num_seqs)]
    sampling_params = [SamplingParams(temperature=0.6, ignore_eos=True, max_tokens=randint(100, max_ouput_len)) for _ in range(num_seqs)]
    # uncomment the following line for vllm
    # prompt_token_ids = [dict(prompt_token_ids=p) for p in prompt_token_ids]

    llm.generate(["Benchmark: "], SamplingParams())            # 热个身，初始化一下
    t = time.time()
    llm.generate(prompt_token_ids, sampling_params, use_tqdm=False)
    t = (time.time() - t)
    total_tokens = sum(sp.max_tokens for sp in sampling_params)
    throughput = total_tokens / t                              # 吞吐量
    print(f"Total: {total_tokens}tok, Time: {t:.2f}s, Throughput: {throughput:.2f}tok/s")


if __name__ == "__main__":
    main()
