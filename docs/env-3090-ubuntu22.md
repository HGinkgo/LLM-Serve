# 3090 Ubuntu 22.04 Environment

Generated: 2026-05-29

- OS: Ubuntu 22.04.5 LTS
- glibc: 2.35
- GPU: 2 x RTX 3090
- NVIDIA driver: 580.95.05
- conda env: nano-vllm
- Python: 3.10.20
- torch: 2.7.1+cu128
- torch CUDA: 12.8
- triton: 3.3.1
- transformers: 4.57.6
- flash-attn: 2.8.3
- model path: local Qwen3-0.6B path supplied with `MODEL_PATH`

## Validation

The following checks passed on GPU 0:

```bash
CUDA_VISIBLE_DEVICES=0 conda run -n nano-vllm python -c "import torch; print(torch.cuda.is_available())"
```

Benchmark/example commands should set:

```bash
export MODEL_PATH=/path/to/Qwen3-0.6B
```

Small `nano-vllm` generation passed with `enforce_eager=True`.

Small non-eager benchmark passed with:

- requests: 8
- generated tokens: 256
- elapsed time: 1.045s
- throughput: 245.03 tok/s
