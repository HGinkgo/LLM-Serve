# Test Tiers

`core` is the default CPU regression: scheduler/KV lifecycle, EAGLE, PD Shared
transport, cancellation/failure convergence, CLI and HTTP service.

```bash
CUDA_VISIBLE_DEVICES='' python -m tests.tier_runner core
```

`extended` additionally verifies benchmark harnesses, including the service-level Poisson
overload runner and the direct-versus-service startup diagnostic and examples. Run it
before a benchmark or a broad refactor.

```bash
CUDA_VISIBLE_DEVICES='' python -m tests.tier_runner extended
```

`gpu` is a focused subset. It requires an explicit model/checkpoint setup where applicable;
the CPU-only environment will skip CUDA-dependent cases.

```bash
CUDA_VISIBLE_DEVICES=0 python -m tests.tier_runner gpu
```
