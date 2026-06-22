"""
    初始化整个推理系统，接受请求并交给调度器
    驱动 prefill/decode 循环直到生成结束
"""
import atexit
from dataclasses import fields
from time import perf_counter
from tqdm.auto import tqdm
from transformers import AutoTokenizer                  # prompt 编码，输出 token 编码
import torch.multiprocessing as mp

from nanovllm.config import Config
from nanovllm.sampling_params import SamplingParams
from nanovllm.engine.sequence import Sequence
from nanovllm.engine.scheduler import Scheduler
from nanovllm.engine.model_runner import ModelRunner


class LLMEngine:

    def __init__(self, model, **kwargs):                # 传入模型和其他参数
        config_fields = {field.name for field in fields(Config)}
        config_kwargs = {k: v for k, v in kwargs.items() if k in config_fields}
        config = Config(model, **config_kwargs)
        Sequence.block_size = config.kvcache_block_size
        self.ps = []
        self.events = []
        ctx = mp.get_context("spawn")                   # 起 GPU 子进程时，用 spawn 比较安全
        for i in range(1, config.tensor_parallel_size):
            event = ctx.Event()
            process = ctx.Process(target=ModelRunner, args=(config, i, event)) 
            process.start()
            self.ps.append(process)
            self.events.append(event)
        self.model_runner = ModelRunner(config, 0, self.events)  # 起 rank0
        self.tokenizer = AutoTokenizer.from_pretrained(config.model, use_fast=True)
        config.eos = self.tokenizer.eos_token_id
        self.scheduler = Scheduler(config)
        self.request_metrics = {}
        self.last_step_events = {}
        self._exited = False
        atexit.register(self.exit)

    @staticmethod
    def _percentile(values: list[float], percentile: float):
        if not values:
            return None
        values = sorted(values)
        if len(values) == 1:
            return values[0]
        rank = (len(values) - 1) * percentile / 100
        low = int(rank)
        high = min(low + 1, len(values) - 1)
        weight = rank - low
        return values[low] * (1 - weight) + values[high] * weight

    @classmethod
    def _summary(cls, values: list[float]):
        if not values:
            return {"mean": None, "p50": None, "p99": None, "max": None}
        return {
            "mean": sum(values) / len(values),
            "p50": cls._percentile(values, 50),
            "p99": cls._percentile(values, 99),
            "max": max(values),
        }

    def exit(self):
        if self._exited:
            return
        self._exited = True
        self.model_runner.call("exit")
        del self.model_runner
        for p in self.ps:
            p.join()

    # 把用户请求放进系统
    def add_request(self, prompt: str | list[int], sampling_params: SamplingParams):
        if isinstance(prompt, str):
            prompt = self.tokenizer.encode(prompt)
        seq = Sequence(prompt, sampling_params)
        now = perf_counter()
        self.request_metrics[seq.seq_id] = {
            "seq_id": seq.seq_id,
            "arrival_time": now,
            "first_token_time": None,
            "token_times": [],
            "finish_time": None,
            "prompt_tokens": len(prompt),
            "output_tokens": 0,
            "success": False,
            "failure_reason": None,
        }
        self.scheduler.add(seq)
        return seq.seq_id

    def step(self):
        step_start = perf_counter()
        seqs, is_prefill = self.scheduler.schedule()
        # ===== 2026-06-07 chunked prefill =====
        num_tokens = sum(max(seq.num_scheduled_tokens, 0) for seq in seqs) if is_prefill else -len(seqs)
        # ===== 2026-06-07 chunked prefill =====
        before_completion_tokens = {seq.seq_id: seq.num_completion_tokens for seq in seqs}
        token_ids = self.model_runner.call("run", seqs, is_prefill)
        # ===== 2026-06-07 chunked prefill =====
        num_prefill_seqs = self.scheduler.last_num_prefill_seqs if is_prefill else 0
        # mixed chunked-prefill 输出顺序是 [prefill..., decode...]。
        # 分段调用 postprocess，可以复用原有 prefill/decode 处理语义。
        if is_prefill and 0 < num_prefill_seqs < len(seqs):
            self.scheduler.postprocess(seqs[:num_prefill_seqs], token_ids[:num_prefill_seqs], True)
            self.scheduler.postprocess(seqs[num_prefill_seqs:], token_ids[num_prefill_seqs:], False)
        else:
            self.scheduler.postprocess(seqs, token_ids, is_prefill)
        # ===== 2026-06-07 chunked prefill =====
        step_end = perf_counter()

        first_token_seq_ids = []
        decode_seq_ids = []
        finished_seq_ids = []
        for seq in seqs:
            metric = self.request_metrics.get(seq.seq_id)
            if metric is None:
                continue
            before = before_completion_tokens[seq.seq_id]
            after = seq.num_completion_tokens
            if after > before:
                if metric["first_token_time"] is None:
                    metric["first_token_time"] = step_end
                    first_token_seq_ids.append(seq.seq_id)
                decode_seq_ids.append(seq.seq_id)
                metric["token_times"].extend([step_end] * (after - before))
                metric["output_tokens"] = after
            if seq.is_finished and metric["finish_time"] is None:
                metric["finish_time"] = step_end
                metric["success"] = True
                finished_seq_ids.append(seq.seq_id)

        self.last_step_events = {
            "step_start": step_start,
            "step_end": step_end,
            "is_prefill": is_prefill,
            "num_tokens": num_tokens,
            "scheduled_seq_ids": [seq.seq_id for seq in seqs],
            "first_token_seq_ids": first_token_seq_ids,
            "decode_seq_ids": decode_seq_ids,
            "finished_seq_ids": finished_seq_ids,
        }
        outputs = [(seq.seq_id, seq.completion_token_ids) for seq in seqs if seq.is_finished]
        return outputs, num_tokens

    def get_metrics(self):
        now = perf_counter()
        requests = []
        ttfts = []
        itls = []
        request_latencies = []
        tpots = []
        wall_start = None
        wall_end = None
        total_output_tokens = 0

        for metric in self.request_metrics.values():
            arrival_time = metric["arrival_time"]
            first_token_time = metric["first_token_time"]
            token_times = metric["token_times"]
            finish_time = metric["finish_time"]
            output_tokens = metric["output_tokens"]
            success = metric["success"]
            failure_reason = metric["failure_reason"]

            wall_start = arrival_time if wall_start is None else min(wall_start, arrival_time)
            finish_or_now_time = finish_time if finish_time is not None else now
            wall_end = finish_or_now_time if wall_end is None else max(wall_end, finish_or_now_time)
            total_output_tokens += output_tokens

            ttft = None
            if first_token_time is not None:
                ttft = first_token_time - arrival_time
                ttfts.append(ttft)

            request_itls = [
                token_times[i] - token_times[i - 1]
                for i in range(1, len(token_times))
            ]
            itls.extend(request_itls)

            latency = None
            if finish_time is not None:
                latency = finish_time - arrival_time
                request_latencies.append(latency)
                if output_tokens > 1 and first_token_time is not None:
                    tpot = (finish_time - first_token_time) / (output_tokens - 1)
                    tpots.append(tpot)
                elif output_tokens == 1:
                    tpots.append(0.0)
            elif not success:
                failure_reason = failure_reason or "unfinished"

            requests.append({
                "seq_id": metric["seq_id"],
                "prompt_tokens": metric["prompt_tokens"],
                "output_tokens": output_tokens,
                "success": success,
                "failure_reason": None if success else failure_reason,
                "arrival_time": arrival_time,
                "first_token_time": first_token_time,
                "token_times": token_times,
                "finish_time": finish_time,
                "ttft": ttft,
                "itl": request_itls,
                "latency": latency,
            })

        wall_time = 0.0
        if wall_start is not None and wall_end is not None:
            wall_time = wall_end - wall_start

        num_requests = len(requests)
        num_finished = sum(1 for request in requests if request["success"])
        return {
            "summary": {
                "num_requests": num_requests,
                "num_finished": num_finished,
                "num_failed": num_requests - num_finished,
                "total_output_tokens": total_output_tokens,
                "wall_time": wall_time,
                "throughput": total_output_tokens / wall_time if wall_time > 0 else 0.0,
                "ttft": self._summary(ttfts),
                "itl": self._summary(itls),
                "tpot": self._summary(tpots),
                "request_latency": self._summary(request_latencies),
            },
            "requests": sorted(requests, key=lambda request: request["seq_id"]),
        }

    def is_finished(self):
        return self.scheduler.is_finished()

    def generate(
        self,
        prompts: list[str] | list[list[int]],
        sampling_params: SamplingParams | list[SamplingParams],
        use_tqdm: bool = True,
    ) -> list[str]:
        pbar = tqdm(total=len(prompts), desc="Generating", dynamic_ncols=True, disable=not use_tqdm)
        if not isinstance(sampling_params, list):
            sampling_params = [sampling_params] * len(prompts)
        for prompt, sp in zip(prompts, sampling_params):
            self.add_request(prompt, sp)
        outputs = {}
        prefill_throughput = decode_throughput = 0.
        while not self.is_finished():
            t = perf_counter()
            output, num_tokens = self.step()
            if num_tokens > 0:
                prefill_throughput = num_tokens / (perf_counter() - t)
            else:
                decode_throughput = -num_tokens / (perf_counter() - t)
            pbar.set_postfix({
                "Prefill": f"{int(prefill_throughput)}tok/s",
                "Decode": f"{int(decode_throughput)}tok/s",
            })
            for seq_id, token_ids in output:
                outputs[seq_id] = token_ids
                pbar.update(1)
        pbar.close()
        outputs = [outputs[seq_id] for seq_id in sorted(outputs.keys())]
        outputs = [{"text": self.tokenizer.decode(token_ids), "token_ids": token_ids} for token_ids in outputs]
        return outputs
