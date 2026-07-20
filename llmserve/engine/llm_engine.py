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

from llmserve.config import Config
from llmserve.sampling_params import SamplingParams
from llmserve.engine.sequence import Sequence
from llmserve.engine.scheduler import Scheduler
from llmserve.engine.model_runner import ModelRunner


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
        self.speculative_batch_calls = 0
        self.speculative_batch_sequences = 0
        self.speculative_max_batch_size = 0
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
            "output_event_times": [],
            "speculative_step_latencies": [],
            "finish_time": None,
            "prompt_tokens": len(prompt),
            "output_tokens": 0,
            "success": False,
            "failure_reason": None,
            "speculative_steps": 0,
            "speculative_draft_tokens": 0,
            "speculative_accepted_tokens": 0,
            "speculative_emitted_tokens": 0,
            "speculative_accept_all_count": 0,
            "speculative_gamma_counts": {},
            "speculative_trace": [],
            "speculative_timing": {},
        }
        self.scheduler.add(seq)
        return seq.seq_id

    @staticmethod
    def _build_speculative_trace_entry(
        debug: dict,
        step: int,
        emitted_token_ids: list[int],
        num_accepted: int,
        accepted_all: bool,
    ):
        entry = {
            "step": step,
            "draft_token_ids": debug.get("draft_token_ids"),
            "matches": debug.get("matches"),
            "emitted_token_ids": [int(token_id) for token_id in emitted_token_ids],
            "num_accepted": int(num_accepted),
            "accepted_all": bool(accepted_all),
        }
        for key in (
            "start_token_id",
            "target_argmax_token_ids",
            "draft_token_target_ranks",
        ):
            if key in debug:
                entry[key] = debug[key]
        return entry

    @staticmethod
    def _accumulate_speculative_timing(metric: dict, timing: dict | None):
        if not timing:
            return
        bucket = metric.setdefault("speculative_timing", {})
        for name, value in timing.items():
            bucket[name] = bucket.get(name, 0.0) + float(value)

    @staticmethod
    def _summarize_speculative_timing(timing: dict, steps: int):
        return {
            name: {
                "total": value,
                "mean": value / steps if steps > 0 else None,
            }
            for name, value in sorted(timing.items())
        }

    def step(self):
        step_start = perf_counter()
        seqs, is_prefill = self.scheduler.schedule()
        if self._can_run_speculative_step(seqs, is_prefill):
            num_reserved_tokens = self.model_runner.speculative_gamma + 2
            block_manager = self.scheduler.block_manager
            if len(seqs) == 1:
                slots_ready = block_manager.ensure_slots(seqs[0], num_reserved_tokens)
            else:
                slots_ready = block_manager.ensure_slots_batch(seqs, num_reserved_tokens)
            if slots_ready:
                if len(seqs) == 1:
                    method_name = (
                        "run_speculative_tree_single"
                        if getattr(self.model_runner, "speculative_tree_nodes", 0)
                        else "run_speculative_single"
                    )
                    speculative_outputs = [
                        self.model_runner.call(method_name, seqs[0])
                    ]
                else:
                    speculative_outputs = self.model_runner.call("run_speculative_batch", seqs)
                return self._postprocess_speculative_step(
                    seqs,
                    speculative_outputs,
                    step_start,
                )

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
                metric.setdefault("output_event_times", []).append(step_end)
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

    def _postprocess_speculative_step(
        self,
        seqs: list[Sequence],
        speculative_outputs,
        step_start: float,
    ):
        if len(seqs) != len(speculative_outputs):
            raise ValueError("speculative output count does not match scheduled sequences")

        before_completion_tokens = {
            seq.seq_id: seq.num_completion_tokens for seq in seqs
        }
        for seq, speculative_output in zip(seqs, speculative_outputs):
            self.scheduler.postprocess_speculative(seq, speculative_output.token_ids)
        step_end = perf_counter()

        self.speculative_batch_calls = getattr(self, "speculative_batch_calls", 0) + 1
        self.speculative_batch_sequences = getattr(self, "speculative_batch_sequences", 0) + len(seqs)
        self.speculative_max_batch_size = max(
            getattr(self, "speculative_max_batch_size", 0),
            len(seqs),
        )

        first_token_seq_ids = []
        decode_seq_ids = []
        finished_seq_ids = []
        debug_by_seq = {}
        total_draft_tokens = 0
        total_accepted_tokens = 0
        total_emitted_tokens = 0

        for seq, speculative_output in zip(seqs, speculative_outputs):
            before = before_completion_tokens[seq.seq_id]
            after = seq.num_completion_tokens
            emitted_tokens = after - before
            total_draft_tokens += speculative_output.num_draft_tokens
            total_accepted_tokens += speculative_output.num_accepted
            total_emitted_tokens += emitted_tokens

            metric = self.request_metrics.get(seq.seq_id)
            if metric is not None:
                metric.setdefault("speculative_step_latencies", []).append(
                    step_end - step_start
                )
                if emitted_tokens > 0:
                    if metric["first_token_time"] is None:
                        metric["first_token_time"] = step_end
                        first_token_seq_ids.append(seq.seq_id)
                    decode_seq_ids.append(seq.seq_id)
                    metric["token_times"].extend([step_end] * emitted_tokens)
                    metric.setdefault("output_event_times", []).append(step_end)
                    metric["output_tokens"] = after
                metric.setdefault("speculative_steps", 0)
                metric.setdefault("speculative_draft_tokens", 0)
                metric.setdefault("speculative_accepted_tokens", 0)
                metric.setdefault("speculative_emitted_tokens", 0)
                metric.setdefault("speculative_accept_all_count", 0)
                metric.setdefault("speculative_gamma_counts", {})
                metric["speculative_steps"] += 1
                metric["speculative_draft_tokens"] += speculative_output.num_draft_tokens
                metric["speculative_accepted_tokens"] += speculative_output.num_accepted
                metric["speculative_emitted_tokens"] += emitted_tokens
                metric["speculative_accept_all_count"] += int(speculative_output.accepted_all)
                gamma_key = str(speculative_output.num_draft_tokens)
                metric["speculative_gamma_counts"][gamma_key] = (
                    metric["speculative_gamma_counts"].get(gamma_key, 0) + 1
                )
                self._accumulate_speculative_timing(metric, speculative_output.timing)
                if speculative_output.debug is not None:
                    metric.setdefault("speculative_trace", []).append(
                        self._build_speculative_trace_entry(
                            speculative_output.debug,
                            metric["speculative_steps"],
                            speculative_output.token_ids,
                            speculative_output.num_accepted,
                            speculative_output.accepted_all,
                        )
                    )
                if seq.is_finished and metric["finish_time"] is None:
                    metric["finish_time"] = step_end
                    metric["success"] = True
                    finished_seq_ids.append(seq.seq_id)
            if speculative_output.debug is not None:
                debug_by_seq[seq.seq_id] = speculative_output.debug

        num_tokens = -total_emitted_tokens
        self.last_step_events = {
            "step_start": step_start,
            "step_end": step_end,
            "is_prefill": False,
            "num_tokens": num_tokens,
            "scheduled_seq_ids": [seq.seq_id for seq in seqs],
            "first_token_seq_ids": first_token_seq_ids,
            "decode_seq_ids": decode_seq_ids,
            "finished_seq_ids": finished_seq_ids,
            "speculative": True,
            "speculative_batch_size": len(seqs),
            "speculative_num_draft_tokens": total_draft_tokens,
            "speculative_num_accepted": total_accepted_tokens,
            "speculative_accepted_all": all(output.accepted_all for output in speculative_outputs),
            "speculative_emitted_tokens": total_emitted_tokens,
        }
        if debug_by_seq:
            self.last_step_events["speculative_debug_by_seq"] = debug_by_seq
            if len(seqs) == 1:
                self.last_step_events["speculative_debug"] = debug_by_seq[seqs[0].seq_id]

        finished_state_ids = [seq.seq_id for seq in seqs if seq.is_finished]
        if finished_state_ids:
            self.model_runner.call("clear_speculative_state", finished_state_ids)

        outputs = [
            (seq.seq_id, seq.completion_token_ids)
            for seq in seqs
            if seq.is_finished
        ]
        return outputs, num_tokens

    def _can_run_speculative_step(self, seqs: list[Sequence], is_prefill: bool):
        if is_prefill or not seqs:
            return False
        if getattr(self.model_runner, "draft_model", None) is None:
            return False
        if not getattr(self.model_runner, "enforce_eager", False):
            return False
        if getattr(self.model_runner, "world_size", 1) != 1:
            return False
        return True

    def get_metrics(self):
        now = perf_counter()
        requests = []
        ttfts = []
        burst_itls = []
        output_event_latencies = []
        speculative_step_latencies = []
        request_latencies = []
        tpots = []
        wall_start = None
        wall_end = None
        total_output_tokens = 0
        total_speculative_steps = 0
        total_speculative_draft_tokens = 0
        total_speculative_accepted_tokens = 0
        total_speculative_emitted_tokens = 0
        total_speculative_accept_all_count = 0
        total_speculative_gamma_counts = {}
        total_speculative_timing = {}

        for metric in self.request_metrics.values():
            arrival_time = metric["arrival_time"]
            first_token_time = metric["first_token_time"]
            token_times = metric["token_times"]
            output_event_times = metric.get("output_event_times", [])
            request_speculative_step_latencies = metric.get(
                "speculative_step_latencies", []
            )
            finish_time = metric["finish_time"]
            output_tokens = metric["output_tokens"]
            success = metric["success"]
            failure_reason = metric["failure_reason"]
            speculative_steps = metric.get("speculative_steps", 0)
            speculative_draft_tokens = metric.get("speculative_draft_tokens", 0)
            speculative_accepted_tokens = metric.get("speculative_accepted_tokens", 0)
            speculative_emitted_tokens = metric.get("speculative_emitted_tokens", 0)
            speculative_accept_all_count = metric.get("speculative_accept_all_count", 0)
            speculative_gamma_counts = metric.get("speculative_gamma_counts", {})
            speculative_trace = metric.get("speculative_trace", [])
            speculative_timing = metric.get("speculative_timing", {})

            wall_start = arrival_time if wall_start is None else min(wall_start, arrival_time)
            finish_or_now_time = finish_time if finish_time is not None else now
            wall_end = finish_or_now_time if wall_end is None else max(wall_end, finish_or_now_time)
            total_output_tokens += output_tokens
            total_speculative_steps += speculative_steps
            total_speculative_draft_tokens += speculative_draft_tokens
            total_speculative_accepted_tokens += speculative_accepted_tokens
            total_speculative_emitted_tokens += speculative_emitted_tokens
            total_speculative_accept_all_count += speculative_accept_all_count
            for gamma, count in speculative_gamma_counts.items():
                total_speculative_gamma_counts[gamma] = (
                    total_speculative_gamma_counts.get(gamma, 0) + count
                )
            for name, value in speculative_timing.items():
                total_speculative_timing[name] = total_speculative_timing.get(name, 0.0) + value

            ttft = None
            if first_token_time is not None:
                ttft = first_token_time - arrival_time
                ttfts.append(ttft)

            request_itls = [
                token_times[i] - token_times[i - 1]
                for i in range(1, len(token_times))
            ]
            request_output_event_latencies = [
                output_event_times[i] - output_event_times[i - 1]
                for i in range(1, len(output_event_times))
            ]
            burst_itls.extend(request_itls)
            output_event_latencies.extend(request_output_event_latencies)
            speculative_step_latencies.extend(request_speculative_step_latencies)

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
                "output_event_times": output_event_times,
                "burst_itl": request_itls,
                "itl": request_itls,
                "output_event_latency": request_output_event_latencies,
                "speculative_step_latency": request_speculative_step_latencies,
                "latency": latency,
                "speculative_steps": speculative_steps,
                "speculative_draft_tokens": speculative_draft_tokens,
                "speculative_accepted_tokens": speculative_accepted_tokens,
                "speculative_emitted_tokens": speculative_emitted_tokens,
                "speculative_accept_all_count": speculative_accept_all_count,
                "speculative_gamma_counts": dict(
                    sorted(speculative_gamma_counts.items(), key=lambda item: int(item[0]))
                ),
                "speculative_trace": speculative_trace,
                "speculative_timing": speculative_timing,
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
                "burst_itl": self._summary(burst_itls),
                "itl": self._summary(burst_itls),
                "output_event_latency": self._summary(output_event_latencies),
                "speculative_step_latency": self._summary(speculative_step_latencies),
                "tpot": self._summary(tpots),
                "request_latency": self._summary(request_latencies),
                "speculative": {
                    "steps": total_speculative_steps,
                    "batch_calls": getattr(self, "speculative_batch_calls", 0),
                    "mean_batch_size": (
                        getattr(self, "speculative_batch_sequences", 0)
                        / getattr(self, "speculative_batch_calls", 0)
                        if getattr(self, "speculative_batch_calls", 0) > 0 else None
                    ),
                    "max_batch_size": getattr(self, "speculative_max_batch_size", 0),
                    "draft_tokens": total_speculative_draft_tokens,
                    "accepted_tokens": total_speculative_accepted_tokens,
                    "emitted_tokens": total_speculative_emitted_tokens,
                    "acceptance_rate": (
                        total_speculative_accepted_tokens / total_speculative_draft_tokens
                        if total_speculative_draft_tokens > 0 else None
                    ),
                    "acceptance_length": (
                        total_speculative_emitted_tokens / total_speculative_steps
                        if total_speculative_steps > 0 else None
                    ),
                    "accepted_length": (
                        total_speculative_accepted_tokens / total_speculative_steps
                        if total_speculative_steps > 0 else None
                    ),
                    "draft_tokens_per_step": (
                        total_speculative_draft_tokens / total_speculative_steps
                        if total_speculative_steps > 0 else None
                    ),
                    "accept_all_count": total_speculative_accept_all_count,
                    "gamma_counts": dict(
                        sorted(total_speculative_gamma_counts.items(), key=lambda item: int(item[0]))
                    ),
                    "timing": self._summarize_speculative_timing(
                        total_speculative_timing,
                        total_speculative_steps,
                    ),
                },
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
