"""
    Scheduler:请求调度器
    决定本轮执行哪些 Sequence，并显式划分 prefill/decode 分组
"""

from collections import deque
from dataclasses import dataclass

from llmserve.config import Config
from llmserve.engine.sequence import Sequence, SequenceStatus
from llmserve.engine.block_manager import BlockManager


@dataclass(slots=True)
class SchedulerOutput:
    """Explicit execution groups and token budget for one scheduler step."""

    scheduled_seqs: list[Sequence]
    prefill_seqs: list[Sequence]
    decode_seqs: list[Sequence]
    num_batched_tokens: int

    def __post_init__(self):
        if self.scheduled_seqs != self.prefill_seqs + self.decode_seqs:
            raise ValueError("scheduled sequences must be ordered as prefill then decode")
        if self.num_batched_tokens < 0:
            raise ValueError("num_batched_tokens cannot be negative")


class Scheduler:

    def __init__(self, config: Config):
        self.max_num_seqs = config.max_num_seqs
        self.max_num_batched_tokens = config.max_num_batched_tokens
        self.eos = config.eos
        self.block_manager = BlockManager(config.num_kvcache_blocks, config.kvcache_block_size)
        self.enable_chunked_prefill = config.enable_chunked_prefill
        self.waiting: deque[Sequence] = deque()
        self.running: deque[Sequence] = deque()
    # 调度器启动时会记住调度预算、建好 block 管理器、准备两个队列：waiting 和 running。

    def is_finished(self):
        return not self.waiting and not self.running

    def add(self, seq: Sequence):
        self.waiting.append(seq)

    def schedule(self) -> SchedulerOutput:
        # 挑出本轮要上 GPU 的序列，并显式区分 prefill/decode 分组。
        # ===== 2026-06-07 chunked prefill =====
        if self.enable_chunked_prefill:
            return self.schedule_chunked_prefill()

        # ===== 2026-06-07 chunked prefill =====
        scheduled_seqs = []
        num_batched_tokens = 0

        # prefill
        while self.waiting and len(scheduled_seqs) < self.max_num_seqs:
            seq = self.waiting[0]
            num_tokens = max(seq.num_tokens - seq.num_cached_tokens, 1)
            remaining = self.max_num_batched_tokens - num_batched_tokens
            if remaining == 0 or (not seq.block_table and not self.block_manager.can_allocate(seq)):    # no budget
                break
            if remaining < num_tokens and scheduled_seqs:    # only allow chunked prefill for the first seq
                break
            if not seq.block_table:
                self.block_manager.allocate(seq)
            seq.num_scheduled_tokens = min(num_tokens, remaining)
            if seq.num_scheduled_tokens == num_tokens:
                seq.status = SequenceStatus.RUNNING
                self.waiting.popleft()
                self.running.append(seq)
            scheduled_seqs.append(seq)
            num_batched_tokens += seq.num_scheduled_tokens
        if scheduled_seqs:
            return self._build_output(scheduled_seqs, [])

        # decode
        while self.running and len(scheduled_seqs) < self.max_num_seqs:
            seq = self.running.popleft()
            while not self.block_manager.can_append(seq):
                if self.running:
                    self.preempt(self.running.pop())
                else:
                    self.preempt(seq)
                    break
            else:
                seq.num_scheduled_tokens = 1
                self.block_manager.may_append(seq)
                scheduled_seqs.append(seq)
        assert scheduled_seqs
        self.running.extendleft(reversed(scheduled_seqs))
        return self._build_output([], scheduled_seqs)

    @staticmethod
    def _build_output(prefill_seqs: list[Sequence], decode_seqs: list[Sequence]):
        scheduled_seqs = prefill_seqs + decode_seqs
        return SchedulerOutput(
            scheduled_seqs=scheduled_seqs,
            prefill_seqs=prefill_seqs,
            decode_seqs=decode_seqs,
            num_batched_tokens=sum(seq.num_scheduled_tokens for seq in scheduled_seqs),
        )

    # ===== 2026-06-07 chunked prefill =====
    def schedule_chunked_prefill(self) -> SchedulerOutput:
        """先调度 decode，再把剩余 token budget 分给 prefill chunk。"""
        scheduled_decodes = []
        scheduled_prefills = []
        completed_prefills = []
        deferred_running = deque()
        num_scheduled_seqs = 0
        token_budget = self.max_num_batched_tokens

        # decode 优先：先保证已经进入 running 的请求能继续出 token。
        while self.running and num_scheduled_seqs < self.max_num_seqs and token_budget > 0:
            seq = self.running.popleft()
            while not self.block_manager.can_append(seq):
                if self.running:
                    self.preempt(self.running.pop())
                    # 如果 running 队列里还有其他请求，就从右边踢掉一个请求，释放它的 KV cache
                else:
                    self.preempt(seq)
                    seq = None
                    break
                    # 没有其他请求可以踢，就只能踢当前请求自己
            if seq is None:
                continue
            seq.num_scheduled_tokens = 1
            self.block_manager.may_append(seq)
            scheduled_decodes.append(seq)
            num_scheduled_seqs += 1
            token_budget -= 1
        deferred_running.extend(self.running)
        self.running.clear()

        # 剩余预算再给 waiting 队列里的长 prompt 做 prefill chunk。
        while self.waiting and num_scheduled_seqs < self.max_num_seqs and token_budget > 0:
            seq = self.waiting[0]
            if not seq.block_table:
                if not self.block_manager.can_allocate(seq):
                    break
                self.block_manager.allocate(seq)
            num_tokens = max(seq.num_tokens - seq.num_cached_tokens, 1)
            chunk_size = min(num_tokens, token_budget)
            if chunk_size == 0:
                break
            seq.num_scheduled_tokens = chunk_size
            scheduled_prefills.append(seq)
            token_budget -= chunk_size
            num_scheduled_seqs += 1
            if chunk_size == num_tokens:
                seq.status = SequenceStatus.RUNNING
                self.waiting.popleft()
                completed_prefills.append(seq)
            else:
                break

        self.running = deque(scheduled_decodes + completed_prefills)
        self.running.extend(deferred_running)

        if scheduled_prefills:
            return self._build_output(scheduled_prefills, scheduled_decodes)
        if scheduled_decodes:
            return self._build_output([], scheduled_decodes)
        assert False, "scheduler has no sequence to schedule"
    # ===== 2026-06-07 chunked prefill =====

    def preempt(self, seq: Sequence):
        seq.status = SequenceStatus.WAITING
        self.block_manager.deallocate(seq)
        self.waiting.appendleft(seq)
    # 资源不够时，把这条请求先踢下场，清掉它的 KV cache，占位释放出来，然后把它插回等待队列前面

    def postprocess(self, output: SchedulerOutput, token_ids: list[int]):
        if len(output.scheduled_seqs) != len(token_ids):
            raise ValueError("token count does not match scheduled sequences")
        num_prefill_seqs = len(output.prefill_seqs)
        for seq, token_id in zip(
            output.prefill_seqs,
            token_ids[:num_prefill_seqs],
        ):
            self._postprocess_sequence(seq, token_id, is_prefill=True)
        for seq, token_id in zip(
            output.decode_seqs,
            token_ids[num_prefill_seqs:],
        ):
            self._postprocess_sequence(seq, token_id, is_prefill=False)

    def _postprocess_sequence(self, seq: Sequence, token_id: int, is_prefill: bool):
        if is_prefill:
            seq.num_cached_tokens = min(
                seq.num_cached_tokens + seq.num_scheduled_tokens,
                seq.num_tokens,
            )
            if seq.num_cached_tokens < seq.num_tokens or seq.num_completion_tokens > 0:
                seq.num_scheduled_tokens = 0
                return
        seq.append_token(token_id)
        seq.num_cached_tokens += 1
        seq.num_scheduled_tokens = 0
        if (not seq.ignore_eos and token_id == self.eos) or seq.num_completion_tokens == seq.max_tokens:
            seq.status = SequenceStatus.FINISHED
            self.block_manager.deallocate(seq)
            self.running.remove(seq)

    # ===== 2026-07-07 EAGLE speculative decoding =====
    def postprocess_speculative(self, seq: Sequence, token_ids: list[int]):
        remaining = max(seq.max_tokens - seq.num_completion_tokens, 0)
        token_ids = token_ids[:remaining]
        for token_id in token_ids:
            seq.append_token(token_id)
            seq.num_cached_tokens += 1
            if not seq.ignore_eos and token_id == self.eos:
                break
        self.block_manager.finalize_full_blocks(seq)
        self.block_manager.release_extra_slots(seq)
        seq.num_scheduled_tokens = 0
        if (
            seq.num_completion_tokens == seq.max_tokens
            or (not seq.ignore_eos and seq.completion_token_ids and seq.completion_token_ids[-1] == self.eos)
        ):
            seq.status = SequenceStatus.FINISHED
            self.block_manager.deallocate(seq)
            if seq in self.running:
                self.running.remove(seq)
    # ===== 2026-07-07 EAGLE speculative decoding =====
