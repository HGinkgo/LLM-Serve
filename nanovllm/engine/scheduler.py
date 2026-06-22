"""
    Scheduler:请求调度器
    决定这轮该跑哪些 Sequence,这轮走 prefill 还是 decode
"""

from collections import deque

from nanovllm.config import Config
from nanovllm.engine.sequence import Sequence, SequenceStatus
from nanovllm.engine.block_manager import BlockManager


class Scheduler:

    def __init__(self, config: Config):
        self.max_num_seqs = config.max_num_seqs
        self.max_num_batched_tokens = config.max_num_batched_tokens
        self.eos = config.eos
        self.block_manager = BlockManager(config.num_kvcache_blocks, config.kvcache_block_size)
        self.enable_chunked_prefill = config.enable_chunked_prefill
        self.waiting: deque[Sequence] = deque()
        self.running: deque[Sequence] = deque()
        # mixed chunked-prefill batch 仍复用旧的 `(seqs, is_prefill)` 接口；
        # 这里记录 prefill/decode 分界，方便 LLMEngine 分段 postprocess。
        self.last_num_prefill_seqs = 0
    # 调度器启动时会记住调度预算、建好 block 管理器、准备两个队列：waiting 和 running。

    def is_finished(self):
        return not self.waiting and not self.running

    def add(self, seq: Sequence):
        self.waiting.append(seq)

    def schedule(self) -> tuple[list[Sequence], bool]:
    # 挑出本轮要上 GPU 的序列，并告诉外面这是 prefill 轮还是 decode 轮。
        # ===== 2026-06-07 chunked prefill =====
        if self.enable_chunked_prefill:
            return self.schedule_chunked_prefill()

        self.last_num_prefill_seqs = 0
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
            # ===== 2026-06-07 chunked prefill =====
            self.last_num_prefill_seqs = len(scheduled_seqs)
            # ===== 2026-06-07 chunked prefill =====
            return scheduled_seqs, True

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
        return scheduled_seqs, False

    # ===== 2026-06-07 chunked prefill =====
    def schedule_chunked_prefill(self) -> tuple[list[Sequence], bool]:
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
            self.last_num_prefill_seqs = len(scheduled_prefills)
            for seq in scheduled_decodes:
                # 负数表示这个 decode token 被打包进了 prefill 形态的 mixed batch。
                seq.num_scheduled_tokens = -1
            return scheduled_prefills + scheduled_decodes, True
        if scheduled_decodes:
            self.last_num_prefill_seqs = 0
            return scheduled_decodes, False
        assert False, "scheduler has no sequence to schedule"
    # ===== 2026-06-07 chunked prefill =====

    def preempt(self, seq: Sequence):
        seq.status = SequenceStatus.WAITING
        self.block_manager.deallocate(seq)
        self.waiting.appendleft(seq)
    # 资源不够时，把这条请求先踢下场，清掉它的 KV cache，占位释放出来，然后把它插回等待队列前面

    def postprocess(self, seqs: list[Sequence], token_ids: list[int], is_prefill: bool):
        for seq, token_id in zip(seqs, token_ids):
            if is_prefill:
                seq.num_cached_tokens = min(seq.num_cached_tokens + seq.num_scheduled_tokens, seq.num_tokens)
                if seq.num_cached_tokens < seq.num_tokens or seq.num_completion_tokens > 0:    # chunked prefill or re prefill after preemption
                    seq.num_scheduled_tokens = 0
                    continue
            seq.append_token(token_id)
            seq.num_cached_tokens += 1
            seq.num_scheduled_tokens = 0
            if (not seq.ignore_eos and token_id == self.eos) or seq.num_completion_tokens == seq.max_tokens:
                seq.status = SequenceStatus.FINISHED
                self.block_manager.deallocate(seq)
                self.running.remove(seq)
