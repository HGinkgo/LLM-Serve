"""
    BlockManager 负责管理 KV cache 的“块”怎么分配、复用、释放
    free:空闲块,ref_count == 0
    used + hashed:已使用且内容固定,hash != -1,可参与 prefix cache 复用
    used + mutable tail:已使用但还是最后那个未封口块,hash == -1
"""

from collections import deque
import xxhash
import numpy as np

from llmserve.engine.sequence import Sequence


class Block:

    def __init__(self, block_id):
        self.block_id = block_id
        self.ref_count = 0      # 如果多个 sequence 复用了同一个前缀块，这个值会大于 1
        self.hash = -1          # -1 表示当前这个块还不能作为可复用前缀块来识别
        self.token_ids = []

    # 这个块代表哪段 token
    def update(self, hash: int, token_ids: list[int]):
        self.hash = hash
        self.token_ids = token_ids

    # 重置块状态
    def reset(self):
        self.ref_count = 1
        self.hash = -1
        self.token_ids = []


class BlockManager:

    def __init__(self, num_blocks: int, block_size: int):
        self.block_size = block_size
        self.blocks: list[Block] = [Block(i) for i in range(num_blocks)]
        self.hash_to_block_id: dict[int, int] = dict()
        self.free_block_ids: deque[int] = deque(range(num_blocks))
        self.used_block_ids: set[int] = set()

    @classmethod
    def compute_hash(cls, token_ids: list[int], prefix: int = -1):
        h = xxhash.xxh64()
        if prefix != -1:
            h.update(prefix.to_bytes(8, "little"))
        h.update(np.array(token_ids).tobytes())
        return h.intdigest()

    def _allocate_block(self, block_id: int) -> Block:
        block = self.blocks[block_id]
        assert block.ref_count == 0
        block.reset()
        self.free_block_ids.remove(block_id)
        self.used_block_ids.add(block_id)
        return block

    # 把一个 block 从 used 放回 free
    def _deallocate_block(self, block_id: int) -> Block:
        assert self.blocks[block_id].ref_count == 0
        self.used_block_ids.remove(block_id)
        self.free_block_ids.append(block_id)

    def can_allocate(self, seq: Sequence) -> bool:
        return len(self.free_block_ids) >= seq.num_blocks

    # 给一条 sequence 简历 block_table 同时尽量利用 prefix cache 复用已有块
    def allocate(self, seq: Sequence):
        assert not seq.block_table
        h = -1
        cache_miss = False
        for i in range(seq.num_blocks):
            token_ids = seq.block(i)
            h = self.compute_hash(token_ids, h) if len(token_ids) == self.block_size else -1
            # 只有满块才参与 prefix cache，或者说只有满块才进行分配
            block_id = self.hash_to_block_id.get(h, -1)
            if block_id == -1 or self.blocks[block_id].token_ids != token_ids:
                cache_miss = True       # 不能复用之前的老前缀复用
            if cache_miss:
                block_id = self.free_block_ids[0]
                block = self._allocate_block(block_id)
            else:
                seq.num_cached_tokens += self.block_size
                if block_id in self.used_block_ids:
                    block = self.blocks[block_id]
                    block.ref_count += 1
                else:
                    block = self._allocate_block(block_id)
            if h != -1:
                block.update(h, token_ids)
                self.hash_to_block_id[h] = block_id
            seq.block_table.append(block_id)
    # 按逻辑块遍历 sequence，优先尝试用 hash 命中 prefix cache，
    # 命不中就分新块，最后建立 sequence 的 block_table

    def deallocate(self, seq: Sequence):
        for block_id in reversed(seq.block_table):
            block = self.blocks[block_id]
            block.ref_count -= 1
            if block.ref_count == 0:
                self._deallocate_block(block_id)
        seq.num_cached_tokens = 0
        seq.block_table.clear()

    def can_append(self, seq: Sequence) -> bool:
        return len(self.free_block_ids) >= (len(seq) % self.block_size == 1)

    # ===== 2026-07-07 EAGLE speculative decoding =====
    def num_required_blocks(self, seq: Sequence, num_tokens: int) -> int:
        future_num_tokens = len(seq) + num_tokens
        future_num_blocks = (future_num_tokens + self.block_size - 1) // self.block_size
        return max(future_num_blocks - len(seq.block_table), 0)

    def can_append_tokens(self, seq: Sequence, num_tokens: int) -> bool:
        return len(self.free_block_ids) >= self.num_required_blocks(seq, num_tokens)

    def _finalize_block_hash(self, seq: Sequence, block_idx: int):
        block_id = seq.block_table[block_idx]
        block = self.blocks[block_id]
        if block.hash != -1:
            return
        token_ids = seq.block(block_idx)
        if len(token_ids) != self.block_size:
            return
        prefix = self.blocks[seq.block_table[block_idx - 1]].hash if block_idx > 0 else -1
        h = self.compute_hash(token_ids, prefix)
        block.update(h, token_ids)
        self.hash_to_block_id[h] = block_id

    def ensure_slots(self, seq: Sequence, num_tokens: int) -> bool:
        if not self.can_append_tokens(seq, num_tokens):
            return False
        if seq.block_table and len(seq) % self.block_size == 0:
            self._finalize_block_hash(seq, seq.num_blocks - 1)
        for _ in range(self.num_required_blocks(seq, num_tokens)):
            block_id = self.free_block_ids[0]
            self._allocate_block(block_id)
            seq.block_table.append(block_id)
        return True

    def ensure_slots_batch(self, seqs: list[Sequence], num_tokens: int) -> bool:
        """先检查整批容量，再为每条序列预留 speculative KV slots。"""
        required_blocks = sum(self.num_required_blocks(seq, num_tokens) for seq in seqs)
        if required_blocks > len(self.free_block_ids):
            return False
        for seq in seqs:
            if not self.ensure_slots(seq, num_tokens):
                raise RuntimeError("speculative batch reservation became non-atomic")
        return True

    def finalize_full_blocks(self, seq: Sequence):
        for block_idx in range(min(seq.num_blocks, len(seq.block_table))):
            self._finalize_block_hash(seq, block_idx)

    def release_extra_slots(self, seq: Sequence):
        while len(seq.block_table) > seq.num_blocks:
            block_id = seq.block_table.pop()
            block = self.blocks[block_id]
            block.ref_count -= 1
            if block.ref_count == 0:
                self._deallocate_block(block_id)
    # ===== 2026-07-07 EAGLE speculative decoding =====

    def may_append(self, seq: Sequence):
        block_table = seq.block_table
        last_block = self.blocks[block_table[-1]]
        if len(seq) % self.block_size == 1 and len(block_table) < seq.num_blocks:
            assert last_block.hash != -1
            block_id = self.free_block_ids[0]
            self._allocate_block(block_id)
            block_table.append(block_id)
        elif len(seq) % self.block_size == 0:
            self._finalize_block_hash(seq, seq.num_blocks - 1)
        else:
            assert last_block.hash == -1
