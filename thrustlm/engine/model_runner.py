import pickle
import torch
import torch.distributed as dist
from dataclasses import dataclass
from multiprocessing.synchronize import Event
from multiprocessing.shared_memory import SharedMemory
from time import perf_counter

from thrustlm.config import Config
from thrustlm.engine.sequence import Sequence
from thrustlm.models.eagle3 import (
    Eagle3Speculator,
    Eagle3TargetVerifyOutput,
    generate_eagle3_draft_tokens,
    generate_eagle3_draft_tokens_batched,
    speculative_accept_greedy_from_logits,
    speculative_accept_reject_from_logits,
)
from thrustlm.models.qwen3 import Qwen3ForCausalLM
from thrustlm.layers.sampler import Sampler
from thrustlm.utils.context import set_context, get_context, reset_context
from thrustlm.utils.loader import load_model


@dataclass(slots=True)
class TargetDecodeAuxOutput:
    token_ids: list[int]
    logits: torch.Tensor
    aux_hidden: torch.Tensor
    positions: torch.Tensor


@dataclass(slots=True)
class SpeculativeDecodeOutput:
    token_ids: list[int]
    num_draft_tokens: int
    num_accepted: int
    accepted_all: bool
    emitted_tokens: int
    timing: dict | None = None
    debug: dict | None = None


class ModelRunner:

    def __init__(self, config: Config, rank: int, event: Event | list[Event]):
        self.config = config
        hf_config = config.hf_config
        self.block_size = config.kvcache_block_size
        self.enforce_eager = config.enforce_eager
        self.world_size = config.tensor_parallel_size
        self.rank = rank
        self.event = event

        dist.init_process_group("nccl", "tcp://localhost:2333", world_size=self.world_size, rank=rank)
        torch.cuda.set_device(rank)
        default_dtype = torch.get_default_dtype()
        torch.set_default_dtype(hf_config.dtype)
        torch.set_default_device("cuda")
        self.model = Qwen3ForCausalLM(hf_config)
        load_model(self.model, config.model)
        self.draft_model = self.load_draft_model()
        self.speculative_gamma = config.speculative_gamma
        self.speculative_accept_mode = config.speculative_accept_mode
        self.speculative_batched_draft = config.speculative_batched_draft
        self.speculative_trace = config.speculative_trace
        self.draft_kv_cache = {}
        self._prefill_aux_chunks = {}
        self._prev_correction = {}
        self.sampler = Sampler()
        self.warmup_model()
        # 真正跑一次模型，把模型执行的峰值显存测出来，计算给 KV cache 留多少空间
        self.allocate_kv_cache()
        if not self.enforce_eager:
            self.capture_cudagraph()
        torch.set_default_device("cpu")
        torch.set_default_dtype(default_dtype)

        if self.world_size > 1:
            if rank == 0:
                self.shm = SharedMemory(name="thrustlm", create=True, size=2**20)
                dist.barrier()
            else:
                dist.barrier()
                self.shm = SharedMemory(name="thrustlm")
                self.loop()

    def exit(self):
        if self.world_size > 1:
            self.shm.close()
            dist.barrier()
            if self.rank == 0:
                self.shm.unlink()
        if not self.enforce_eager:
            del self.graphs, self.graph_pool
        torch.cuda.synchronize()
        dist.destroy_process_group()

    def loop(self):
        while True:
            method_name, args = self.read_shm()
            self.call(method_name, *args)
            if method_name == "exit":
                break

    def read_shm(self):
        assert self.world_size > 1 and self.rank > 0
        self.event.wait()
        n = int.from_bytes(self.shm.buf[0:4], "little")
        method_name, *args = pickle.loads(self.shm.buf[4:n+4])
        self.event.clear()
        return method_name, args

    def write_shm(self, method_name, *args):
        assert self.world_size > 1 and self.rank == 0
        data = pickle.dumps([method_name, *args])
        n = len(data)
        self.shm.buf[0:4] = n.to_bytes(4, "little")
        self.shm.buf[4:n+4] = data
        for event in self.event:
            event.set()

    def call(self, method_name, *args):
        if self.world_size > 1 and self.rank == 0:
            self.write_shm(method_name, *args)
        method = getattr(self, method_name, None)
        return method(*args)

    def load_draft_model(self):
        if self.config.speculative_model is None:
            return None
        draft_model = Eagle3Speculator.from_pretrained(
            self.config.speculative_model,
            target_model_path=self.config.model,
        )
        return draft_model.eval()

    @torch.inference_mode()
    def _draft_prefill(self, seq: Sequence, prompt_aux_hidden: torch.Tensor):
        if self.draft_model is None:
            return
        if not hasattr(self, "draft_kv_cache"):
            self.draft_kv_cache = {}
        if seq.num_prompt_tokens <= 1:
            return
        device = prompt_aux_hidden.device
        # EAGLE draft 输入 token x_t 应搭配 target 在上一位置预测出它的 hidden h_{t-1}。
        # prompt prefill 因此使用 x_1..x_n 和 h_0..h_{n-1}，避免同位置 hidden/token 错位。
        prompt_ids = torch.tensor(seq.prompt_token_ids[1:], dtype=torch.long, device=device).unsqueeze(0)
        positions = torch.arange(seq.num_prompt_tokens - 1, dtype=torch.long, device=device).unsqueeze(0)
        aux_hidden = prompt_aux_hidden[:-1].unsqueeze(0)
        _, _, draft_kv = self.draft_model(prompt_ids, aux_hidden, positions, past_kv=None)
        self.draft_kv_cache[seq.seq_id] = (draft_kv[0], draft_kv[1])

    @torch.inference_mode()
    def _accumulate_draft_prefill(self, seqs: list[Sequence], aux_hidden: torch.Tensor):
        if self.draft_model is None:
            return
        if not hasattr(self, "_prefill_aux_chunks"):
            self._prefill_aux_chunks = {}
        offset = 0
        for seq in seqs:
            is_decode = seq.num_scheduled_tokens < 0
            chunk = 1 if is_decode else seq.num_scheduled_tokens
            if not is_decode and seq.num_cached_tokens < seq.num_prompt_tokens:
                chunk_aux = aux_hidden[offset:offset + chunk]
                self._prefill_aux_chunks.setdefault(seq.seq_id, []).append(chunk_aux)
                if seq.num_cached_tokens + chunk >= seq.num_prompt_tokens:
                    chunks = self._prefill_aux_chunks.pop(seq.seq_id)
                    prompt_aux_hidden = torch.cat(chunks, dim=0) if len(chunks) > 1 else chunks[0]
                    self._draft_prefill(seq, prompt_aux_hidden)
            offset += chunk

    def _get_single_draft_kv(self, seq: Sequence):
        if not hasattr(self, "draft_kv_cache"):
            self.draft_kv_cache = {}
        return self.draft_kv_cache.get(seq.seq_id)

    def clear_speculative_state(self, seq_ids: list[int]):
        for seq_id in seq_ids:
            self.draft_kv_cache.pop(seq_id, None)
            self._prefill_aux_chunks.pop(seq_id, None)
            self._prev_correction.pop(seq_id, None)

    def _update_single_draft_kv(
        self,
        seq: Sequence,
        past_kv: tuple[torch.Tensor, torch.Tensor] | None,
        old_len: int,
        emitted_token_ids: list[int],
        verify_aux_hidden: torch.Tensor,
        num_accepted: int,
    ):
        if past_kv is None:
            return
        if not hasattr(self, "draft_kv_cache"):
            self.draft_kv_cache = {}

        # EAGLE 的 draft KV 只覆盖“已经喂进 draft layer 的 token”。
        # 如果 target 最后返回 correction/bonus token，这些 token 已经进入 Sequence，
        # 但还没有进入 draft KV；这里先裁掉未接受分支，再把缺失尾部补回去。
        keep = old_len + min(num_accepted + 1, self.speculative_gamma)
        keep = min(keep, past_kv[0].shape[2])
        current_kv = (
            past_kv[0][:, :, :keep, :].contiguous(),
            past_kv[1][:, :, :keep, :].contiguous(),
        )

        self.draft_kv_cache[seq.seq_id] = current_kv

    @torch.inference_mode()
    def _fill_prefill_sampled_tokens(
        self,
        seqs: list[Sequence],
        aux_hidden: torch.Tensor,
        token_ids: list[int] | None,
    ):
        if self.draft_model is None or token_ids is None:
            return
        offset = 0
        for index, seq in enumerate(seqs):
            is_decode = seq.num_scheduled_tokens < 0
            chunk = 1 if is_decode else seq.num_scheduled_tokens
            if (
                not is_decode
                and seq.num_completion_tokens == 0
                and seq.num_cached_tokens < seq.num_prompt_tokens
                and seq.num_cached_tokens + chunk >= seq.num_prompt_tokens
            ):
                draft_kv = self._get_single_draft_kv(seq)
                if draft_kv is not None:
                    device = draft_kv[0].device
                    current_len = draft_kv[0].shape[2]
                    input_ids = torch.tensor([[int(token_ids[index])]], dtype=torch.long, device=device)
                    positions = torch.tensor([[current_len]], dtype=torch.long, device=device)
                    token_aux = aux_hidden[offset + chunk - 1].to(device).view(1, 1, -1)
                    kv_valid_lens = torch.tensor([current_len], dtype=torch.long, device=device)
                    _, _, draft_kv = self.draft_model(
                        input_ids,
                        token_aux,
                        positions,
                        past_kv=draft_kv,
                        kv_valid_lens=kv_valid_lens,
                    )
                    self.draft_kv_cache[seq.seq_id] = draft_kv
            offset += chunk

    def _build_speculative_debug(
        self,
        *,
        accept_mode: str,
        start_token_id: int,
        draft_token_ids: list[int],
        target_logits: torch.Tensor,
        sample_token_ids: list[int],
        num_accepted: int,
        accepted_all: bool,
        draft_kv_len: int,
    ) -> dict:
        target_argmax = target_logits.argmax(dim=-1).tolist()
        matches = [
            int(target_argmax[i]) == int(draft_token_ids[i])
            for i in range(len(draft_token_ids))
        ]
        logits = target_logits.float()
        ranks = []
        target_logits_for_draft_tokens = []
        for i, token_id in enumerate(draft_token_ids):
            if token_id < 0 or token_id >= logits.size(1):
                ranks.append(None)
                target_logits_for_draft_tokens.append(None)
                continue
            token_logit = logits[i, token_id]
            ranks.append(int((logits[i] > token_logit).sum().item()) + 1)
            target_logits_for_draft_tokens.append(float(token_logit.item()))
        top_k = min(5, logits.size(1))
        target_top = torch.topk(logits, k=top_k, dim=-1)
        return {
            "accept_mode": accept_mode,
            "start_token_id": int(start_token_id),
            "draft_token_ids": [int(x) for x in draft_token_ids],
            "target_argmax_token_ids": [int(x) for x in target_argmax],
            "matches": matches,
            "num_accepted": int(num_accepted),
            "accepted_all": bool(accepted_all),
            "emitted_token_ids": [int(x) for x in sample_token_ids],
            "draft_kv_len_before": int(draft_kv_len),
            "draft_token_target_ranks": ranks,
            "draft_token_target_logits": target_logits_for_draft_tokens,
            "target_top_token_ids": target_top.indices.tolist(),
            "target_top_logits": target_top.values.tolist(),
        }

    @torch.inference_mode()
    def run_target_decode_with_eagle3_aux(self, seqs: list[Sequence]) -> TargetDecodeAuxOutput:
        assert seqs
        input_ids, positions = self.prepare_decode(seqs)
        temperatures = self.prepare_sample(seqs)
        hidden_states, aux_hidden = self.model.forward_with_eagle3_aux(input_ids, positions)
        logits = self.model.compute_logits(hidden_states)
        token_ids = self.sampler(logits, temperatures).tolist() if getattr(self, "rank", 0) == 0 else None
        return TargetDecodeAuxOutput(token_ids, logits, aux_hidden, positions)

    def _build_target_verify_batch_metadata(
        self,
        seqs: list[Sequence],
        start_token_ids: list[int],
        draft_token_ids: list[list[int]],
        base_offsets: list[int],
    ) -> dict:
        if not (len(seqs) == len(start_token_ids) == len(draft_token_ids) == len(base_offsets)):
            raise ValueError("speculative verify batch metadata has inconsistent sizes")
        if not seqs:
            raise ValueError("speculative verify batch cannot be empty")

        input_ids = []
        positions = []
        slot_mapping = []
        cu_seqlens_q = [0]
        cu_seqlens_k = [0]
        verify_lengths = []
        max_seqlen_q = 0
        max_seqlen_k = 0

        for seq, start_token_id, drafts, base_offset in zip(
            seqs,
            start_token_ids,
            draft_token_ids,
            base_offsets,
        ):
            verify_tokens = [start_token_id] + drafts
            num_verify = len(verify_tokens)
            base_pos = len(seq) + base_offset
            if base_pos < 0:
                raise ValueError(f"invalid speculative verify base_pos: {base_pos}")

            input_ids.extend(verify_tokens)
            positions.extend(range(base_pos, base_pos + num_verify))
            verify_lengths.append(num_verify)
            cu_seqlens_q.append(cu_seqlens_q[-1] + num_verify)
            seqlen_k = base_pos + num_verify
            cu_seqlens_k.append(cu_seqlens_k[-1] + seqlen_k)
            max_seqlen_q = max(max_seqlen_q, num_verify)
            max_seqlen_k = max(max_seqlen_k, seqlen_k)

            for absolute_pos in range(base_pos, base_pos + num_verify):
                block_idx = absolute_pos // self.block_size
                offset = absolute_pos % self.block_size
                slot_mapping.append(seq.block_table[block_idx] * self.block_size + offset)

        return {
            "input_ids": input_ids,
            "positions": positions,
            "slot_mapping": slot_mapping,
            "cu_seqlens_q": cu_seqlens_q,
            "cu_seqlens_k": cu_seqlens_k,
            "max_seqlen_q": max_seqlen_q,
            "max_seqlen_k": max_seqlen_k,
            "verify_lengths": verify_lengths,
        }

    @torch.inference_mode()
    def run_target_verify_batch_with_eagle3_aux(
        self,
        seqs: list[Sequence],
        start_token_ids: list[int],
        draft_token_ids: list[list[int]],
        base_offsets: list[int],
    ) -> list[Eagle3TargetVerifyOutput]:
        metadata = self._build_target_verify_batch_metadata(
            seqs,
            start_token_ids,
            draft_token_ids,
            base_offsets,
        )
        input_ids = torch.tensor(metadata["input_ids"], dtype=torch.int64, pin_memory=True).cuda(non_blocking=True)
        positions = torch.tensor(metadata["positions"], dtype=torch.int64, pin_memory=True).cuda(non_blocking=True)
        slot_mapping = torch.tensor(metadata["slot_mapping"], dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
        cu_seqlens_q = torch.tensor(metadata["cu_seqlens_q"], dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
        cu_seqlens_k = torch.tensor(metadata["cu_seqlens_k"], dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
        block_tables = self.prepare_block_tables(seqs)

        set_context(
            True,
            cu_seqlens_q=cu_seqlens_q,
            cu_seqlens_k=cu_seqlens_k,
            max_seqlen_q=metadata["max_seqlen_q"],
            max_seqlen_k=metadata["max_seqlen_k"],
            slot_mapping=slot_mapping,
            block_tables=block_tables,
        )
        try:
            hidden_states, aux_hidden = self.model.forward_with_eagle3_aux(input_ids, positions)
            logits = self.model.compute_logits(hidden_states, all_tokens=True)
        finally:
            reset_context()

        logits_by_seq = torch.split(logits, metadata["verify_lengths"], dim=0)
        aux_by_seq = torch.split(aux_hidden, metadata["verify_lengths"], dim=0)
        return [
            Eagle3TargetVerifyOutput(seq_logits, seq_aux)
            for seq_logits, seq_aux in zip(logits_by_seq, aux_by_seq)
        ]

    @torch.inference_mode()
    def run_target_verify_with_eagle3_aux(
        self,
        seq: Sequence,
        start_token_id: int,
        draft_token_ids: list[int],
        base_offset: int = 0,
    ) -> Eagle3TargetVerifyOutput:
        return self.run_target_verify_batch_with_eagle3_aux(
            [seq],
            [start_token_id],
            [draft_token_ids],
            [base_offset],
        )[0]

    @torch.inference_mode()
    def run_speculative_single(self, seq: Sequence) -> SpeculativeDecodeOutput:
        assert self.draft_model is not None
        total_start = perf_counter()
        target_decode_time = 0.0
        trace_time = 0.0
        prev_correction = getattr(self, "_prev_correction", {})
        merged = seq.seq_id in prev_correction
        if merged:
            # 上轮 correction/bonus 已经追加到 Sequence，但 target KV 里还没有对应 token。
            # 本轮复用它作为 start token，并在 verify 阶段从 len(seq)-1 覆盖该 KV 槽。
            start_token_id, start_aux_hidden = prev_correction[seq.seq_id]
        else:
            stage_start = perf_counter()
            target_decode = self.run_target_decode_with_eagle3_aux([seq])
            reset_context()
            target_decode_time = perf_counter() - stage_start
            start_token_id = target_decode.token_ids[0]
            start_aux_hidden = target_decode.aux_hidden.view(1, 1, -1)
        temperature = seq.temperature
        draft_past_kv = self._get_single_draft_kv(seq)
        if draft_past_kv is not None:
            draft_kv_len = draft_past_kv[0].shape[2]
        elif merged:
            draft_kv_len = len(seq)
        else:
            draft_kv_len = int(target_decode.positions[-1].item()) + 1
        start_position = draft_kv_len
        accept_mode = getattr(self, "speculative_accept_mode", "greedy")
        stage_start = perf_counter()
        draft_sequence = generate_eagle3_draft_tokens(
            self.draft_model,
            start_token_id=start_token_id,
            start_aux_hidden=start_aux_hidden.view(1, 1, -1),
            start_position=start_position,
            gamma=self.speculative_gamma,
            temperature=temperature,
            past_kv=draft_past_kv,
            kv_valid_len=draft_kv_len if draft_past_kv is not None else None,
            draft_sampling_mode="greedy" if accept_mode == "greedy" else "sample",
        )
        draft_proposal_time = perf_counter() - stage_start
        stage_start = perf_counter()
        verify_output = self.run_target_verify_with_eagle3_aux(
            seq,
            start_token_id,
            draft_sequence.draft_token_ids,
            base_offset=-1 if merged else 0,
        )
        target_verify_time = perf_counter() - stage_start
        stage_start = perf_counter()
        draft_token_ids = torch.tensor(
            draft_sequence.draft_token_ids,
            dtype=torch.long,
            device=verify_output.target_logits.device,
        )
        if accept_mode == "greedy":
            sample_result = speculative_accept_greedy_from_logits(
                verify_output.target_logits,
                draft_token_ids,
            )
        elif accept_mode == "rejection":
            sample_result = speculative_accept_reject_from_logits(
                verify_output.target_logits,
                draft_sequence.draft_target_logits,
                draft_token_ids,
                temperature=temperature,
            )
        else:
            raise ValueError(f"unsupported speculative_accept_mode: {accept_mode}")
        reset_context()
        token_ids = ([] if merged else [start_token_id]) + sample_result.token_ids
        if not hasattr(self, "_prev_correction"):
            self._prev_correction = {}
        correction_aux = verify_output.target_aux_hidden[sample_result.num_accepted]
        self._prev_correction[seq.seq_id] = (
            sample_result.final_token_id,
            correction_aux.detach(),
        )
        accept_time = perf_counter() - stage_start
        debug = None
        if getattr(self, "speculative_trace", False):
            stage_start = perf_counter()
            debug = self._build_speculative_debug(
                accept_mode=accept_mode,
                start_token_id=start_token_id,
                draft_token_ids=draft_sequence.draft_token_ids,
                target_logits=verify_output.target_logits,
                sample_token_ids=token_ids,
                num_accepted=sample_result.num_accepted,
                accepted_all=sample_result.accepted_all,
                draft_kv_len=draft_kv_len,
            )
            trace_time = perf_counter() - stage_start
        stage_start = perf_counter()
        self._update_single_draft_kv(
            seq,
            draft_sequence.past_kv,
            draft_kv_len,
            token_ids,
            verify_output.target_aux_hidden,
            sample_result.num_accepted,
        )
        kv_update_time = perf_counter() - stage_start
        timing = {
            "target_decode_time": target_decode_time,
            "draft_proposal_time": draft_proposal_time,
            "target_verify_time": target_verify_time,
            "accept_time": accept_time,
            "kv_update_time": kv_update_time,
            "trace_time": trace_time,
            "total_time": perf_counter() - total_start,
        }
        return SpeculativeDecodeOutput(
            token_ids=token_ids,
            num_draft_tokens=len(draft_sequence.draft_token_ids),
            num_accepted=sample_result.num_accepted,
            accepted_all=sample_result.accepted_all,
            emitted_tokens=len(token_ids),
            timing=timing,
            debug=debug,
        )

    @torch.inference_mode()
    def _generate_speculative_draft_sequences(self, states: list[dict], accept_mode: str):
        seqs = [state["seq"] for state in states]
        use_batched_draft = (
            accept_mode == "greedy"
            and getattr(self, "speculative_batched_draft", True)
        )
        if use_batched_draft:
            stage_start = perf_counter()
            draft_sequences = generate_eagle3_draft_tokens_batched(
                self.draft_model,
                start_token_ids=[state["start_token_id"] for state in states],
                start_aux_hidden=torch.cat([state["start_aux_hidden"] for state in states], dim=0),
                start_positions=[state["draft_kv_len"] for state in states],
                gamma=self.speculative_gamma,
                temperature=seqs[0].temperature,
                past_kv=[state["draft_past_kv"] for state in states],
                draft_sampling_mode="greedy",
            )
            proposal_time_share = (perf_counter() - stage_start) / len(seqs)
            for state, draft_sequence in zip(states, draft_sequences):
                state["draft_sequence"] = draft_sequence
                state["draft_proposal_time"] = proposal_time_share
            return

        for state in states:
            seq = state["seq"]
            stage_start = perf_counter()
            state["draft_sequence"] = generate_eagle3_draft_tokens(
                self.draft_model,
                start_token_id=state["start_token_id"],
                start_aux_hidden=state["start_aux_hidden"],
                start_position=state["draft_kv_len"],
                gamma=self.speculative_gamma,
                temperature=seq.temperature,
                past_kv=state["draft_past_kv"],
                kv_valid_len=(
                    state["draft_kv_len"]
                    if state["draft_past_kv"] is not None else None
                ),
                draft_sampling_mode="greedy" if accept_mode == "greedy" else "sample",
            )
            state["draft_proposal_time"] = perf_counter() - stage_start

    @torch.inference_mode()
    def run_speculative_batch(self, seqs: list[Sequence]) -> list[SpeculativeDecodeOutput]:
        """合批生成 draft，再把所有候选打包成一次 target verify。"""
        assert self.draft_model is not None
        if not seqs:
            return []
        if len(seqs) == 1:
            return [self.run_speculative_single(seqs[0])]

        total_start = perf_counter()
        if not hasattr(self, "_prev_correction"):
            self._prev_correction = {}
        prev_correction = getattr(self, "_prev_correction", {})
        merged_by_seq = {seq.seq_id: seq.seq_id in prev_correction for seq in seqs}
        decode_seqs = [seq for seq in seqs if not merged_by_seq[seq.seq_id]]
        target_decode_time = 0.0
        decoded_starts = {}

        if decode_seqs:
            stage_start = perf_counter()
            target_decode = self.run_target_decode_with_eagle3_aux(decode_seqs)
            reset_context()
            target_decode_time = perf_counter() - stage_start
            for index, seq in enumerate(decode_seqs):
                decoded_starts[seq.seq_id] = (
                    target_decode.token_ids[index],
                    target_decode.aux_hidden[index].view(1, 1, -1),
                    int(target_decode.positions[index].item()),
                )

        states = []
        accept_mode = getattr(self, "speculative_accept_mode", "greedy")
        for seq in seqs:
            merged = merged_by_seq[seq.seq_id]
            if merged:
                start_token_id, start_aux_hidden = prev_correction[seq.seq_id]
                start_aux_hidden = start_aux_hidden.view(1, 1, -1)
                decoded_position = None
            else:
                start_token_id, start_aux_hidden, decoded_position = decoded_starts[seq.seq_id]

            draft_past_kv = self._get_single_draft_kv(seq)
            if draft_past_kv is not None:
                draft_kv_len = draft_past_kv[0].shape[2]
            elif merged:
                draft_kv_len = len(seq)
            else:
                draft_kv_len = decoded_position + 1

            states.append({
                "seq": seq,
                "merged": merged,
                "start_token_id": start_token_id,
                "start_aux_hidden": start_aux_hidden,
                "draft_kv_len": draft_kv_len,
                "draft_past_kv": draft_past_kv,
            })

        self._generate_speculative_draft_sequences(states, accept_mode)

        stage_start = perf_counter()
        verify_outputs = self.run_target_verify_batch_with_eagle3_aux(
            seqs,
            [state["start_token_id"] for state in states],
            [state["draft_sequence"].draft_token_ids for state in states],
            [-1 if state["merged"] else 0 for state in states],
        )
        target_verify_time = perf_counter() - stage_start
        target_decode_share = target_decode_time / len(decode_seqs) if decode_seqs else 0.0
        target_verify_share = target_verify_time / len(seqs)

        outputs = []
        for state, verify_output in zip(states, verify_outputs):
            seq = state["seq"]
            draft_sequence = state["draft_sequence"]
            stage_start = perf_counter()
            draft_ids = torch.tensor(
                draft_sequence.draft_token_ids,
                dtype=torch.long,
                device=verify_output.target_logits.device,
            )
            if accept_mode == "greedy":
                sample_result = speculative_accept_greedy_from_logits(
                    verify_output.target_logits,
                    draft_ids,
                )
            elif accept_mode == "rejection":
                sample_result = speculative_accept_reject_from_logits(
                    verify_output.target_logits,
                    draft_sequence.draft_target_logits,
                    draft_ids,
                    temperature=seq.temperature,
                )
            else:
                raise ValueError(f"unsupported speculative_accept_mode: {accept_mode}")

            token_ids = ([] if state["merged"] else [state["start_token_id"]]) + sample_result.token_ids
            correction_aux = verify_output.target_aux_hidden[sample_result.num_accepted]
            self._prev_correction[seq.seq_id] = (
                sample_result.final_token_id,
                correction_aux.detach(),
            )
            accept_time = perf_counter() - stage_start

            trace_time = 0.0
            debug = None
            if getattr(self, "speculative_trace", False):
                stage_start = perf_counter()
                debug = self._build_speculative_debug(
                    accept_mode=accept_mode,
                    start_token_id=state["start_token_id"],
                    draft_token_ids=draft_sequence.draft_token_ids,
                    target_logits=verify_output.target_logits,
                    sample_token_ids=token_ids,
                    num_accepted=sample_result.num_accepted,
                    accepted_all=sample_result.accepted_all,
                    draft_kv_len=state["draft_kv_len"],
                )
                trace_time = perf_counter() - stage_start

            stage_start = perf_counter()
            self._update_single_draft_kv(
                seq,
                draft_sequence.past_kv,
                state["draft_kv_len"],
                token_ids,
                verify_output.target_aux_hidden,
                sample_result.num_accepted,
            )
            kv_update_time = perf_counter() - stage_start
            timing = {
                "target_decode_time": target_decode_share if not state["merged"] else 0.0,
                "draft_proposal_time": state["draft_proposal_time"],
                "target_verify_time": target_verify_share,
                "accept_time": accept_time,
                "kv_update_time": kv_update_time,
                "trace_time": trace_time,
                "total_time": 0.0,
            }
            outputs.append(SpeculativeDecodeOutput(
                token_ids=token_ids,
                num_draft_tokens=len(draft_sequence.draft_token_ids),
                num_accepted=sample_result.num_accepted,
                accepted_all=sample_result.accepted_all,
                emitted_tokens=len(token_ids),
                timing=timing,
                debug=debug,
            ))

        total_time_share = (perf_counter() - total_start) / len(seqs)
        for output in outputs:
            output.timing["total_time"] = total_time_share
        return outputs

    def warmup_model(self):
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        max_num_batched_tokens, max_model_len = self.config.max_num_batched_tokens, self.config.max_model_len
        seq_len = min(max_num_batched_tokens, max_model_len)
        num_seqs = min(max_num_batched_tokens // seq_len, self.config.max_num_seqs)
        seqs = [Sequence([0] * seq_len) for _ in range(num_seqs)]
        for seq in seqs:
            seq.num_scheduled_tokens = seq_len
        self.run(seqs, True)
        self.draft_kv_cache.clear()
        self._prefill_aux_chunks.clear()
        self._prev_correction.clear()
        torch.cuda.empty_cache()

    # 把预热用过的临时显存清掉
    # 但峰值统计已经保留下来，后面分配 KV cache 会用

    def allocate_kv_cache(self):
        config = self.config
        hf_config = config.hf_config
        free, total = torch.cuda.mem_get_info()
        used = total - free
        peak = torch.cuda.memory_stats()["allocated_bytes.all.peak"]
        current = torch.cuda.memory_stats()["allocated_bytes.all.current"]
        num_kv_heads = hf_config.num_key_value_heads // self.world_size
        head_dim = getattr(hf_config, "head_dim", hf_config.hidden_size // hf_config.num_attention_heads)
        block_bytes = 2 * hf_config.num_hidden_layers * self.block_size * num_kv_heads * head_dim * hf_config.dtype.itemsize
        config.num_kvcache_blocks = int(total * config.gpu_memory_utilization - used - peak + current) // block_bytes
        assert config.num_kvcache_blocks > 0
        self.kv_cache = torch.empty(2, hf_config.num_hidden_layers, config.num_kvcache_blocks, self.block_size, num_kv_heads, head_dim)
        layer_id = 0
        for module in self.model.modules():
            if hasattr(module, "k_cache") and hasattr(module, "v_cache"):
                module.k_cache = self.kv_cache[0, layer_id]
                module.v_cache = self.kv_cache[1, layer_id]
                layer_id += 1

    def prepare_block_tables(self, seqs: list[Sequence]):
        max_len = max(len(seq.block_table) for seq in seqs)
        block_tables = [seq.block_table + [-1] * (max_len - len(seq.block_table)) for seq in seqs]
        block_tables = torch.tensor(block_tables, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
        return block_tables

    def prepare_prefill(self, seqs: list[Sequence]):
        input_ids = []
        positions = []
        cu_seqlens_q = [0]
        cu_seqlens_k = [0]
        max_seqlen_q = 0
        max_seqlen_k = 0
        slot_mapping = []
        block_tables = None
        for seq in seqs:
            # ===== 2026-06-07 chunked prefill =====
            # mixed batch 里 decode 请求按 1-token prefill 处理，
            # 这样 attention 可以走 flash_attn_varlen_func + block_table。
            is_decode = seq.num_scheduled_tokens < 0
            seqlen = len(seq)
            start = seqlen - 1 if is_decode else min(seq.num_cached_tokens, seqlen - 1)
            seqlen_q = 1 if is_decode else seq.num_scheduled_tokens
            end = start + seqlen_q
            seqlen_k = end
            # ===== 2026-06-07 chunked prefill =====
            input_ids.extend(seq[start:end])
            positions.extend(range(start, end))
            cu_seqlens_q.append(cu_seqlens_q[-1] + seqlen_q)
            cu_seqlens_k.append(cu_seqlens_k[-1] + seqlen_k)
            max_seqlen_q = max(seqlen_q, max_seqlen_q)
            max_seqlen_k = max(seqlen_k, max_seqlen_k)
            if not seq.block_table:    # warmup
                continue
            start_block = start // self.block_size
            end_block = (end + self.block_size - 1) // self.block_size
            for i in range(start_block, end_block):
                slot_start = seq.block_table[i] * self.block_size
                if i == start_block:
                    slot_start += start % self.block_size
                if i != end_block - 1:
                    slot_end = seq.block_table[i] * self.block_size + self.block_size
                else:
                    slot_end = seq.block_table[i] * self.block_size + end - i * self.block_size
                slot_mapping.extend(range(slot_start, slot_end))
        if cu_seqlens_k[-1] > cu_seqlens_q[-1]:    # prefix cache
            block_tables = self.prepare_block_tables(seqs)
        input_ids = torch.tensor(input_ids, dtype=torch.int64, pin_memory=True).cuda(non_blocking=True)
        positions = torch.tensor(positions, dtype=torch.int64, pin_memory=True).cuda(non_blocking=True)
        cu_seqlens_q = torch.tensor(cu_seqlens_q, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
        cu_seqlens_k = torch.tensor(cu_seqlens_k, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
        slot_mapping = torch.tensor(slot_mapping, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
        set_context(True, cu_seqlens_q, cu_seqlens_k, max_seqlen_q, max_seqlen_k, slot_mapping, None, block_tables)
        return input_ids, positions

    def prepare_decode(self, seqs: list[Sequence]):
        input_ids = []
        positions = []
        slot_mapping = []
        context_lens = []
        for seq in seqs:
            input_ids.append(seq.last_token)
            positions.append(len(seq) - 1)
            context_lens.append(len(seq))
            slot_mapping.append(seq.block_table[-1] * self.block_size + seq.last_block_num_tokens  - 1)
        input_ids = torch.tensor(input_ids, dtype=torch.int64, pin_memory=True).cuda(non_blocking=True)
        positions = torch.tensor(positions, dtype=torch.int64, pin_memory=True).cuda(non_blocking=True)
        slot_mapping = torch.tensor(slot_mapping, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
        context_lens = torch.tensor(context_lens, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
        block_tables = self.prepare_block_tables(seqs)
        set_context(False, slot_mapping=slot_mapping, context_lens=context_lens, block_tables=block_tables)
        return input_ids, positions

    def prepare_sample(self, seqs: list[Sequence]):
        temperatures = [seq.temperature for seq in seqs]
        temperatures = torch.tensor(temperatures, dtype=torch.float32, pin_memory=True).cuda(non_blocking=True)
        return temperatures

    @torch.inference_mode()
    def run_model(self, input_ids: torch.Tensor, positions: torch.Tensor, is_prefill: bool):
        if is_prefill or self.enforce_eager or input_ids.size(0) > 512:
            return self.model.compute_logits(self.model(input_ids, positions))
        else:
            bs = input_ids.size(0)
            context = get_context()
            graph = self.graphs[next(x for x in self.graph_bs if x >= bs)]
            graph_vars = self.graph_vars
            graph_vars["input_ids"][:bs] = input_ids
            graph_vars["positions"][:bs] = positions
            graph_vars["slot_mapping"].fill_(-1)
            graph_vars["slot_mapping"][:bs] = context.slot_mapping
            graph_vars["context_lens"].zero_()
            graph_vars["context_lens"][:bs] = context.context_lens
            graph_vars["block_tables"][:bs, :context.block_tables.size(1)] = context.block_tables
            graph.replay()
            return self.model.compute_logits(graph_vars["outputs"][:bs])

    @torch.inference_mode()
    def run(self, seqs: list[Sequence], is_prefill: bool) -> list[int]:
        input_ids, positions = self.prepare_prefill(seqs) if is_prefill else self.prepare_decode(seqs)
        temperatures = self.prepare_sample(seqs) if self.rank == 0 else None
        if is_prefill and self.draft_model is not None:
            hidden_states, aux_hidden = self.model.forward_with_eagle3_aux(input_ids, positions)
            logits = self.model.compute_logits(hidden_states)
            self._accumulate_draft_prefill(seqs, aux_hidden)
        else:
            logits = self.run_model(input_ids, positions, is_prefill)
        token_ids = self.sampler(logits, temperatures).tolist() if self.rank == 0 else None
        if is_prefill and self.draft_model is not None:
            self._fill_prefill_sampled_tokens(seqs, aux_hidden, token_ids)
        reset_context()
        return token_ids

    @torch.inference_mode()
    def capture_cudagraph(self):
        config = self.config
        hf_config = config.hf_config
        max_bs = min(self.config.max_num_seqs, 512)
        max_num_blocks = (config.max_model_len + self.block_size - 1) // self.block_size
        input_ids = torch.zeros(max_bs, dtype=torch.int64)
        positions = torch.zeros(max_bs, dtype=torch.int64)
        slot_mapping = torch.zeros(max_bs, dtype=torch.int32)
        context_lens = torch.zeros(max_bs, dtype=torch.int32)
        block_tables = torch.zeros(max_bs, max_num_blocks, dtype=torch.int32)
        outputs = torch.zeros(max_bs, hf_config.hidden_size)
        self.graph_bs = [1, 2, 4, 8] + list(range(16, max_bs + 1, 16))
        self.graphs = {}
        self.graph_pool = None

        for bs in reversed(self.graph_bs):
            graph = torch.cuda.CUDAGraph()
            set_context(False, slot_mapping=slot_mapping[:bs], context_lens=context_lens[:bs], block_tables=block_tables[:bs])
            outputs[:bs] = self.model(input_ids[:bs], positions[:bs])    # warmup
            with torch.cuda.graph(graph, self.graph_pool):
                outputs[:bs] = self.model(input_ids[:bs], positions[:bs])    # capture
            if self.graph_pool is None:
                self.graph_pool = graph.pool()
            self.graphs[bs] = graph
            torch.cuda.synchronize()
            reset_context()

        self.graph_vars = dict(
            input_ids=input_ids,
            positions=positions,
            slot_mapping=slot_mapping,
            context_lens=context_lens,
            block_tables=block_tables,
            outputs=outputs,
        )
