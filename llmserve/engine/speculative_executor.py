import torch
from time import perf_counter

from llmserve.engine.sequence import Sequence
from llmserve.models.eagle3 import Eagle3Speculator
from llmserve.speculative.draft import (
    generate_eagle3_draft_tokens,
    generate_eagle3_draft_tokens_batched,
)
from llmserve.speculative.sampling import (
    speculative_accept_greedy_from_logits,
    speculative_accept_reject_from_logits,
)
from llmserve.speculative.types import Eagle3TargetVerifyOutput
from llmserve.speculative.types import SpeculativeDecodeOutput, TargetDecodeAuxOutput
from llmserve.utils.context import set_context, reset_context
from llmserve.utils.stage_profiler import StageProfiler


class SpeculativeExecutor:

    draft_stage_timing_names = (
        "draft_pack_time",
        "draft_forward_time",
        "draft_sample_time",
        "draft_compact_time",
    )

    def __init__(self, host):
        object.__setattr__(self, "host", host)

    def __getattribute__(self, name):
        if name not in {"host", "__dict__", "__class__"}:
            host = object.__getattribute__(self, "host")
            # Preserve instance-level hooks used by the scheduler and focused tests.
            override = vars(host).get(name)
            if callable(override):
                return override
        return object.__getattribute__(self, name)

    def __getattr__(self, name):
        return getattr(self.host, name)

    def __setattr__(self, name, value):
        setattr(self.host, name, value)

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
    def _accumulate_draft_prefill(
        self,
        seqs: list[Sequence],
        aux_hidden: torch.Tensor,
        decode_seqs: list[Sequence] | None = None,
    ):
        if self.draft_model is None:
            return
        if not hasattr(self, "_prefill_aux_chunks"):
            self._prefill_aux_chunks = {}
        decode_seq_ids = {seq.seq_id for seq in (decode_seqs or [])}
        offset = 0
        for seq in seqs:
            is_decode = seq.seq_id in decode_seq_ids
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

    @classmethod
    def _draft_stage_timing(cls, draft_sequence) -> dict[str, float]:
        proposal_timing = draft_sequence.proposal_timing or {}
        return {
            name: float(proposal_timing.get(name, 0.0))
            for name in cls.draft_stage_timing_names
        }

    def _new_stage_profiler(self, *stage_names: str) -> StageProfiler:
        if torch.cuda.is_available():
            device = torch.device("cuda", getattr(self, "rank", torch.cuda.current_device()))
        else:
            device = torch.device("cpu")
        return StageProfiler(device, stage_names)

    def _update_single_draft_kv(
        self,
        seq: Sequence,
        past_kv: tuple[torch.Tensor, torch.Tensor] | None,
        old_len: int,
        emitted_token_ids: list[int],
        verify_aux_hidden: torch.Tensor,
        num_accepted: int,
        gamma: int,
    ):
        if past_kv is None:
            return
        if not hasattr(self, "draft_kv_cache"):
            self.draft_kv_cache = {}

        # EAGLE 的 draft KV 只覆盖“已经喂进 draft layer 的 token”。
        # 如果 target 最后返回 correction/bonus token，这些 token 已经进入 Sequence，
        # 但还没有进入 draft KV；这里先裁掉未接受分支，再把缺失尾部补回去。
        keep = old_len + min(num_accepted + 1, gamma)
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
        decode_seqs: list[Sequence] | None = None,
    ):
        if self.draft_model is None or token_ids is None:
            return
        decode_seq_ids = {seq.seq_id for seq in (decode_seqs or [])}
        offset = 0
        for index, seq in enumerate(seqs):
            is_decode = seq.seq_id in decode_seq_ids
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
        block_tables = self.prepare_block_tables(seqs)

        graph_backend = getattr(self, "_target_verify_graph_backend", None)
        graph_output = (
            graph_backend.run_if_supported(metadata, block_tables)
            if graph_backend is not None
            else None
        )
        if graph_output is not None:
            logits, aux_hidden = graph_output
            logits_by_seq = torch.split(logits, metadata["verify_lengths"], dim=0)
            aux_by_seq = torch.split(aux_hidden, metadata["verify_lengths"], dim=0)
            return [
                Eagle3TargetVerifyOutput(seq_logits, seq_aux)
                for seq_logits, seq_aux in zip(logits_by_seq, aux_by_seq)
            ]

        input_ids = torch.tensor(metadata["input_ids"], dtype=torch.int64, pin_memory=True).cuda(non_blocking=True)
        positions = torch.tensor(metadata["positions"], dtype=torch.int64, pin_memory=True).cuda(non_blocking=True)
        slot_mapping = torch.tensor(metadata["slot_mapping"], dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
        cu_seqlens_q = torch.tensor(metadata["cu_seqlens_q"], dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
        cu_seqlens_k = torch.tensor(metadata["cu_seqlens_k"], dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)

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
        profiler = self._new_stage_profiler("target_decode", "target_verify", "kv_update")
        trace_time = 0.0
        prev_correction = getattr(self, "_prev_correction", {})
        merged = seq.seq_id in prev_correction
        if merged:
            # 上轮 correction/bonus 已经追加到 Sequence，但 target KV 里还没有对应 token。
            # 本轮复用它作为 start token，并在 verify 阶段从 len(seq)-1 覆盖该 KV 槽。
            start_token_id, start_aux_hidden = prev_correction[seq.seq_id]
        else:
            with profiler.stage("target_decode"):
                target_decode = self.run_target_decode_with_eagle3_aux([seq])
            reset_context()
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
        gamma = self.speculative_gamma
        accept_mode = getattr(self, "speculative_accept_mode", "greedy")
        stage_start = perf_counter()
        draft_sequence = generate_eagle3_draft_tokens(
            self.draft_model,
            start_token_id=start_token_id,
            start_aux_hidden=start_aux_hidden.view(1, 1, -1),
            start_position=start_position,
            gamma=gamma,
            temperature=temperature,
            past_kv=draft_past_kv,
            kv_valid_len=draft_kv_len if draft_past_kv is not None else None,
            draft_sampling_mode="greedy" if accept_mode == "greedy" else "sample",
        )
        draft_proposal_time = perf_counter() - stage_start
        with profiler.stage("target_verify"):
            verify_output = self.run_target_verify_with_eagle3_aux(
                seq,
                start_token_id,
                draft_sequence.draft_token_ids,
                base_offset=-1 if merged else 0,
            )
        profiler.finish()
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
        with profiler.stage("kv_update"):
            self._update_single_draft_kv(
                seq,
                draft_sequence.past_kv,
                draft_kv_len,
                token_ids,
                verify_output.target_aux_hidden,
                sample_result.num_accepted,
                gamma,
            )
        stage_timings = profiler.finish()
        timing = {
            "target_decode_time": stage_timings["target_decode"].device_seconds,
            "target_decode_host_time": stage_timings["target_decode"].host_seconds,
            "draft_proposal_time": draft_proposal_time,
            "target_verify_time": stage_timings["target_verify"].device_seconds,
            "target_verify_host_time": stage_timings["target_verify"].host_seconds,
            "accept_time": accept_time,
            "kv_update_time": stage_timings["kv_update"].device_seconds,
            "kv_update_host_time": stage_timings["kv_update"].host_seconds,
            "trace_time": trace_time,
            "total_time": perf_counter() - total_start,
        }
        timing.update(self._draft_stage_timing(draft_sequence))
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
        use_batched_draft = accept_mode == "greedy"
        if use_batched_draft:
            gammas = [state.get("gamma", self.speculative_gamma) for state in states]
            stage_start = perf_counter()
            draft_sequences = generate_eagle3_draft_tokens_batched(
                self.draft_model,
                start_token_ids=[state["start_token_id"] for state in states],
                start_aux_hidden=torch.cat([state["start_aux_hidden"] for state in states], dim=0),
                start_positions=[state["draft_kv_len"] for state in states],
                gamma=max(gammas),
                temperature=seqs[0].temperature,
                past_kv=[state["draft_past_kv"] for state in states],
                draft_sampling_mode="greedy",
                gammas=gammas,
            )
            proposal_time_share = (perf_counter() - stage_start) / len(seqs)
            for state, draft_sequence in zip(states, draft_sequences):
                state["draft_sequence"] = draft_sequence
                state["draft_proposal_time"] = proposal_time_share
            return

        for state in states:
            seq = state["seq"]
            gamma = state.get("gamma", self.speculative_gamma)
            stage_start = perf_counter()
            state["draft_sequence"] = generate_eagle3_draft_tokens(
                self.draft_model,
                start_token_id=state["start_token_id"],
                start_aux_hidden=state["start_aux_hidden"],
                start_position=state["draft_kv_len"],
                gamma=gamma,
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
        profiler = self._new_stage_profiler("target_decode", "target_verify", "kv_update")
        if not hasattr(self, "_prev_correction"):
            self._prev_correction = {}
        prev_correction = getattr(self, "_prev_correction", {})
        merged_by_seq = {seq.seq_id: seq.seq_id in prev_correction for seq in seqs}
        decode_seqs = [seq for seq in seqs if not merged_by_seq[seq.seq_id]]
        decoded_starts = {}

        if decode_seqs:
            with profiler.stage("target_decode"):
                target_decode = self.run_target_decode_with_eagle3_aux(decode_seqs)
            reset_context()
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
                "gamma": self.speculative_gamma,
            })

        self._generate_speculative_draft_sequences(states, accept_mode)

        with profiler.stage("target_verify"):
            verify_outputs = self.run_target_verify_batch_with_eagle3_aux(
                seqs,
                [state["start_token_id"] for state in states],
                [state["draft_sequence"].draft_token_ids for state in states],
                [-1 if state["merged"] else 0 for state in states],
            )
        profiler.finish()

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

            with profiler.stage("kv_update"):
                self._update_single_draft_kv(
                    seq,
                    draft_sequence.past_kv,
                    state["draft_kv_len"],
                    token_ids,
                    verify_output.target_aux_hidden,
                    sample_result.num_accepted,
                    state["gamma"],
                )
            timing = {
                "draft_proposal_time": state["draft_proposal_time"],
                "accept_time": accept_time,
                "trace_time": trace_time,
                "total_time": 0.0,
            }
            timing.update(self._draft_stage_timing(draft_sequence))
            outputs.append(SpeculativeDecodeOutput(
                token_ids=token_ids,
                num_draft_tokens=len(draft_sequence.draft_token_ids),
                num_accepted=sample_result.num_accepted,
                accepted_all=sample_result.accepted_all,
                emitted_tokens=len(token_ids),
                timing=timing,
                debug=debug,
            ))

        stage_timings = profiler.finish()
        target_decode_divisor = len(decode_seqs) if decode_seqs else 1
        target_decode_time = stage_timings["target_decode"].device_seconds / target_decode_divisor
        target_decode_host_time = stage_timings["target_decode"].host_seconds / target_decode_divisor
        target_verify_time = stage_timings["target_verify"].device_seconds / len(seqs)
        target_verify_host_time = stage_timings["target_verify"].host_seconds / len(seqs)
        kv_update_time = stage_timings["kv_update"].device_seconds / len(seqs)
        kv_update_host_time = stage_timings["kv_update"].host_seconds / len(seqs)
        total_time_share = (perf_counter() - total_start) / len(seqs)
        for output, state in zip(outputs, states):
            merged = state["merged"]
            output.timing["target_decode_time"] = 0.0 if merged else target_decode_time
            output.timing["target_decode_host_time"] = 0.0 if merged else target_decode_host_time
            output.timing["target_verify_time"] = target_verify_time
            output.timing["target_verify_host_time"] = target_verify_host_time
            output.timing["kv_update_time"] = kv_update_time
            output.timing["kv_update_host_time"] = kv_update_host_time
            output.timing["total_time"] = total_time_share
        return outputs
