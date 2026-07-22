import unittest
from types import SimpleNamespace

from llmserve.engine.llm_engine import LLMEngine
from llmserve.engine.model_runner import SpeculativeDecodeOutput
from llmserve.engine.scheduler import SchedulerOutput
from llmserve.engine.sequence import Sequence, SequenceStatus


class FakeBlockManager:

    def __init__(self):
        self.calls = []
        self.batch_calls = []

    def ensure_slots(self, seq, num_tokens):
        self.calls.append((seq.seq_id, num_tokens))
        return True

    def ensure_slots_batch(self, seqs, num_tokens):
        self.batch_calls.append(([seq.seq_id for seq in seqs], num_tokens))
        return True


class FakeScheduler:

    def __init__(self, seq):
        self.seq = seq
        self.block_manager = FakeBlockManager()
        self.waiting = [object(), object()]
        self.running = [seq]

    def schedule(self):
        self.seq.num_scheduled_tokens = 1
        return SchedulerOutput([self.seq], [], [self.seq], 1)

    def postprocess_speculative(self, seq, token_ids):
        for token_id in token_ids:
            seq.append_token(token_id)
            seq.num_cached_tokens += 1
        seq.num_scheduled_tokens = 0


class FakeBatchScheduler(FakeScheduler):

    def __init__(self, seqs):
        self.seqs = seqs
        self.block_manager = FakeBlockManager()

    def schedule(self):
        for seq in self.seqs:
            seq.num_scheduled_tokens = 1
        return SchedulerOutput(self.seqs, [], self.seqs, len(self.seqs))


class FakeModelRunner:
    draft_model = object()
    enforce_eager = True
    world_size = 1
    speculative_gamma = 3

    def __init__(self):
        self.calls = []

    def call(self, method_name, *args):
        self.calls.append((method_name, args))
        if method_name == "run_speculative_single":
            return SpeculativeDecodeOutput(
                token_ids=[10, 11, 12],
                num_draft_tokens=3,
                num_accepted=2,
                accepted_all=False,
                emitted_tokens=3,
                timing={
                    "target_decode_time": 0.01,
                    "draft_proposal_time": 0.02,
                    "target_verify_time": 0.03,
                    "accept_time": 0.004,
                    "kv_update_time": 0.005,
                    "trace_time": 0.006,
                    "total_time": 0.075,
                },
                debug={"draft_token_ids": [11, 12], "matches": [True, True]},
            )
        raise AssertionError(method_name)


class FakeNormalModelRunner(FakeModelRunner):
    draft_model = None

    def call(self, method_name, *args):
        self.calls.append((method_name, args))
        if method_name == "run":
            return [10]
        raise AssertionError(method_name)


class FakeMixedScheduler:

    def __init__(self, output):
        self.output = output
        self.block_manager = FakeBlockManager()
        self.postprocess_output = None

    def schedule(self):
        return self.output

    def postprocess(self, output, token_ids):
        self.postprocess_output = output
        prefill_count = len(output.prefill_seqs)
        for seq in output.prefill_seqs:
            seq.num_cached_tokens += seq.num_scheduled_tokens
            seq.num_scheduled_tokens = 0
        for seq, token_id in zip(output.decode_seqs, token_ids[prefill_count:]):
            seq.append_token(token_id)
            seq.num_scheduled_tokens = 0


class FakeMixedModelRunner(FakeNormalModelRunner):

    def call(self, method_name, *args):
        self.calls.append((method_name, args))
        if method_name == "run":
            return [10, 20]
        raise AssertionError(method_name)


class FakeTreeModelRunner(FakeModelRunner):
    speculative_tree_nodes = 6

    def call(self, method_name, *args):
        self.calls.append((method_name, args))
        if method_name == "run_speculative_tree_single":
            return SpeculativeDecodeOutput(
                token_ids=[10, 11, 12],
                num_draft_tokens=6,
                num_accepted=2,
                accepted_all=False,
                emitted_tokens=3,
            )
        raise AssertionError(method_name)


class FakeBatchModelRunner(FakeModelRunner):

    def call(self, method_name, *args):
        self.calls.append((method_name, args))
        if method_name == "run_speculative_batch":
            return [
                SpeculativeDecodeOutput(
                    token_ids=[10, 11, 12],
                    num_draft_tokens=3,
                    num_accepted=2,
                    accepted_all=False,
                    emitted_tokens=3,
                    timing={"target_verify_time": 0.02, "total_time": 0.04},
                    debug={"draft_token_ids": [11, 12, 13], "matches": [True, True, False]},
                ),
                SpeculativeDecodeOutput(
                    token_ids=[20, 21],
                    num_draft_tokens=1,
                    num_accepted=1,
                    accepted_all=False,
                    emitted_tokens=2,
                    timing={"target_verify_time": 0.02, "total_time": 0.03},
                ),
            ]
        raise AssertionError(method_name)


class LLMEngineSpeculativeTest(unittest.TestCase):
    def test_reset_metrics_requires_idle_engine_and_clears_counters(self):
        engine = object.__new__(LLMEngine)
        engine.scheduler = SimpleNamespace(is_finished=lambda: True)
        engine.request_metrics = {1: {"success": True}}
        engine.last_step_events = {"step_end": 1.0}
        engine.speculative_batch_calls = 3
        engine.speculative_batch_sequences = 7
        engine.speculative_max_batch_size = 4

        engine.reset_metrics()

        self.assertEqual(engine.request_metrics, {})
        self.assertEqual(engine.last_step_events, {})
        self.assertEqual(engine.speculative_batch_calls, 0)
        self.assertEqual(engine.speculative_batch_sequences, 0)
        self.assertEqual(engine.speculative_max_batch_size, 0)

    @staticmethod
    def make_metric(seq):
        return {
            "seq_id": seq.seq_id,
            "arrival_time": 1.0,
            "first_token_time": None,
            "token_times": [],
            "output_event_times": [],
            "speculative_step_latencies": [],
            "finish_time": None,
            "prompt_tokens": len(seq),
            "output_tokens": 0,
            "success": False,
            "failure_reason": None,
            "speculative_steps": 0,
            "speculative_draft_tokens": 0,
            "speculative_accepted_tokens": 0,
            "speculative_emitted_tokens": 0,
            "speculative_accept_all_count": 0,
            "speculative_trace": [],
            "speculative_timing": {},
        }

    def test_step_uses_speculative_branch_for_single_eager_decode(self):
        seq = Sequence([1, 2, 3])
        seq.status = SequenceStatus.RUNNING
        seq.num_cached_tokens = len(seq)

        engine = LLMEngine.__new__(LLMEngine)
        engine.scheduler = FakeScheduler(seq)
        engine.model_runner = FakeModelRunner()
        engine.request_metrics = {
            seq.seq_id: {
                "seq_id": seq.seq_id,
                "arrival_time": 1.0,
                "first_token_time": None,
                "token_times": [],
                "output_event_times": [],
                "speculative_step_latencies": [],
                "finish_time": None,
                "prompt_tokens": len(seq),
                "output_tokens": 0,
                "success": False,
                "failure_reason": None,
                "speculative_steps": 0,
                "speculative_draft_tokens": 0,
                "speculative_accepted_tokens": 0,
                "speculative_emitted_tokens": 0,
                "speculative_accept_all_count": 0,
                "speculative_trace": [],
                "speculative_timing": {},
            }
        }

        outputs, num_tokens = LLMEngine.step(engine)

        self.assertEqual(outputs, [])
        self.assertEqual(num_tokens, 3)
        self.assertEqual(seq.completion_token_ids, [10, 11, 12])
        self.assertEqual(engine.scheduler.block_manager.calls, [(seq.seq_id, 5)])
        self.assertEqual(engine.model_runner.calls[0][0], "run_speculative_single")
        self.assertEqual(engine.request_metrics[seq.seq_id]["output_tokens"], 3)
        self.assertEqual(engine.request_metrics[seq.seq_id]["speculative_steps"], 1)
        self.assertEqual(engine.request_metrics[seq.seq_id]["speculative_draft_tokens"], 3)
        self.assertEqual(engine.request_metrics[seq.seq_id]["speculative_accepted_tokens"], 2)
        self.assertEqual(engine.request_metrics[seq.seq_id]["speculative_emitted_tokens"], 3)
        self.assertEqual(engine.request_metrics[seq.seq_id]["speculative_accept_all_count"], 0)
        self.assertEqual(engine.request_metrics[seq.seq_id]["speculative_gamma_counts"], {"3": 1})
        self.assertEqual(engine.request_metrics[seq.seq_id]["speculative_timing"]["total_time"], 0.075)
        self.assertIsNotNone(engine.request_metrics[seq.seq_id]["first_token_time"])
        self.assertEqual(len(engine.request_metrics[seq.seq_id]["token_times"]), 3)
        self.assertEqual(len(engine.request_metrics[seq.seq_id]["output_event_times"]), 1)
        self.assertEqual(len(engine.request_metrics[seq.seq_id]["speculative_step_latencies"]), 1)
        self.assertTrue(engine.last_step_events["speculative"])
        self.assertEqual(engine.last_step_events["speculative_num_draft_tokens"], 3)
        self.assertEqual(engine.last_step_events["speculative_num_accepted"], 2)
        self.assertFalse(engine.last_step_events["speculative_accepted_all"])
        self.assertEqual(engine.last_step_events["speculative_emitted_tokens"], 3)
        self.assertEqual(engine.last_step_events["speculative_debug"]["draft_token_ids"], [11, 12])

        metrics = LLMEngine.get_metrics(engine)
        speculative = metrics["summary"]["speculative"]
        self.assertEqual(speculative["steps"], 1)
        self.assertEqual(speculative["batch_calls"], 1)
        self.assertEqual(speculative["mean_batch_size"], 1.0)
        self.assertEqual(speculative["max_batch_size"], 1)
        self.assertEqual(speculative["draft_tokens"], 3)
        self.assertEqual(speculative["accepted_tokens"], 2)
        self.assertEqual(speculative["emitted_tokens"], 3)
        self.assertAlmostEqual(speculative["acceptance_rate"], 2 / 3)
        self.assertAlmostEqual(speculative["acceptance_length"], 3.0)
        self.assertAlmostEqual(speculative["accepted_length"], 2.0)
        self.assertAlmostEqual(speculative["draft_tokens_per_step"], 3.0)
        self.assertEqual(speculative["accept_all_count"], 0)
        self.assertEqual(speculative["gamma_counts"], {"3": 1})
        self.assertEqual(speculative["timing"]["total_time"]["total"], 0.075)
        self.assertEqual(speculative["timing"]["total_time"]["mean"], 0.075)
        self.assertEqual(speculative["timing"]["draft_proposal_time"]["total"], 0.02)
        self.assertEqual(speculative["timing"]["target_verify_time"]["total"], 0.03)
        self.assertEqual(metrics["requests"][0]["speculative_steps"], 1)
        self.assertEqual(metrics["requests"][0]["output_event_times"], [
            engine.request_metrics[seq.seq_id]["output_event_times"][0]
        ])
        self.assertEqual(len(metrics["requests"][0]["output_event_latency"]), 0)
        self.assertEqual(len(metrics["requests"][0]["speculative_step_latency"]), 1)
        self.assertIsNotNone(metrics["summary"]["speculative_step_latency"]["mean"])
        self.assertEqual(metrics["requests"][0]["speculative_gamma_counts"], {"3": 1})
        self.assertEqual(metrics["requests"][0]["speculative_timing"]["trace_time"], 0.006)
        self.assertEqual(
            metrics["requests"][0]["speculative_trace"],
            [
                {
                    "step": 1,
                    "draft_token_ids": [11, 12],
                    "matches": [True, True],
                    "emitted_token_ids": [10, 11, 12],
                    "num_accepted": 2,
                    "accepted_all": False,
                }
            ],
        )

    def test_step_uses_tree_branch_for_single_request_when_enabled(self):
        seq = Sequence([1, 2, 3])
        seq.status = SequenceStatus.RUNNING
        seq.num_cached_tokens = len(seq)
        engine = LLMEngine.__new__(LLMEngine)
        engine.scheduler = FakeScheduler(seq)
        engine.model_runner = FakeTreeModelRunner()
        engine.request_metrics = {seq.seq_id: self.make_metric(seq)}

        LLMEngine.step(engine)

        self.assertEqual(engine.model_runner.calls[0][0], "run_speculative_tree_single")
        self.assertEqual(engine.scheduler.block_manager.calls, [(seq.seq_id, 5)])

    def test_step_falls_back_to_normal_decode_without_draft_model(self):
        seq = Sequence([1, 2, 3])
        seq.status = SequenceStatus.RUNNING
        seq.num_cached_tokens = len(seq)

        scheduler = FakeScheduler(seq)
        scheduler.postprocess = lambda output, token_ids: [
            output.decode_seqs[0].append_token(token_ids[0])
        ]

        engine = LLMEngine.__new__(LLMEngine)
        engine.scheduler = scheduler
        engine.model_runner = FakeNormalModelRunner()
        engine.request_metrics = {seq.seq_id: self.make_metric(seq)}

        _, num_tokens = LLMEngine.step(engine)

        self.assertEqual(num_tokens, 1)
        self.assertEqual(seq.completion_token_ids, [10])
        self.assertEqual(engine.model_runner.calls[0][0], "run")
        self.assertEqual(engine.scheduler.block_manager.calls, [])
        self.assertNotIn("speculative", engine.last_step_events)
        self.assertEqual(len(engine.request_metrics[seq.seq_id]["output_event_times"]), 1)
        self.assertEqual(engine.last_step_events["waiting_queue_size"], 2)
        self.assertEqual(engine.last_step_events["running_queue_size"], 1)

    def test_step_passes_explicit_groups_for_mixed_batch(self):
        prefill = Sequence([1, 2, 3])
        prefill.num_scheduled_tokens = 3
        decode = Sequence([4, 5, 6])
        decode.status = SequenceStatus.RUNNING
        decode.num_cached_tokens = len(decode)
        decode.num_scheduled_tokens = 1
        scheduler_output = SchedulerOutput(
            [prefill, decode], [prefill], [decode], 4
        )
        scheduler = FakeMixedScheduler(scheduler_output)
        engine = LLMEngine.__new__(LLMEngine)
        engine.scheduler = scheduler
        engine.model_runner = FakeMixedModelRunner()
        engine.request_metrics = {
            prefill.seq_id: self.make_metric(prefill),
            decode.seq_id: self.make_metric(decode),
        }

        _, num_tokens = LLMEngine.step(engine)

        self.assertEqual(num_tokens, 4)
        self.assertIs(scheduler.postprocess_output, scheduler_output)
        self.assertIs(engine.model_runner.calls[0][1][0], scheduler_output)
        self.assertEqual(engine.last_step_events["prefill_seq_ids"], [prefill.seq_id])
        self.assertEqual(engine.last_step_events["scheduled_seq_ids"], [prefill.seq_id, decode.seq_id])
        self.assertEqual(decode.completion_token_ids, [20])

    def test_step_runs_batched_speculative_decode_for_multiple_sequences(self):
        seq1 = Sequence([1, 2, 3])
        seq2 = Sequence([4, 5, 6])
        for seq in (seq1, seq2):
            seq.status = SequenceStatus.RUNNING
            seq.num_cached_tokens = len(seq)

        engine = LLMEngine.__new__(LLMEngine)
        engine.scheduler = FakeBatchScheduler([seq1, seq2])
        engine.model_runner = FakeBatchModelRunner()
        engine.request_metrics = {
            seq1.seq_id: self.make_metric(seq1),
            seq2.seq_id: self.make_metric(seq2),
        }

        outputs, num_tokens = LLMEngine.step(engine)

        self.assertEqual(outputs, [])
        self.assertEqual(num_tokens, 5)
        self.assertEqual(seq1.completion_token_ids, [10, 11, 12])
        self.assertEqual(seq2.completion_token_ids, [20, 21])
        self.assertEqual(
            engine.scheduler.block_manager.batch_calls,
            [([seq1.seq_id, seq2.seq_id], 5)],
        )
        self.assertEqual(engine.model_runner.calls[0][0], "run_speculative_batch")
        self.assertEqual(engine.request_metrics[seq1.seq_id]["speculative_accepted_tokens"], 2)
        self.assertEqual(engine.request_metrics[seq2.seq_id]["speculative_accepted_tokens"], 1)
        self.assertEqual(engine.request_metrics[seq1.seq_id]["speculative_gamma_counts"], {"3": 1})
        self.assertEqual(engine.request_metrics[seq2.seq_id]["speculative_gamma_counts"], {"1": 1})
        self.assertEqual(engine.last_step_events["scheduled_seq_ids"], [seq1.seq_id, seq2.seq_id])
        self.assertEqual(engine.last_step_events["speculative_num_accepted"], 3)
        self.assertEqual(engine.last_step_events["speculative_emitted_tokens"], 5)
        self.assertEqual(
            engine.last_step_events["speculative_debug_by_seq"][seq1.seq_id]["draft_token_ids"],
            [11, 12, 13],
        )
        speculative = LLMEngine.get_metrics(engine)["summary"]["speculative"]
        self.assertEqual(speculative["batch_calls"], 1)
        self.assertEqual(speculative["mean_batch_size"], 2.0)
        self.assertEqual(speculative["max_batch_size"], 2)
        self.assertEqual(speculative["gamma_counts"], {"1": 1, "3": 1})


if __name__ == "__main__":
    unittest.main()
