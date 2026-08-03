import unittest

from llmserve.pd.protocol import (
    InvalidLifecycleTransition,
    KVTransferDescriptor,
    RequestEnvelope,
    RequestLifecycle,
    RequestState,
)


class TestRequestEnvelope(unittest.TestCase):

    def test_round_trip_preserves_wire_fields(self):
        envelope = RequestEnvelope(
            request_id=7,
            prompt_token_ids=(11, 22, 33),
            max_tokens=16,
            temperature=0.01,
            ignore_eos=True,
        )

        payload = envelope.to_payload()
        restored = RequestEnvelope.from_payload(payload)

        self.assertEqual(restored, envelope)
        self.assertEqual(
            payload,
            {
                "request_id": 7,
                "prompt_token_ids": [11, 22, 33],
                "max_tokens": 16,
                "temperature": 0.01,
                "ignore_eos": True,
            },
        )

    def test_rejects_invalid_sampling_and_prompt_values(self):
        with self.assertRaises(ValueError):
            RequestEnvelope(1, (), 16, 0.01, False)
        with self.assertRaises(ValueError):
            RequestEnvelope(1, (1,), 0, 0.01, False)
        with self.assertRaises(ValueError):
            RequestEnvelope(1, (1,), 16, 0.0, False)


class TestKVTransferDescriptor(unittest.TestCase):

    def test_round_trip_preserves_layout_and_transfer_identity(self):
        descriptor = KVTransferDescriptor(
            request_id=7,
            transfer_id="handoff-7-1",
            num_tokens=128,
            num_layers=36,
            num_kv_heads=8,
            head_dim=128,
            dtype="bfloat16",
            block_size=256,
            payload_nbytes=18874368,
        )

        restored = KVTransferDescriptor.from_payload(descriptor.to_payload())

        self.assertEqual(restored, descriptor)
        self.assertEqual(restored.source_worker, "prefill")
        self.assertEqual(restored.target_worker, "decode")

    def test_rejects_invalid_shape_or_payload_metadata(self):
        with self.assertRaises(ValueError):
            KVTransferDescriptor(1, "x", 0, 36, 8, 128, "bfloat16", 256, 1)
        with self.assertRaises(ValueError):
            KVTransferDescriptor(1, "x", 128, 36, 8, 128, "", 256, 1)
        with self.assertRaises(ValueError):
            KVTransferDescriptor(1, "x", 128, 36, 8, 128, "bfloat16", 256, -1)


class TestRequestLifecycle(unittest.TestCase):

    def test_valid_pd_lifecycle_reaches_decoding_and_completion(self):
        lifecycle = RequestLifecycle(request_id=7)

        for state in (
            RequestState.PREFILLING,
            RequestState.HANDOFF,
            RequestState.DECODING,
            RequestState.FINISHED,
        ):
            lifecycle.transition(state)

        self.assertEqual(lifecycle.state, RequestState.FINISHED)
        self.assertTrue(lifecycle.is_terminal)

    def test_invalid_transition_is_rejected(self):
        lifecycle = RequestLifecycle(request_id=7)
        lifecycle.transition(RequestState.PREFILLING)

        with self.assertRaises(InvalidLifecycleTransition):
            lifecycle.transition(RequestState.DECODING)

    def test_failure_cleanup_is_idempotent(self):
        lifecycle = RequestLifecycle(request_id=7)

        lifecycle.fail("decode worker exited")
        lifecycle.fail("duplicate cleanup")

        self.assertEqual(lifecycle.state, RequestState.FAILED)
        self.assertEqual(lifecycle.terminal_reason, "decode worker exited")
        self.assertTrue(lifecycle.is_terminal)


if __name__ == "__main__":
    unittest.main()
