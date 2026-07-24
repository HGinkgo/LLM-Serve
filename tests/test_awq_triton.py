import unittest
from unittest import mock

import torch


AWQ_PACK_ORDER = (0, 2, 4, 6, 1, 3, 5, 7)


def pack_awq_int4(values: torch.Tensor) -> torch.Tensor:
    chunks = values.to(torch.int64).view(values.shape[0], -1, 8)
    chunks = chunks[:, :, AWQ_PACK_ORDER]
    shifts = torch.arange(0, 32, 4, dtype=torch.int64)
    return torch.sum((chunks & 0xF) << shifts, dim=-1).to(torch.int32)


class AWQTritonTests(unittest.TestCase):
    def test_kernel_config_adds_split_k_only_for_underfilled_shapes(self):
        from llmserve.layers.quantization.awq_triton import (
            select_awq_kernel_config,
        )

        self.assertEqual(select_awq_kernel_config(4096, 24576), (64, 1))
        self.assertEqual(select_awq_kernel_config(4096, 6144), (32, 4))
        self.assertEqual(select_awq_kernel_config(4096, 4096), (32, 1))
        self.assertEqual(select_awq_kernel_config(12288, 4096), (32, 8))
        self.assertEqual(select_awq_kernel_config(128, 64), (32, 1))

    def test_kernel_rejects_cpu_tensors(self):
        from llmserve.layers.quantization import awq_triton

        for triton_value in (awq_triton.triton, None):
            with self.subTest(triton_available=triton_value is not None):
                with mock.patch.object(awq_triton, "triton", triton_value):
                    with self.assertRaisesRegex(ValueError, "CUDA"):
                        awq_triton.awq_triton_linear(
                            torch.zeros((1, 128), dtype=torch.bfloat16),
                            torch.zeros((128, 8), dtype=torch.int32),
                            torch.zeros((1, 8), dtype=torch.int32),
                            torch.ones((1, 64), dtype=torch.bfloat16),
                            group_size=128,
                        )

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA is required")
    def test_kernel_matches_reference_for_decode_token_counts(self):
        from llmserve.layers.quantization.awq import awq_reference_linear
        from llmserve.layers.quantization.awq_triton import awq_triton_linear

        torch.manual_seed(7)
        device = torch.device("cuda")
        qvalues = torch.randint(0, 16, (128, 64), dtype=torch.int32)
        zeros = torch.randint(0, 16, (1, 64), dtype=torch.int32)
        scales = torch.rand((1, 64), dtype=torch.bfloat16) * 0.05
        qweight = pack_awq_int4(qvalues).to(device)
        qzeros = pack_awq_int4(zeros).to(device)
        scales = scales.to(device)

        for num_tokens in (1, 4, 8, 16):
            with self.subTest(num_tokens=num_tokens):
                inputs = torch.randn(
                    (num_tokens, 128),
                    dtype=torch.bfloat16,
                    device=device,
                )
                expected = awq_reference_linear(
                    inputs,
                    qweight,
                    qzeros,
                    scales,
                    group_size=128,
                )
                actual = awq_triton_linear(
                    inputs,
                    qweight,
                    qzeros,
                    scales,
                    group_size=128,
                )

                self.assertEqual(actual.dtype, torch.bfloat16)
                torch.testing.assert_close(actual, expected, rtol=0.02, atol=0.05)

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA is required")
    def test_single_token_gemv_matches_bfloat16_reference_semantics(self):
        from llmserve.layers.quantization.awq import awq_reference_linear
        from llmserve.layers.quantization.awq_triton import awq_triton_linear

        torch.manual_seed(11)
        device = torch.device("cuda")
        qvalues = torch.randint(0, 16, (128, 16384), dtype=torch.int32)
        zeros = torch.randint(0, 16, (1, 16384), dtype=torch.int32)
        scales = (torch.rand((1, 16384)) * 0.05).to(torch.bfloat16)
        qweight = pack_awq_int4(qvalues).to(device)
        qzeros = pack_awq_int4(zeros).to(device)
        scales = scales.to(device)
        inputs = torch.randn((1, 128), dtype=torch.bfloat16, device=device)

        expected = awq_reference_linear(
            inputs,
            qweight,
            qzeros,
            scales,
            group_size=128,
        )
        actual = awq_triton_linear(
            inputs,
            qweight,
            qzeros,
            scales,
            group_size=128,
        )

        torch.testing.assert_close(actual, expected, rtol=0.01, atol=0.02)


if __name__ == "__main__":
    unittest.main()
