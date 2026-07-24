import unittest

import torch


AWQ_PACK_ORDER = (0, 2, 4, 6, 1, 3, 5, 7)


def pack_awq_int4(values: torch.Tensor) -> torch.Tensor:
    chunks = values.to(torch.int64).view(values.shape[0], -1, 8)
    chunks = chunks[:, :, AWQ_PACK_ORDER]
    shifts = torch.arange(0, 32, 4, dtype=torch.int64)
    return torch.sum((chunks & 0xF) << shifts, dim=-1).to(torch.int32)


def unpack_sequential_int4(packed: torch.Tensor) -> torch.Tensor:
    shifts = torch.arange(0, 32, 4, dtype=torch.int64)
    return (
        torch.bitwise_right_shift(packed.to(torch.int64).unsqueeze(-1), shifts)
        & 0xF
    ).reshape(packed.shape[0], -1)


class AWQCudaRepackTests(unittest.TestCase):
    def test_split_k_targets_underfilled_qwen3_shapes(self):
        from llmserve.layers.quantization.awq_cuda import (
            select_awq_cuda_split_k,
        )

        self.assertEqual(select_awq_cuda_split_k(1, 4096, 24576), 1)
        self.assertEqual(select_awq_cuda_split_k(4, 4096, 6144), 4)
        self.assertEqual(select_awq_cuda_split_k(8, 4096, 4096), 4)
        self.assertEqual(select_awq_cuda_split_k(16, 12288, 4096), 8)
        self.assertEqual(select_awq_cuda_split_k(32, 12288, 4096), 1)
        self.assertEqual(select_awq_cuda_split_k(128, 4096, 6144), 1)

    def test_reference_repack_builds_register_fragment_words(self):
        from llmserve.layers.quantization.awq_cuda import (
            repack_awq_qweight_reference,
        )

        k, n = 32, 128
        values = torch.arange(k * n, dtype=torch.int32).view(k, n) % 16
        packed = pack_awq_int4(values)

        repacked = repack_awq_qweight_reference(packed)

        self.assertEqual(repacked.shape, (k // 16, n // 64, 4, 32))
        actual_words = unpack_sequential_int4(repacked.reshape(-1, 1))
        expected_words = []
        for k_tile in range(k // 16):
            for n_tile in range(n // 64):
                for warp in range(4):
                    for lane in range(32):
                        group = lane // 4
                        thread = lane % 4
                        rows = (
                            k_tile * 16 + thread * 2,
                            k_tile * 16 + thread * 2 + 1,
                            k_tile * 16 + thread * 2 + 8,
                            k_tile * 16 + thread * 2 + 9,
                        )
                        first_col = n_tile * 64 + warp * 16 + group
                        second_col = first_col + 8
                        expected_words.append(
                            torch.tensor(
                                [
                                    values[rows[0], first_col],
                                    values[rows[1], first_col],
                                    values[rows[2], first_col],
                                    values[rows[3], first_col],
                                    values[rows[0], second_col],
                                    values[rows[1], second_col],
                                    values[rows[2], second_col],
                                    values[rows[3], second_col],
                                ],
                                dtype=torch.int64,
                            )
                        )
        expected = torch.stack(expected_words)
        torch.testing.assert_close(actual_words, expected, rtol=0, atol=0)

    def test_reference_repack_rejects_shapes_outside_kernel_contract(self):
        from llmserve.layers.quantization.awq_cuda import (
            repack_awq_qweight_reference,
        )

        with self.assertRaisesRegex(ValueError, "K.*16"):
            repack_awq_qweight_reference(torch.zeros((15, 8), dtype=torch.int32))
        with self.assertRaisesRegex(ValueError, "N.*64"):
            repack_awq_qweight_reference(torch.zeros((16, 4), dtype=torch.int32))
        with self.assertRaisesRegex(ValueError, "int32"):
            repack_awq_qweight_reference(torch.zeros((16, 8), dtype=torch.int64))

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA is required")
    def test_cuda_repack_matches_reference_layout(self):
        from llmserve.layers.quantization.awq_cuda import (
            repack_awq_qweight_cuda,
            repack_awq_qweight_reference,
        )

        values = torch.arange(32 * 128, dtype=torch.int32).view(32, 128) % 16
        packed = pack_awq_int4(values).cuda()

        actual = repack_awq_qweight_cuda(packed)
        expected = repack_awq_qweight_reference(packed)

        torch.testing.assert_close(actual, expected, rtol=0, atol=0)

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA is required")
    def test_cuda_linear_matches_reference_for_decode_token_counts(self):
        from llmserve.layers.quantization.awq import awq_reference_linear
        from llmserve.layers.quantization.awq_cuda import (
            awq_cuda_linear,
            repack_awq_qweight_cuda,
            reorder_awq_qzeros_cuda,
        )

        torch.manual_seed(17)
        k, n = 128, 64
        qvalues = torch.randint(0, 16, (k, n), dtype=torch.int32)
        zeros = torch.randint(0, 16, (k // 128, n), dtype=torch.int32)
        qweight = pack_awq_int4(qvalues).cuda()
        qzeros = pack_awq_int4(zeros).cuda()
        scales = (torch.rand((k // 128, n)) * 0.05).to(
            dtype=torch.bfloat16,
            device="cuda",
        )
        repacked_qweight = repack_awq_qweight_cuda(qweight)
        reordered_qzeros = reorder_awq_qzeros_cuda(qzeros)

        for num_tokens in (1, 4, 8, 16, 32, 65, 128):
            with self.subTest(num_tokens=num_tokens):
                inputs = torch.randn(
                    (num_tokens, k),
                    dtype=torch.bfloat16,
                    device="cuda",
                )
                expected = awq_reference_linear(
                    inputs,
                    qweight,
                    qzeros,
                    scales,
                    group_size=128,
                )
                actual = awq_cuda_linear(
                    inputs,
                    repacked_qweight,
                    reordered_qzeros,
                    scales,
                    group_size=128,
                )

                self.assertEqual(actual.dtype, torch.bfloat16)
                torch.testing.assert_close(actual, expected, rtol=0.02, atol=0.05)


if __name__ == "__main__":
    unittest.main()
