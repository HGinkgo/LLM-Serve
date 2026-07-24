import unittest
from types import SimpleNamespace

import torch


AWQ_PACK_ORDER = (0, 2, 4, 6, 1, 3, 5, 7)


def pack_awq_int4(values: torch.Tensor) -> torch.Tensor:
    if values.ndim != 2 or values.shape[1] % 8:
        raise ValueError("test values must be two-dimensional with N divisible by 8")
    chunks = values.to(torch.int64).view(values.shape[0], -1, 8)
    chunks = chunks[:, :, AWQ_PACK_ORDER]
    shifts = torch.arange(0, 32, 4, dtype=torch.int64)
    return torch.sum((chunks & 0xF) << shifts, dim=-1).to(torch.int32)


class AWQConfigTests(unittest.TestCase):
    def test_from_hf_config_accepts_only_the_pinned_qwen3_format(self):
        from llmserve.layers.quantization.awq import AWQRuntimeConfig

        hf_config = SimpleNamespace(
            dtype=torch.bfloat16,
            quantization_config={
                "quant_method": "awq",
                "backend": "autoawq",
                "bits": 4,
                "group_size": 128,
                "version": "gemm",
                "zero_point": True,
                "do_fuse": False,
            }
        )

        config = AWQRuntimeConfig.from_hf_config(hf_config)

        self.assertEqual(config.bits, 4)
        self.assertEqual(config.group_size, 128)
        self.assertEqual(config.pack_factor, 8)
        self.assertTrue(config.zero_point)
        self.assertEqual(config.version, "gemm")
        self.assertEqual(config.backend, "autoawq")
        self.assertEqual(config.activation_dtype, torch.bfloat16)

    def test_from_hf_config_rejects_unsupported_layouts(self):
        from llmserve.layers.quantization.awq import AWQRuntimeConfig

        base = {
            "quant_method": "awq",
            "backend": "autoawq",
            "bits": 4,
            "group_size": 128,
            "version": "gemm",
            "zero_point": True,
        }
        invalid_values = {
            "quant_method": "gptq",
            "backend": "llm-awq",
            "bits": 8,
            "group_size": 64,
            "version": "gemv",
            "zero_point": False,
        }

        for field, value in invalid_values.items():
            with self.subTest(field=field):
                quantization_config = dict(base)
                quantization_config[field] = value
                with self.assertRaisesRegex(ValueError, field):
                    AWQRuntimeConfig.from_hf_config(
                        SimpleNamespace(
                            quantization_config=quantization_config
                        )
                    )

    def test_from_hf_config_requires_quantization_metadata(self):
        from llmserve.layers.quantization.awq import AWQRuntimeConfig

        with self.assertRaisesRegex(ValueError, "quantization_config"):
            AWQRuntimeConfig.from_hf_config(SimpleNamespace())

    def test_from_hf_config_rejects_non_native_activation_dtype(self):
        from llmserve.layers.quantization.awq import AWQRuntimeConfig

        with self.assertRaisesRegex(ValueError, "dtype"):
            AWQRuntimeConfig.from_hf_config(
                SimpleNamespace(
                    dtype=torch.float16,
                    quantization_config={
                        "quant_method": "awq",
                        "backend": "autoawq",
                        "bits": 4,
                        "group_size": 128,
                        "version": "gemm",
                        "zero_point": True,
                    },
                )
            )


class AWQReferenceTests(unittest.TestCase):
    def test_unpack_restores_autoawq_gemm_channel_order(self):
        from llmserve.layers.quantization.awq import unpack_awq_int4

        expected = torch.tensor(
            [
                list(range(16)),
                list(reversed(range(16))),
            ],
            dtype=torch.int32,
        )

        unpacked = unpack_awq_int4(pack_awq_int4(expected))

        torch.testing.assert_close(unpacked, expected, rtol=0, atol=0)

    def test_dequantize_applies_groupwise_asymmetric_zero_and_scale(self):
        from llmserve.layers.quantization.awq import dequantize_awq_gemm

        qvalues = torch.tensor(
            [
                [1, 2, 3, 4, 5, 6, 7, 8],
                [8, 7, 6, 5, 4, 3, 2, 1],
                [2, 4, 6, 8, 10, 12, 14, 15],
                [15, 14, 12, 10, 8, 6, 4, 2],
            ],
            dtype=torch.int32,
        )
        zeros = torch.tensor(
            [
                [1, 1, 2, 2, 3, 3, 4, 4],
                [4, 4, 3, 3, 2, 2, 1, 1],
            ],
            dtype=torch.int32,
        )
        scales = torch.tensor(
            [
                [0.5, 0.5, 1.0, 1.0, 1.5, 1.5, 2.0, 2.0],
                [2.0, 2.0, 1.5, 1.5, 1.0, 1.0, 0.5, 0.5],
            ],
            dtype=torch.float32,
        )
        expected = (
            qvalues - zeros.repeat_interleave(2, dim=0)
        ) * scales.repeat_interleave(2, dim=0)

        actual = dequantize_awq_gemm(
            pack_awq_int4(qvalues),
            pack_awq_int4(zeros),
            scales,
            group_size=2,
        )

        self.assertEqual(actual.dtype, scales.dtype)
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)

    def test_reference_linear_matches_explicit_dequantize_and_matmul(self):
        from llmserve.layers.quantization.awq import awq_reference_linear

        qvalues = torch.tensor(
            [
                [1, 2, 3, 4, 5, 6, 7, 8],
                [8, 7, 6, 5, 4, 3, 2, 1],
                [2, 4, 6, 8, 10, 12, 14, 15],
                [15, 14, 12, 10, 8, 6, 4, 2],
            ],
            dtype=torch.int32,
        )
        zeros = torch.tensor(
            [
                [1, 1, 2, 2, 3, 3, 4, 4],
                [4, 4, 3, 3, 2, 2, 1, 1],
            ],
            dtype=torch.int32,
        )
        scales = torch.tensor(
            [
                [0.5, 0.5, 1.0, 1.0, 1.5, 1.5, 2.0, 2.0],
                [2.0, 2.0, 1.5, 1.5, 1.0, 1.0, 0.5, 0.5],
            ],
            dtype=torch.float32,
        )
        inputs = torch.tensor(
            [[[1.0, -1.0, 0.5, 2.0], [0.0, 1.0, 2.0, -1.0]]],
            dtype=torch.float32,
        )
        weight = (
            qvalues - zeros.repeat_interleave(2, dim=0)
        ) * scales.repeat_interleave(2, dim=0)
        expected = torch.matmul(inputs, weight)

        actual = awq_reference_linear(
            inputs,
            pack_awq_int4(qvalues),
            pack_awq_int4(zeros),
            scales,
            group_size=2,
        )

        self.assertEqual(actual.shape, (1, 2, 8))
        self.assertEqual(actual.dtype, inputs.dtype)
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)

    def test_reference_rejects_incompatible_checkpoint_tensors(self):
        from llmserve.layers.quantization.awq import dequantize_awq_gemm

        valid_qweight = torch.zeros((4, 1), dtype=torch.int32)
        valid_qzeros = torch.zeros((2, 1), dtype=torch.int32)
        valid_scales = torch.ones((2, 8), dtype=torch.float16)
        invalid_cases = (
            (
                valid_qweight.to(torch.int64),
                valid_qzeros,
                valid_scales,
                2,
                "qweight.*int32",
            ),
            (
                valid_qweight,
                valid_qzeros.to(torch.int64),
                valid_scales,
                2,
                "qzeros.*int32",
            ),
            (
                valid_qweight,
                valid_qzeros,
                torch.ones((2, 7), dtype=torch.float16),
                2,
                "scales",
            ),
            (
                torch.zeros((5, 1), dtype=torch.int32),
                valid_qzeros,
                valid_scales,
                2,
                "divisible",
            ),
        )

        for qweight, qzeros, scales, group_size, message in invalid_cases:
            with self.subTest(message=message):
                with self.assertRaisesRegex(ValueError, message):
                    dequantize_awq_gemm(
                        qweight,
                        qzeros,
                        scales,
                        group_size=group_size,
                    )

    def test_reference_linear_rejects_activation_scale_dtype_mismatch(self):
        from llmserve.layers.quantization.awq import awq_reference_linear

        with self.assertRaisesRegex(ValueError, "activation dtype"):
            awq_reference_linear(
                torch.zeros((1, 4), dtype=torch.float16),
                torch.zeros((4, 1), dtype=torch.int32),
                torch.zeros((2, 1), dtype=torch.int32),
                torch.ones((2, 8), dtype=torch.bfloat16),
                group_size=2,
            )


if __name__ == "__main__":
    unittest.main()
