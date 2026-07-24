import unittest

import torch
from torch import nn


class AWQWeightQuantizerTests(unittest.TestCase):
    def test_quantize_weight_exports_runtime_native_autoawq_layout(self):
        from llmserve.quantization.awq_calibration import quantize_awq_weight
        from llmserve.layers.quantization.awq import dequantize_awq_gemm

        weight = torch.tensor(
            [[-7.0, 8.0, -14.0, 16.0]] * 8,
            dtype=torch.bfloat16,
        )

        quantized = quantize_awq_weight(weight, group_size=2)

        self.assertEqual(quantized.qweight.shape, (4, 1))
        self.assertEqual(quantized.qzeros.shape, (2, 1))
        self.assertEqual(quantized.scales.shape, (2, 8))
        self.assertEqual(quantized.qweight.dtype, torch.int32)
        self.assertEqual(quantized.qzeros.dtype, torch.int32)
        self.assertEqual(quantized.scales.dtype, torch.bfloat16)
        expected_scales = torch.tensor(
            [[1.0] * 8, [2.0] * 8],
            dtype=torch.bfloat16,
        )
        torch.testing.assert_close(
            quantized.scales,
            expected_scales,
            rtol=0,
            atol=0,
        )
        restored = dequantize_awq_gemm(
            quantized.qweight,
            quantized.qzeros,
            quantized.scales,
            group_size=2,
        ).t()
        torch.testing.assert_close(restored, weight, rtol=0, atol=0)

    def test_quantize_weight_matches_explicit_asymmetric_reference(self):
        from llmserve.quantization.awq_calibration import quantize_awq_weight
        from llmserve.layers.quantization.awq import dequantize_awq_gemm

        torch.manual_seed(7)
        weight = torch.randn(16, 128, dtype=torch.float32)
        grouped = weight.reshape(-1, 128)
        minimum = grouped.amin(dim=1, keepdim=True)
        maximum = grouped.amax(dim=1, keepdim=True)
        scales = (maximum - minimum).clamp(min=1e-5) / 15
        zeros = (-torch.round(minimum / scales)).clamp(0, 15)
        expected = (
            torch.clamp(torch.round(grouped / scales) + zeros, 0, 15)
            - zeros
        ) * scales

        quantized = quantize_awq_weight(weight, group_size=128)
        actual = dequantize_awq_gemm(
            quantized.qweight,
            quantized.qzeros,
            quantized.scales,
            group_size=128,
        ).t()

        torch.testing.assert_close(
            actual.float(),
            expected.reshape_as(weight),
            rtol=0.01,
            atol=0.01,
        )

    def test_quantize_weight_rejects_unsupported_shapes(self):
        from llmserve.quantization.awq_calibration import quantize_awq_weight

        with self.assertRaisesRegex(ValueError, "two-dimensional"):
            quantize_awq_weight(torch.zeros(8), group_size=2)
        with self.assertRaisesRegex(ValueError, "floating"):
            quantize_awq_weight(
                torch.zeros((8, 4), dtype=torch.int32),
                group_size=2,
            )
        with self.assertRaisesRegex(ValueError, "input size"):
            quantize_awq_weight(torch.zeros((8, 5)), group_size=2)
        with self.assertRaisesRegex(ValueError, "output size"):
            quantize_awq_weight(torch.zeros((7, 4)), group_size=2)


class AWQScaleSearchTests(unittest.TestCase):
    def test_activation_aware_scale_does_not_regress_linear_reconstruction(self):
        from llmserve.quantization.awq_calibration import (
            fake_quantize_awq_weight,
            search_awq_scale,
        )

        torch.manual_seed(19)
        inputs = torch.randn(64, 128)
        inputs[:, 0] *= 40
        inputs[:, 1] *= 0.02
        weight = torch.randn(32, 128)
        weight[:, 0] *= 0.02
        weight[:, 1] *= 30

        reference = inputs @ weight.t()
        baseline = inputs @ fake_quantize_awq_weight(weight, group_size=128).t()
        scales = search_awq_scale(inputs, [weight], group_size=128, n_grid=20)
        calibrated_weight = fake_quantize_awq_weight(
            weight * scales.view(1, -1),
            group_size=128,
        ) / scales.view(1, -1)
        calibrated = inputs @ calibrated_weight.t()

        baseline_error = torch.mean((reference - baseline).float().pow(2))
        calibrated_error = torch.mean((reference - calibrated).float().pow(2))
        self.assertLessEqual(calibrated_error.item(), baseline_error.item())
        self.assertFalse(torch.allclose(scales, torch.ones_like(scales)))

    def test_rmsnorm_scale_folding_preserves_full_precision_output(self):
        from llmserve.quantization.awq_calibration import fold_rmsnorm_scale

        torch.manual_seed(23)
        norm = nn.RMSNorm(8, eps=1e-6)
        linears = [nn.Linear(8, 16, bias=False), nn.Linear(8, 4, bias=False)]
        inputs = torch.randn(3, 8)
        expected = [linear(norm(inputs)) for linear in linears]
        scales = torch.linspace(0.5, 1.5, 8)

        fold_rmsnorm_scale(norm, linears, scales)

        actual = [linear(norm(inputs)) for linear in linears]
        for expected_output, actual_output in zip(expected, actual):
            torch.testing.assert_close(actual_output, expected_output, rtol=1e-5, atol=1e-5)

    def test_linear_scale_folding_preserves_full_precision_chain(self):
        from llmserve.quantization.awq_calibration import fold_linear_scale

        torch.manual_seed(29)
        previous = nn.Linear(8, 16, bias=False)
        following = nn.Linear(16, 4, bias=False)
        inputs = torch.randn(3, 8)
        expected = following(previous(inputs))
        scales = torch.linspace(0.5, 1.5, 16)

        fold_linear_scale(previous, [following], scales)

        actual = following(previous(inputs))
        torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-5)


class AWQClipSearchTests(unittest.TestCase):
    def test_activation_aware_clip_does_not_regress_group_reconstruction(self):
        from llmserve.quantization.awq_calibration import (
            apply_awq_clip,
            fake_quantize_awq_weight,
            search_awq_clip,
        )

        torch.manual_seed(31)
        inputs = torch.randn(256, 8)
        inputs[:, 0] *= 0.001
        weight = torch.randn(8, 8)
        weight[:, 0] = 100
        reference = inputs @ weight.t()
        baseline = inputs @ fake_quantize_awq_weight(weight, group_size=8).t()

        clip_values = search_awq_clip(
            weight,
            inputs,
            group_size=8,
            n_grid=20,
            max_shrink=0.5,
        )
        clipped = weight.clone()
        apply_awq_clip(clipped, clip_values, group_size=8)
        calibrated = inputs @ fake_quantize_awq_weight(clipped, group_size=8).t()

        baseline_error = torch.mean((reference - baseline).float().pow(2))
        calibrated_error = torch.mean((reference - calibrated).float().pow(2))
        self.assertLessEqual(calibrated_error.item(), baseline_error.item())
        self.assertTrue(torch.any(clip_values < 100))

    def test_clip_shape_matches_output_channels_and_groups(self):
        from llmserve.quantization.awq_calibration import search_awq_clip

        weight = torch.randn(16, 128)
        inputs = torch.randn(4, 128)
        clip_values = search_awq_clip(weight, inputs, group_size=64, n_grid=2)
        self.assertEqual(clip_values.shape, (16, 2, 1))


if __name__ == "__main__":
    unittest.main()
