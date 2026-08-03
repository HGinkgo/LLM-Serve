import unittest

from examples.pd_batch import build_parser


class TestPDBatchExample(unittest.TestCase):

    def test_chunked_prefill_and_prompt_repeat_are_explicit_flags(self):
        args = build_parser().parse_args(
            [
                "--model",
                "/models/qwen3",
                "--enable-chunked-prefill",
                "--prompt-repeat",
                "8",
                "--print-metrics",
            ]
        )

        self.assertTrue(args.enable_chunked_prefill)
        self.assertEqual(args.prompt_repeat, 8)
        self.assertTrue(args.print_metrics)

    def test_chunked_prefill_is_disabled_by_default(self):
        args = build_parser().parse_args(["--model", "/models/qwen3"])

        self.assertFalse(args.enable_chunked_prefill)
        self.assertEqual(args.prompt_repeat, 1)
        self.assertFalse(args.print_metrics)


if __name__ == "__main__":
    unittest.main()
