import unittest

import torch
from torch import nn

from llmserve.layers.linear import MergedColumnParallelLinear, QKVParallelLinear


class MergedColumnParallelLinearTest(unittest.TestCase):

    def test_weight_loader_places_each_unpacked_shard_at_its_output_offset(self):
        linear = MergedColumnParallelLinear.__new__(MergedColumnParallelLinear)
        nn.Module.__init__(linear)
        linear.output_sizes = [2, 2]
        linear.tp_size = 1
        linear.tp_rank = 0
        linear.tp_dim = 0
        param = nn.Parameter(torch.empty(4, 3))

        linear.weight_loader(
            param,
            torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]),
            loaded_shard_id=0,
        )
        linear.weight_loader(
            param,
            torch.tensor([[7.0, 8.0, 9.0], [10.0, 11.0, 12.0]]),
            loaded_shard_id=1,
        )

        self.assertTrue(torch.equal(
            param.data,
            torch.tensor([
                [1.0, 2.0, 3.0],
                [4.0, 5.0, 6.0],
                [7.0, 8.0, 9.0],
                [10.0, 11.0, 12.0],
            ]),
        ))

    def test_qkv_weight_loader_uses_unpacked_tensor_parallel_shards(self):
        linear = QKVParallelLinear.__new__(QKVParallelLinear)
        nn.Module.__init__(linear)
        linear.tp_size = 2
        linear.tp_rank = 1
        linear.tp_dim = 0
        linear.num_heads = 2
        linear.num_kv_heads = 1
        linear.head_size = 1
        param = nn.Parameter(torch.empty(4, 2))

        linear.weight_loader(
            param,
            torch.tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0]]),
            loaded_shard_id="q",
        )
        linear.weight_loader(
            param,
            torch.tensor([[9.0, 10.0], [11.0, 12.0]]),
            loaded_shard_id="k",
        )
        linear.weight_loader(
            param,
            torch.tensor([[13.0, 14.0], [15.0, 16.0]]),
            loaded_shard_id="v",
        )

        self.assertTrue(torch.equal(
            param.data,
            torch.tensor([
                [5.0, 6.0],
                [7.0, 8.0],
                [11.0, 12.0],
                [15.0, 16.0],
            ]),
        ))


if __name__ == "__main__":
    unittest.main()
