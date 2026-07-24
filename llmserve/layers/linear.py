import torch
from torch import nn
import torch.nn.functional as F
import torch.distributed as dist

from llmserve.layers.quantization.awq import AWQLinearMethod, AWQRuntimeConfig


def divide(numerator, denominator):
    assert numerator % denominator == 0
    return numerator // denominator


def validate_awq_output_shards(
    output_sizes: list[int],
    quant_config: AWQRuntimeConfig | None,
) -> None:
    if quant_config is None:
        return
    if any(size % quant_config.pack_factor for size in output_sizes):
        raise ValueError("each AWQ output shard must be divisible by pack_factor")


class LinearBase(nn.Module):

    def __init__(
        self,
        input_size: int,
        output_size: int,
        bias: bool = False,
        tp_dim: int | None = None,
        quant_config: AWQRuntimeConfig | None = None,
    ):
        super().__init__()
        self.tp_dim = tp_dim
        self.tp_rank = dist.get_rank()
        self.tp_size = dist.get_world_size()
        self.quant_method = AWQLinearMethod(quant_config) if quant_config else None
        if self.quant_method is None:
            self.weight = nn.Parameter(torch.empty(output_size, input_size))
            self.weight.weight_loader = self.weight_loader
        else:
            if bias:
                raise ValueError("AWQ Linear bias is not supported")
            self.quant_method.create_weights(self, input_size, output_size)
            for param in (self.qweight, self.qzeros, self.scales):
                param.weight_loader = self.weight_loader
        if bias:
            self.bias = nn.Parameter(torch.empty(output_size))
            self.bias.weight_loader = self.weight_loader
        else:
            self.register_parameter("bias", None)

    def apply_linear(self, x: torch.Tensor) -> torch.Tensor:
        if self.quant_method is not None:
            return self.quant_method.apply(self, x)
        return F.linear(x, self.weight, self.bias)

    @staticmethod
    def get_output_dim(param: nn.Parameter, default: int) -> int:
        return getattr(param, "output_dim", default)

    @staticmethod
    def get_packed_factor(param: nn.Parameter, dim: int) -> int:
        if getattr(param, "packed_dim", None) == dim:
            return getattr(param, "pack_factor")
        return 1

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError


class ReplicatedLinear(LinearBase):

    def __init__(
        self,
        input_size: int,
        output_size: int,
        bias: bool = False,
        quant_config: AWQRuntimeConfig | None = None,
    ):
        super().__init__(input_size, output_size, bias, quant_config=quant_config)

    def weight_loader(self, param: nn.Parameter, loaded_weight: torch.Tensor):
        param.data.copy_(loaded_weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.apply_linear(x)


class ColumnParallelLinear(LinearBase):

    def __init__(
        self,
        input_size: int,
        output_size: int,
        bias: bool = False,
        quant_config: AWQRuntimeConfig | None = None,
    ):
        tp_size = dist.get_world_size()
        super().__init__(
            input_size,
            divide(output_size, tp_size),
            bias,
            0,
            quant_config,
        )

    def weight_loader(self, param: nn.Parameter, loaded_weight: torch.Tensor):
        param_data = param.data
        output_dim = self.get_output_dim(param, self.tp_dim)
        shard_size = param_data.size(output_dim)
        start_idx = self.tp_rank * shard_size
        loaded_weight = loaded_weight.narrow(output_dim, start_idx, shard_size)
        param_data.copy_(loaded_weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.apply_linear(x)


class MergedColumnParallelLinear(ColumnParallelLinear):

    def __init__(
        self,
        input_size: int,
        output_sizes: list[int],
        bias: bool = False,
        quant_config: AWQRuntimeConfig | None = None,
    ):
        validate_awq_output_shards(output_sizes, quant_config)
        self.output_sizes = output_sizes
        super().__init__(input_size, sum(output_sizes), bias, quant_config)
        for param in self.parameters(recurse=False):
            param.expected_shard_ids = tuple(range(len(output_sizes)))

    def weight_loader(self, param: nn.Parameter, loaded_weight: torch.Tensor, loaded_shard_id: int):
        param_data = param.data
        output_dim = self.get_output_dim(param, self.tp_dim)
        pack_factor = self.get_packed_factor(param, output_dim)
        shard_offset = (
            sum(self.output_sizes[:loaded_shard_id]) // self.tp_size // pack_factor
        )
        shard_size = self.output_sizes[loaded_shard_id] // self.tp_size // pack_factor
        param_data = param_data.narrow(output_dim, shard_offset, shard_size)
        loaded_weight = loaded_weight.chunk(self.tp_size, output_dim)[self.tp_rank]
        param_data.copy_(loaded_weight)


class QKVParallelLinear(ColumnParallelLinear):

    def __init__(
        self,
        hidden_size: int,
        head_size: int,
        total_num_heads: int,
        total_num_kv_heads: int | None = None,
        bias: bool = False,
        quant_config: AWQRuntimeConfig | None = None,
    ):
        tp_size = dist.get_world_size()
        total_num_kv_heads = total_num_kv_heads or total_num_heads
        self.head_size = head_size
        self.num_heads = divide(total_num_heads, tp_size)
        self.num_kv_heads = divide(total_num_kv_heads, tp_size)
        validate_awq_output_shards(
            [
                total_num_heads * self.head_size,
                total_num_kv_heads * self.head_size,
                total_num_kv_heads * self.head_size,
            ],
            quant_config,
        )
        output_size = (total_num_heads + 2 * total_num_kv_heads) * self.head_size
        super().__init__(hidden_size, output_size, bias, quant_config)
        for param in self.parameters(recurse=False):
            param.expected_shard_ids = ("q", "k", "v")

    def weight_loader(self, param: nn.Parameter, loaded_weight: torch.Tensor, loaded_shard_id: str):
        param_data = param.data
        assert loaded_shard_id in ["q", "k", "v"]
        output_dim = self.get_output_dim(param, self.tp_dim)
        pack_factor = self.get_packed_factor(param, output_dim)
        if loaded_shard_id == "q":
            shard_size = self.num_heads * self.head_size
            shard_offset = 0
        elif loaded_shard_id == "k":
            shard_size = self.num_kv_heads * self.head_size
            shard_offset = self.num_heads * self.head_size
        else:
            shard_size = self.num_kv_heads * self.head_size
            shard_offset = self.num_heads * self.head_size + self.num_kv_heads * self.head_size
        shard_size //= pack_factor
        shard_offset //= pack_factor
        param_data = param_data.narrow(output_dim, shard_offset, shard_size)
        loaded_weight = loaded_weight.chunk(self.tp_size, output_dim)[self.tp_rank]
        param_data.copy_(loaded_weight)


class RowParallelLinear(LinearBase):

    def __init__(
        self,
        input_size: int,
        output_size: int,
        bias: bool = False,
        quant_config: AWQRuntimeConfig | None = None,
    ):
        tp_size = dist.get_world_size()
        super().__init__(
            divide(input_size, tp_size),
            output_size,
            bias,
            1,
            quant_config,
        )

    def weight_loader(self, param: nn.Parameter, loaded_weight: torch.Tensor):
        param_data = param.data
        if param_data.ndim == 1:
            param_data.copy_(loaded_weight)
            return
        input_dim = getattr(param, "input_dim", self.tp_dim)
        shard_size = param_data.size(input_dim)
        start_idx = self.tp_rank * shard_size
        loaded_weight = loaded_weight.narrow(input_dim, start_idx, shard_size)
        param_data.copy_(loaded_weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.quant_method is None:
            bias = self.bias if self.tp_rank == 0 else None
            y = F.linear(x, self.weight, bias)
        else:
            y = self.quant_method.apply(self, x)
        if self.tp_size > 1:
            dist.all_reduce(y)
        return y
