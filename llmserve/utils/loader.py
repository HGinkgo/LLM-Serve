import os
from glob import glob
import torch
from torch import nn
from safetensors import safe_open


def default_weight_loader(param: nn.Parameter, loaded_weight: torch.Tensor):
    param.data.copy_(loaded_weight)


def load_model(model: nn.Module, path: str):
    packed_modules_mapping = getattr(model, "packed_modules_mapping", {})
    loaded_param_ids = set()
    loaded_storages = set()
    loaded_shards = {}
    for file in glob(os.path.join(path, "*.safetensors")):
        with safe_open(file, "pt", "cpu") as f:
            for weight_name in f.keys():
                for k in packed_modules_mapping:
                    if k in weight_name:
                        v, shard_id = packed_modules_mapping[k]
                        param_name = weight_name.replace(k, v)
                        param = model.get_parameter(param_name)
                        weight_loader = getattr(param, "weight_loader")
                        weight_loader(param, f.get_tensor(weight_name), shard_id)
                        loaded_param_ids.add(id(param))
                        loaded_shards.setdefault(id(param), set()).add(shard_id)
                        break
                else:
                    param = model.get_parameter(weight_name)
                    weight_loader = getattr(param, "weight_loader", default_weight_loader)
                    weight_loader(param, f.get_tensor(weight_name))
                    loaded_param_ids.add(id(param))
                storage = param.untyped_storage()
                loaded_storages.add((str(param.device), storage.data_ptr()))

    missing = []
    for param_name, param in model.named_parameters():
        expected_shards = getattr(param, "expected_shard_ids", None)
        if expected_shards is not None:
            missing_shards = [
                shard_id
                for shard_id in expected_shards
                if shard_id not in loaded_shards.get(id(param), set())
            ]
            missing.extend(
                f"{param_name}[{shard_id}]" for shard_id in missing_shards
            )
            continue
        storage = param.untyped_storage()
        storage_key = (str(param.device), storage.data_ptr())
        if id(param) not in loaded_param_ids and storage_key not in loaded_storages:
            missing.append(param_name)
    if missing:
        raise RuntimeError(f"missing checkpoint parameters or shards: {missing}")

    for module in model.modules():
        quant_method = getattr(module, "quant_method", None)
        if quant_method is not None:
            quant_method.process_weights_after_loading(module)
