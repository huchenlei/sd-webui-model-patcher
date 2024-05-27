import torch

from lib_modelpatcher.model_patcher import (
    ModelPatcher
)

def test_model_patcher_creation():
    model = torch.nn.Linear(10, 10)
    load_device = torch.device("cuda:0")
    offload_device = torch.device("cpu")

    model_patcher = ModelPatcher(
        model=model,
        load_device=load_device,
        offload_device=offload_device,
    )

    assert model_patcher.model == model
    assert model_patcher.load_device == load_device
    assert model_patcher.offload_device == offload_device
    assert model_patcher.is_patched is False

