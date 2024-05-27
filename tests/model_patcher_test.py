import torch

from lib_modelpatcher.model_patcher import ModelPatcher, ModulePatch, WeightPatch


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


class SampleModel(torch.nn.Module):
    def __init__(self):
        super(SampleModel, self).__init__()
        self.fc1 = torch.nn.Linear(10, 10)
        self.fc2 = torch.nn.Linear(10, 10)

    def init_identity_linear(self):
        with torch.no_grad():
            self.fc1.weight.copy_(torch.eye(10))
            self.fc1.bias.copy_(torch.zeros(10))
            self.fc2.weight.copy_(torch.eye(10))
            self.fc2.bias.copy_(torch.zeros(10))

    def forward(self, x):
        x = self.fc1(x)
        x = self.fc2(x)
        return x


def test_model_patcher_module_patch():
    load_device = torch.device("cpu")  # TODO: Change to cuda:0 when running locally
    offload_device = torch.device("cpu")

    model = SampleModel()
    model.init_identity_linear()

    model = model.to(load_device)
    input_tensor = torch.randn(10, 10).to(load_device)
    assert torch.allclose(
        model(input_tensor), input_tensor
    ), "Model is identity transformation before patching"

    model_patcher = ModelPatcher(
        model=model,
        load_device=load_device,
        offload_device=offload_device,
    ).to(load_device)

    def create_new_forward(module, old_forward):
        assert isinstance(module, torch.nn.Linear)

        def new_forward(x):
            return old_forward(x) + 1.0

        return new_forward

    model_patcher.add_module_patch(
        key="fc1", module_patch=ModulePatch(create_new_forward_func=create_new_forward)
    )
    assert model_patcher.is_patched is False
    model_patcher.patch_model()
    assert model_patcher.is_patched is True
    assert torch.allclose(
        model(input_tensor), input_tensor + 1.0
    ), "Model is not identity transformation after patching"

    model_patcher.unpatch_model()
    assert model_patcher.is_patched is False
    assert torch.allclose(
        model(input_tensor), input_tensor
    ), "Model is identity transformation after unpatching"


def test_model_patcher_weight_patch():
    load_device = torch.device("cpu")  # TODO: Change to cuda:0 when running locally
    offload_device = torch.device("cpu")

    model = SampleModel()
    model.init_identity_linear()

    model = model.to(load_device)
    input_tensor = torch.randn(10, 10).to(load_device)
    assert torch.allclose(
        model(input_tensor), input_tensor
    ), "Model is identity transformation before patching"

    model_patcher = ModelPatcher(
        model=model,
        load_device=load_device,
        offload_device=offload_device,
    ).to(load_device)

    model_patcher.add_weight_patch(
        key="fc1.bias", weight_patch=WeightPatch(weight=2.0 * torch.ones_like(model.fc1.bias))
    )
    assert model_patcher.is_patched is False
    model_patcher.patch_model()
    assert model_patcher.is_patched is True
    assert torch.allclose(
        model(input_tensor), input_tensor + 2.0
    ), "Model is not identity transformation after patching"

    model_patcher.unpatch_model()
    assert model_patcher.is_patched is False
    assert torch.allclose(
        model(input_tensor), input_tensor
    ), "Model is identity transformation after unpatching"
