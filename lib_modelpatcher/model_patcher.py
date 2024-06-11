# Original version from:
# https://github.com/comfyanonymous/ComfyUI/blob/ffc4b7c30e35eb2773ace52a0b00e0ca5c1f4362/comfy/model_patcher.py

from __future__ import annotations
from collections import defaultdict
from enum import Enum
from typing import (
    Any,
    Callable,
    Dict,
    Generic,
    List,
    Optional,
    Set,
    Tuple,
    TypeVar,
    ClassVar,
    Union,
    NamedTuple,
)

import torch
import logging
from pydantic import BaseModel, ConfigDict, Field, field_validator


def set_attr(obj, attr, value):
    attrs = attr.split(".")
    for name in attrs[:-1]:
        obj = getattr(obj, name)
    prev = getattr(obj, attrs[-1])
    setattr(obj, attrs[-1], value)
    return prev


def get_attr(obj, attr):
    attrs = attr.split(".")
    for name in attrs:
        obj = getattr(obj, name)
    return obj


def module_size(module: torch.nn.Module) -> int:
    """Get the memory size of a module."""
    module_mem = 0
    sd = module.state_dict()
    for k in sd:
        t = sd[k]
        module_mem += t.nelement() * t.element_size()
    return module_mem


def apply_weight_decompose(dora_scale, weight):
    weight_norm = (
        weight.transpose(0, 1)
        .reshape(weight.shape[1], -1)
        .norm(dim=1, keepdim=True)
        .reshape(weight.shape[1], *[1] * (weight.dim() - 1))
        .transpose(0, 1)
    )

    return weight * (dora_scale / weight_norm).type(weight.dtype)


class PatchType(Enum):
    DIFF = "diff"
    LORA = "lora"
    LOKR = "lokr"
    LOHA = "loha"
    GLORA = "glora"


class LoRAWeight(NamedTuple):
    down: torch.Tensor
    up: torch.Tensor
    alpha_scale: Optional[float] = None
    # locon mid weights
    mid: Optional[torch.Tensor] = None
    dora_scale: Optional[torch.Tensor] = None


# Represent the model to patch.
ModelType = TypeVar("ModelType", bound=torch.nn.Module)
# Represent the sub-module of the model to patch.
ModuleType = TypeVar("ModuleType", bound=torch.nn.Module)
WeightPatchWeight = Union[torch.Tensor, LoRAWeight, Tuple[torch.Tensor, ...]]
CastToDeviceFunc = Callable[[torch.Tensor, torch.device, torch.dtype], torch.Tensor]


class WeightPatch(BaseModel):
    """Patch to apply on model weight."""

    model_config = ConfigDict(
        arbitrary_types_allowed=True,
        json_schema_extra="ignore",
    )

    cls_logger: ClassVar[logging.Logger] = logging.Logger("WeightPatch")
    cls_cast_to_device: ClassVar[CastToDeviceFunc] = lambda t, device, dtype: t.to(
        device, dtype
    )

    weight: WeightPatchWeight
    patch_type: PatchType = PatchType.DIFF
    # The scale applied on patch weight value.
    alpha: float = 1.0
    # The scale applied on the model weight value.
    strength_model: float = 1.0

    def apply(
        self, model_weight: torch.Tensor, key: Optional[str] = None
    ) -> torch.Tensor:
        """Apply the patch to model weight."""
        if self.strength_model != 1.0:
            model_weight *= self.strength_model

        try:
            if self.patch_type == PatchType.DIFF:
                assert isinstance(self.weight, torch.Tensor)
                return self._patch_diff(model_weight, key)
            elif self.patch_type == PatchType.LORA:
                assert isinstance(self.weight, LoRAWeight)
                return self._patch_lora(model_weight)
            else:
                raise NotImplementedError(
                    f"Patch type {self.patch_type} is not implemented."
                )
        except ValueError as e:
            logging.error("ERROR {} {} {}".format(self.patch_type, key, e))
            return model_weight

    def _patch_diff_expand(self, model_weight: torch.Tensor, key: str) -> torch.Tensor:
        """Unet input only. Used for the model to accept more input concats."""
        new_shape = [max(n, m) for n, m in zip(self.weight.shape, model_weight.shape)]
        WeightPatch.cls_logger.info(
            f"Merged with {key} channel changed from {model_weight.shape} to {new_shape}"
        )
        new_diff = self.alpha * WeightPatch.cls_cast_to_device(
            self.weight, model_weight.device, model_weight.dtype
        )
        new_weight = torch.zeros(size=new_shape).to(model_weight)
        new_weight[
            : model_weight.shape[0],
            : model_weight.shape[1],
            : model_weight.shape[2],
            : model_weight.shape[3],
        ] = model_weight
        new_weight[
            : new_diff.shape[0],
            : new_diff.shape[1],
            : new_diff.shape[2],
            : new_diff.shape[3],
        ] += new_diff
        return new_weight.contiguous().clone()

    def _patch_diff(self, model_weight: torch.Tensor, key: str) -> torch.Tensor:
        """Apply the diff patch to model weight."""
        if self.alpha != 0.0:
            if self.weight.shape != model_weight.shape:
                if model_weight.ndim == self.weight.ndim == 4:
                    return self._patch_diff_expand(model_weight, key)

                raise ValueError(
                    "WARNING SHAPE MISMATCH WEIGHT NOT MERGED {} != {}".format(
                        self.weight.shape, model_weight.shape
                    )
                )
            else:
                return model_weight + self.alpha * self.weight.to(model_weight.device)
        return model_weight

    def _patch_lora(self, model_weight: torch.Tensor) -> torch.Tensor:
        """Apply the lora/locon patch to model weight."""
        v: LoRAWeight = self.weight
        alpha = self.alpha
        weight = model_weight

        mat1 = WeightPatch.cls_cast_to_device(v.down, weight.device, torch.float32)
        mat2 = WeightPatch.cls_cast_to_device(v.up, weight.device, torch.float32)
        dora_scale = v.dora_scale

        if v.alpha_scale is not None:
            alpha *= v.alpha_scale / mat2.shape[0]
        if v.mid is not None:
            # locon mid weights, hopefully the math is fine because I didn't properly test it
            mat3 = WeightPatch.cls_cast_to_device(v.mid, weight.device, torch.float32)
            final_shape = [mat2.shape[1], mat2.shape[0], mat3.shape[2], mat3.shape[3]]
            mat2 = (
                torch.mm(
                    mat2.transpose(0, 1).flatten(start_dim=1),
                    mat3.transpose(0, 1).flatten(start_dim=1),
                )
                .reshape(final_shape)
                .transpose(0, 1)
            )
        weight += (
            (alpha * torch.mm(mat1.flatten(start_dim=1), mat2.flatten(start_dim=1)))
            .reshape(weight.shape)
            .type(weight.dtype)
        )
        if dora_scale is not None:
            weight = apply_weight_decompose(
                WeightPatch.cls_cast_to_device(
                    dora_scale, weight.device, torch.float32
                ),
                weight,
            )
        return weight


class ModulePatch(BaseModel, Generic[ModuleType]):
    """Patch to replace a module in the model."""

    create_new_forward_func: Callable[[ModuleType, Callable], Callable]


class ModelPatcher(BaseModel, Generic[ModelType]):
    model_config = ConfigDict(
        arbitrary_types_allowed=True,
        json_schema_extra="ignore",
        protected_namespaces=(),
    )

    cls_logger: ClassVar[logging.Logger] = logging.Logger(
        "ModelPatcher", level=logging.INFO
    )
    cls_strict: ClassVar[bool] = False

    # The managed model of the model patcher.
    model: ModelType = Field(frozen=True)
    # The device to run inference on.
    load_device: torch.device = Field(frozen=True)
    # The device to offload the model to.
    offload_device: torch.device = Field(frozen=True)
    # Whether to update weight in place.
    weight_inplace_update: bool = Field(frozen=True, default=False)

    # The current device the model is stored on.
    current_device: torch.device = None

    # The size of the model in number of bytes.
    model_size: int = None

    model_keys: Set[str] = None

    # The optional name of the ModelPatcher for debug purpose.
    name: str = Field(frozen=True, default="ModelPatcher")

    # Patches applied to module weights.
    weight_patches: Dict[str, List[WeightPatch]] = Field(
        default_factory=lambda: defaultdict(list)
    )
    # Store weights before patching.
    weight_backup: Dict[str, torch.Tensor] = Field(default_factory=dict)

    # Patches applied to model's torch modules.
    module_patches: Dict[str, List[ModulePatch]] = Field(
        default_factory=lambda: defaultdict(list)
    )
    # Store modules before patching.
    module_backup: Dict[str, Callable] = Field(default_factory=dict)
    # Whether the model is patched.
    is_patched: bool = False

    def __init__(self, **data):
        super().__init__(**data)
        self.current_device = (
            self.offload_device if self.current_device is None else self.current_device
        )
        self.model_size = (
            module_size(self.model) if self.model_size is None else self.model_size
        )
        self.model_keys = (
            set(self.model.state_dict().keys())
            if self.model_keys is None
            else self.model_keys
        )

    def add_weight_patch(self, key: str, weight_patch: WeightPatch) -> bool:
        if key not in self.model_keys:
            if self.cls_strict:
                raise ValueError(f"Key {key} not found in model.")
            else:
                return False
        self.weight_patches[key].append(weight_patch)

    def add_weight_patches(self, weight_patches: Dict[str, WeightPatch]) -> List[str]:
        return [
            key
            for key, weight_patch in weight_patches.items()
            if self.add_weight_patch(key, weight_patch)
        ]

    def add_patches(
        self,
        patches: Dict[str, Union[Tuple[torch.Tensor], Tuple[str, torch.Tensor]]],
        strength_patch: float = 1.0,
        strength_model: float = 1.0,
    ):
        """ComfyUI-compatible interface to add weight patches."""

        def parse_value(
            v: Union[Tuple[torch.Tensor], Tuple[str, torch.Tensor]]
        ) -> Tuple[torch.Tensor, PatchType]:
            if len(v) == 1:
                return dict(weight=v[0], patch_type=PatchType.DIFF)
            else:
                assert len(v) == 2, f"Invalid patch value {v}."
                return dict(weight=v[1], patch_type=PatchType(v[0]))

        return self.add_weight_patches(
            {
                key: WeightPatch(
                    **parse_value(value),
                    alpha=strength_patch,
                    strength_model=strength_model,
                )
                for key, value in patches.items()
            }
        )

    def clone(self):
        """ComfyUI-compatible interface to clone the model patcher."""
        # TODO: Check everything works as expected.
        # Some fields might needs explicit copy.
        return self.model_copy()

    def __repr__(self):
        return f"ModelPatcher(model={self.model.__class__}, model_size={self.model_size}, is_patched={self.is_patched})"

    def to(
        self,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ):
        self.model.to(device=device, dtype=dtype)
        if device is not None:
            self.current_device = device
        return self

    def get_attr(self, key: str) -> Optional[Any]:
        if key == ".":
            return self.model
        return get_attr(self.model, key)

    def set_attr(self, key: str, value: Any) -> Any:
        if key == ".":
            value = getattr(self.model, key)
            setattr(self.model, key, value)
            return value

        return set_attr(self.model, key, value)

    def set_attr_param(self, attr, value):
        return self.set_attr(attr, torch.nn.Parameter(value, requires_grad=False))

    def copy_to_param(self, attr, value):
        """inplace update tensor instead of replacing it"""
        attrs = attr.split(".")
        obj = self.model
        for name in attrs[:-1]:
            obj = getattr(obj, name)
        prev = getattr(obj, attrs[-1])
        prev.data.copy_(value)

    def add_module_patch(self, key: str, module_patch: ModulePatch) -> bool:
        target_module = self.get_attr(key)
        if target_module is None:
            if self.cls_strict:
                raise ValueError(f"Key {key} not found in model.")
            return False

        self.module_patches[key].append(module_patch)
        return True

    def _patch_modules(self):
        for key, module_patches in self.module_patches.items():
            module = self.get_attr(key)
            old_forward = module.forward
            self.module_backup[key] = old_forward
            for module_patch in module_patches:
                module.forward = module_patch.create_new_forward_func(
                    module, module.forward
                )

    def _patch_weights(self):
        for key, weight_patches in self.weight_patches.items():
            assert key in self.model_keys, f"Key {key} not found in model."
            old_weight = self.get_attr(key)
            self.weight_backup[key] = old_weight

            new_weight = old_weight
            for weight_patch in weight_patches:
                new_weight = weight_patch.apply(new_weight, key)

            if self.weight_inplace_update:
                self.copy_to_param(key, new_weight)
            else:
                self.set_attr_param(key, new_weight)

    def patch_model(self, patch_weights: bool = True):
        assert not self.is_patched, "Model is already patched."
        self._patch_modules()
        if patch_weights:
            self._patch_weights()
        self.is_patched = True
        return self.model

    def _unpatch_weights(self):
        for k, v in self.weight_backup.items():
            if self.weight_inplace_update:
                self.copy_to_param(k, v)
            else:
                self.set_attr_param(k, v)
        self.weight_backup.clear()

    def _unpatch_modules(self):
        for k, v in self.module_backup.items():
            module = self.get_attr(k)
            module.forward = v
        self.module_backup.clear()

    def unpatch_model(self, unpatch_weights=True):
        assert self.is_patched, "Model is not patched."
        if unpatch_weights:
            self._unpatch_weights()
        self.is_patched = False
        self._unpatch_modules()

    def close(self):
        """Properly free VRAM by clearing reference to tensors and modules."""
        assert not self.is_patched
        assert len(self.weight_backup) == 0
        assert len(self.module_backup) == 0
        self.module_patches.clear()
        self.weight_patches.clear()
