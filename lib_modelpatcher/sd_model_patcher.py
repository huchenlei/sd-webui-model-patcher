from typing import NamedTuple

from ldm.modules.diffusionmodules.openaimodel import UNetModel
from ldm.models.autoencoder import AutoencoderKL

from .model_patcher import ModelPatcher


class StableDiffusionModelPatchers(NamedTuple):
    """A class that contains all the model patchers for Stable Diffusion models."""

    unet_patcher: ModelPatcher[UNetModel]
    vae_patcher: ModelPatcher[AutoencoderKL]
    # TODO: Add more model patchers if necessary
