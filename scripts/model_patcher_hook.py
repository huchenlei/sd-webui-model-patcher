import logging
import functools
from typing import Callable

from modules.processing import (
    StableDiffusionProcessingTxt2Img,
    StableDiffusionProcessingImg2Img,
)
from modules import devices

from lib_modelpatcher.model_patcher import ModelPatcher
from lib_modelpatcher.sd_model_patcher import StableDiffusionModelPatchers


def model_patcher_hook(logger: logging.Logger):
    """Patches StableDiffusionProcessing to add
    - model_patchers
    - hr_model_patchers
    fields to StableDiffusionProcessing classes, and apply patches before
    calling sample methods
    """
    def hook_init(patcher_field_name: str):
        def decorator(func: Callable) -> Callable:
            @functools.wraps(func)
            def wrapped_init_func(self, *args, **kwargs):
                result = func(self, *args, **kwargs)

                sd_ldm = self.sd_model
                assert sd_ldm is not None
                load_device = devices.get_optimal_device()
                offload_device = devices.cpu

                setattr(
                    self,
                    patcher_field_name,
                    StableDiffusionModelPatchers(
                        unet_patcher=ModelPatcher(
                            model=sd_ldm.model.diffusion_model,
                            load_device=load_device,
                            offload_device=offload_device,
                        ),
                        vae_patcher=ModelPatcher(
                            model=sd_ldm.first_stage_model,
                            load_device=load_device,
                            offload_device=offload_device,
                        ),
                    ),
                )
                logger.info(f"Init p.{patcher_field_name}.")
                return result

            return wrapped_init_func

        return decorator

    StableDiffusionProcessingTxt2Img.__init__ = hook_init("model_patchers")(
        StableDiffusionProcessingTxt2Img.__init__
    )
    StableDiffusionProcessingTxt2Img.__init__ = hook_init("hr_model_patchers")(
        StableDiffusionProcessingTxt2Img.__init__
    )
    StableDiffusionProcessingImg2Img.__init__ = hook_init("model_patchers")(
        StableDiffusionProcessingImg2Img.__init__
    )
    logger.info("__init__ hooks applied")

    def hook_sample(patcher_field_name: str):
        def decorator(func: Callable) -> Callable:
            @functools.wraps(func)
            def wrapped_sample_func(self, *args, **kwargs):
                sd_patchers: StableDiffusionModelPatchers = getattr(
                    self, patcher_field_name
                )
                assert isinstance(sd_patchers, StableDiffusionModelPatchers)
                for patcher in sd_patchers:
                    assert isinstance(patcher, ModelPatcher)
                    patcher.patch_model()
                logger.info(f"Patch p.{patcher_field_name}.")

                try:
                    return func(self, *args, **kwargs)
                finally:
                    patcher.unpatch_model()
                    logger.info(f"Unpatch p.{patcher_field_name}.")

            return wrapped_sample_func

        return decorator

    StableDiffusionProcessingTxt2Img.sample = hook_sample("model_patchers")(
        StableDiffusionProcessingTxt2Img.sample
    )
    StableDiffusionProcessingImg2Img.sample = hook_sample("model_patchers")(
        StableDiffusionProcessingImg2Img.sample
    )
    StableDiffusionProcessingTxt2Img.sample_hr_pass = hook_sample("hr_model_patchers")(
        StableDiffusionProcessingTxt2Img.sample_hr_pass
    )
    logger.info("sample hooks applied")


logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
model_patcher_hook(logger)
