import sys
import logging
import functools
from typing import Callable

from modules.processing import (
    StableDiffusionProcessing,
    StableDiffusionProcessingTxt2Img,
    StableDiffusionProcessingImg2Img,
)
from modules.sd_samplers_common import Sampler
from modules import devices

from lib_modelpatcher.model_patcher import ModelPatcher


def model_patcher_hook(logger: logging.Logger):
    """Patches StableDiffusionProcessing to add
    - model_patcher
    - hr_model_patcher
    - get_model_patcher
    fields to StableDiffusionProcessing classes, and apply patches before
    calling sample methods
    """

    def hook_init(patcher_field_name: str):
        def decorator(func: Callable) -> Callable:
            @functools.wraps(func)
            def wrapped_init_func(self: StableDiffusionProcessing, *args, **kwargs):
                result = func(self, *args, **kwargs)

                sd_ldm = self.sd_model
                assert sd_ldm is not None
                load_device = devices.get_optimal_device()
                offload_device = devices.cpu

                setattr(
                    self,
                    patcher_field_name,
                    ModelPatcher(
                        model=sd_ldm.model,
                        load_device=load_device,
                        offload_device=offload_device,
                        name=f"{patcher_field_name} of {self.__class__.__name__}",
                    ),
                )
                logger.info(f"Init p.{patcher_field_name}.")
                return result

            return wrapped_init_func

        return decorator

    StableDiffusionProcessingTxt2Img.__init__ = hook_init("model_patcher")(
        StableDiffusionProcessingTxt2Img.__init__
    )
    StableDiffusionProcessingTxt2Img.__init__ = hook_init("hr_model_patcher")(
        StableDiffusionProcessingTxt2Img.__init__
    )
    StableDiffusionProcessingImg2Img.__init__ = hook_init("model_patcher")(
        StableDiffusionProcessingImg2Img.__init__
    )
    logger.info("__init__ hooks applied")

    def hook_close(patcher_field_name: str):
        def decorator(func: Callable) -> Callable:
            @functools.wraps(func)
            def wrapped_close_func(self: StableDiffusionProcessing, *args, **kwargs):
                patcher: ModelPatcher = getattr(self, patcher_field_name)
                assert isinstance(patcher, ModelPatcher)
                patcher.close()
                logger.info(f"Close p.{patcher_field_name}.")
                return func(self, *args, **kwargs)

            return wrapped_close_func

        return decorator

    StableDiffusionProcessingTxt2Img.close = hook_close("model_patcher")(
        StableDiffusionProcessingTxt2Img.close
    )
    StableDiffusionProcessingTxt2Img.close = hook_close("hr_model_patcher")(
        StableDiffusionProcessingTxt2Img.close
    )
    StableDiffusionProcessingImg2Img.close = hook_close("model_patcher")(
        StableDiffusionProcessingImg2Img.close
    )
    logger.info("close hooks applied")

    def hook_sample():
        def decorator(func: Callable) -> Callable:
            @functools.wraps(func)
            def wrapped_sample_func(self: Sampler, *args, **kwargs):
                patcher: ModelPatcher = self.p.get_model_patcher()
                assert isinstance(patcher, ModelPatcher)
                patcher.patch_model()
                logger.info(f"Patch {patcher.name}.")

                try:
                    return func(self, *args, **kwargs)
                finally:
                    patcher.unpatch_model()
                    logger.info(f"Unpatch {patcher.name}.")

            return wrapped_sample_func

        return decorator

    Sampler.launch_sampling = hook_sample()(Sampler.launch_sampling)
    logger.info("sample hooks applied")

    def get_model_patcher(self: StableDiffusionProcessing) -> ModelPatcher:
        if isinstance(self, StableDiffusionProcessingTxt2Img) and self.is_hr_pass:
            return self.hr_model_patcher
        return self.model_patcher

    StableDiffusionProcessing.get_model_patcher = get_model_patcher


def create_logger():
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(
            logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
        )
        logger.addHandler(handler)
    return logger


model_patcher_hook(create_logger())
