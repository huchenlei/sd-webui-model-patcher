# sd-webui-model-patcher
Patch LDM in ComfyUI style in A1111

## Road Map

- Support all exposed APIs in ComfyUI's `ModelPatcher`.
- Support patching of diffusers pipeline.

## Example

Here is a code snippet from A1111's IC-Light extension that demonstrates how to use `ModelPatcher`.
```python
def apply_c_concat(unet, old_forward: Callable) -> Callable:
    def new_forward(x, timesteps=None, context=None, **kwargs):
        # Expand according to batch number.
        c_concat = torch.cat(
            ([concat_conds.to(x.device)] * (x.shape[0] // concat_conds.shape[0])),
            dim=0,
        )
        new_x = torch.cat([x, c_concat], dim=1)
        return old_forward(new_x, timesteps, context, **kwargs)

    return new_forward

# Patch unet forward.
p.model_patcher.add_module_patch(
    "diffusion_model", ModulePatch(create_new_forward_func=apply_c_concat)
)
# Patch weights.
p.model_patcher.add_patches(
    patches={
        "diffusion_model." + key: (value.to(dtype=dtype, device=device),)
        for key, value in ic_model_state_dict.items()
    }
)
```