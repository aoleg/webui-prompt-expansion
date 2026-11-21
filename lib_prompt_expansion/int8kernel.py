"""While a text is written, an encoder's int8 ConvRot linears run through comfy-kitchen's fused int8 kernel.

Forge runs a text encoder's int8 layers with full-precision activations: every call turns the int8 weight into
bf16 and multiplies in bf16. That is the right choice for the conditioning, and it is slow for writing, where
the same weights are read once per token: on an RTX 5090 more than half of a token's GPU time went into that
conversion. The fused kernel rotates and quantises the activation row by row and multiplies int8 by int8, one
byte read per weight. Forge uses the same kernel in its int8 image models.

The numbers change slightly, because the activation is now quantised too. Measured against the full-precision
path with the same tokens fed to both, the next token was the same in 98 % of the places on Qwen3-VL 4B and in
95 % on Qwen3.5 9B; where it differed it was the other path's second choice, a near tie. Writing is a sampled
process, so this is well inside its own noise. The conditioning is not touched: the layers are switched for the
writing session only, and Forge encodes the finished prompts its usual way.

Only ConvRot weights are switched. Without the rotation an int8 activation loses too much.
"""

import torch

TAG = "[Prompt Expansion]"
ROWS_CHECKED = (1, 2, 4)  # the kernel has its own code path for one row, for two, and for more
AGREEMENT = 0.99  # cosine similarity of the kernel's output and Forge's, for a random activation

_said: set[str] = set()


def _say(reason: str):
    if reason not in _said:
        _said.add(reason)
        print(f"{TAG} The int8 kernel is not used for writing: {reason}.")


def _parts(module):
    """(int8 weight, scale, ConvRot group size) of a linear the kernel can run, else None."""
    from backend.quant_ops import QuantizedTensor, TensorWiseINT8Layout

    weight = getattr(module, "weight", None)
    if type(module).__name__ != "Linear" or not isinstance(weight, QuantizedTensor) or weight._layout_cls != "TensorWiseINT8Layout":
        return None
    params = weight._params
    if not getattr(params, "convrot", False) or getattr(params, "transposed", False):
        return None
    if len(getattr(module, "weight_function", ())) or len(getattr(module, "bias_function", ())) or getattr(module, "forge_force_cast_weights", False):
        return None  # a patched weight (a LoRA on the text encoder): Forge's own path knows how
    qdata, scale = TensorWiseINT8Layout.get_plain_tensors(weight)
    if not qdata.is_cuda:
        return None
    return qdata, scale, int(getattr(params, "convrot_groupsize", 256))


def _kernel(module, parts):
    from backend.quant_ops import ck

    qdata, scale, group = parts
    original = type(module).forward

    def forward(x, *args, **kwargs):
        if args or kwargs or not torch.is_tensor(x) or x.device != qdata.device or not x.is_floating_point():
            return original(module, x, *args, **kwargs)
        bias = module.bias
        if bias is not None:
            bias = bias.to(device=x.device, dtype=x.dtype)
        return ck.int8_linear(x, qdata, scale, bias, x.dtype, convrot=True, convrot_groupsize=group)

    return forward


def _agrees(module, forward, dtype) -> bool:
    """The kernel against Forge's own layer, on a random activation, for each row count it treats differently."""
    weight = module.weight
    generator = torch.Generator(device="cpu").manual_seed(0)
    for rows in ROWS_CHECKED:
        x = torch.randn((rows, 1, weight.shape[1]), generator=generator).to(device=weight.device, dtype=dtype)
        ours, theirs = forward(x).float(), module(x).float()
        similarity = torch.nn.functional.cosine_similarity(ours.flatten(1), theirs.flatten(1), dim=-1).min()
        if not float(similarity) >= AGREEMENT:  # also false for NaN
            return False
    return True


def switch_on(stack, dtype) -> list:
    """Route the int8 ConvRot linears of the layer stack through the fused kernel. Returns what switch_off() needs.

    Returns an empty list, and changes nothing, when the encoder has no such layers or the kernel is not there.
    """
    try:
        from backend.quant_ops import ck

        if not hasattr(ck, "int8_linear"):
            return []
        found = [(module, parts) for module in stack.layers.modules() for parts in [_parts(module)] if parts is not None]
        if not found:
            return []
        first, parts = found[0]
        if not _agrees(first, _kernel(first, parts), dtype):
            _say("its result does not agree with Forge's own layer")
            return []
    except Exception as problem:  # noqa: BLE001 - a speed-up is not worth a failed job
        _say(f"{type(problem).__name__}: {str(problem)[:200]}")
        return []
    for module, parts in found:
        module.forward = _kernel(module, parts)
    return [module for module, _ in found]


def switch_off(modules: list):
    for module in modules:
        module.__dict__.pop("forward", None)
