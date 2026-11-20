"""The output head of an encoder whose input and output tables are separate, read from the encoder file.

Forge drops `lm_head` when it loads such a text encoder (it only needs hidden states). The weights are still in
the file, so they are read from there when a writing session starts. A quantised head is built with Forge's own
layer classes and then asked for its full matrix, so every storage format Forge can load is covered.
"""

import json
import os
import struct
from typing import Iterable, Optional

import torch

KEY = "lm_head.weight"
_headers: dict[tuple, Optional[list]] = {}


def head_shape(path: str) -> Optional[list]:
    """The shape of lm_head.weight in a .safetensors file, or None. Only the file's header is read."""
    try:
        stamp = (os.path.abspath(path), os.path.getmtime(path), os.path.getsize(path))
    except OSError:
        return None
    if stamp not in _headers:
        shape = None
        if path.lower().endswith((".safetensors", ".sft")):
            try:
                with open(path, "rb") as handle:
                    (size,) = struct.unpack("<Q", handle.read(8))
                    if size < 100_000_000:
                        shape = json.loads(handle.read(size)).get(KEY, {}).get("shape")
            except (OSError, ValueError, struct.error):
                shape = None
        _headers[stamp] = shape
    return _headers[stamp]


def find_head_file(paths: Iterable[str], vocab_size: int, hidden_size: int) -> Optional[str]:
    """The first of `paths` that holds an output head of this size."""
    for path in paths:
        if path and head_shape(path) == [vocab_size, hidden_size]:
            return path
    return None


def loaded_files() -> list[str]:
    """The files Forge loaded the current model from: text encoders and VAE first, then the checkpoint."""
    try:
        from modules import sd_models

        parameters = sd_models.model_data.forge_loading_parameters or {}
    except Exception:  # noqa: BLE001 - outside Forge there is nothing to look at
        return []
    files = [str(path) for path in parameters.get("additional_modules", []) or []]
    info = parameters.get("checkpoint_info")
    if getattr(info, "filename", None):
        files.append(str(info.filename))
    return files


def load_head(path: str, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    """The output matrix (vocabulary, hidden) on `device` in `dtype`."""
    import safetensors
    from backend import utils
    from backend.operations import using_forge_operations
    from backend.state_dict import detect_quantization, load_state_dict

    with safetensors.safe_open(path, framework="pt", device="cpu") as handle:
        tensors = {key: handle.get_tensor(key) for key in handle.keys() if key.startswith("lm_head.")}
    weight = tensors[KEY]
    vocab_size, hidden_size = weight.shape
    if set(tensors) == {KEY} and weight.dtype in (torch.bfloat16, torch.float16, torch.float32):
        return weight.to(device=device, dtype=dtype)

    # a quantised head: let Forge's layer hold it, then read the matrix off it one block of unit vectors at a time
    quantisation = detect_quantization(tensors)
    floating = utils.weight_dtype(tensors)
    if quantisation is not None:
        quantisation.pop("layout", None)

    class Holder(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.lm_head = torch.nn.Linear(hidden_size, vocab_size, bias=False)

    with using_forge_operations(device=torch.device("cpu"), dtype=floating or dtype, manual_cast_enabled=True, sd_dtype=floating, extra_dtype=quantisation):
        holder = Holder()
    load_state_dict(holder, tensors, log_name="lm_head")
    holder = holder.eval().to(device)

    matrix = torch.empty((vocab_size, hidden_size), device=device, dtype=dtype)
    block = 256
    with torch.inference_mode():
        for start in range(0, hidden_size, block):
            stop = min(start + block, hidden_size)
            units = torch.zeros((stop - start, hidden_size), device=device, dtype=dtype)
            units[torch.arange(stop - start), torch.arange(start, stop)] = 1.0
            matrix[:, start:stop] = holder.lm_head(units).to(dtype).T
    del holder
    return matrix
