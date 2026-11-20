"""The output head of an encoder whose input and output tables are separate, read from a file.

Forge drops `lm_head` when it loads such a text encoder (it only needs hidden states). When the weights are still
in the encoder file they are read from there as a writing session starts. A quantised head is built with Forge's
own layer classes and then asked for its full matrix, so every storage format Forge can load is covered.

Some encoder files were saved without the head (the fp8 Qwen3-8B file for FLUX.2 Klein 9B is one). Then a head of
the same size is borrowed from another text encoder file that Forge knows about. That works when the two models
are close relatives: Qwen3-VL 8B is built on Qwen3-8B (their input tables match at 0.98 cosine), and Qwen3-8B
writes well through its head. A head of the right size from an unrelated model would write nonsense; the loop
guard and the clean-up then leave the prompts as typed.

The model's own head may also sit in a file of its own: Qwen publishes the head of Qwen3-8B as one shard of its
repository, a file that holds nothing else. Such a file is recognised by a fingerprint of its first megabyte and is
then the encoder's true head, not a borrowed one.
"""

import hashlib
import json
import os
import struct
from typing import Iterable, Optional

import torch

KEY = "lm_head.weight"
PRINT_BYTES = 1 << 20  # how much of the head's data a fingerprint covers
_headers: dict[tuple, tuple] = {}
_prints: dict[tuple, Optional[str]] = {}


def _header(path: str) -> tuple:
    """(shape of lm_head.weight or None, number of tensors, where its data starts) of a .safetensors file.

    Only the header is read.
    """
    try:
        stamp = (os.path.abspath(path), os.path.getmtime(path), os.path.getsize(path))
    except OSError:
        return None, 0, 0
    if stamp not in _headers:
        found = (None, 0, 0)
        if path.lower().endswith((".safetensors", ".sft")):
            try:
                with open(path, "rb") as handle:
                    (size,) = struct.unpack("<Q", handle.read(8))
                    if size < 100_000_000:
                        header = json.loads(handle.read(size))
                        entry = header.get(KEY) or {}
                        found = (entry.get("shape"), len(header) - ("__metadata__" in header), 8 + size + (entry.get("data_offsets") or [0])[0])
            except (OSError, ValueError, struct.error, AttributeError, TypeError):
                found = (None, 0, 0)
        _headers[stamp] = found
    return _headers[stamp]


def head_shape(path: str) -> Optional[list]:
    """The shape of lm_head.weight in a .safetensors file, or None."""
    return _header(path)[0]


def fingerprint(path: str) -> Optional[str]:
    """SHA-256 of the first megabyte of lm_head.weight as stored in the file, or None when the file has no head."""
    shape, _, start = _header(path)
    if shape is None:
        return None
    stamp = (os.path.abspath(path), os.path.getmtime(path), os.path.getsize(path))
    if stamp not in _prints:
        try:
            with open(path, "rb") as handle:
                handle.seek(start)
                _prints[stamp] = hashlib.sha256(handle.read(PRINT_BYTES)).hexdigest()
        except OSError:
            _prints[stamp] = None
    return _prints[stamp]


def find_head_file(paths: Iterable[str], vocab_size: int, hidden_size: int) -> Optional[str]:
    """The first of `paths` that holds an output head of this size."""
    for path in paths:
        if path and head_shape(path) == [vocab_size, hidden_size]:
            return path
    return None


def library_files() -> list[str]:
    """Every text encoder and VAE file Forge lists in its module dropdown (its model folders and --text-encoder-dirs)."""
    try:
        from modules_forge import main_entry

        return [str(path) for path in main_entry.module_list.values()]
    except Exception:  # noqa: BLE001 - outside Forge there is nothing to look at
        return []


def find_head(vocab_size: int, hidden_size: int, loaded: Iterable[str], library: Iterable[str], own_heads: Iterable[str] = ()) -> tuple[Optional[str], bool]:
    """Where to read a head of this size from, and whether it is borrowed from another model.

    The loaded encoder's own file comes first. Among the other files, one that carries the model's own head
    (a fingerprint in `own_heads`) is taken before any other and is not borrowed. Then a file that holds little
    besides a head (one saved for this purpose) is preferred to a whole encoder that happens to have a head of the
    same size.
    """
    own_heads = set(own_heads)
    loaded = [path for path in loaded if path]
    own = find_head_file(loaded, vocab_size, hidden_size)
    if own is not None:
        return own, False
    known = {os.path.abspath(path) for path in loaded}
    others = [path for path in library if path and os.path.abspath(path) not in known and head_shape(path) == [vocab_size, hidden_size]]
    if own_heads:
        true = sorted((path for path in others if fingerprint(path) in own_heads), key=lambda path: os.path.basename(path).lower())
        if true:
            return true[0], False
    others.sort(key=lambda path: (_header(path)[1] > 4, os.path.basename(path).lower()))
    return (others[0], True) if others else (None, False)


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
