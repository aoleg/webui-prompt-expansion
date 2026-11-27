"""Molmo2-8B as a writer: a file of its own from the text encoder folders, run on Forge Neo's Qwen3-8B.

Molmo2's language model is Qwen3-8B (36 pre-norm layers of 4096, 32 heads of 128 over 8 key/value heads, q/k norms,
plain RoPE) under allenai's own tensor names, with the query, key and value projections fused into one matrix
(att_proj) and the two halves of the SwiGLU into another (ff_proj). convert() renames it to the layout of Forge Neo's
Qwen3-8B (backend.nn.llm.llama) and splits the fused matrices by rows. Nothing is copied: the result holds views of
the file's tensors.

The INT8 files store each linear as "qweight" (int8, the weight rotated by a block Hadamard matrix over its inputs)
and "scales" (one per output row): the same regular Hadamard blocks of 256 and the same per-row scale as
comfy-kitchen's INT8 ConvRot, so each one becomes a comfy_quant layer without touching the numbers. A row split of
such a matrix is exact: the rotation runs over the inputs, the scales over the outputs.

A file is recognised by its tensors, never by its name. Only a complete file can write: all 36 layers, the final norm
and the output head. The copies that are cut down for MiniMax-H3-Image (it reads the state after layer 24) still
carry the norm and the head, but text written without the last layers is a loop of garbage, so they are refused.

Text only. The vision tower (model.vision_backbone.*) and the 128 picture-token rows (wte.new_embedding) are not
read. The chat format is Qwen's, and so is the tokenizer: Molmo2's vocabulary and merges are those of the Qwen
tokenizer files Forge bundles.

The conversion is the one of the MiniMax H3 extension (forge_h3/native/molmo.py), which uses the first layers of the
same files as a text encoder.
"""

import json
import os
import re
import struct
from dataclasses import dataclass
from typing import Iterable, Optional

from .generator import UnsupportedEncoder

LABEL = "Molmo2-8B"
VOCAB, HIDDEN, LAYERS = 151936, 4096, 36

EMBEDDING = "model.transformer.wte.embedding"
BLOCKS = "model.transformer.blocks."
ATTENTION = "self_attn.att_proj"
QUANTIZED, PLAIN, SCALES = ".qweight", ".weight", ".scales"
FINAL_NORM = "model.transformer.ln_f.weight"
HEAD = "lm_head"
SKIPPED = ("model.vision_backbone.", "model.transformer.wte.new_embedding")
# the language model's Hadamard block (the vision tower of these files uses 64 and 16)
CONVROT_GROUP = 256

_BLOCK = re.compile(re.escape(BLOCKS) + r"(\d+)\.")
# Forge's bundled Qwen tokenizers, the one of Qwen3-8B first; all three hold the same vocabulary and merges
_TOKENIZERS = (("black-forest-labs", "FLUX.2-klein-9B"), ("black-forest-labs", "FLUX.2-klein-4B"), ("Tongyi-MAI", "Z-Image-Turbo"))

_looked: dict[tuple, Optional["Found"]] = {}
_tokenizer = None


# --- recognising a file ---------------------------------------------------------------------------------------


def is_molmo(keys) -> bool:
    return EMBEDDING in keys and any(f"{BLOCKS}0.{ATTENTION}{suffix}" in keys for suffix in (QUANTIZED, PLAIN))


def molmo_layers(keys) -> int:
    found = {int(match.group(1)) for key in keys if (match := _BLOCK.match(key))}
    return max(found) + 1 if found else 0


@dataclass(frozen=True)
class Found:
    """A Molmo2 file, as its header shows it."""

    path: str
    layers: int
    quantised: bool
    problem: str = ""  # why this file cannot write, as the rest of a sentence that begins with its name; empty when it can
    brief: str = ""  # the same in a few words, for a list

    @property
    def name(self) -> str:
        return os.path.basename(self.path)

    @property
    def choice(self) -> str:
        """How the file is named in the panel's list of writers."""
        return f"{LABEL}: {self.name}"

    def refusal(self) -> str:
        return f"{self.name} {self.problem}"


def _problem(tensors: dict) -> tuple[str, str]:
    """(why a Molmo2 file cannot write, the same in a few words); two empty texts when it can."""
    shape = tensors[EMBEDDING].get("shape")
    layers = molmo_layers(tensors)
    if shape != [VOCAB, HIDDEN] or layers > LAYERS:
        size = " x ".join(str(side) for side in shape or [])
        return f"is a Molmo2 model of another size (input table {size}, {layers} layers); only {LABEL} is supported", "a Molmo2 model of another size"
    present = {int(match.group(1)) for key in tensors if (match := _BLOCK.match(key))}
    if len(present) != layers:
        return f"is not a whole {LABEL}: {layers - len(present)} of its first {layers} layers are missing", f"a {LABEL} with layers missing"
    if layers < LAYERS:
        return (
            f"is a cut-down {LABEL}: it holds {layers} of the model's {LAYERS} layers. That is enough to encode for an image model "
            f"that reads an earlier layer, but text written without the last layers is a loop of garbage. The complete file ({LAYERS} layers) is needed"
        ), f"a {LABEL} cut down to {layers} of {LAYERS} layers"
    if FINAL_NORM not in tensors:
        return f"is a {LABEL} without its final norm, so it cannot write text", f"a {LABEL} without its final norm"
    if HEAD + PLAIN not in tensors and not (HEAD + QUANTIZED in tensors and HEAD + SCALES in tensors):
        return f"is a {LABEL} without its output head (lm_head), so it cannot write text", f"a {LABEL} without its output head"
    return "", ""


def look(path: str) -> Optional[Found]:
    """What a .safetensors file is, if it is a Molmo2 file; None for any other file. Only the header is read."""
    try:
        stamp = (os.path.normcase(os.path.abspath(path)), os.path.getmtime(path), os.path.getsize(path))
    except (OSError, TypeError):
        return None
    if stamp not in _looked:
        found = None
        if str(path).lower().endswith((".safetensors", ".sft")):
            try:
                with open(path, "rb") as handle:
                    (size,) = struct.unpack("<Q", handle.read(8))
                    raw = handle.read(size) if size < 100_000_000 else b""
                # most files are not Molmo2: their headers are not parsed at all
                if EMBEDDING.encode("ascii") in raw:
                    tensors = json.loads(raw)
                    tensors.pop("__metadata__", None)
                    if is_molmo(tensors):
                        found = Found(str(path), molmo_layers(tensors), f"{BLOCKS}0.{ATTENTION}{QUANTIZED}" in tensors, *_problem(tensors))
            except (OSError, ValueError, struct.error, AttributeError, TypeError):
                found = None
        _looked[stamp] = found
    return _looked[stamp]


def files(paths: Iterable[str]) -> tuple[list[Found], list[Found]]:
    """(the Molmo2 files among `paths` that can write, the ones that cannot), each sorted by name."""
    usable, refused, seen = [], [], set()
    for path in paths:
        found = look(path) if path else None
        if found is None or os.path.normcase(os.path.abspath(found.path)) in seen:
            continue
        seen.add(os.path.normcase(os.path.abspath(found.path)))
        (refused if found.problem else usable).append(found)
    return sorted(usable, key=lambda item: item.name.lower()), sorted(refused, key=lambda item: item.name.lower())


# --- the conversion -------------------------------------------------------------------------------------------


def _comfy_quant():
    import torch

    config = {"format": "int8_tensorwise", "convrot": True, "convrot_groupsize": CONVROT_GROUP}
    return torch.tensor(list(json.dumps(config).encode("utf-8")), dtype=torch.uint8)


def _shape(state_dict: dict, source: str) -> tuple[int, int]:
    """(outputs, inputs) of a linear of the file, quantized or plain."""
    key = source + QUANTIZED if source + QUANTIZED in state_dict else source + PLAIN
    return tuple(state_dict[key].shape)


def _linear(out: dict, target: str, state_dict: dict, source: str, rows: Optional[slice] = None) -> None:
    """One linear of the file, or a run of its output rows, under Forge's name: views of the file's tensors."""
    quantized = state_dict.get(source + QUANTIZED)
    if quantized is None:
        weight = state_dict[source + PLAIN]
        out[f"{target}.weight"] = weight if rows is None else weight[rows]
        return
    scales = state_dict[source + SCALES]
    if rows is not None:
        quantized, scales = quantized[rows], scales[rows]
    if quantized.shape[1] % CONVROT_GROUP:
        raise ValueError(f"{source}: {quantized.shape[1]} inputs do not divide into ConvRot blocks of {CONVROT_GROUP}")
    out[f"{target}.weight"] = quantized
    out[f"{target}.weight_scale"] = scales.reshape(-1, 1)
    out[f"{target}.comfy_quant"] = _comfy_quant()


def convert(state_dict: dict, layers: Optional[int] = None, head: bool = True) -> dict:
    """A Molmo2 state dict as Forge Neo's Qwen3-8B reads it: the first `layers` decoder layers (all of them when
    None), and with head the final norm and the lm_head, which text generation needs. The vision tower is left out.
    Nothing is copied."""
    blocks = molmo_layers(state_dict)
    layers = blocks if layers is None else layers
    if not 0 < layers <= blocks:
        raise ValueError(f"the Molmo2 file has {blocks} layers, {layers} are needed")
    # the text vocabulary; the 128 rows of wte.new_embedding are Molmo2's picture tokens
    out = {"model.embed_tokens.weight": state_dict[EMBEDDING]}
    for i in range(layers):
        source, target = f"{BLOCKS}{i}.", f"model.layers.{i}."
        # att_proj: the query rows (heads x head_dim, the input width of attn_out), then the keys, then the values
        attention = source + ATTENTION
        total = _shape(state_dict, attention)[0]
        query = _shape(state_dict, source + "self_attn.attn_out")[1]
        key_value = (total - query) // 2
        if key_value <= 0 or query + 2 * key_value != total:
            raise ValueError(f"{attention}: {total} rows do not split into query, key and value")
        _linear(out, target + "self_attn.q_proj", state_dict, attention, slice(0, query))
        _linear(out, target + "self_attn.k_proj", state_dict, attention, slice(query, query + key_value))
        _linear(out, target + "self_attn.v_proj", state_dict, attention, slice(query + key_value, total))
        _linear(out, target + "self_attn.o_proj", state_dict, source + "self_attn.attn_out")
        # ff_proj: Molmo2 computes silu(second half) * first half, Qwen3 silu(gate_proj) * up_proj
        half = _shape(state_dict, source + "mlp.ff_proj")[0] // 2
        _linear(out, target + "mlp.up_proj", state_dict, source + "mlp.ff_proj", slice(0, half))
        _linear(out, target + "mlp.gate_proj", state_dict, source + "mlp.ff_proj", slice(half, 2 * half))
        _linear(out, target + "mlp.down_proj", state_dict, source + "mlp.ff_out")
        for name, forge in (("attn_norm", "input_layernorm"), ("ff_norm", "post_attention_layernorm"), ("self_attn.q_norm", "self_attn.q_norm"), ("self_attn.k_norm", "self_attn.k_norm")):
            out[f"{target}{forge}.weight"] = state_dict[f"{source}{name}.weight"]
    if head:
        out["model.norm.weight"] = state_dict[FINAL_NORM]
        _linear(out, "model.lm_head", state_dict, HEAD)
    return out


# --- the model ------------------------------------------------------------------------------------------------


def build(converted: dict):
    """Forge Neo's Qwen3-8B with an output head, holding a converted state dict. On the CPU, as Forge's loader leaves
    a text encoder.

    The sizes are read off the tensors (a Molmo2-8B file gives Qwen3-8B's own), so the conversion can be checked on
    a miniature model.
    """
    import torch
    from backend import memory_management, utils
    from backend.nn.llm import llama
    from backend.operations import using_forge_operations
    from backend.state_dict import detect_quantization, load_state_dict
    from transformers.modeling_utils import no_init_weights

    first = "model.layers.0."
    vocab, hidden = converted["model.embed_tokens.weight"].shape
    head_dim = converted[first + "self_attn.q_norm.weight"].shape[0]
    config = llama.Qwen3_8BConfig(
        vocab_size=vocab,
        hidden_size=hidden,
        intermediate_size=converted[first + "mlp.down_proj.weight"].shape[1],
        num_hidden_layers=sum(1 for key in converted if key.endswith(".input_layernorm.weight")),
        num_attention_heads=converted[first + "self_attn.q_proj.weight"].shape[0] // head_dim,
        num_key_value_heads=converted[first + "self_attn.k_proj.weight"].shape[0] // head_dim,
    )
    config.head_dim = head_dim
    config.lm_head = True  # Forge's Qwen3-8B is an encoder and has none; the final norm is on by default

    # the name is how adapters.SPECS knows the model
    class Molmo2_8B(llama.BaseLlama, torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.num_layers = config.num_hidden_layers
            self.model = llama.Llama2_(config)

    # as Forge's loader builds a Qwen3 text encoder: the file's own floating type, int8 layers as they are stored
    floating = utils.weight_dtype(converted)
    quantisation = detect_quantization(converted)
    if quantisation is not None:
        quantisation.pop("layout", None)
    with torch.inference_mode():
        with no_init_weights():
            with using_forge_operations(device=memory_management.cpu, dtype=floating, manual_cast_enabled=True, sd_dtype=floating, extra_dtype=quantisation):
                model = Molmo2_8B()
        load_state_dict(model, converted, log_name=LABEL)
    return model.eval()


def load(path: str):
    """The model of a complete Molmo2-8B file, on the CPU. Raises UnsupportedEncoder with the reason for any other file."""
    import safetensors
    import torch

    found = look(path)
    if found is None:
        raise UnsupportedEncoder(f"{os.path.basename(str(path))} is not a Molmo2 file" if os.path.isfile(str(path)) else f"the writer file {os.path.basename(str(path))} is no longer there")
    if found.problem:
        raise UnsupportedEncoder(found.refusal())
    # Inside inference mode, as Forge creates every model's weights (see writer.py).
    with torch.inference_mode():
        with safetensors.safe_open(path, framework="pt", device="cpu") as handle:
            state = {key: handle.get_tensor(key) for key in handle.keys() if not key.startswith(SKIPPED)}
        try:
            return build(convert(state))
        except (KeyError, ValueError) as problem:
            raise UnsupportedEncoder(f"{found.name} does not have the layout of a {LABEL} file ({problem})") from problem


def tokenizer():
    """The Qwen tokenizer Forge bundles for its own Qwen3 text encoders."""
    global _tokenizer
    if _tokenizer is None:
        from backend import memory_management
        from transformers import AutoTokenizer

        # `backend` itself is a package without a file of its own, so one of its modules gives the folder
        root = os.path.join(os.path.dirname(os.path.abspath(memory_management.__file__)), "huggingface")
        for owner, repository in _TOKENIZERS:
            folder = os.path.join(root, owner, repository, "tokenizer")
            if os.path.isdir(folder):
                _tokenizer = AutoTokenizer.from_pretrained(folder)
                break
        else:
            raise UnsupportedEncoder(f"{LABEL} needs the Qwen tokenizer that Forge Neo bundles, and it is not in {root}")
    return _tokenizer
