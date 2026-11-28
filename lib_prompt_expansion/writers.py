"""Writer files: which files of the text encoder folders can write, and how one is loaded as a model of its own.

A text encoder file is a chat LLM without its vision tower in use. Any of them that belongs to a family this extension
can write with is offered as a writer, whatever checkpoint is loaded: Qwen3-4B and 8B, Qwen3-VL 4B and 8B, Qwen3.5 2B,
4B and 9B, and Molmo2-8B (molmo.py). T5, CLIP and the families that are not supported yet are passed over.

A file is recognised by its tensors, never by its name: the vocabulary and width of the input table, the kind of its
layers (Qwen3.5 has linear-attention layers, Qwen3 has q/k norms), whether a vision tower rides along (that is what
tells Qwen3-VL from Qwen3), and the number of layers. Only the header is read, and what it says is remembered:
in memory for the session, and in Forge's own disk cache across restarts (use_store), keyed by the file's time and
size, so a folder is read once.

The model is built on Forge Neo's own layer stacks, as its loader builds a text encoder, but without the vision
tower and with the output head where the file has one: the file's tensors under their Forge names, the int8, fp8 and
4-bit layouts as they are stored.
"""

import json
import os
import struct
from dataclasses import dataclass
from typing import Iterable, Optional

from . import molmo
from .generator import UnsupportedEncoder

SCHEMA = 2  # of what is remembered about a file; a change of the rules here must raise it
STORE_KEY = "files"


@dataclass(frozen=True)
class Family:
    key: str
    label: str
    spec: str  # the entry of adapters.SPECS: chat format and reasoning
    vocab: int
    hidden: int
    layers: int
    tied: bool  # the output table is the input table
    config: str = ""  # the configuration class in backend.nn.llm.llama
    tokenizer: str = "qwen"  # which of Forge's bundled tokenizers
    linear: bool = False  # a Qwen3.5 stack: linear-attention layers, every fourth one full attention


QWEN, QWEN35 = 151936, 248320
FAMILIES = {
    family.key: family
    for family in (
        Family("qwen3_4b", "Qwen3-4B", "Qwen3_4B", QWEN, 2560, 36, True, "Qwen3_4BConfig"),
        Family("qwen3_8b", "Qwen3-8B", "Qwen3_8B", QWEN, 4096, 36, False, "Qwen3_8BConfig"),
        Family("qwen3vl_4b", "Qwen3-VL 4B", "Qwen3VL", QWEN, 2560, 36, True, "Qwen3VL_4BConfig"),
        Family("qwen3vl_8b", "Qwen3-VL 8B", "Qwen3VL", QWEN, 4096, 36, False, "Qwen3VL_8BConfig"),
        # Qwen3.5: the 2B and 4B models tie their tables (Qwen's own configurations), the 9B one does not
        Family("qwen35_2b", "Qwen3.5 2B", "Qwen35_9B", QWEN35, 2048, 24, True, "Qwen35_9BConfig", "qwen35", True),
        Family("qwen35_4b", "Qwen3.5 4B", "Qwen35_9B", QWEN35, 2560, 32, True, "Qwen35_9BConfig", "qwen35", True),
        Family("qwen35_9b", "Qwen3.5 9B", "Qwen35_9B", QWEN35, 4096, 32, False, "Qwen35_9BConfig", "qwen35", True),
        Family("molmo2_8b", molmo.LABEL, "Molmo2_8B", molmo.VOCAB, molmo.HIDDEN, molmo.LAYERS, False),
    )
}

# where a file keeps its language model: as Qwen publishes a vision-language model, as Forge and ComfyUI save a text
# encoder, and bare
PREFIXES = ("model.language_model.", "model.", "")
VISION = ("model.visual.", "visual.")
CONNECTOR = "text_connector.proj.weight"  # Pruna's connector behind a Qwen3-VL 4B
HEAD = "lm_head.weight"
# Forge's bundled tokenizers, by vocabulary; the Qwen3 ones hold the same vocabulary and merges
TOKENIZERS = {
    "qwen": (("black-forest-labs", "FLUX.2-klein-9B"), ("black-forest-labs", "FLUX.2-klein-4B"), ("Tongyi-MAI", "Z-Image-Turbo")),
    "qwen35": (("OpenMOSS-Team", "MOSS-Image-SFT"),),
}


@dataclass(frozen=True)
class Found:
    """A file of a family that can write, as its header shows it."""

    path: str
    family: str
    label: str
    layers: int
    head: bool = True  # the file holds what turns states into words: its own output head, or tied tables
    problem: str = ""  # why this file cannot write, as the rest of a sentence that begins with its name; empty when it can
    brief: str = ""  # the same in a few words, for a list

    @property
    def name(self) -> str:
        return os.path.basename(self.path)

    @property
    def choice(self) -> str:
        """How the file is named in the panel's list of writers."""
        return f"{self.label}: {self.name}"

    def refusal(self) -> str:
        return f"{self.name} {self.problem}"


# --- what a header says ---------------------------------------------------------------------------------------


def prefix_of(tensors) -> Optional[str]:
    for prefix in PREFIXES:
        if prefix + "embed_tokens.weight" in tensors and prefix + "layers.0.input_layernorm.weight" in tensors:
            return prefix
    return None


def describe(tensors: dict, metadata: Optional[dict] = None) -> Optional[dict]:
    """What a file's tensors make it: {"family", "label", "layers", "head", "problem", "brief"}, or None for a file
    of no family that can write. tensors: name to {"shape": [...]}, a safetensors header."""
    if molmo.is_molmo(tensors):
        problem, brief = molmo.problem_of(tensors)
        return {"family": "molmo2_8b", "label": molmo.LABEL, "layers": molmo.molmo_layers(tensors), "head": True, "problem": problem, "brief": brief}
    prefix = prefix_of(tensors)
    if prefix is None:
        return None
    shape = tensors[prefix + "embed_tokens.weight"].get("shape") or []
    if len(shape) != 2:
        return None
    vocab, hidden = shape
    layer = prefix + "layers.0."
    linear = layer + "linear_attn.in_proj_qkv.weight" in tensors
    if not linear and not (layer + "self_attn.q_norm.weight" in tensors and layer + "mlp.gate_proj.weight" in tensors):
        return None  # no q/k norms: Qwen2.5, Llama, Mistral, Gemma
    # a vision tower in the file is what tells Qwen3-VL from Qwen3: the two differ in how they count positions
    vision = any(key.startswith(VISION) for key in tensors)
    family = None
    for item in FAMILIES.values():
        if item.config and item.linear == linear and (item.vocab, item.hidden) == (vocab, hidden) and (linear or (item.spec == "Qwen3VL") == vision):
            family = item
            break
    if family is None:
        return None  # a size no supported model has: Qwen3 0.6B, the 32B encoder of MiniMax H3
    label = family.label
    if CONNECTOR in tensors and not ((metadata or {}).get("built_by") or (metadata or {}).get("built_from")):
        label += ", Pruna's retrained"  # P-Image-OSS's own encoder; a file built here holds the stock model
    stem = len(prefix + "layers.")
    present = {int(number) for key in tensors if key.startswith(prefix + "layers.") and (number := key[stem:].split(".", 1)[0]).isdigit()}
    layers = max(present) + 1 if present else 0
    head = family.tied or HEAD in tensors
    problem = brief = ""
    if layers > family.layers:
        return None
    if len(present) != layers:
        problem, brief = f"is not a whole {family.label}: {layers - len(present)} of its first {layers} layers are missing", f"a {family.label} with layers missing"
    elif layers < family.layers:
        problem = f"is a cut-down {family.label}: it holds {layers} of the model's {family.layers} layers, and text written without the last layers is a loop of garbage"
        brief = f"a {family.label} cut down to {layers} of {family.layers} layers"
    elif prefix + "norm.weight" not in tensors:
        problem, brief = f"is a {family.label} without its final norm, so it cannot write text", f"a {family.label} without its final norm"
    return {"family": family.key, "label": label, "layers": layers, "head": head, "problem": problem, "brief": brief}


def read_header(path: str) -> tuple[dict, dict]:
    """(tensors, metadata) of a .safetensors file. Only the header is read."""
    with open(path, "rb") as handle:
        (size,) = struct.unpack("<Q", handle.read(8))
        if size > 100_000_000:
            raise ValueError("not a safetensors header")
        raw = handle.read(size)
    # nearly every file is told apart here, without parsing: a T5 or a CLIP has no table of this name and no head
    if b"embed_tokens.weight" not in raw and molmo.EMBEDDING.encode("ascii") not in raw and HEAD.encode("ascii") not in raw:
        return {}, {}
    tensors = json.loads(raw)
    metadata = tensors.pop("__metadata__", None) or {}
    return tensors, metadata


def _entry(path: str) -> dict:
    """What is remembered about a file: {"said": what describe() says or None, "head": the shape of an output head
    in it or None}."""
    tensors, metadata = read_header(path)
    shape = (tensors.get(HEAD) or {}).get("shape")
    return {"said": describe(tensors, metadata) if tensors else None, "head": list(shape) if shape else None}


# --- remembering ----------------------------------------------------------------------------------------------

_looked: dict[tuple, dict] = {}  # this session: (path, time, size) to the file's entry
_store = None  # across restarts: a mapping with get() and item assignment (Forge's disk cache), or None
_table: Optional[dict] = None  # the store's content: path to [time, size, the file's entry]
_dirty = False
headers_read = 0  # how many file headers were read from disk in this session, for the tests and the curious


def use_store(store) -> None:
    """Keep what the headers say across restarts in `store` (modules.cache.cache(...) in the web UI)."""
    global _store, _table, _dirty
    _store, _table, _dirty = store, None, False


def _remembered() -> dict:
    global _table
    if _table is None:
        _table = {}
        if _store is not None:
            try:
                kept = _store.get(STORE_KEY) or {}
                if kept.get("schema") == SCHEMA and isinstance(kept.get("files"), dict):
                    _table = dict(kept["files"])
            except Exception:  # noqa: BLE001 - a cache that cannot be read is an empty cache
                _table = {}
    return _table


def flush() -> None:
    """Write what was learned since the last call to the store. Files that are gone are forgotten."""
    global _dirty
    if _store is None or not _dirty or _table is None:
        return
    _dirty = False
    try:
        _store[STORE_KEY] = {"schema": SCHEMA, "files": {path: entry for path, entry in _table.items() if os.path.isfile(path)}}
    except Exception:  # noqa: BLE001 - remembering is a convenience
        pass


def forget() -> None:
    """Drop everything remembered in this session (the store keeps its own)."""
    global _table
    _looked.clear()
    _table = None


def _known(path: str) -> Optional[dict]:
    """The remembered entry of a file as it is now on disk (see _entry), read from its header when it is new or
    has changed; None for a file that is not there."""
    global _dirty, headers_read
    try:
        key = os.path.normcase(os.path.abspath(path))
        stamp = (key, os.path.getmtime(path), os.path.getsize(path))
    except (OSError, TypeError):
        return None
    if stamp not in _looked:
        table = _remembered()
        kept = table.get(key)
        if isinstance(kept, (list, tuple)) and len(kept) == 3 and (kept[0], kept[1]) == stamp[1:] and isinstance(kept[2], dict):
            entry = kept[2]
        else:
            entry = {"said": None, "head": None}
            if str(path).lower().endswith((".safetensors", ".sft")):
                try:
                    headers_read += 1
                    entry = _entry(path)
                except (OSError, ValueError, struct.error, AttributeError, TypeError, KeyError):
                    pass
            table[key] = [stamp[1], stamp[2], entry]
            _dirty = True
        _looked[stamp] = entry
    return _looked[stamp]


def look(path: str) -> Optional[Found]:
    """What a file is, if it belongs to a family that can write; None for any other file."""
    entry = _known(path)
    said = entry.get("said") if entry else None
    try:
        return Found(path=str(path), **said) if said else None
    except TypeError:  # remembered in another shape
        return None


def head_shape(path: str) -> Optional[list]:
    """The shape of the output head (lm_head.weight) a file holds, or None."""
    entry = _known(path)
    return entry.get("head") if entry else None


def files(paths: Iterable[str]) -> list[Found]:
    """The files among `paths` that belong to a family that can write, each once, sorted by model and name."""
    out, seen = [], set()
    for path in paths:
        found = look(path) if path else None
        key = os.path.normcase(os.path.abspath(found.path)) if found is not None else None
        if found is None or key in seen:
            continue
        seen.add(key)
        out.append(found)
    flush()
    return sorted(out, key=lambda item: (item.label.lower(), item.name.lower()))


# --- the model ------------------------------------------------------------------------------------------------


def _config(family: Family, state: dict):
    """The stack's configuration: the family's own kind (how positions are counted, which norms), with the sizes
    read off the tensors. For a published model they are the configuration's own; reading them also covers the
    Qwen3.5 sizes Forge has no configuration for, and lets a miniature model stand in for a real one in a test."""
    from backend.nn.llm import llama

    config = getattr(llama, family.config)()
    first = "model.layers.0."
    # the first layer with ordinary attention: every layer of Qwen3, every fourth of Qwen3.5
    full = next((f"model.layers.{index}." for index in range(family.layers) if f"model.layers.{index}.self_attn.q_proj.weight" in state), None)
    if full is None:
        raise ValueError("no attention layer")
    head_dim = state[full + "self_attn.q_norm.weight"].shape[0]
    config.vocab_size, config.hidden_size = state["model.embed_tokens.weight"].shape
    config.intermediate_size = state[first + "mlp.down_proj.weight"].shape[1]
    config.num_hidden_layers = family.layers
    config.head_dim = head_dim
    # Qwen3.5's query projection also gives the gate of each head: twice the rows
    config.num_attention_heads = state[full + "self_attn.q_proj.weight"].shape[0] // (head_dim * (2 if family.linear else 1))
    config.num_key_value_heads = state[full + "self_attn.k_proj.weight"].shape[0] // head_dim
    if family.linear:
        value_dim = state[first + "linear_attn.in_proj_z.weight"].shape[0]
        value_head_dim = state[first + "linear_attn.norm.weight"].shape[0]
        key_dim = (state[first + "linear_attn.in_proj_qkv.weight"].shape[0] - value_dim) // 2
        config.linear_value_head_dim = value_head_dim
        config.linear_num_value_heads = value_dim // value_head_dim
        # key heads are as wide as value heads in every published size (128)
        config.linear_key_head_dim = value_head_dim
        config.linear_num_key_heads = key_dim // value_head_dim
        config.conv_kernel_size = state[first + "linear_attn.conv1d.weight"].shape[-1]
        config.layer_types = llama._qwen35_layer_types(family.layers)
        kinds = ["full_attention" if f"model.layers.{index}.self_attn.q_proj.weight" in state else "linear_attention" for index in range(family.layers)]
        if kinds != config.layer_types:
            raise ValueError("the layers are not three of linear attention and one of full attention")
    config.lm_head = "model.lm_head.weight" in state  # Forge builds none for most encoders; a writer wants it
    return config


def build(family: Family, state: dict, floating=None):
    """Forge Neo's layer stack of this family holding `state` (tensors under Forge's names, "model..."). On the CPU,
    as Forge's loader leaves a text encoder; no vision tower.

    floating: the floating-point type to build in; by default the one most of the tensors have.
    """
    import torch
    from backend import memory_management, utils
    from backend.nn.llm import llama
    from backend.operations import using_forge_operations
    from backend.state_dict import detect_quantization, load_state_dict
    from transformers.modeling_utils import no_init_weights

    config = _config(family, state)
    stack = llama.Qwen35Transformer if family.linear else llama.Llama2_

    def __init__(self):
        torch.nn.Module.__init__(self)
        self.num_layers = config.num_hidden_layers
        self.model = stack(config)

    # named after Forge's own class of the family: its log lines then say what is being moved
    wrapper = type(family.spec, (llama.BaseLlama, torch.nn.Module), {"__init__": __init__})

    # as Forge's loader builds a text encoder: the file's own floating type, quantised layers as they are stored
    floating = floating or utils.weight_dtype(state)
    quantisation = detect_quantization(state)
    if quantisation is not None:
        quantisation.pop("layout", None)
    with torch.inference_mode():
        with no_init_weights():
            with using_forge_operations(device=memory_management.cpu, dtype=floating, manual_cast_enabled=True, sd_dtype=floating, extra_dtype=quantisation):
                model = wrapper()
        load_state_dict(model, state, log_name=family.label)
    return model.eval()


def rename(state: dict) -> dict:
    """The language model's tensors of a text encoder file under Forge's names; the vision tower, a connector and
    everything else stay behind. Nothing is copied."""
    prefix = prefix_of(state)
    if prefix is None:
        raise ValueError("no language model")
    out = {}
    for key, value in state.items():
        if key.startswith(VISION):
            continue
        if key.startswith("lm_head."):
            out["model." + key] = value
        elif prefix and key.startswith(prefix):
            out["model." + key[len(prefix) :]] = value
        elif not prefix and key.startswith(("embed_tokens.", "layers.", "norm.")):
            out["model." + key] = value
    return out


def load(path: str):
    """The model of a writer file, on the CPU. Raises UnsupportedEncoder with the reason for any other file."""
    import torch
    from backend import utils
    from backend.state_dict import convert_quantization

    found = look(path)
    name = os.path.basename(str(path))
    if found is None:
        raise UnsupportedEncoder(f"{name} is not a model this extension can write with" if os.path.isfile(str(path)) else f"the writer file {name} is no longer there")
    if found.problem:
        raise UnsupportedEncoder(found.refusal())
    if found.family == "molmo2_8b":
        return molmo.load(path)
    family = FAMILIES[found.family]
    # Inside inference mode, as Forge creates every model's weights (see writer.py). The file is read as Forge reads
    # a text encoder: every tensor as a view of the file, then the older ways of marking a quantised layer
    # brought to the one its layers understand.
    with torch.inference_mode():
        state, metadata = utils.load_torch_file(str(path), return_metadata=True)
        state, _ = convert_quantization(state, metadata)
        floating = utils.weight_dtype(state)
        try:
            return build(family, rename(state), floating)
        except (KeyError, ValueError) as problem:
            raise UnsupportedEncoder(f"{name} does not have the layout of a {family.label} file ({problem})") from problem


_tokenizers: dict = {}


def tokenizer(kind: str = "qwen"):
    """The tokenizer Forge bundles for its own text encoders of this vocabulary."""
    if kind not in _tokenizers:
        from backend import memory_management
        from transformers import AutoTokenizer

        # `backend` itself is a package without a file of its own, so one of its modules gives the folder
        root = os.path.join(os.path.dirname(os.path.abspath(memory_management.__file__)), "huggingface")
        for owner, repository in TOKENIZERS[kind]:
            folder = os.path.join(root, owner, repository, "tokenizer")
            if os.path.isdir(folder):
                _tokenizers[kind] = AutoTokenizer.from_pretrained(folder)
                break
        else:
            raise UnsupportedEncoder(f"the tokenizer that Forge Neo bundles for this model is not in {root}")
    return _tokenizers[kind]
