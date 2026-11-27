"""Which text encoders can write, and how to talk to each: chat format, stop tokens, output head.

The writer is the loaded checkpoint's own text encoder (find_adapter), or a model file of its own that is loaded
for writing only (file_adapter); choose() decides between the two.
"""

import gc
import os
import re
from dataclasses import dataclass
from typing import Any, Callable, Iterable, Optional

from . import molmo
from .fetch import HeadDownload, existing, failure
from .generator import UnsupportedEncoder
from .headfile import find_head, library_files, loaded_files

QWEN_STOP_IDS = (151645, 151643)  # <|im_end|>, <|endoftext|> in the Qwen3 vocabulary
QWEN_PAD_ID = 151643  # what Forge's own engines pad Qwen prompts with

# control tokens a request must not carry into the chat template
_CONTROL = re.compile(r"<\|[A-Za-z0-9_]+\|>|</?think>")


class HeadMissing(UnsupportedEncoder):
    """The encoder can write, but its output head is not on disk; `download` says where it can be had."""

    def __init__(self, text: str, label: str, download: HeadDownload):
        super().__init__(text)
        self.label, self.download = label, download


def plain(text: str) -> str:
    return _CONTROL.sub("", text or "").strip()


def chatml(system: str, user: str, think: Optional[bool]) -> str:
    """Qwen chat format. think: None = the model has no reasoning block, False = closed empty block, True = open."""
    text = ""
    if system:
        text += f"<|im_start|>system\n{system}<|im_end|>\n"
    text += f"<|im_start|>user\n{user}<|im_end|>\n<|im_start|>assistant\n"
    if think is False:
        text += "<think>\n\n</think>\n\n"
    return text


@dataclass(frozen=True)
class Spec:
    label: str
    tied: Optional[bool]  # True: output table = input table. False: separate head. None: read the stack's config.
    think: str  # "never": the model has no reasoning block. "optional": offered in the UI. "closed": always an empty block.
    own_heads: tuple[str, ...] = ()  # headfile.fingerprint of the model's own output head, as published
    head_download: Optional[HeadDownload] = None  # where that head comes from when the encoder file was saved without it


# keyed by the class name of the encoder in backend/nn/llm/llama.py
SPECS = {
    "Qwen3_4B": Spec("Qwen3-4B", tied=True, think="optional"),
    # Qwen publishes this model's head as a shard of its own: one tensor, bf16, 1.24 GB, Apache 2.0. The encoder
    # in the FLUX.2 Klein 9B files is the unmodified Qwen3-8B (weights equal bit for bit), so it is the true head.
    "Qwen3_8B": Spec(
        "Qwen3-8B",
        tied=False,
        think="optional",
        own_heads=("acaf662d0d0b5f25d89228ea13faaa958516262966d4bb032247d6137d9f642f",),
        head_download=HeadDownload(
            repo="Qwen/Qwen3-8B",
            revision="b968826d9c46dd6066d109eabc6255188de91218",
            filename="model-00005-of-00005.safetensors",
            size=1244659840,
            sha256="20c2d6366ab85c90786ccdd829cd2b9e7d30ef3b2ebbb998280e7e4014b542ff",
            save_as="qwen_3_8b_lm_head.safetensors",
            licence="Apache 2.0",
        ),
    ),
    "Qwen3VL": Spec("Qwen3-VL", tied=None, think="never"),
    # Qwen3.5 reasons by default and at great length (1,536 tokens were not enough for one image request on the
    # MOSS-Image encoder), so its reasoning block is always closed and thinking is not offered.
    "Qwen35_9B": Spec("Qwen3.5 9B", tied=False, think="closed"),
    # Molmo2's language model is Qwen3-8B, trained on without a reasoning block: its chat format has none. It is
    # never a checkpoint's text encoder as a whole; molmo.py loads it from its file, with its own head (file_adapter).
    "Molmo2_8B": Spec(molmo.LABEL, tied=False, think="never"),
}

NOT_YET = {
    "Qwen25_7BVLI": "Qwen2.5-VL is not supported yet",
    "Ministral3_3B": "Ministral 3 is not supported yet",
}


@dataclass
class Adapter:
    label: str
    transformer: Any
    tokenizer: Any  # the Hugging Face tokenizer
    tied: bool
    can_think: bool
    stop_ids: tuple[int, ...]
    patcher: Any = None  # Forge's model patcher for the text encoder, to load it on the device
    pad_id: int = QWEN_PAD_ID  # fills the left of shorter rows in a batched pass; never attended to
    head_file: Optional[str] = None  # the file to read a separate output head from
    head_borrowed: bool = False  # the head comes from another model's file, not from the loaded encoder's own
    think: str = "optional"  # see Spec.think
    head_download: Optional[HeadDownload] = None  # see Spec.head_download
    own_file: Optional[str] = None  # the writer is a model of its own, loaded from this file; None: the text encoder

    def prompt(self, system: str, user: str, thinking: bool = False) -> str:
        think = None if self.think == "never" else bool(thinking and self.can_think)
        return chatml(plain(system), plain(user), think)

    def encode(self, text: str) -> list[int]:
        return list(self.tokenizer(text, add_special_tokens=False)["input_ids"])

    def decode(self, ids: list[int]) -> str:
        return self.tokenizer.decode(ids, skip_special_tokens=False)


def _engines(sd_model):
    for name, value in getattr(sd_model, "__dict__", {}).items():
        if name.startswith("text_processing_engine") and value is not None:
            yield value


def _token_id(tokenizer, token: str) -> Optional[int]:
    try:
        found = tokenizer.convert_tokens_to_ids(token)
    except Exception:  # noqa: BLE001 - a tokenizer without the method or the token
        return None
    unknown = getattr(tokenizer, "unk_token_id", None)
    return found if isinstance(found, int) and found >= 0 and found != unknown else None


def find_adapter(sd_model, head_files: Optional[Iterable[str]] = None, library: Optional[Iterable[str]] = None, fetch: Optional[Callable] = None) -> Adapter:
    """The adapter for the loaded checkpoint's text encoder. Raises UnsupportedEncoder with the reason.

    head_files: where to look for a separate output head first; by default the files Forge loaded the model from.
    library: other files a head of the same size may be borrowed from; by default every text encoder file Forge lists.
    fetch: called as fetch(download, loaded files, encoder name, what happens otherwise) when the model's own head
        is not on disk and can be downloaded; returns the path of the downloaded file or None. Without it nothing is
        downloaded (the panel's status line must never start a download).
    """
    reasons, missing = [], None
    for engine in _engines(sd_model):
        transformer = getattr(getattr(engine, "text_encoder", None), "transformer", None)
        tokenizer = getattr(getattr(engine, "tokenizer", None), "tokenizer", None)
        if transformer is None or tokenizer is None:
            continue
        # An extension may subclass one of Forge's encoders (the P-H3-Image encoder is a Qwen3VL with a connector on
        # top), so the class is looked up along its ancestry.
        name = next((cls.__name__ for cls in type(transformer).__mro__ if cls.__name__ in SPECS or cls.__name__ in NOT_YET), type(transformer).__name__)
        spec = SPECS.get(name)
        if spec is None:
            reasons.append(NOT_YET.get(name, f"{name} cannot write text"))
            continue
        stack = getattr(transformer, "model", None)
        config = getattr(stack, "config", None)
        if config is None:
            reasons.append(f"{type(transformer).__name__} cannot write text")
            continue
        tied = spec.tied if spec.tied is not None else not getattr(config, "lm_head", False)
        label = spec.label
        if name == "Qwen3VL":
            label += {2560: " 4B", 4096: " 8B"}.get(config.hidden_size, "")
        if getattr(stack, "norm", None) is None or getattr(config, "final_norm", True) is False:
            # MiniMax H3's own encoder: Qwen3-VL 32B cut off after 50 of its 64 layers, with no final norm
            reasons.append(f"{label} is a truncated encoder (no final layers), so it cannot write text".replace("  ", " "))
            continue

        head_file, head_borrowed = None, False
        if not tied and getattr(transformer.model, "lm_head", None) is None:
            loaded = list(loaded_files() if head_files is None else head_files)
            shelf = list(library_files() if library is None else library)
            if library is None and spec.head_download is not None:
                shelf += existing(spec.head_download, loaded)  # downloaded in this session; Forge's list is older
            head_file, head_borrowed = find_head(config.vocab_size, config.hidden_size, loaded, shelf, spec.own_heads)
            if (head_file is None or head_borrowed) and spec.head_download is not None and fetch is not None:
                otherwise = f"Until then {label} writes through the head in {os.path.basename(head_file)}, which is clearly worse." if head_file else "Until then the prompts are used as typed."
                fetched = fetch(spec.head_download, loaded, label, otherwise)
                if fetched:
                    head_file, head_borrowed = fetched, False
            if head_file is None:
                source = f"; put {spec.head_download.describe()} into a text encoder folder" if spec.head_download else ""
                reasons.append(f"{label} needs its output head: the loaded encoder file has none, and no other text encoder file holds one of that size ({config.vocab_size} x {config.hidden_size}){source}")
                if spec.head_download is not None:
                    missing = (label, spec.head_download)
                continue

        # Through a borrowed head the reasoning mode does not work: on Qwen3-8B with the head of Qwen3-VL 8B the
        # opening tag comes out as a stray character and some answers never leave the reasoning block (with its own
        # head: four of four reason and answer cleanly). The block is then always closed, and thinking not offered.
        think = "closed" if head_borrowed and spec.think == "optional" else spec.think

        # the vocabulary decides the ids: Qwen3 and Qwen3.5 number their control tokens differently
        end, stop = _token_id(tokenizer, "<|im_end|>"), _token_id(tokenizer, "<|endoftext|>")
        stop_ids = tuple(token for token in (end, stop) if token is not None) or QWEN_STOP_IDS
        clip = getattr(getattr(sd_model, "forge_objects", None), "clip", None)
        return Adapter(label, transformer, tokenizer, tied, think == "optional", stop_ids, getattr(clip, "patcher", None), stop if stop is not None else QWEN_PAD_ID, head_file, head_borrowed, think, spec.head_download)
    if missing is not None:
        raise HeadMissing("; ".join(reasons), *missing)
    raise UnsupportedEncoder("; ".join(reasons) or "this checkpoint's text encoder cannot write text")


# --- a writer of its own --------------------------------------------------------------------------------------

_own: Optional[tuple] = None  # (the file as it was when loaded, its Adapter): the one writer file that is loaded


def _stamp(path: str) -> Optional[tuple]:
    try:
        return os.path.normcase(os.path.abspath(path)), os.path.getmtime(path), os.path.getsize(path)
    except (OSError, TypeError):
        return None


def release_file_adapter() -> None:
    """Let go of the loaded writer file. Forge's memory management drops a model that nothing holds any more."""
    global _own
    if _own is None:
        return
    _own = None
    gc.collect()
    try:
        from backend import memory_management

        memory_management.soft_empty_cache()
    except Exception:  # noqa: BLE001 - the memory comes back at the next load at the latest
        pass


def file_adapter(path: str) -> Adapter:
    """The adapter of a writer that is a model file of its own. Raises UnsupportedEncoder with the reason.

    The file is read at the first call, and the model is kept for the next job, under Forge's memory management
    like any text encoder: it is moved to the GPU for a writing session and gives way when Forge needs the memory.
    Another file takes its place.
    """
    global _own
    stamp = _stamp(path)
    if _own is not None and stamp is not None and _own[0] == stamp:
        return _own[1]
    release_file_adapter()  # the old model goes before the new one is read
    transformer = molmo.load(path)
    tokenizer = molmo.tokenizer()

    from backend import memory_management
    from backend.patcher.base import ModelPatcher

    patcher = ModelPatcher(transformer, load_device=memory_management.text_encoder_device(), offload_device=memory_management.text_encoder_offload_device())
    spec = SPECS[type(transformer).__name__]
    end, stop = _token_id(tokenizer, "<|im_end|>"), _token_id(tokenizer, "<|endoftext|>")
    stop_ids = tuple(token for token in (end, stop) if token is not None) or QWEN_STOP_IDS
    adapter = Adapter(spec.label, transformer, tokenizer, bool(spec.tied), spec.think == "optional", stop_ids, patcher, stop if stop is not None else QWEN_PAD_ID, think=spec.think, own_file=str(path))
    _own = (stamp, adapter)
    return adapter


def loaded_writer_file(loaded: Optional[Iterable[str]] = None) -> tuple[Optional[str], Optional[str]]:
    """(a file among the loaded ones that is a writer of its own, why such a file cannot write): at most one is set.

    MiniMax-H3-Image can take Molmo2-8B as its text encoder, through the H3 extension, which loads the layers up to
    the one its projection reads. That encoder cannot write; the file it came from can, if it is complete.
    """
    usable, refused = molmo.files(loaded_files() if loaded is None else loaded)
    if usable:
        return usable[0].path, None
    return None, (refused[0].refusal() if refused else None)


def choose(sd_model, writer_file: Optional[str] = None, fetch: Optional[Callable] = None, head_files: Optional[Iterable[str]] = None) -> Adapter:
    """The adapter that writes this job: the chosen writer file, else the loaded checkpoint's text encoder, else
    the text encoder's own file where that is a writer (see loaded_writer_file). Raises UnsupportedEncoder.

    A writer file that is not used any more is let go of here.
    """
    if writer_file:
        return file_adapter(writer_file)
    try:
        adapter = find_adapter(sd_model, head_files=head_files, fetch=fetch)
    except HeadMissing:
        raise
    except UnsupportedEncoder:
        own, problem = loaded_writer_file(head_files)
        if own is not None:
            return file_adapter(own)
        if problem is not None:
            raise UnsupportedEncoder(problem) from None
        raise
    release_file_adapter()
    return adapter


OWN_MODEL_NOTE = "a model of its own: it is loaded for the writing, next to the checkpoint's text encoder"


def _own_head(download: HeadDownload) -> str:
    """For the status line: what will happen about a head that can be downloaded."""
    if failure(download) is not None:
        return f"Its own head could not be downloaded ({failure(download)}); the console says how to get it by hand, and the refresh button lets the next job try again."
    return f"Its own head ({download.describe()}, {download.gigabytes()}) is downloaded when you first generate with this panel on."


def file_status(path: str) -> tuple[bool, str]:
    """For the UI: whether this writer file can write, and one line saying so. Reads the file's header only."""
    found = molmo.look(path)
    if found is None:
        return False, f"Not available: {os.path.basename(str(path))} is {'not a model this extension can write with' if os.path.isfile(str(path)) else 'no longer there'}."
    if found.problem:
        return False, f"Not available: {found.refusal()}."
    return True, f"Writer: {molmo.LABEL}, from {found.name} ({OWN_MODEL_NOTE})."


def status(sd_model, writer_file: Optional[str] = None) -> tuple[bool, str]:
    """For the UI: whether expansion works with the chosen writer, and one line saying so. Downloads and loads nothing."""
    if writer_file:
        return file_status(writer_file)
    if sd_model is None:
        return False, "No checkpoint is loaded."
    try:
        adapter = find_adapter(sd_model)
        borrowed = ""
        if adapter.head_borrowed:
            borrowed = f" Its file has no output head; the one in {os.path.basename(adapter.head_file)} is used."
            if adapter.head_download is not None:
                borrowed += " " + _own_head(adapter.head_download)
        return True, f"Writer: {adapter.label} (this checkpoint's text encoder).{borrowed}"
    except HeadMissing as reason:
        ready = failure(reason.download) is None
        return ready, f"{'Writer' if ready else 'Not available for this checkpoint'}: {reason.label} (this checkpoint's text encoder). Its file has no output head. {_own_head(reason.download)}"
    except UnsupportedEncoder as reason:
        own, problem = loaded_writer_file()
        if own is not None:
            return True, f"Writer: {molmo.LABEL}, from this checkpoint's text encoder file {os.path.basename(own)}. The encoder as loaded holds only the layers the image model reads, so the file is loaded once more, whole, for the writing."
        return False, f"Not available for this checkpoint: {problem or reason}."
