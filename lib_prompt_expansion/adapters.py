"""Which text encoders can write, and how to talk to each: chat format, stop tokens, output head."""

import re
from dataclasses import dataclass
from typing import Any, Iterable, Optional

from .generator import UnsupportedEncoder
from .headfile import find_head_file, loaded_files

QWEN_STOP_IDS = (151645, 151643)  # <|im_end|>, <|endoftext|> in the Qwen3 vocabulary
QWEN_PAD_ID = 151643  # what Forge's own engines pad Qwen prompts with

# control tokens a request must not carry into the chat template
_CONTROL = re.compile(r"<\|[A-Za-z0-9_]+\|>|</?think>")


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


# keyed by the class name of the encoder in backend/nn/llm/llama.py
SPECS = {
    "Qwen3_4B": Spec("Qwen3-4B", tied=True, think="optional"),
    "Qwen3_8B": Spec("Qwen3-8B", tied=False, think="optional"),
    "Qwen3VL": Spec("Qwen3-VL", tied=None, think="never"),
    # Qwen3.5 reasons by default and at great length (1,536 tokens were not enough for one image request on the
    # MOSS-Image encoder), so its reasoning block is always closed and thinking is not offered.
    "Qwen35_9B": Spec("Qwen3.5 9B", tied=False, think="closed"),
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
    head_file: Optional[str] = None  # the encoder file to read a separate output head from
    think: str = "optional"  # see Spec.think

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


def find_adapter(sd_model, head_files: Optional[Iterable[str]] = None) -> Adapter:
    """The adapter for the loaded checkpoint's text encoder. Raises UnsupportedEncoder with the reason.

    head_files: where to look for a separate output head; by default the files Forge loaded the model from.
    """
    reasons = []
    for engine in _engines(sd_model):
        transformer = getattr(getattr(engine, "text_encoder", None), "transformer", None)
        tokenizer = getattr(getattr(engine, "tokenizer", None), "tokenizer", None)
        if transformer is None or tokenizer is None:
            continue
        name = type(transformer).__name__
        spec = SPECS.get(name)
        if spec is None:
            reasons.append(NOT_YET.get(name, f"{name} cannot write text"))
            continue
        config = transformer.model.config
        tied = spec.tied if spec.tied is not None else not getattr(config, "lm_head", False)
        label = spec.label
        if name == "Qwen3VL":
            label += " 8B" if config.hidden_size == 4096 else " 4B"

        head_file = None
        if not tied and getattr(transformer.model, "lm_head", None) is None:
            head_file = find_head_file(loaded_files() if head_files is None else head_files, config.vocab_size, config.hidden_size)
            if head_file is None:
                reasons.append(f"{label} needs its output head, and none of the loaded files contains it")
                continue

        # the vocabulary decides the ids: Qwen3 and Qwen3.5 number their control tokens differently
        end, stop = _token_id(tokenizer, "<|im_end|>"), _token_id(tokenizer, "<|endoftext|>")
        stop_ids = tuple(token for token in (end, stop) if token is not None) or QWEN_STOP_IDS
        clip = getattr(getattr(sd_model, "forge_objects", None), "clip", None)
        return Adapter(label, transformer, tokenizer, tied, spec.think == "optional", stop_ids, getattr(clip, "patcher", None), stop if stop is not None else QWEN_PAD_ID, head_file, spec.think)
    raise UnsupportedEncoder("; ".join(reasons) or "this checkpoint's text encoder cannot write text")


def status(sd_model) -> tuple[bool, str]:
    """For the UI: whether expansion works with the loaded checkpoint, and one line saying so."""
    if sd_model is None:
        return False, "No checkpoint is loaded."
    try:
        return True, f"Writer: {find_adapter(sd_model).label} (this checkpoint's text encoder)."
    except UnsupportedEncoder as reason:
        return False, f"Not available for this checkpoint: {reason}."
