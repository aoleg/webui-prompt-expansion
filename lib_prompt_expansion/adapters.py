"""Which text encoders can write, and how to talk to each: chat format, stop tokens, output head."""

import re
from dataclasses import dataclass
from typing import Any, Optional

from .generator import UnsupportedEncoder

QWEN_STOP_IDS = (151645, 151643)  # <|im_end|>, <|endoftext|>
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
    tied: Optional[bool]  # None = read it from the stack's config (lm_head)
    can_think: bool
    stop_ids: tuple[int, ...] = QWEN_STOP_IDS
    pad_id: int = QWEN_PAD_ID


# keyed by the class name of the encoder in backend/nn/llm/llama.py
SPECS = {
    "Qwen3_4B": Spec("Qwen3-4B", tied=True, can_think=True),
    "Qwen3VL": Spec("Qwen3-VL", tied=None, can_think=False),
}

NOT_YET = {
    "Qwen3_8B": "Qwen3-8B needs its output head, which Forge does not load",
    "Qwen25_7BVLI": "Qwen2.5-VL needs its output head, which Forge does not load",
    "Ministral3_3B": "Ministral 3 is not supported yet",
    "Qwen35_9B": "Qwen3.5 is not supported yet",
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

    def prompt(self, system: str, user: str, thinking: bool = False) -> str:
        think = (True if thinking else False) if self.can_think else None
        return chatml(plain(system), plain(user), think)

    def encode(self, text: str) -> list[int]:
        return list(self.tokenizer(text, add_special_tokens=False)["input_ids"])

    def decode(self, ids: list[int]) -> str:
        return self.tokenizer.decode(ids, skip_special_tokens=False)


def _engines(sd_model):
    for name, value in getattr(sd_model, "__dict__", {}).items():
        if name.startswith("text_processing_engine") and value is not None:
            yield value


def find_adapter(sd_model) -> Adapter:
    """The adapter for the loaded checkpoint's text encoder. Raises UnsupportedEncoder with the reason."""
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
        clip = getattr(getattr(sd_model, "forge_objects", None), "clip", None)
        return Adapter(label, transformer, tokenizer, tied, spec.can_think, spec.stop_ids, getattr(clip, "patcher", None), spec.pad_id)
    raise UnsupportedEncoder("; ".join(reasons) or "this checkpoint's text encoder cannot write text")


def status(sd_model) -> tuple[bool, str]:
    """For the UI: whether expansion works with the loaded checkpoint, and one line saying so."""
    if sd_model is None:
        return False, "No checkpoint is loaded."
    try:
        return True, f"Writer: {find_adapter(sd_model).label} (this checkpoint's text encoder)."
    except UnsupportedEncoder as reason:
        return False, f"Not available for this checkpoint: {reason}."
