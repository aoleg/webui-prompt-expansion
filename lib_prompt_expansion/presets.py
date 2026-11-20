"""Expansion prompts kept as text files: presets/*.txt ship with the extension, presets/user/*.txt are the user's.

File name: "NN Name.txt"; the number only sets the order. Optional first lines: "# words: 150" gives the default
length, and "# begin: request" says that the expansion starts with the request itself and continues it (the request
is then put at the start of the encoder's answer, so the instruction need not ask for it). "{words}" in the text is
replaced by the length when a request is written.
"""

import math
import re
from dataclasses import dataclass
from pathlib import Path

PRESET_DIR = Path(__file__).resolve().parent.parent / "presets"
FALLBACK_WORDS = 150
TOKENS_PER_WORD = 1.6  # English prose in the Qwen and Gemma tokenizers, punctuation included
SLACK = 2.0  # the length is a guide: the model may write up to twice it before the token limit cuts in

_HEADER = re.compile(r"^#\s*words\s*:\s*(\d+)\s*$", re.I)
_BEGIN = re.compile(r"^#\s*begin\s*:\s*(\w+)\s*$", re.I)
_ORDER = re.compile(r"^\d+[\s_.-]*")


@dataclass(frozen=True)
class Preset:
    name: str
    text: str
    words: int  # 0 = the preset sets no length
    begin: bool = False  # the expansion starts with the request itself, word for word


def read_preset(path: Path) -> Preset:
    words, begin, lines = 0, False, []
    for line in path.read_text(encoding="utf-8-sig").splitlines():
        header = _HEADER.match(line.strip())
        start = _BEGIN.match(line.strip())
        if header and not lines:
            words = int(header.group(1))
        elif start and not lines:
            begin = start.group(1).lower() in ("request", "yes", "true", "on")
        elif line.lstrip().startswith("#") and not lines:
            continue
        else:
            lines.append(line.rstrip())
    name = _ORDER.sub("", path.stem).strip() or path.stem
    return Preset(name, "\n".join(lines).strip(), words, begin)


def load(folder: Path = PRESET_DIR) -> list[Preset]:
    """Shipped presets in file order, then the user's. A user preset replaces a shipped one of the same name."""
    presets: dict[str, Preset] = {}
    for directory in (folder, folder / "user"):
        if not directory.is_dir():
            continue
        for path in sorted(directory.glob("*.txt"), key=lambda p: p.name.lower()):
            try:
                preset = read_preset(path)
            except (OSError, UnicodeDecodeError):
                continue
            if preset.text:
                presets[preset.name] = preset
    return list(presets.values())


def find(presets: list[Preset], name: str):
    return next((preset for preset in presets if preset.name == name), None)


def name_of(presets: list[Preset], text: str) -> str:
    """The name of the preset whose text this is, or "Custom" for edited or free text."""
    wanted = text.strip()
    return next((preset.name for preset in presets if preset.text == wanted), "Custom")


def effective_words(text: str, chosen: int, preset_words: int) -> int:
    """The length to ask for: the slider when set, else the preset's own, else a fallback if the text wants one."""
    if chosen > 0:
        return int(chosen)
    if preset_words > 0:
        return int(preset_words)
    return FALLBACK_WORDS if "{words}" in text else 0


def fill(text: str, words: int) -> str:
    return text.replace("{words}", str(words if words > 0 else FALLBACK_WORDS))


def token_budget(words: int, max_new_tokens: int, thinking: bool = False) -> int:
    """Tokens the model may write. With a length target the limit sits at twice the target; reasoning needs the full room."""
    if words <= 0 or thinking:
        return int(max_new_tokens)
    return int(min(max_new_tokens, math.ceil(words * SLACK * TOKENS_PER_WORD) + 8))
