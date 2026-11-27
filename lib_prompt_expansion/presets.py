"""Expansion prompts kept as text files: presets/*.txt ship with the extension, presets/user/*.txt are the user's.

File name: "NN Name.txt"; the number only sets the order. Optional first lines: "# words: 150" gives the default
length, and "# begin: request" says that the expansion starts with the request itself and continues it (the request
is then put at the start of the encoder's answer, so the instruction need not ask for it). "{words}" in the text is
replaced by the length when a request is written.

Lines between "# if: term, term*" and "# end" (with an optional "# else") go to the encoder only for requests that
contain one of the terms, so that a small encoder is not shown rules it would apply to every picture. A term
"!boy*" holds when the request has no such word. Lines between "# add if missing: term, term*" and "# end" never go to
the encoder: they are put after the expanded text when it contains none of the terms, for what must be said.
"# replace in request: *children => *children, *boys and *girls" changes the request the encoder sees ("*" is the rest
of a word); "# replace in answer: each child wears => each boy wears" changes the encoder's text, and inside an
"# add if missing:" block only when that block is used.

"{random:47-19 ЛДА}" is replaced by a new value of the sample's form for every request (each digit by a random digit,
each Cyrillic or Latin capital by a random capital of the same alphabet, the rest kept), and "{number:2-99}" by a
number in that range. A small encoder copies such samples, so the plates and route numbers it writes then differ from
picture to picture; they come from the expansion seed, so a fixed seed gives the same text again.
"""

import math
import random
import re
from dataclasses import dataclass
from typing import Optional
from pathlib import Path

PRESET_DIR = Path(__file__).resolve().parent.parent / "presets"
FALLBACK_WORDS = 150
TOKENS_PER_WORD = 1.6  # English prose in the Qwen and Gemma tokenizers, punctuation included
SLACK = 2.0  # the length is a guide: the model may write up to twice it before the token limit cuts in

_HEADER = re.compile(r"^#\s*words\s*:\s*(\d+)\s*$", re.I)
_BEGIN = re.compile(r"^#\s*begin\s*:\s*(\w+)\s*$", re.I)
_ORDER = re.compile(r"^\d+[\s_.-]*")
_IF = re.compile(r"^#\s*if\s*:\s*(.*?)\s*$", re.I)
_ELSE = re.compile(r"^#\s*else\s*$", re.I)
_END = re.compile(r"^#\s*end\s*$", re.I)
_ADD = re.compile(r"^#\s*add\s+if\s+missing\s*:\s*(.*?)\s*$", re.I)
_IN_REQUEST = re.compile(r"^#\s*replace\s+in\s+request\s*:\s*(.+?)\s*=>\s*(.*?)\s*$", re.I)
_IN_ANSWER = re.compile(r"^#\s*replace\s+in\s+answer\s*:\s*(.+?)\s*=>\s*(.*?)\s*$", re.I)
_CONDITION = re.compile(r"^#\s*(?:if\s*:.*|add\s+if\s+missing\s*:.*|replace\s+in\s+(?:request|answer)\s*:.*|else|end)\s*$", re.I)


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
        elif line.lstrip().startswith("#") and not lines and not _CONDITION.match(line.strip()):
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


def has_conditions(text: str) -> bool:
    """Does the expansion prompt hold "# if:" or "# add if missing:" blocks, so that each request gets its own instruction?"""
    return any(_CONDITION.match(line.strip()) and not _END.match(line.strip()) and not _ELSE.match(line.strip()) for line in text.splitlines())


def _term(term: str) -> re.Pattern:
    """A whole word or phrase of the request, any case; with a trailing "*", any word that begins so."""
    prefix = term.endswith("*")
    body = re.escape(term.rstrip("*").strip())
    return re.compile(rf"(?<!\w){body}" + ("" if prefix else r"(?!\w)"), re.I)


def matches(terms: str, request: str) -> bool:
    """Does the request contain any of the comma-separated terms? A term written "!boy*" holds when the request
    has no such word: the condition holds when any plain term is found, or when none of the "!" terms is
    ("girl*, !boy*" = girls are named, or no boys are)."""
    plain, absent = [], []
    for term in (part.strip() for part in terms.split(",")):
        if term.startswith("!") and term[1:].strip():
            absent.append(term[1:].strip())
        elif term:
            plain.append(term)
    if any(_term(term).search(request) for term in plain):
        return True
    return bool(absent) and not any(_term(term).search(request) for term in absent)


@dataclass(frozen=True)
class Addition:
    """A "# add if missing:" block: when the expanded text has none of the terms, its "# replace in answer:" rules are
    applied and its lines are put after the text."""

    terms: str
    text: str
    replacements: tuple = ()  # (pattern, replacement) pairs


@dataclass(frozen=True)
class Plan:
    """What a preset does for one request."""

    instruction: str  # the expansion prompt the encoder gets
    additions: tuple = ()  # Addition
    request_rules: tuple = ()  # (pattern, replacement): the request as the encoder sees it
    answer_rules: tuple = ()  # (pattern, replacement): the encoder's text after the clean-up


def _walk(text: str, request: str) -> Plan:
    kept: list[str] = []
    found: list[list] = []  # [terms, lines, replacements]
    request_rules: list[tuple] = []
    answer_rules: list[tuple] = []
    # (the lines of the enclosing level are kept, the condition held, the addition this level writes to or None)
    stack: list[tuple[bool, bool, Optional[int]]] = []
    keeping, target = True, None
    for line in text.splitlines():
        stripped = line.strip()
        condition, missing = _IF.match(stripped), _ADD.match(stripped)
        in_request, in_answer = _IN_REQUEST.match(stripped), _IN_ANSWER.match(stripped)
        if condition:
            held = matches(condition.group(1), request)
            stack.append((keeping, held, target))
            keeping = keeping and held
        elif missing:
            stack.append((keeping, True, target))
            if keeping:
                found.append([missing.group(1), [], []])
                target = len(found) - 1
            else:
                target = None
        elif _ELSE.match(stripped):
            if stack:
                outer, held, _ = stack[-1]
                keeping = outer and not held
        elif _END.match(stripped):
            if stack:
                keeping, _, target = stack.pop()
        elif in_request:
            if keeping:
                request_rules.append((in_request.group(1), in_request.group(2)))
        elif in_answer:
            if keeping:
                (found[target][2] if target is not None else answer_rules).append((in_answer.group(1), in_answer.group(2)))
        elif keeping and target is not None:
            found[target][1].append(stripped)
        elif keeping:
            kept.append(line)
    joined = re.sub(r"\n{3,}", "\n\n", "\n".join(kept)).strip()
    extra = tuple(Addition(terms, " ".join(part for part in lines if part), tuple(rules)) for terms, lines, rules in found if any(lines) or rules)
    return Plan(joined, extra, tuple(request_rules), tuple(answer_rules))


def plan(text: str, request: str) -> Plan:
    return _walk(text, request)


def select(text: str, request: str) -> str:
    """The expansion prompt for one request: the lines of every "# if: a, b*" block whose terms the request
    contains, else those of its "# else" part. Blocks nest and close with "# end"; an unclosed block ends with the text.
    The lines of "# add if missing:" blocks and the "# replace in" rules are not part of it."""
    return _walk(text, request).instruction


def additions(text: str, request: str) -> list:
    """The "# add if missing: a, b*" blocks this request reaches: each one's lines, joined into one piece of text,
    are put after the expanded text when it contains none of the terms. "# if:" blocks inside them work as above."""
    return list(_walk(text, request).additions)


def _phrase(pattern: str) -> re.Pattern:
    """A "# replace in" pattern: words in any case, any run of spaces for a space, "*" for the rest of a word."""
    parts = re.split(r"(\*)", pattern.strip())
    body = "".join(r"(\w*)" if part == "*" else r"\s+".join(re.escape(word) for word in part.split(" ")) for part in parts)
    # word edges only where the pattern begins or ends with a word: '" in double quotes' follows a closing quote
    before = r"(?<!\w)" if re.match(r"[\w*]", pattern.strip()) else ""
    after = r"(?!\w)" if re.search(r"[\w*]$", pattern.strip()) else ""
    return re.compile(before + body + after, re.I)


def replace(text: str, pattern: str, replacement: str) -> tuple[str, int]:
    """Every match of the pattern in the text, replaced. A "*" of the replacement is the text the pattern's "*" stood
    for (the first for the first, and so on; the last one for any further); a match that begins with a capital gives a
    replacement that does too."""
    def one(match: re.Match) -> str:
        caught = list(match.groups())
        out, used = [], 0
        for part in re.split(r"(\*)", replacement):
            if part == "*":
                piece = caught[min(used, len(caught) - 1)] if caught else ""
                out.append(piece if used == 0 else piece.lower())
                used += 1
            else:
                out.append(part)
        result = "".join(out)
        return result[:1].upper() + result[1:] if match.group(0)[:1].isupper() else result
    return _phrase(pattern).subn(one, text)


def rewrite_request(text: str, request: str) -> str:
    """The request as the encoder gets it: the preset's "# replace in request:" rules applied in order."""
    for pattern, replacement in _walk(text, request).request_rules:
        request = replace(request, pattern, replacement)[0]
    return request


def complete(expanded: str, extra: list, rules: tuple = ()) -> tuple[str, list]:
    """The expanded text after the preset's own work: its "# replace in answer:" rules, then every addition whose terms
    the text lacks (that addition's own replace rules first, its lines after the text). Returns the text and the
    additions that were used."""
    for pattern, replacement in rules:
        expanded = replace(expanded, pattern, replacement)[0]
    used = []
    for addition in extra:
        if matches(addition.terms, expanded):
            continue
        used.append(addition)
        for pattern, replacement in addition.replacements:
            expanded = replace(expanded, pattern, replacement)[0]
        if addition.text:
            expanded = expanded.rstrip() + " " + addition.text
    return expanded, used


def finish(text: str, request: str, expanded: str) -> tuple[str, list]:
    """The preset's work on the encoder's cleaned text for this request (see complete)."""
    found = _walk(text, request)
    return complete(expanded, list(found.additions), found.answer_rules)


_RANDOM = re.compile(r"\{random:([^{}]+)\}")
_NUMBER = re.compile(r"\{number:\s*(\d+)\s*-\s*(\d+)\s*\}")
# letters of the plates and signs a sample stands for: no Ё, Й, Ъ, Ы, Ь (never on a plate); Latin without I, O, Q
_CYRILLIC = "АБВГДЕЖЗИКЛМНОПРСТУФХЦЧШЭЮЯ"
_LATIN = "ABCDEFGHJKLMNPRSTUVWXYZ"


def _random_like(sample: str, chance: random.Random) -> str:
    out = []
    for char in sample:
        if char.isdigit():
            out.append(chance.choice("0123456789"))
        elif "А" <= char <= "Я" or char == "Ё":
            out.append(chance.choice(_CYRILLIC))
        elif "A" <= char <= "Z":
            out.append(chance.choice(_LATIN))
        else:
            out.append(char)
    return "".join(out)


def has_random(text: str) -> bool:
    return bool(_RANDOM.search(text) or _NUMBER.search(text))


# Digits in a request that are not a number of its own: a year or decade ("1957", "1970s", "70s", "'70s", "1970-е",
# "70-х"), an age ("7-year-old", "40 years", "8 лет") and a model name ("GAZ-21", "Tu-154", "ZiU-9", "Ikarus 280").
_NOT_OWN = re.compile(
    r"(?<!\w)(?:1[89]\d\d|20\d\d)(?:'?s|-[ех])?(?!\d)"
    r"|(?<!\w)'?\d0(?:'?s|-[ех])(?!\w)"
    r"|\d+(?=\s*-?\s*(?:years?|year-old|yrs?|лет|год)\b)"
    r"|(?<=[A-Za-zА-Яа-яЁё]-)\d+[A-Za-z]?\b"
    r"|(?<=\bIkarus )\d+",
    re.I,
)
_SENTENCE = re.compile(r"(?<=[.!?])\s+")


def own_numbers(request: str) -> list:
    """The numbers the request gives itself (a route, a plate, a house number), beyond years, decades, ages and models."""
    return re.findall(r"\d+", _NOT_OWN.sub(" ", request))


def _without_samples(text: str) -> str:
    """The instruction without the sentences that carry a random sample (the request brings its own number)."""
    lines = []
    for line in text.split("\n"):
        if not has_random(line):
            lines.append(line)
            continue
        kept = [sentence for sentence in _SENTENCE.split(line) if not has_random(sentence)]
        if kept:
            lines.append(" ".join(kept))
    return re.sub(r"\n{3,}", "\n\n", "\n".join(lines)).strip()


def fill(text: str, words: int, seed: int = 0, request: str = "") -> str:
    """The instruction as it goes to the encoder: the length, and new values for "{random:...}" and "{number:a-b}".

    A request that gives a number of its own (a route, a plate) gets no samples: the sentences that carry them are left
    out, so that the encoder keeps the request's number instead of one drawn here."""
    text = text.replace("{words}", str(words if words > 0 else FALLBACK_WORDS))
    if not has_random(text):
        return text
    if own_numbers(request):
        return _without_samples(text)
    chance = random.Random(seed)
    text = _NUMBER.sub(lambda m: str(chance.randint(min(int(m.group(1)), int(m.group(2))), max(int(m.group(1)), int(m.group(2))))), text)
    return _RANDOM.sub(lambda m: _random_like(m.group(1), chance), text)


def token_budget(words: int, max_new_tokens: int, thinking: bool = False) -> int:
    """Tokens the model may write. With a length target the limit sits at twice the target; reasoning needs the full room."""
    if words <= 0 or thinking:
        return int(max_new_tokens)
    return int(min(max_new_tokens, math.ceil(words * SLACK * TOKENS_PER_WORD) + 8))
