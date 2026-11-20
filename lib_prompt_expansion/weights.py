"""Prompt weights across an expansion: (red car:1.4), ((fog)), [crowd], (dress:-1).

The text encoder that writes the expansion must not see this syntax: it would copy it, drop it or spell it out.
So the weights are taken out of the request before it is written, and marked again in the finished text:

- a weighted phrase that the text kept word for word is wrapped again where it first occurs;
- one whose words were kept but moved apart gets each of its words wrapped;
- one that the text lost is added at the end as it was typed;
- a phrase with a weight of zero or less (a NegPiP "less of this") is never shown to the encoder at all, which
  would only write more about it, and goes to the end as typed.

The syntax is the web UI's own, read the same way as Forge reads it (backend/text_processing/parsing.py). Forge
Neo's engines for the models this extension supports treat it as plain characters; it acts through an extension
such as NegPiP, and nothing here depends on which.
"""

import math
import re
from dataclasses import dataclass, field

_ATTENTION = re.compile(
    r"""
\\\(|
\\\)|
\\\[|
\\]|
\\\\|
\\|
\(|
\[|
:\s*([+-]?[.\d]+)\s*\)|
\)|
]|
[^\\()\[\]:]+|
:
""",
    re.X,
)
_BREAK = re.compile(r"\s*\bBREAK\b\s*", re.S)
_EXPLICIT = re.compile(r":\s*[+-]?[.\d]+\s*\)|\\[()\[\]]")
_WORD = re.compile(r"[^\W_]+(?:['’-][^\W_]+)*")
_STOP = frozenset("a an the of in on at to by for with and or as is are its his her their this that".split())


def parse(text: str) -> list[tuple[str, float]]:
    """(text, weight) pieces of a prompt, as Forge's parse_prompt_attention gives them. BREAK is a plain word here."""
    pieces: list[list] = []
    round_open, square_open = [], []

    def multiply(start, factor):
        for position in range(start, len(pieces)):
            pieces[position][1] *= factor

    for match in _ATTENTION.finditer(text):
        token, weight = match.group(0), match.group(1)
        if token.startswith("\\"):
            pieces.append([token[1:], 1.0])
        elif token == "(":
            round_open.append(len(pieces))
        elif token == "[":
            square_open.append(len(pieces))
        elif weight is not None and round_open:
            try:
                multiply(round_open.pop(), float(weight))
            except ValueError:  # "(a:1.2.3)" is not a number; Forge would fail here, we read it as text
                pieces.append([token, 1.0])
        elif token == ")" and round_open:
            multiply(round_open.pop(), 1.1)
        elif token == "]" and square_open:
            multiply(square_open.pop(), 1 / 1.1)
        else:
            parts = _BREAK.split(token)
            for number, part in enumerate(parts):
                if number > 0:
                    pieces.append(["BREAK", -1])
                pieces.append([part, 1.0])
    for position in round_open:
        multiply(position, 1.1)
    for position in square_open:
        multiply(position, 1 / 1.1)
    if not pieces:
        pieces = [["", 1.0]]
    position = 0
    while position + 1 < len(pieces):
        if pieces[position][1] == pieces[position + 1][1]:
            pieces[position][0] += pieces[position + 1][0]
            pieces.pop(position + 1)
        else:
            position += 1
    return [(piece, float(weight)) for piece, weight in pieces]


@dataclass(frozen=True)
class Phrase:
    text: str
    weight: float


@dataclass
class Stripped:
    plain: str  # the request as the encoder is given it
    phrases: list[Phrase] = field(default_factory=list)  # weighted above zero: marked again in the answer
    withheld: list[Phrase] = field(default_factory=list)  # zero or less: kept from the encoder, put back at the end
    explicit: bool = False  # the request wrote a number or an escape, so the syntax is surely meant as syntax

    @property
    def any(self) -> bool:
        return bool(self.phrases or self.withheld)


def _tidy(text: str) -> str:
    """Spacing and commas after pieces were taken out of a prompt."""
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"\s+([,.;:!?])", r"\1", text)
    text = re.sub(r"([,;:])(?:\s*[,;:])+", r"\1", text)
    return text.strip(" ,;:")


def strip(request: str) -> Stripped:
    """Take the weights out of a request."""
    if not request or not re.search(r"[()\[\]\\]", request) or request.lstrip().startswith(("{", "[{")):
        return Stripped(request)  # nothing to do, or a layout in JSON, whose brackets are not weights
    plain, phrases, withheld = [], [], []
    for text, weight in parse(request):
        if text.strip() == "BREAK" and weight == -1:
            plain.append(" BREAK ")
            continue
        if weight == 1.0:
            plain.append(text)
            continue
        phrase = Phrase(text.strip(" \t\r\n,;:."), weight)
        if not _WORD.search(phrase.text):
            plain.append(text)  # "( , :1.2)": nothing to weight
        elif weight <= 0:
            withheld.append(phrase)
            plain.append(" , " if text.strip().endswith((",", ";")) or text.strip().startswith((",", ";")) else " ")
        else:
            phrases.append(phrase)
            plain.append(text)
    if not phrases and not withheld:
        return Stripped(_tidy("".join(plain)) if "\\" in request else request, explicit=bool(_EXPLICIT.search(request)))
    return Stripped(_tidy("".join(plain)), phrases, withheld, bool(_EXPLICIT.search(request)))


def _number(weight: float) -> str:
    return f"{weight:.4g}"


def wrap(text: str, weight: float, escape: bool = True) -> str:
    """The phrase in weight syntax; a weight that is a plain power of 1.1 is written as typed: (x), ((x)), [x]."""
    inner = _escape(text) if escape else text
    if weight > 0:
        power = round(math.log(weight) / math.log(1.1))
        if 1 <= abs(power) <= 4 and math.isclose(1.1**power, weight, rel_tol=1e-9):
            return ("(" * power + inner + ")" * power) if power > 0 else ("[" * -power + inner + "]" * -power)
    return f"({inner}:{_number(weight)})"


def _escape(text: str) -> str:
    return re.sub(r"(?<!\\)([()\[\]])", r"\\\1", text)


def _pattern(words: list[str]) -> re.Pattern:
    return re.compile(r"(?<![^\W_])" + r"[\s,]+".join(re.escape(word) for word in words) + r"(?![^\W_])", re.I)


def _free(pattern: re.Pattern, text: str, taken: list[tuple[int, int, float]]):
    for match in pattern.finditer(text):
        if all(match.end() <= start or match.start() >= end for start, end, _ in taken):
            return match
    return None


@dataclass
class Applied:
    text: str
    placed: int = 0  # phrases wrapped again where the text kept them
    spread: int = 0  # phrases whose words were wrapped one by one
    added: int = 0  # phrases the text had lost, put at the end
    withheld: int = 0  # phrases with a weight of zero or less, put at the end

    def notes(self) -> list[str]:
        notes = []
        kept = self.placed + self.spread
        if kept:
            notes.append(f"{kept} weight{'s' if kept != 1 else ''} marked again")
        if self.added:
            notes.append(f"{self.added} weighted phrase{'s' if self.added != 1 else ''} added at the end (the text had not kept {'them' if self.added != 1 else 'it'})")
        if self.withheld:
            notes.append(f"{self.withheld} negative phrase{'s' if self.withheld != 1 else ''} kept out of the writing and put back")
        return notes


def apply(text: str, stripped: Stripped) -> Applied:
    """Mark the weights of the request again in the finished text."""
    if not stripped.any and not stripped.explicit:
        return Applied(text)
    done = Applied("")
    taken: list[tuple[int, int, float]] = []
    lost: list[Phrase] = []
    for phrase in sorted(stripped.phrases, key=lambda phrase: -len(phrase.text)):
        words = _WORD.findall(phrase.text)
        whole = _free(_pattern(words), text, taken) if words else None
        if whole is not None:
            taken.append((whole.start(), whole.end(), phrase.weight))
            done.placed += 1
            continue
        content = list(dict.fromkeys(word for word in words if word.lower() not in _STOP)) or list(dict.fromkeys(words))
        found, trial = [], list(taken)
        for word in content:
            match = _free(_pattern([word]), text, trial)
            if match is None:
                break
            found.append(match)
            trial.append((match.start(), match.end(), phrase.weight))
        if content and len(found) == len(content):
            taken = trial
            done.spread += 1
        else:
            lost.append(phrase)

    # the text's own brackets would be read as weights too, once the syntax is surely in use: they become literal
    literal = _escape if stripped.explicit else (lambda piece: piece)
    out, last = [], 0
    for start, end, weight in sorted(taken):
        out.append(literal(text[last:start]))
        out.append(wrap(text[start:end], weight, escape=stripped.explicit))
        last = end
    out.append(literal(text[last:]))
    result = "".join(out)
    for phrase in [phrase for phrase in stripped.phrases if phrase in lost]:
        result += " " + wrap(phrase.text, phrase.weight, escape=stripped.explicit)
        done.added += 1
    for phrase in stripped.withheld:
        result += " " + wrap(phrase.text, phrase.weight, escape=stripped.explicit)
        done.withheld += 1
    done.text = result.strip()
    return done
