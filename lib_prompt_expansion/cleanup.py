"""Turn the model's answer into a prompt: one clean paragraph, or nothing.

When there is nothing usable the caller keeps the prompt as it was typed, so an expansion can never break a job.
"""

import re
from dataclasses import dataclass, field
from typing import Optional

# a sentence-final mark, optionally followed by closing quotes or brackets (straight and curly)
SENTENCE_END = "[.!?…。！？][\"'”’)\\]]*"

_FENCE = re.compile(r"^```[^\n]*\n(.*?)(?:\n```\s*)?$", re.S)
_PREAMBLE = re.compile(r"^(?:here(?:['’]s| is| are)|sure|certainly|okay|ok|of course|absolutely)\b[^:\n]{0,80}:\s*", re.I)
_LABEL = re.compile(r"^(?:the\s+)?(?:(?:rewritten|expanded|revised|enhanced|final|detailed|new)\s+)?(?:image\s+)?(?:prompt|description|request)\s*:\s*", re.I)
_REFUSAL = re.compile(r"^(?:i['’]m sorry|i am sorry|sorry,|i cannot|i can['’]t|i can not|i am unable|i['’]m unable|as an ai\b)", re.I)
_BULLET = re.compile(r"^\s*(?:[-*•]|\d+[.)])\s+")
_HEADING = re.compile(r"^\s*#{1,6}\s*")
_QUOTED = re.compile("\"[^\"]*\"|“[^”]*”|«[^»]*»|「[^」]*」")
_QUOTE_PAIRS = (('"', '"'), ("“", "”"), ("'", "'"), ("‘", "’"))

# Writing systems a small multilingual model may drift into. Latin is not listed: it is never a drift.
_SCRIPTS = (
    ("CJK", ((0x2E80, 0x2FFF), (0x3040, 0x30FF), (0x3400, 0x4DBF), (0x4E00, 0x9FFF), (0xF900, 0xFAFF), (0xFF00, 0xFFEF), (0x3001, 0x3002))),
    ("Hangul", ((0x1100, 0x11FF), (0xAC00, 0xD7AF))),
    ("Cyrillic", ((0x0400, 0x052F),)),
    ("Greek", ((0x0370, 0x03FF),)),
    ("Arabic", ((0x0600, 0x06FF),)),
    ("Hebrew", ((0x0590, 0x05FF),)),
    ("Devanagari", ((0x0900, 0x097F),)),
    ("Thai", ((0x0E00, 0x0E7F),)),
)


@dataclass
class Cleaned:
    text: Optional[str]  # None = keep the prompt as typed
    notes: list[str] = field(default_factory=list)  # what was changed or why it was refused, for the log


def scripts_in(text: str) -> set[str]:
    found = set()
    for char in text:
        code = ord(char)
        if code < 0x0370:
            continue
        for name, ranges in _SCRIPTS:
            if any(low <= code <= high for low, high in ranges):
                found.add(name)
                break
    return found


def _strip_scripts(text: str, names: set[str]) -> str:
    """`text` without the characters of the named writing systems, except inside quoted lettering."""
    ranges = [span for name, spans in _SCRIPTS if name in names for span in spans]

    def plain(piece: str) -> str:
        return "".join(char for char in piece if not any(low <= ord(char) <= high for low, high in ranges))

    pieces, last = [], 0
    for quoted in _QUOTED.finditer(text):
        pieces += [plain(text[last : quoted.start()]), quoted.group(0)]
        last = quoted.end()
    pieces.append(plain(text[last:]))
    return "".join(pieces)


def split_sentences(text: str) -> list[str]:
    """Sentences with their trailing space removed; a tail without a final mark is the last item."""
    parts, start = [], 0
    for match in re.finditer(SENTENCE_END + r"(?=\s)", text):
        parts.append(text[start : match.end()].strip())
        start = match.end()
    tail = text[start:].strip()
    if tail:
        parts.append(tail)
    return [part for part in parts if part]


def trim_to_sentence(text: str) -> str:
    """Drop an unfinished last sentence. A single unfinished sentence is cut at its last clause and closed."""
    if re.search(SENTENCE_END + "$", text):
        return text
    ends = [match.end() for match in re.finditer(SENTENCE_END + r"(?=\s)", text)]
    if ends:
        return text[: ends[-1]]
    clauses = [match.start() for match in re.finditer(r"[,;:]\s", text)]
    if clauses and clauses[-1] > len(text) // 2:
        text = text[: clauses[-1]]
    return text.rstrip(",;:-— ") + "."


def clean(answer: str, finish: str = "stop", source: str = "") -> Cleaned:
    """answer: the model's text without its reasoning block. finish: why writing ended. source: the typed request."""
    notes: list[str] = []
    if finish == "interrupted":
        return Cleaned(None, ["interrupted"])
    text = (answer or "").strip()
    if not text:
        return Cleaned(None, ["empty answer" if finish != "loop" else "nothing before a runaway loop"])

    fenced = _FENCE.match(text)
    if fenced:
        text = fenced.group(1).strip()
        notes.append("code fence removed")

    lines = []
    for line in text.splitlines():
        line = _BULLET.sub("", _HEADING.sub("", line)).strip()
        if line:
            lines.append(line)
    text = re.sub(r"\s+", " ", " ".join(lines)).replace("**", "").replace("__", "").strip()

    for pattern, note in ((_PREAMBLE, "preamble removed"), (_LABEL, "label removed")):
        shorter = pattern.sub("", text, count=1).strip()
        if shorter != text:
            text = shorter
            notes.append(note)

    for opening, closing in _QUOTE_PAIRS:
        if len(text) > 1 and text.startswith(opening) and text.endswith(closing) and opening not in text[1:-1] and closing not in text[1:-1]:
            text = text[1:-1].strip()
            notes.append("wrapping quotes removed")
            break

    if not text:
        return Cleaned(None, notes + ["empty answer"])
    if _REFUSAL.match(text):
        return Cleaned(None, notes + ["refusal"])

    if finish in ("length", "loop"):
        trimmed = trim_to_sentence(text)
        if trimmed != text:
            text = trimmed
            notes.append("cut back to a complete sentence after " + ("the token limit" if finish == "length" else "a runaway loop"))

    # A sentence that drifts into a writing system the request does not use is dropped. Text inside quotes is
    # lettering for the picture and may be in any script.
    allowed = scripts_in(source)
    kept, dropped = [], set()
    for sentence in split_sentences(text):
        foreign = scripts_in(_QUOTED.sub("", sentence)) - allowed
        if foreign:
            dropped |= foreign
        else:
            kept.append(sentence)
    if dropped:
        remaining = " ".join(kept)
        if len(remaining.split()) * 2 >= len(text.split()):
            text = remaining
            notes.append("dropped a sentence written in " + ", ".join(sorted(dropped)))
        else:
            # the drift sits in a sentence that carries most of the text: take out the foreign words only
            text = re.sub(r"\s{2,}", " ", _strip_scripts(text, dropped)).strip()
            notes.append("removed words written in " + ", ".join(sorted(dropped)))

    # "No text or signs appear in the image": an encoder that is told to add no lettering may say so in its answer,
    # which puts those very words into the prompt. Such statements go, unless the request itself asks for them.
    if not _ABSENT_ANY.search(source):
        total = len(text.split())
        # whole sentences first; if that would take more than half of the text (a long run-on sentence),
        # only the clauses are cut; if even that takes too much, the text stays as it is
        for whole_sentences in (True, False):
            kept, removed = [], 0
            for sentence in split_sentences(text):
                if _ABSENT_SENTENCE.match(sentence) or _ABSENT_VERB.search(_QUOTED.sub("", sentence)):
                    if whole_sentences:
                        removed += 1
                    else:
                        kept.append(sentence)  # the statement is the sentence's backbone: cutting a piece would break it
                    continue
                shorter = _ABSENT_CLAUSE.sub("", sentence)
                if shorter != sentence:
                    removed += 1
                    shorter = re.sub(r"\s+([.,;!?])", r"\1", shorter).strip()
                    if shorter and shorter[-1] not in ".!?\"'":
                        shorter += "."
                if len(shorter.split()) >= 3:
                    kept.append(shorter)
            result = " ".join(kept)
            if not removed:
                break
            if len(result.split()) * 2 >= total:
                text = result
                notes.append("removed a statement that there is no text")
                break

    if len(text.split()) < 3:
        return Cleaned(None, notes + ["too little text left"])
    return Cleaned(text, notes)


_LETTERING = r"(?:text|signage|signs?|lettering|writing|words|labels?|captions?|logos?|watermarks?)\b(?!\s+of\b)"
_ABSENT_ANY = re.compile(r"\b(?:no|without|free of|devoid of)\b[^.;]{0,30}\b" + _LETTERING, re.I)
# a sentence that opens by saying there is no text: the whole sentence is about that
_ABSENT_SENTENCE = re.compile(r"^(?:there (?:is|are) )?no (?:other |additional |visible |further |added |written |printed |overlaid )*(?:objects or |elements or )?" + _LETTERING + r"(?:(?:,| or| and| nor)\s+(?:other |visible )?" + _LETTERING + r")*\b", re.I)
# "The image contains no lettering", "The scene is devoid of any signs": the statement is the sentence's main clause
_ABSENT_VERB = re.compile(
    r"\b(?:is|are|was|were|remains?|appears?|stays?|contains?|shows?|displays?|has|have|bears?|features?|includes?|carr(?:y|ies))\s+(?:no|devoid of(?: any)?|free of(?: any)?|without(?: any)?)\s+"
    r"(?:other |additional |visible |further |added |written |printed |overlaid )*(?:objects or |elements or )?" + _LETTERING,
    re.I,
)
# a trailing or inserted clause: ", without any visible text or signs present in the frame"
_ABSENT_CLAUSE = re.compile(
    r",?\s*(?:and |but |yet )?(?:with no|without(?: any)?|free of(?: any)?|devoid of(?: any)?)\s+(?:other |additional |visible |surrounding |extra |further |added |written |printed |overlaid |superimposed )*(?:objects or |elements or )?"
    + _LETTERING
    + r"(?:(?:,| or| and| nor)\s+(?:other |visible )?"
    + _LETTERING
    + r")*(?:\s+(?:present|visible|anywhere|at all|of any kind))?(?:\s+(?:in|on|within|across)\s+(?:the |this )?(?:frame|view|scene|image|picture|composition|structure|shot|background|foreground))?",
    re.I,
)


_AND = re.compile(r"\bAND\b")
_EXTRA_NETWORK = re.compile(r"<\w+:[^>]+>")


def neutralise(text: str, comments_follow: bool = False) -> str:
    """Make a written paragraph safe from the web UI's own prompt syntax, which is applied after the expansion.

    - An upper-case AND splits a prompt into sub-prompts: it becomes "&" inside quoted lettering and "and" elsewhere.
    - Square brackets are prompt scheduling ([a:b:0.5], [a|b]): they become round ones.
    - Anything shaped like <name:value> would be read as an extra network and is removed.
    - comments_follow: Forge's comment remover runs after us and would cut the paragraph at "#" or "//".
    """
    text = _EXTRA_NETWORK.sub("", text)
    pieces, last = [], 0
    for quoted in _QUOTED.finditer(text):
        pieces.append(_AND.sub("and", text[last : quoted.start()]))
        pieces.append(_AND.sub("&", quoted.group(0)))
        last = quoted.end()
    pieces.append(_AND.sub("and", text[last:]))
    text = "".join(pieces).replace("[", "(").replace("]", ")")
    if comments_follow:
        text = re.sub(r"#\s*(?=\d)", "No. ", text).replace("#", "")
        text = text.replace("/*", "/ *").replace("//", "/")
    return re.sub(r"\s{2,}", " ", text).strip()
