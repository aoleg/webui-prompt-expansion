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

    lines, broke_off = [], False
    for line in text.splitlines():
        line = _BULLET.sub("", _HEADING.sub("", line)).strip()
        if not line:
            continue
        if lines and (_SELF_TALK.match(line) or _SELF_TALK_ANYWHERE.search(_QUOTED.sub("", line))):
            broke_off = True  # a new line that opens with self-talk: the text ends before it (see below)
            break
        lines.append(line)
    text = re.sub(r"\s+", " ", " ".join(lines)).replace("**", "").replace("__", "").strip()
    if broke_off:
        whole = [sentence for sentence in split_sentences(text) if re.search(SENTENCE_END + "$", sentence)]
        text = " ".join(whole)  # the sentence the encoder broke off in goes too
        notes.append("cut where the encoder began to talk to itself")
        if len(text.split()) < 12:
            return Cleaned(None, notes + ["too little text left"])

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

    # "Wait - I need to finish the sentence. Let me complete it.": some encoders break off and talk to themselves.
    # Everything from the first such sentence on is dropped, with the unfinished sentence before it.
    sentences = split_sentences(text)
    for position, sentence in enumerate(sentences):
        if _SELF_TALK.match(sentence) or _SELF_TALK_ANYWHERE.search(_QUOTED.sub("", sentence)):
            kept = " ".join(sentences[:position])
            if position and not re.search(SENTENCE_END + "$", sentences[position - 1]):
                kept = " ".join(sentences[: position - 1])
            text = kept
            notes.append("cut where the encoder began to talk to itself")
            if len(text.split()) < 12:
                return Cleaned(None, notes + ["too little text left"])
            break

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


# the opening of a sentence in which the model speaks about its own writing, and phrases that give one away anywhere
_SELF_TALK = re.compile(r"^[\s—–\-*(]*(?:Wait\b|Let me\b|I need to\b|I(?:'|’)m\b|I am\b|I think\b|I(?:'|’)ll\b|I will\b|I should\b|I have to\b|I apologi|Sorry\b|Okay\b|Actually,|Hmm\b|But I\b|No,|Note:)", re.I)
_SELF_TALK_ANYWHERE = re.compile(r"\b(?:the instructions?\b|my previous\b|the user\b|rewritten prompt|let me (?:correct|try|complete|start|rewrite)|that(?:'|’)s not right)", re.I)


def limit_words(text: str, limit: int) -> str:
    """Whole sentences from the start, as many as fit in `limit` words.

    A sentence that does not fit is left out, unless less than half the limit would then remain: a request that was
    continued as one long chain of clauses is cut at its last clause inside the limit and closed. When there is no
    clause to cut at, the first sentence is kept whole.
    """
    kept, count = [], 0
    for sentence in split_sentences(text):
        words = len(sentence.split())
        if count + words > limit:
            if count < limit // 2:
                head = " ".join(sentence.split()[: limit - count])
                breaks = [match.start() for match in re.finditer(r"[,;:]\s", head)]
                if breaks:
                    kept.append(head[: breaks[-1]].rstrip(",;:-— ") + ".")
                elif not kept:
                    kept.append(sentence)
            break
        kept.append(sentence)
        count += words
    return " ".join(kept)


# words of a request that carry no picture content of their own
_FILLER = frozenset(
    "with without from that this these those into onto over under above below behind beside between outside inside near next along across around through "
    "their there some very which while being says reads called named style kind sort also then than have has here where what when and the".split()
)


def keeps_request(text: str, request: str) -> bool:
    """Has everything in the request survived in the text?

    True when the text opens with the request itself, or when every content word of the request (by its first five
    letters, so "children" finds "child") and every piece of quoted lettering is somewhere in the text. A paraphrase
    that kept it all is left alone; one that lost a word is not.
    """
    squeeze = lambda value: re.sub(r"[^\w]+", " ", value.lower()).strip()  # noqa: E731
    if squeeze(text).startswith(squeeze(request)):
        return True
    for quoted in _QUOTED.finditer(request):
        if quoted.group(0)[1:-1] not in text:
            return False
    low = text.lower()
    wanted = [word for word in re.findall(r"[^\W_]+", request.lower()) if len(word) >= 4 and word not in _FILLER]
    return all(word[:5] in low for word in wanted)


def lead_with_request(text: str, request: str) -> str:
    """The request, then the text: for an encoder that was told to begin with the request and paraphrased it away."""
    request = re.sub(r"\s+", " ", request).strip()
    if not request:
        return text
    return request + ("" if request[-1] in ".!?" else ".") + " " + text


# An instruction that says one of these is held to it: the request is put in front when the encoder lost it ...
LEADS_WITH_REQUEST = ("the request itself first", "begin with the request")
# ... and sounds and smells are removed for certain.
LEAVES_OUT_SOUNDS = ("never mentions sounds", "leave out sounds")

# Words that are only heard or smelled. Left out on purpose: quiet, silent, still, howling (a storm), crash (waves),
# rumble (a tram rumbles into view), bell, music (it is on posters) - they are seen, or said of things that are seen, at least as often.
_UNSEEN = re.compile(
    r"\b(?:sounds?|noises?|hum|hums|hummed|humming|murmur\w*|chatter\w*|clink\w*|clatter\w*|rustl\w*|whisper\w*|echo(?:es|ed|ing)?|buzz\w*|creak\w*|hiss\w*|laughter|voices?|"
    r"audible|heard|silence|click(?:s|ed|ing)?|crackl\w*|chirp\w*|chim(?:e|es|ed|ing)|roar(?:s|ed|ing)?|toll(?:s|ed|ing)?|ticking|pattering|"
    r"smell\w*|scent\w*|aroma\w*|fragran\w*|odou?rs?)\b",
    re.I,
)
_CLAUSE_BREAK = re.compile(r",\s+|;\s+|\s+[—–]\s+|—")
_JOINED = re.compile(r"^(?:and|but|while|as|yet)\s+(?=(?:the|a|an|its|his|her|their|two|three|several|one)\b)", re.I)


def _heard(piece: str) -> bool:
    """Does this piece of a sentence speak of a sound or a smell? Quoted lettering does not count."""
    return bool(_UNSEEN.search(_QUOTED.sub(" ", piece)))


def _join(clauses: list, heard: list) -> str:
    """The clauses that are not about a sound, each with the break that stood in front of it."""
    out = ""
    for number, ((clause, _), bad) in enumerate(zip(clauses, heard)):
        if bad:
            continue
        out += (clauses[number - 1][1] if out and number > 0 else "") + clause
    return out


def drop_unseen(text: str, source: str = "") -> tuple[str, int]:
    """Take out what cannot be seen: sounds and smells. Returns the text and how many places were changed.

    A sentence that is about a sound goes as a whole. Where only a later clause is ("..., their voices barely
    audible"), that clause goes. Where the sentence opens with the sound and goes on with ", and the horizon glows",
    the rest is kept. The first sentence (the request), a sentence with quoted lettering in the part that would be
    lost, and anything whose loss would cost a word of the request are left alone.
    """
    sentences = split_sentences(text)
    kept_before = keeps_request(text, source) if source else True
    out, changed = [], 0
    for number, sentence in enumerate(sentences):
        if not _heard(sentence):
            out.append(sentence)
            continue
        clauses, last = [], 0
        for match in _CLAUSE_BREAK.finditer(sentence):
            clauses.append((sentence[last : match.start()], match.group(0)))
            last = match.end()
        clauses.append((sentence[last:], ""))
        heard = [_heard(clause) for clause, _ in clauses]
        replacement = None  # None = no way found, leave the sentence alone
        if number > 0 and (all(heard) or (heard[0] and len(clauses) == 1)):
            replacement = ""
        elif not heard[0]:
            # later clauses only: they go, each with the break in front of it
            lost = "".join(clause for (clause, _), bad in zip(clauses, heard) if bad)
            if not _QUOTED.search(lost):
                replacement = _join(clauses, heard)
        elif number > 0:
            rest = [(clause, bad) for (clause, _), bad in zip(clauses[1:], heard[1:])]
            first_kept = next((clause for clause, bad in rest if not bad), None)
            if first_kept is not None and _JOINED.match(first_kept.strip()) and not _QUOTED.search("".join(clause for clause, bad in zip([c for c, _ in clauses], heard) if bad)):
                replacement = _JOINED.sub("", _join(clauses, heard).lstrip())
                replacement = replacement[:1].upper() + replacement[1:]
            elif not _QUOTED.search(sentence):
                replacement = ""
        if replacement is None:
            out.append(sentence)
            continue
        if replacement:
            replacement = replacement.rstrip(" ,;:")
            if not re.search(SENTENCE_END + r"$", replacement):
                replacement += "."
        candidate = out + ([replacement] if replacement else []) + sentences[number + 1 :]
        if kept_before and source and not keeps_request(" ".join(candidate), source):
            out.append(sentence)  # the only place a word of the request stood
            continue
        if replacement:
            out.append(replacement)
        changed += 1
    return " ".join(out), changed


# ... and a black-and-white picture gets no colour words: a "red pioneer tie" in a black-and-white prompt makes the image
# model draw a grey picture with one red tie in it.
LEAVES_OUT_COLOURS = ("no colour words at all", "no color words at all")
# white, black, grey, dark and light are tones; gold and silver are materials
_HUE = (
    r"(?:red|reddish|blue|bluish|green|greenish|brown|brownish|yellow|yellowish|orange|pink|pinkish|purple|navy|beige|golden|crimson|scarlet|"
    r"turquoise|violet|maroon|khaki|teal|amber|ochre|burgundy|lilac|magenta|cyan|emerald|azure)"
)
_TONE = r"(?:white|black|grey|gray|dark|light|pale|deep|bright)"
# one colour word, or a compound of colour words ("navy-blue", "red-and-white", "blue-grey"), in front of the word it describes
_HUED = re.compile(
    rf"(?<![\w-])(?P<words>(?:{_HUE}|{_TONE})(?:(?:-and-|-| and )(?:{_HUE}|{_TONE}))*)(?:-(?:coloured|colored|toned|tinted))?(?P<gap>[ -])(?=(?P<next>[a-z]+))",
    re.I,
)
_HUE_WORD = re.compile(rf"\b{_HUE}\b", re.I)
_NOT_A_NOUN = {"and", "or", "with", "in", "on", "at", "against", "of", "to", "from", "as", "under", "over", "near", "by", "behind", "than", "but", "while"}
_PREDICATE = re.compile(r"\b(?:is|are|was|were|be|been|painted|coloured|colored|turns|turned|looks|appears|stays|remains)\s+$", re.I)


def drop_colours(text: str, source: str = "") -> tuple[str, int]:
    """Take colour words out of a text that must have none: "a red pioneer tie" becomes "a pioneer tie". Only a colour
    word in front of the word it describes goes; quoted lettering and colour words of the request are left alone."""
    asked = {word.lower() for word in _HUE_WORD.findall(source)}
    changed = 0

    def one(piece: str) -> str:
        nonlocal changed
        out, last = [], 0
        for match in _HUED.finditer(piece):
            hues = [word.lower() for word in _HUE_WORD.findall(match.group("words"))]
            if not hues or any(hue in asked for hue in hues) or match.group("next").lower() in _NOT_A_NOUN or _PREDICATE.search(piece[: match.start()]):
                continue
            before = piece[last : match.start()]
            # the tones stay: "a dark red tie" -> "a dark tie", "blue-grey coat" -> "grey coat"
            tones = [word for word in re.split(r"-and-|-| and ", match.group("words")) if re.fullmatch(_TONE, word, re.I)]
            kept = "-".join(tones) + " " if tones else ""
            # "an orange scarf" -> "a scarf", "a red apron" -> "an apron"
            article = re.search(r"\b(an?) $", before, re.I)
            if article:
                wanted = "an" if (kept or match.group("next"))[:1].lower() in "aeiou" else "a"
                wanted = wanted.capitalize() if article.group(1)[0].isupper() else wanted
                before = before[: article.start(1)] + wanted + " "
            out.append(before + kept)
            last = match.end()
            changed += 1
        out.append(piece[last:])
        return "".join(out)

    parts, last = [], 0
    for quoted in _QUOTED.finditer(text):
        parts.append(one(text[last : quoted.start()]))
        parts.append(quoted.group(0))
        last = quoted.end()
    parts.append(one(text[last:]))
    return "".join(parts), changed


OVERLONG = 1.5  # a text longer than this many times the length asked for is cut back to it


def prepare(answer: str, finish: str = "stop", source: str = "", instruction: str = "", words: int = 0, comments_follow: bool = False, begins: bool = False) -> Cleaned:
    """Everything between the encoder's answer and the prompt that is used: clean-up, the request in front where the
    instruction asked for it and did not get it, a bound on the length, and protection from web UI syntax.

    instruction: the expansion prompt that was sent. words: the length that was asked for, 0 for none.
    begins: the expansion is meant to start with the request (the answer was begun with it); should the text have
        lost it all the same, as after reasoning, it is put in front.
    """
    cleaned = clean(answer, finish, source)
    if not cleaned.text:
        return cleaned
    text, notes = cleaned.text, cleaned.notes
    told = instruction.lower()
    if any(marker in told for marker in LEAVES_OUT_SOUNDS):
        shorter, changed = drop_unseen(text, source)
        if changed and shorter.strip():
            text = shorter
            notes.append("sounds or smells removed")
    if any(marker in told for marker in LEAVES_OUT_COLOURS):
        text, changed = drop_colours(text, source)
        if changed:
            notes.append("colour words removed (black-and-white)")
    if (begins or any(marker in told for marker in LEADS_WITH_REQUEST)) and not keeps_request(text, source):
        text = lead_with_request(text, source)
        notes.append("the request was put in front (the encoder had not kept it)")
    # with no length asked for (a rewrite, not an expansion) the bound follows the request's own length
    limit = round(OVERLONG * words) if words > 0 else max(80, 12 * len(source.split()))
    if len(text.split()) > limit:
        text = limit_words(text, limit)
        notes.append("cut back to the length asked for" if words > 0 else "cut back: far longer than the request")
    return Cleaned(neutralise(text, comments_follow), notes)


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
    r",?\s*(?:and |but |yet )?(?:with no|without(?: (?:introducing|adding|including|showing|featuring|displaying|using))?(?: any)?|free of(?: any)?|devoid of(?: any)?)\s+"
    r"(?:other |additional |visible |surrounding |extra |extraneous |external |unnecessary |distracting |further |added |written |printed |overlaid |superimposed )*(?:objects or |elements or )?"
    + _LETTERING
    + r"(?:(?:,| or| and| nor)\s+(?:other |visible )?"
    + _LETTERING
    + r")*(?:\s+(?:present|visible|anywhere|at all|of any kind))?(?:\s+(?:in|on|within|across|to)\s+(?:the |this )?(?:frame|view|scene|image|picture|composition|structure|shot|background|foreground))?"
    # "without any text or abstract interpretation": what follows "or" is absent as well, up to the end of the clause
    r"(?:\s+(?:or|nor)\s+[^,.;]{1,40}?(?=\s*(?:[,.;]|$)))?",
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
