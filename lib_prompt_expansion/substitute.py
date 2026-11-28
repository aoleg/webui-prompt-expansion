"""What happens at Generate: every prompt of the job is rewritten by the text encoder and put back in place.

Whatever goes wrong, the job goes on with the prompts as typed.
"""

import logging
import os
import re
import secrets
import time
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Any, Callable, Optional

from . import cleanup
from . import fetch as head_fetch
from . import presets as preset_files
from . import weights as prompt_weights
from .adapters import after_writing, choose, tidy
from .generator import UnsupportedEncoder
from .sampling import SamplingOptions
from .writer import WriteOptions, Writer

log = logging.getLogger("prompt_expansion")

# <lora:name:1> and the like, as modules/extra_networks.py reads them
EXTRA_NETWORK = re.compile(r"<\w+:[^>]+>")

CACHE_SIZE = 64
_cache: "OrderedDict[tuple, Optional[str]]" = OrderedDict()

last_start_seed: Optional[int] = None  # for the "reuse" button of the UI


@dataclass(frozen=True)
class Settings:
    instruction: str  # the expansion prompt, "{words}" not yet filled in
    preset_name: str = "Custom"
    words: int = 0  # the length to ask for; 0 = none
    seed: int = -1  # -1 = a new random start for every job
    sampling: SamplingOptions = SamplingOptions()
    max_new_tokens: int = 512
    rows_per_pass: int = 10
    thinking: bool = False
    begin: bool = False  # the expansion starts with the prompt itself: the encoder's answer is begun with it
    replay: bool = True  # write from a recorded decoding step (replay.py)
    int8_kernel: bool = True  # int8 ConvRot encoders write through the fused int8 kernel (int8kernel.py)
    writer_file: str = ""  # a model file of its own to write with (adapters.file_adapter); empty: the checkpoint's text encoder
    after_writing: str = "keep"  # what becomes of a writer of its own when the prompts are written: keep, ram, release (adapters.after_writing)


@dataclass
class Report:
    writer: str = ""
    total: int = 0  # prompts in the job
    expanded: int = 0
    kept: int = 0  # left as typed: empty, failed, refused or interrupted
    reused: int = 0  # taken from the cache or shared with an equal request
    start_seed: Optional[int] = None
    seconds: float = 0.0
    notes: list[str] = field(default_factory=list)
    weighted: bool = False  # a prompt of the job, as it will be encoded, carries a weight
    tokens: int = 0  # tokens written in this job
    writing: float = 0.0  # seconds spent writing them

    def line(self) -> str:
        if not self.writer:
            return "; ".join(self.notes) or "nothing done"
        text = f"{self.expanded} of {self.total} prompts expanded by {self.writer} in {self.seconds:.1f} s"
        if self.tokens and self.writing > 0:
            text += f" ({self.tokens} tokens, {self.tokens / self.writing:.0f} a second)"
        if self.start_seed is not None:
            text += f", seeds from {self.start_seed}"
        if self.reused:
            text += f", {self.reused} reused"
        if self.kept:
            text += f", {self.kept} kept as typed"
        if self.notes:
            text += " (" + "; ".join(dict.fromkeys(self.notes)) + ")"
        return text


class PerImage:
    """An infotext value that differs from image to image, found by the image's own prompt.

    Forge indexes a list in extra_generation_params by the position inside the current batch, so with a batch count
    above one every image would get the values of the first batch. A callable is asked for each image instead, and
    is handed that image's prompt.
    """

    def __init__(self, prompts: list[str], values: list):
        self.values = list(values)
        self._by_prompt = {self._key(prompt): value for prompt, value in zip(prompts, values)}

    @staticmethod
    def _key(prompt) -> str:
        return re.sub(r"\s+", " ", EXTRA_NETWORK.sub("", str(prompt or ""))).strip()

    def __call__(self, **names):
        key = self._key(names.get("prompt_text"))
        if key in self._by_prompt:
            return self._by_prompt[key]
        index = names.get("index")
        return self.values[index] if isinstance(index, int) and 0 <= index < len(self.values) else None

    def __iter__(self):
        return iter(self.values)

    def __len__(self):
        return len(self.values)

    def __getitem__(self, index):
        return self.values[index]

    def count(self, value) -> int:
        return self.values.count(value)

    def __eq__(self, other):
        return self.values == (other.values if isinstance(other, PerImage) else other)

    def __repr__(self):
        return f"PerImage({self.values!r})"


def split_tags(prompt: str) -> tuple[str, list[str]]:
    """The prompt without its extra-network tags, and the tags in order."""
    tags = EXTRA_NETWORK.findall(prompt)
    return re.sub(r"\s{2,}", " ", EXTRA_NETWORK.sub("", prompt)).strip(), tags


def join_tags(text: str, tags: list[str]) -> str:
    return text if not tags else (text + " " + " ".join(tags)).strip()


def clear_cache():
    _cache.clear()


def _remember(key, value):
    _cache[key] = value
    _cache.move_to_end(key)
    while len(_cache) > CACHE_SIZE:
        _cache.popitem(last=False)


def expand(
    p,
    settings: Settings,
    *,
    state: Any = None,
    adapter: Any = None,
    comments_follow: bool = False,
    writer_factory: Callable = Writer,
    weights: bool = True,
) -> Report:
    """Rewrite p.all_prompts in place. Never raises: on any failure the prompts stay as typed and the report says why.

    weights: (phrase:1.4) and the like are taken out before the writing and marked again in the result. False when
        Forge's Emphasis setting is "None", where brackets are plain characters for every extension too.
    """
    report = Report(total=len(getattr(p, "all_prompts", None) or []))
    started = time.perf_counter()
    try:
        _expand(p, settings, state, adapter, comments_follow, writer_factory, report, weights)
    except UnsupportedEncoder as reason:
        report.writer = ""
        report.notes = [f"not available{'' if settings.writer_file else ' for this checkpoint'}: {reason}"]
    except Exception as problem:  # noqa: BLE001 - an expansion must never stop a generation
        log.exception("Prompt expansion failed; the prompts are used as typed.")
        report.notes.append(f"failed: {type(problem).__name__}: {problem}")
    tidy()  # a writer that was released during the job goes now that the job has let go of it
    report.seconds = time.perf_counter() - started
    return report


def _expand(p, settings, state, adapter, comments_follow, writer_factory, report, weights=True):
    global last_start_seed
    originals = list(p.all_prompts or [])
    if not originals:
        report.notes.append("no prompts")
        return
    if adapter is None:

        def stopped():
            return bool(state is not None and (getattr(state, "interrupted", False) or getattr(state, "skipped", False) or getattr(state, "stopping_generation", False)))

        def announce(line):
            if state is not None:
                state.textinfo = line

        def fetch(download, loaded, encoder, otherwise):
            try:
                return head_fetch.ensure(download, loaded, encoder=encoder, fallback=otherwise, should_stop=stopped, announce=announce)
            finally:
                announce(None)

        adapter = choose(getattr(p, "sd_model", None), settings.writer_file or None, fetch=fetch)
    report.writer = adapter.label
    if getattr(adapter, "head_borrowed", False):
        report.notes.append("output head from " + os.path.basename(adapter.head_file))

    sampling = settings.sampling
    sampled = sampling.do_sample and sampling.temperature > 0.0
    thinking = settings.thinking and adapter.can_think
    instruction = settings.instruction.strip()
    # "# if:" blocks and "{random:...}" samples give each request an instruction of its own
    conditional = preset_files.has_conditions(instruction) or preset_files.has_random(instruction)
    options = WriteOptions(max_new_tokens=preset_files.token_budget(settings.words, settings.max_new_tokens, thinking), thinking=thinking, sampling=sampling)

    start_seed = int(settings.seed) if settings.seed is not None and int(settings.seed) >= 0 else secrets.randbelow(2**32)
    last_start_seed = start_seed
    if sampled:
        report.start_seed = start_seed

    # one request per prompt; with sampling off equal requests share one answer
    # extra-network tags and weights are not for the encoder: they are taken out here and put back at the end
    tagged = [split_tags(original) for original in originals]
    marks = [prompt_weights.strip(text) if weights else prompt_weights.Stripped(text) for text, _tags in tagged]
    parts = [(mark.plain, tags) for mark, (_text, tags) in zip(marks, tagged)]
    seeds = [start_seed + index for index in range(len(originals))]
    # the random samples follow the request's seed; with sampling off a request always gets the same text, so one fixed draw
    systems = [preset_files.fill(preset_files.select(instruction, text) if conditional else instruction, settings.words, seed if sampled else 0, request=text) for (text, _tags), seed in zip(parts, seeds)]
    # the request as the encoder gets it: the preset's "# replace in request:" rules (conditions look at the request as typed)
    asked = [preset_files.rewrite_request(instruction, text) if conditional and text else text for text, _tags in parts]
    keys: list[Optional[tuple]] = []
    for (text, _tags), seed, system in zip(parts, seeds, systems):
        keys.append((id(adapter.transformer), adapter.label, system, text, options.max_new_tokens, thinking, bool(settings.begin), sampling.key(), seed if sampled else None) if text else None)

    answers: dict[tuple, Optional[str]] = {}
    todo: list[int] = []
    for index, key in enumerate(keys):
        if key is None or key in answers:
            continue
        if key in _cache:
            answers[key] = _cache[key]
            _cache.move_to_end(key)
            report.reused += 1
        else:
            answers[key] = None
            todo.append(index)
    report.reused += sum(1 for key in keys if key is not None) - len(answers)

    if todo:
        def should_stop() -> bool:
            return bool(state is not None and (getattr(state, "interrupted", False) or getattr(state, "skipped", False) or getattr(state, "stopping_generation", False)))

        def progress(number: int, passes: int, count: int):
            if state is not None:
                state.textinfo = f"Prompt expansion: pass {number} of {passes}, {count} tokens"

        try:
            with writer_factory(adapter) as session:
                begun = {"starts": [asked[index] for index in todo]} if settings.begin else {}
                # named only when switched off, as `starts` is named only when used: a writer need not know them
                speed = {name: False for name in ("replay", "int8_kernel") if not getattr(settings, name)}
                started = time.perf_counter()
                written = session.write_many([systems[index] for index in todo] if conditional else systems[todo[0]], [asked[index] for index in todo], options, [seeds[index] for index in todo], rows_per_pass=settings.rows_per_pass, should_stop=should_stop, on_progress=progress, **begun, **speed)
                report.writing = time.perf_counter() - started
        finally:
            # a writer of its own makes room for the image model now, when the settings say so; also after a failure
            freed = after_writing(adapter, getattr(settings, "after_writing", "keep"))
            if freed:
                report.notes.append(freed)
        if state is not None:
            state.textinfo = None
        report.tokens = sum(int(getattr(result, "tokens", 0) or 0) for result in written)
        # the slow way is worth a word in every job's line, not only once per session
        if getattr(settings, "replay", True) and report.tokens and not any(getattr(result, "replayed", False) for result in written):
            from . import replay as recorded_step

            reason = getattr(recorded_step, "last_reason", None)
            if reason and reason != recorded_step.NO_CUDA:
                report.notes.append(f"written token by token: {reason}")

        for index, result in zip(todo, written):
            cleaned = cleanup.prepare(result.text, result.finish, source=asked[index], instruction=systems[index], words=settings.words, comments_follow=comments_follow, begins=settings.begin)
            report.notes += cleaned.notes
            text = cleaned.text or None
            if text and conditional:
                # the preset's own work on the answer: "# replace in answer:" rules, and what must be in the text
                # ("# add if missing:") but was left out
                text, used = preset_files.finish(instruction, parts[index][0], text)
                if used:
                    report.notes.append("a missing description was added")
            answers[keys[index]] = text
            if result.finish != "interrupted":
                _remember(keys[index], text)

    expanded: list[str] = []
    used_seeds: list[Optional[int]] = []
    sources: list[Optional[str]] = []
    for original, (text, tags), key, seed, mark in zip(originals, parts, keys, seeds, marks):
        answer = answers.get(key) if key is not None else None
        if answer:
            marked = prompt_weights.apply(answer, mark)
            report.notes += marked.notes()
            expanded.append(join_tags(marked.text, tags))
            used_seeds.append(seed if sampled else None)
            sources.append(original)
            report.expanded += 1
        else:
            expanded.append(original)
            used_seeds.append(None)
            sources.append(None)
            report.kept += 1
    if not report.expanded:
        return

    p.all_prompts = expanded
    p.main_prompt = expanded[0]
    if weights:
        report.weighted = any(weight != 1.0 for prompt in expanded for _text, weight in prompt_weights.parse(EXTRA_NETWORK.sub("", prompt)))
    hires = getattr(p, "all_hr_prompts", None)
    if hires and len(hires) == len(originals):
        # the hires pass follows the expansion unless the user typed a separate hires prompt
        p.all_hr_prompts = [new if old == original else old for old, original, new in zip(hires, originals, expanded)]

    extra = p.extra_generation_params
    extra["Prompt expansion"] = PerImage(expanded, [settings.preset_name if source is not None else None for source in sources])
    # which model wrote: the text encoder's name, or the model and file of a writer of its own
    wrote = adapter.describe() if callable(getattr(adapter, "describe", None)) else report.writer
    extra["Expansion writer"] = PerImage(expanded, [wrote if source is not None else None for source in sources])
    if sampled:
        extra["Expansion seed"] = PerImage(expanded, used_seeds)
    extra["Expansion source"] = PerImage(expanded, sources)
