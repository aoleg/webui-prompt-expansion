"""What happens at Generate: every prompt of the job is rewritten by the text encoder and put back in place.

Whatever goes wrong, the job goes on with the prompts as typed.
"""

import logging
import re
import secrets
import time
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Any, Callable, Optional

from . import cleanup
from . import presets as preset_files
from .adapters import find_adapter
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

    def line(self) -> str:
        if not self.writer:
            return "; ".join(self.notes) or "nothing done"
        text = f"{self.expanded} of {self.total} prompts expanded by {self.writer} in {self.seconds:.1f} s"
        if self.start_seed is not None:
            text += f", seeds from {self.start_seed}"
        if self.reused:
            text += f", {self.reused} reused"
        if self.kept:
            text += f", {self.kept} kept as typed"
        if self.notes:
            text += " (" + "; ".join(dict.fromkeys(self.notes)) + ")"
        return text


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
) -> Report:
    """Rewrite p.all_prompts in place. Never raises: on any failure the prompts stay as typed and the report says why."""
    report = Report(total=len(getattr(p, "all_prompts", None) or []))
    started = time.perf_counter()
    try:
        _expand(p, settings, state, adapter, comments_follow, writer_factory, report)
    except UnsupportedEncoder as reason:
        report.writer = ""
        report.notes = [f"not available for this checkpoint: {reason}"]
    except Exception as problem:  # noqa: BLE001 - an expansion must never stop a generation
        log.exception("Prompt expansion failed; the prompts are used as typed.")
        report.notes.append(f"failed: {type(problem).__name__}: {problem}")
    report.seconds = time.perf_counter() - started
    return report


def _expand(p, settings, state, adapter, comments_follow, writer_factory, report):
    global last_start_seed
    originals = list(p.all_prompts or [])
    if not originals:
        report.notes.append("no prompts")
        return
    if adapter is None:
        adapter = find_adapter(p.sd_model)
    report.writer = adapter.label

    sampling = settings.sampling
    sampled = sampling.do_sample and sampling.temperature > 0.0
    thinking = settings.thinking and adapter.can_think
    system = preset_files.fill(settings.instruction.strip(), settings.words)
    options = WriteOptions(max_new_tokens=preset_files.token_budget(settings.words, settings.max_new_tokens, thinking), thinking=thinking, sampling=sampling)

    start_seed = int(settings.seed) if settings.seed is not None and int(settings.seed) >= 0 else secrets.randbelow(2**32)
    last_start_seed = start_seed
    if sampled:
        report.start_seed = start_seed

    # one request per prompt; with sampling off equal requests share one answer
    parts = [split_tags(original) for original in originals]
    seeds = [start_seed + index for index in range(len(originals))]
    keys: list[Optional[tuple]] = []
    for (text, _tags), seed in zip(parts, seeds):
        keys.append((id(adapter.transformer), adapter.label, system, text, options.max_new_tokens, thinking, sampling.key(), seed if sampled else None) if text else None)

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

        with writer_factory(adapter) as session:
            written = session.write_many(system, [parts[index][0] for index in todo], options, [seeds[index] for index in todo], rows_per_pass=settings.rows_per_pass, should_stop=should_stop, on_progress=progress)
        if state is not None:
            state.textinfo = None

        for index, result in zip(todo, written):
            cleaned = cleanup.clean(result.text, result.finish, source=parts[index][0])
            report.notes += cleaned.notes
            text = cleanup.neutralise(cleaned.text, comments_follow) if cleaned.text else None
            answers[keys[index]] = text
            if result.finish != "interrupted":
                _remember(keys[index], text)

    expanded: list[str] = []
    used_seeds: list[Optional[int]] = []
    sources: list[Optional[str]] = []
    for original, (text, tags), key, seed in zip(originals, parts, keys, seeds):
        answer = answers.get(key) if key is not None else None
        if answer:
            expanded.append(join_tags(answer, tags))
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
    hires = getattr(p, "all_hr_prompts", None)
    if hires and len(hires) == len(originals):
        # the hires pass follows the expansion unless the user typed a separate hires prompt
        p.all_hr_prompts = [new if old == original else old for old, original, new in zip(hires, originals, expanded)]

    extra = p.extra_generation_params
    extra["Prompt expansion"] = [settings.preset_name if source is not None else None for source in sources]
    if sampled:
        extra["Expansion seed"] = used_seeds
    extra["Expansion source"] = sources
