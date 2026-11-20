"""One writing session with the text encoder: load it, hold the output head, answer requests in batched passes."""

import time
from dataclasses import dataclass
from typing import Callable, Optional

from backend import memory_management

from .adapters import Adapter
from .generator import Head, generate_batch
from .sampling import SamplingOptions


@dataclass(frozen=True)
class WriteOptions:
    max_new_tokens: int = 512
    thinking: bool = False
    sampling: SamplingOptions = SamplingOptions()


@dataclass
class Written:
    text: str  # the answer without the reasoning block; empty when the reasoning never finished
    raw: str  # everything the model wrote
    finish: str  # stop | length | interrupted | loop
    tokens: int
    seconds: float  # of the pass this text was written in


def split_reasoning(raw: str, thinking: bool) -> str:
    if "</think>" in raw:
        return raw.rpartition("</think>")[2].strip()
    if thinking or raw.lstrip().startswith("<think>"):
        return ""  # the token limit cut the reasoning off before any answer
    return raw.strip()


class Writer:
    """Use as a context manager around all requests of one job: the head is built once and released at the end."""

    def __init__(self, adapter: Adapter):
        self.adapter = adapter
        self.head: Optional[Head] = None

    def __enter__(self):
        if self.adapter.patcher is not None:
            memory_management.load_model_gpu(self.adapter.patcher)
        self.head = Head(self.adapter.transformer, self.adapter.tied)
        return self

    def __exit__(self, *_):
        if self.head is not None:
            self.head.release()
            self.head = None
        return False

    def write_many(
        self,
        system: str,
        users: list[str],
        options: WriteOptions = WriteOptions(),
        seeds: Optional[list[int]] = None,
        rows_per_pass: int = 10,
        should_stop: Optional[Callable[[], bool]] = None,
        on_progress: Optional[Callable[[int, int, int], None]] = None,
    ) -> list[Written]:
        """One answer per request, in passes of at most `rows_per_pass` requests. Request i is sampled with seeds[i].

        on_progress(pass number, passes, tokens written in this pass) is called after every token step.
        After an interrupt the requests of later passes come back empty with finish "interrupted".
        """
        adapter = self.adapter
        seeds = list(seeds) if seeds is not None else list(range(len(users)))
        if len(seeds) != len(users):
            raise ValueError("one seed per request is needed")
        thinking = options.thinking and adapter.can_think
        rows_per_pass = max(1, int(rows_per_pass))
        passes = [range(start, min(start + rows_per_pass, len(users))) for start in range(0, len(users), rows_per_pass)]
        written: list[Written] = []
        stopped = False
        for number, rows in enumerate(passes, start=1):
            if stopped or (should_stop is not None and should_stop()):
                stopped = True
                written += [Written("", "", "interrupted", 0, 0.0) for _ in rows]
                continue
            prompts = [adapter.encode(adapter.prompt(system, users[row], thinking)) for row in rows]
            started = time.perf_counter()
            results = generate_batch(
                adapter.transformer,
                self.head,
                prompts,
                max_new_tokens=options.max_new_tokens,
                stop_ids=adapter.stop_ids,
                seeds=[seeds[row] for row in rows],
                pad_id=adapter.pad_id,
                sampling=options.sampling,
                should_stop=should_stop,
                on_token=(lambda count, number=number: on_progress(number, len(passes), count)) if on_progress is not None else None,
            )
            seconds = time.perf_counter() - started
            for result in results:
                raw = adapter.decode(result.tokens)
                written.append(Written(split_reasoning(raw, thinking), raw, result.finish, len(result.tokens), seconds))
            stopped = any(result.finish == "interrupted" for result in results)
        return written

    def write(self, system: str, user: str, options: WriteOptions = WriteOptions(), seed: int = 0, should_stop: Optional[Callable[[], bool]] = None, on_token: Optional[Callable[[int], None]] = None) -> Written:
        """One request, one answer."""
        progress = (lambda _number, _passes, count: on_token(count)) if on_token is not None else None
        return self.write_many(system, [user], options, [seed], should_stop=should_stop, on_progress=progress)[0]
