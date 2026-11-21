"""One writing session with the text encoder: load it, hold the output head, answer requests in batched passes."""

import time
from dataclasses import dataclass
from typing import Callable, Optional

import torch
from backend import memory_management

from . import int8kernel
from .adapters import Adapter, plain
from .generator import Head, execution_dtype, generate_batch
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
    replayed: bool = False  # the pass ran from a recorded step


BEGUN_MINIMUM = 12  # an answer that was begun with the request must add at least this many tokens


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
        self._mode = None
        self._kernel: Optional[list] = None  # the layers switched to the int8 kernel; None = not decided yet

    def __enter__(self):
        # The whole session runs in inference mode, as Forge's own encode path does. Forge creates a model's
        # weights in inference mode; a script's process() hook runs outside it. Asking Forge to move the encoder
        # to the GPU from out there leaves the weights half-converted (they look like ordinary tensors but have
        # no version counter), and the first operation that takes a view of one fails with "Inference tensors do
        # not track version counter". Qwen3.5's 1-D convolution does that.
        self._mode = torch.inference_mode()
        self._mode.__enter__()
        try:
            if self.adapter.patcher is not None:
                memory_management.load_model_gpu(self.adapter.patcher)
            self.head = Head(self.adapter.transformer, self.adapter.tied, head_file=self.adapter.head_file)
        except BaseException:
            self._leave()
            raise
        return self

    def _leave(self):
        if self._mode is not None:
            mode, self._mode = self._mode, None
            mode.__exit__(None, None, None)

    def __exit__(self, *_):
        if self._kernel:
            int8kernel.switch_off(self._kernel)
        self._kernel = None
        if self.head is not None:
            self.head.release()
            self.head = None
        self._leave()
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
        starts: Optional[list[str]] = None,
        replay: bool = True,
        int8_kernel: bool = True,
    ) -> list[Written]:
        """One answer per request, in passes of at most `rows_per_pass` requests. Request i is sampled with seeds[i].

        starts: for each request, the first words of its answer. They are put where the encoder's answer begins, so
        that it continues them (word for word, with nothing to obey or to forget); the returned text includes them.
        Not used while the encoder reasons first: its answer then begins with the reasoning.

        replay: write from a recorded step (replay.py). int8_kernel: run an int8 ConvRot encoder's layers through
        the fused int8 kernel while writing (int8kernel.py); decided at the first request of the session.

        on_progress(pass number, passes, tokens written in this pass) is called after every token step.
        After an interrupt the requests of later passes come back empty with finish "interrupted".
        """
        adapter = self.adapter
        seeds = list(seeds) if seeds is not None else list(range(len(users)))
        if len(seeds) != len(users):
            raise ValueError("one seed per request is needed")
        thinking = options.thinking and adapter.can_think
        rows_per_pass = max(1, int(rows_per_pass))
        if self._kernel is None:
            stack = getattr(adapter.transformer, "model", None)
            wanted = int8_kernel and getattr(stack, "layers", None) is not None
            self._kernel = int8kernel.switch_on(stack, execution_dtype(next(adapter.transformer.parameters()).device)) if wanted else []
        passes = [range(start, min(start + rows_per_pass, len(users))) for start in range(0, len(users), rows_per_pass)]
        written: list[Written] = []
        stopped = False
        for number, rows in enumerate(passes, start=1):
            if stopped or (should_stop is not None and should_stop()):
                stopped = True
                written += [Written("", "", "interrupted", 0, 0.0) for _ in rows]
                continue
            begun = ["" if thinking or starts is None else plain(starts[row]) for row in rows]
            prompts = [adapter.encode(adapter.prompt(system, users[row], thinking) + start) for row, start in zip(rows, begun)]
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
                min_new_tokens=BEGUN_MINIMUM if any(begun) else 0,
                replay=replay,
            )
            seconds = time.perf_counter() - started
            for result, start in zip(results, begun):
                raw = start + adapter.decode(result.tokens)
                written.append(Written(split_reasoning(raw, thinking), raw, result.finish, len(result.tokens), seconds, result.replayed))
            stopped = any(result.finish == "interrupted" for result in results)
        return written

    def write(self, system: str, user: str, options: WriteOptions = WriteOptions(), seed: int = 0, should_stop: Optional[Callable[[], bool]] = None, on_token: Optional[Callable[[int], None]] = None) -> Written:
        """One request, one answer."""
        progress = (lambda _number, _passes, count: on_token(count)) if on_token is not None else None
        return self.write_many(system, [user], options, [seed], should_stop=should_stop, on_progress=progress)[0]
