"""Token-by-token text generation with a Forge Neo LLM text encoder.

Forge Neo runs its text encoders as one forward pass and has no text generation. Its transformer blocks
still accept a key/value cache (the code came from ComfyUI with it), but Llama2_.forward never passes one.
This module repeats the steps of Llama2_.forward with that cache, so one new token costs one token's worth
of work. The design follows ComfyUI's BaseGenerate (comfy/text_encoders/llama.py, GPL-3.0).

Several texts are written in one pass: the time of a step is almost all per-call overhead, so ten rows cost
about what one row costs.

Nothing here patches Forge. If backend/nn/llm/llama.py changes its block interface, `Decoder` is the only
place to adapt.
"""

from dataclasses import dataclass, field
from typing import Callable, Optional

import torch
from backend import memory_management
from backend.nn.llm import llama as forge_llama

from .sampling import BatchSampler, SamplingOptions


class UnsupportedEncoder(Exception):
    """The loaded text encoder cannot write text with this module. The message says why."""


@dataclass
class Generation:
    tokens: list[int] = field(default_factory=list)
    finish: str = "stop"  # stop | length | interrupted | loop


# (cycle length, repeats) that count as a runaway loop. A small model can lock onto one token or a short
# phrase and never leave it; the real Qwen3-4B does (measured against the reference implementation).
# One token five times in a row leaves room for numbers such as "10000", whose digits are single tokens.
LOOP_RULES = ((1, 5), (2, 4), (3, 3), (4, 3), (5, 3), (6, 3), (7, 3), (8, 3))


def loop_start(tokens: list[int]) -> Optional[int]:
    """If `tokens` ends in a runaway loop, the index where the repeating stretch begins, else None."""
    total = len(tokens)
    for period, repeats in LOOP_RULES:
        span = period * repeats
        if total < span or any(tokens[total - 1 - i] != tokens[total - 1 - i - period] for i in range(span - period)):
            continue
        start = total - span
        while start > 0 and tokens[start - 1] == tokens[start - 1 + period]:
            start -= 1
        return start
    return None


def execution_dtype(device: torch.device) -> torch.dtype:
    """bf16 where the device runs it fast, else fp32 (as ComfyUI's generation does)."""
    return torch.bfloat16 if memory_management.should_use_bf16(device) else torch.float32


def encoder_device(transformer) -> torch.device:
    # the same rule the encode path uses (SDClipModel.forward)
    return transformer.get_input_embeddings().weight.device


class Decoder:
    """An encoder's layer stack with a key/value cache, for one or more rows written side by side.

    Rows of different lengths are left-padded by the caller; `pads` gives the pad count of each row. Pad positions
    stay hidden from attention for the whole pass. All rows share the positions 0..n: RoPE is relative, so a row
    shifted right by its padding attends exactly as it would alone.
    """

    def __init__(self, transformer, capacity: int, dtype: Optional[torch.dtype] = None, rows: int = 1, pads: Optional[list[int]] = None):
        stack = getattr(transformer, "model", None)
        layers = getattr(stack, "layers", None)
        if layers is None or not all(isinstance(layer, (forge_llama.TransformerBlock, forge_llama.TransformerBlockGemma2)) for layer in layers):
            raise UnsupportedEncoder(f"{type(transformer).__name__} has layers without a key/value cache")
        if getattr(stack.config, "transformer_type", "llama") == "gemma3":
            raise UnsupportedEncoder("Gemma 3 layer stacks are not supported")

        self.stack = stack
        self.config = stack.config
        self.embed = transformer.get_input_embeddings()
        self.device = encoder_device(transformer)
        self.dtype = dtype or execution_dtype(self.device)
        self.capacity = capacity
        self.rows = rows
        self.length = 0

        shape = (rows, self.config.num_key_value_heads, capacity, self.config.head_dim)
        self.cache = [(torch.empty(shape, device=self.device, dtype=self.dtype), torch.empty(shape, device=self.device, dtype=self.dtype), 0) for _ in layers]

        self.low = torch.finfo(self.dtype).min / 4
        self.key_mask = None
        if pads is not None and any(pads):
            if len(pads) != rows:
                raise ValueError("one pad count per row is needed")
            self.key_mask = torch.zeros((rows, 1, 1, capacity), device=self.device, dtype=self.dtype)
            for row, pad in enumerate(pads):
                self.key_mask[row, :, :, :pad] = self.low

    def feed_rows(self, tokens: list[list[int]]) -> torch.Tensor:
        """Run one equal-length list of tokens per row after everything fed before.

        Returns the normalised hidden state of each row's last token, shape (rows, hidden).
        """
        if len(tokens) != self.rows:
            raise ValueError(f"{self.rows} rows expected, {len(tokens)} given")
        past, count = self.length, len(tokens[0])
        if count == 0 or any(len(row) != count for row in tokens):
            raise ValueError("every row needs the same number of tokens, at least one")
        if past + count > self.capacity:
            raise ValueError(f"the cache holds {self.capacity} tokens, {past + count} needed")

        config = self.config
        x = self.embed(torch.tensor(tokens, device=self.device, dtype=torch.long)).to(self.dtype)
        if getattr(self.stack, "normalize_in", False):
            x = x * (config.hidden_size**0.5)

        position_ids = torch.arange(past, past + count, device=x.device).unsqueeze(0)
        freqs_cis = forge_llama.precompute_freqs_cis(config.head_dim, position_ids, config.rope_theta, config.rope_scale, config.rope_dims, interleaved_mrope=getattr(config, "interleaved_mrope", False), device=x.device)
        # Forge builds the rotation matrix from these in every layer; built once here it is passed through as it is.
        # Same numbers, 14 % less time per token on Qwen3-4B.
        freqs_cis = forge_llama.rope_matrix(freqs_cis)

        mask = None
        if count > 1:
            # row = new token, column = every token so far: a new token sees the past and itself, not the ones after it
            mask = torch.full((count, past + count), self.low, dtype=x.dtype, device=x.device).triu_(past + 1)
        if self.key_mask is not None:
            keys = self.key_mask[:, :, :, : past + count].expand(self.rows, 1, count, past + count)
            mask = keys if mask is None else keys + mask

        for index, layer in enumerate(self.stack.layers):
            x, self.cache[index] = layer(x=x, attention_mask=mask, freqs_cis=freqs_cis, optimized_attention=forge_llama.attention_function, past_key_value=self.cache[index])

        self.length = past + count
        if self.stack.norm is not None:
            x = self.stack.norm(x)
        return x[:, -1]

    def feed(self, ids: list[int]) -> torch.Tensor:
        """The single-row form: returns the last hidden state with shape (hidden,)."""
        if self.rows != 1:
            raise ValueError("feed() is for one row; use feed_rows()")
        return self.feed_rows([list(ids)])[0]


def open_decoder(transformer, capacity: int, dtype: Optional[torch.dtype] = None, rows: int = 1, pads: Optional[list[int]] = None):
    """The decoder that fits the encoder's layer stack."""
    if isinstance(getattr(transformer, "model", None), forge_llama.Qwen35Transformer):
        from .qwen35 import Qwen35Decoder

        return Qwen35Decoder(transformer, capacity, dtype=dtype, rows=rows, pads=pads)
    return Decoder(transformer, capacity, dtype=dtype, rows=rows, pads=pads)


class Head:
    """Hidden state to vocabulary logits. Holds device memory: call release() when the batch is done.

    Three sources, in this order: an lm_head that Forge loaded with the encoder; the input embedding table when
    the encoder ties the two; `head_file`, an encoder file that still contains the head Forge dropped.
    """

    def __init__(self, transformer, tied: bool, dtype: Optional[torch.dtype] = None, head_file: Optional[str] = None):
        stack = transformer.model
        self.vocab_size = stack.config.vocab_size
        self.module = getattr(stack, "lm_head", None)
        self.weight = None
        self.checked = False
        if self.module is not None:
            return
        if not tied:
            if not head_file:
                raise UnsupportedEncoder(f"{type(transformer).__name__} needs its output head, and no loaded text encoder file contains one")
            from .headfile import load_head

            device = encoder_device(transformer)
            self.weight = load_head(head_file, device, dtype or execution_dtype(device))
            return
        # Tied embeddings: the output matrix is the input embedding table. Asking the embedding layer for every
        # row goes through Forge's own casting, so quantised, GGUF and offloaded tables resolve the usual way.
        embed = transformer.get_input_embeddings()
        device = encoder_device(transformer)
        rows = torch.arange(self.vocab_size, device=device, dtype=torch.long)
        self.weight = embed(rows).to(dtype or execution_dtype(device))

    def __call__(self, hidden: torch.Tensor) -> torch.Tensor:
        """hidden: (hidden,) or (rows, hidden). Returns logits of the same leading shape."""
        single = hidden.ndim == 1
        states = hidden.unsqueeze(0) if single else hidden
        if self.module is not None:
            logits = self.module(states)
        else:
            logits = torch.nn.functional.linear(states, self.weight.to(states.dtype))
        if not self.checked:
            # an output head whose weights were not in the encoder file is left at zero by the loader
            if not bool(torch.isfinite(logits).any()) or float(logits.float().abs().max()) == 0.0:
                raise UnsupportedEncoder("the output head has no weights (the encoder file does not contain it)")
            self.checked = True
        return logits[0] if single else logits

    def release(self):
        self.weight = None


@torch.inference_mode()
def generate_batch(
    transformer,
    head: Head,
    prompts: list[list[int]],
    *,
    max_new_tokens: int,
    stop_ids: tuple[int, ...],
    seeds: list[int],
    pad_id: int = 0,
    sampling: SamplingOptions = SamplingOptions(),
    dtype: Optional[torch.dtype] = None,
    stop_on_loop: bool = True,
    should_stop: Optional[Callable[[], bool]] = None,
    on_token: Optional[Callable[[int], None]] = None,
) -> list[Generation]:
    """Continue every prompt, all in one pass. Row i is sampled with seeds[i].

    A stop token is not part of a result, and neither is a runaway loop (finish "loop"). The pass ends when every
    row has finished; a finished row is fed the pad token and ignored.
    """
    rows = len(prompts)
    if len(seeds) != rows:
        raise ValueError("one seed per prompt is needed")
    results = [Generation() for _ in prompts]
    if rows == 0:
        return results
    if max_new_tokens <= 0:
        for result in results:
            result.finish = "length"
        return results

    width = max(len(prompt) for prompt in prompts)
    pads = [width - len(prompt) for prompt in prompts]
    decoder = open_decoder(transformer, capacity=width + max_new_tokens, dtype=dtype, rows=rows, pads=pads)
    sampler = BatchSampler(sampling, seeds, getattr(head, "vocab_size", None) or decoder.config.vocab_size, decoder.device)
    stops = set(stop_ids)
    active = [True] * rows

    hidden = decoder.feed_rows([[pad_id] * pad + list(prompt) for pad, prompt in zip(pads, prompts)])
    written = 0
    while True:
        tokens = sampler(head(hidden), active)
        feed = [pad_id] * rows
        for row, token in enumerate(tokens):
            if not active[row]:
                continue
            result = results[row]
            if token in stops:
                result.finish, active[row] = "stop", False
                continue
            result.tokens.append(token)
            feed[row] = token
            cut = loop_start(result.tokens) if stop_on_loop else None
            if cut is not None:
                del result.tokens[cut:]  # the first round of the loop is already off the rails; drop it too
                result.finish, active[row] = "loop", False
            elif len(result.tokens) >= max_new_tokens:
                result.finish, active[row] = "length", False
        written += 1
        if on_token is not None:
            on_token(written)
        if not any(active):
            break
        if should_stop is not None and should_stop():
            for row in range(rows):
                if active[row]:
                    results[row].finish = "interrupted"
            break
        sampler.record(feed, active)
        hidden = decoder.feed_rows([[token] for token in feed])
    return results


def generate(
    transformer,
    head: Head,
    input_ids: list[int],
    *,
    max_new_tokens: int,
    stop_ids: tuple[int, ...],
    sampling: SamplingOptions = SamplingOptions(),
    seed: int = 0,
    dtype: Optional[torch.dtype] = None,
    stop_on_loop: bool = True,
    should_stop: Optional[Callable[[], bool]] = None,
    on_token: Optional[Callable[[int], None]] = None,
) -> Generation:
    """Continue one prompt: generate_batch with a single row."""
    return generate_batch(transformer, head, [list(input_ids)], max_new_tokens=max_new_tokens, stop_ids=stop_ids, seeds=[seed], sampling=sampling, dtype=dtype, stop_on_loop=stop_on_loop, should_stop=should_stop, on_token=on_token)[0]
