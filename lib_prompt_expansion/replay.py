"""One decoding step recorded as a CUDA graph and replayed for every further token.

A text is written one token at a time, and every token is one trip through the layer stack: 1,200 to 2,300 small
GPU operations, each started from Python. Starting them takes longer than running them. Measured on an RTX 5090
with ten rows, a token took 41 to 52 ms as the decoders write it, of which the GPU was busy for 13 to 36 ms and
idle for the rest. A CUDA graph records the operations of one step once, and a replay starts all of them with
one call, so a token costs what the GPU needs and no more.

A recording repeats the same operations on the same memory. The step here is therefore written for fixed
buffers: the token and its position are read from tensors, the key and the value are written through an index
tensor, and attention runs over the whole cache behind a mask that is opened by one position per token.

Nothing is taken on trust. The first token after the request is written twice, by the decoder as before and by
the first replay, and the recording is used only if the two results agree. In every other case (no CUDA device,
part of the encoder not on the GPU, a layer stack that looks different, an error while recording) the decoder
goes on writing the way it always has, and one line in the console says why.

Forge's --cuda-stream. With that flag every layer call takes one of Forge's side streams and waits for it, so that
weights can be brought to the GPU while the previous layer still computes. A wait for another stream cannot be
recorded, and it costs time on every call even when there is nothing to bring: here the whole text encoder is on
the GPU already (that is checked). So for the time of a writing pass the side streams are switched off
(`streams_off`), and switched on again when the pass ends, whatever happens. Without that the recorded step could
not be used with --cuda-stream at all, and the decoder's own steps took half as long again (78 ms against 51 ms
on the MOSS-Image encoder).

The recording ends at the hidden state. The output head runs after it in the ordinary way: it is one operation,
and it may be anything (a layer Forge loaded, a matrix read from another file).

On the plain llama stacks a replay gives the decoder's numbers bit for bit, in every test so far. On Qwen3.5 the
last bit of an attention result can differ, because attention over the whole cache behind a mask does not round
like attention over the part written so far. That is one step of bf16, it does not grow, and it moves about one
first choice in a hundred to what was the second, a near tie.
"""

from typing import Optional

import torch
from backend import memory_management
from backend.nn.llm import llama as forge_llama

TAG = "[Prompt Expansion]"

# Cosine similarity between the decoder's hidden state and the first replay's, the worst row. The two do the
# same arithmetic and differ by rounding only (measured: 0.9999 and better on every supported encoder).
AGREEMENT = 0.999

NO_CUDA = "the text encoder is not on a CUDA device"  # nothing to tell anybody: there is no faster way here

_said: set[str] = set()
last_reason: Optional[str] = None  # why the last pass was written without the recorded step; None when it was used


class Unsuited(Exception):
    """The step cannot be recorded for this decoder. The message says why."""


def _say(reason: str):
    global last_reason
    last_reason = reason
    if reason != NO_CUDA and reason not in _said:
        _said.add(reason)
        print(f"{TAG} Writing without the recorded step: {reason}.")


def unsuited(decoder) -> Optional[str]:
    """Why no step can be recorded for this decoder, or None. Changes nothing."""
    device = decoder.device
    if device.type != "cuda" or not hasattr(torch.cuda, "CUDAGraph"):
        return NO_CUDA
    if not on_device(decoder):
        return "part of the text encoder is not on the GPU"
    return None


def on_device(decoder) -> bool:
    """Is every weight and buffer of the layer stack on the decoder's device?"""
    device = decoder.device
    return all(tensor.device == device for tensor in list(decoder.stack.parameters()) + list(decoder.stack.buffers()))


class streams_off:
    """Forge's side streams for weights (--cuda-stream), switched off for the time of a writing pass.

    Does nothing without that flag, off a CUDA device, or while part of the text encoder is not on the GPU (then
    the streams have work to do). Forge decides per call from this one number, so nothing else needs to know.
    """

    def __init__(self, decoder):
        self.decoder, self.kept = decoder, None

    def __enter__(self):
        streams = getattr(memory_management, "NUM_STREAMS", 0)
        if streams and self.decoder.device.type == "cuda" and on_device(self.decoder):
            self.kept = streams
            memory_management.NUM_STREAMS = 0
        return self

    def __exit__(self, *_):
        if self.kept is not None:
            memory_management.NUM_STREAMS, self.kept = self.kept, None
        return False


class Step:
    """The fixed buffers of a recorded step, and the recording itself. Subclasses give compute() for a stack."""

    def __init__(self, decoder):
        self.decoder = decoder
        device, rows, capacity, length = decoder.device, decoder.rows, decoder.capacity, decoder.length
        self.tokens = torch.zeros((rows, 1), dtype=torch.long, device=device)
        self.position = torch.full((1,), length, dtype=torch.long, device=device)
        # everything not written yet is hidden; so are the pad positions, for the whole pass
        self.mask = torch.full((rows, 1, 1, capacity), decoder.low, dtype=decoder.dtype, device=device)
        self.mask[..., :length] = 0
        if decoder.key_mask is not None:
            self.mask += decoder.key_mask
        self.rope = None
        self.graph = None
        self.hidden = None
        self.agreement = None

    def rope_table(self, dim: int, dims, interleaved: bool) -> torch.Tensor:
        """The rotation of every position the cache can hold; a step picks its own row through the index tensor."""
        decoder, config = self.decoder, self.decoder.config
        ids = torch.arange(decoder.capacity, device=decoder.device).unsqueeze(0)
        table = forge_llama.rope_matrix(forge_llama.precompute_freqs_cis(dim, ids, config.rope_theta, config.rope_scale, dims, interleaved_mrope=interleaved, device=decoder.device))
        if not torch.is_tensor(table) or table.ndim < 4 or table.shape[2] != decoder.capacity:
            raise Unsuited("the rotary table has an unexpected layout")
        return table

    # --- what a subclass provides

    def compute(self) -> torch.Tensor:
        """One token per row, from the buffers. Returns the normalised hidden state, shape (rows, hidden)."""
        raise NotImplementedError

    def caches(self) -> list:
        """(keys, values) of every layer that has a key/value cache."""
        raise NotImplementedError

    def before(self):
        """A copy of whatever a step changes and cannot simply write again. Taken before the decoder's step."""
        return None

    def rewind(self, saved):
        """Make compute() start from `saved`."""

    def adopt(self):
        """The recording is in use: the decoder's own state follows the buffers from here on."""

    # --- recording and replaying

    def load(self, tokens: list[int]):
        self.tokens.copy_(torch.tensor(tokens, dtype=torch.long).unsqueeze(1))
        self.mask.index_fill_(3, self.position, 0.0)

    def record(self, expected: torch.Tensor, saved):
        """Record the step and replay it once for the token the decoder has just written.

        expected: the decoder's hidden state for that token. Raises Unsuited, with the cache as the decoder left
        it, when the replay does not arrive at the same state.
        """
        slot = int(self.position)
        kept = [(keys[:, :, slot : slot + 1].clone(), values[:, :, slot : slot + 1].clone()) for keys, values in self.caches()]
        try:
            self.rewind(saved)
            graph = torch.cuda.CUDAGraph()
            with streams_off(self.decoder), torch.cuda.graph(graph, capture_error_mode="thread_local"):
                self.hidden = self.compute()
            graph.replay()
            agreement = float(torch.nn.functional.cosine_similarity(self.hidden.float(), expected.float(), dim=-1).min())
            if not agreement >= AGREEMENT:  # also false for NaN
                raise Unsuited(f"the first replay did not agree with the decoder (similarity {agreement:.4f})")
        except BaseException:
            for (keys, values), (key, value) in zip(self.caches(), kept):
                keys[:, :, slot : slot + 1] = key
                values[:, :, slot : slot + 1] = value
            self.hidden = None
            raise
        global last_reason
        last_reason = None
        self.agreement = agreement
        self.graph = graph
        self.adopt()
        self.position += 1

    def feed(self, tokens: list[int]) -> torch.Tensor:
        """The hidden state for one more token per row. It is overwritten by the next call."""
        self.load(tokens)
        self.graph.replay()
        self.position += 1
        self.decoder.length += 1
        return self.hidden


class LlamaStep(Step):
    """The plain block of Forge's llama.py (Qwen3, Qwen3-VL): Attention.forward and TransformerBlock.forward."""

    def __init__(self, decoder):
        if not all(type(layer) is forge_llama.TransformerBlock for layer in decoder.stack.layers):
            raise Unsuited("the layer stack is not made of plain blocks")
        super().__init__(decoder)
        config = decoder.config
        self.rope = self.rope_table(config.head_dim, config.rope_dims, getattr(config, "interleaved_mrope", False))

    def caches(self):
        return [(keys, values) for keys, values, _ in self.decoder.cache]

    def compute(self):
        decoder = self.decoder
        stack, rows = decoder.stack, decoder.rows
        x = decoder.embed(self.tokens).to(decoder.dtype)
        if getattr(stack, "normalize_in", False):
            x = x * (decoder.config.hidden_size**0.5)
        rope = self.rope.index_select(2, self.position)
        for layer, (keys, values, _) in zip(stack.layers, decoder.cache):
            attn = layer.self_attn
            hidden = layer.input_layernorm(x)
            query = attn.q_proj(hidden).view(rows, 1, attn.num_heads, attn.head_dim).transpose(1, 2)
            key = attn.k_proj(hidden).view(rows, 1, attn.num_kv_heads, attn.head_dim).transpose(1, 2)
            value = attn.v_proj(hidden).view(rows, 1, attn.num_kv_heads, attn.head_dim).transpose(1, 2)
            if attn.q_norm is not None:
                query = attn.q_norm(query)
            if attn.k_norm is not None:
                key = attn.k_norm(key)
            query, key = forge_llama.apply_rope(query, key, freqs_cis=rope)
            keys.index_copy_(2, self.position, key.to(keys.dtype))
            values.index_copy_(2, self.position, value.to(values.dtype))
            extra = {"enable_gqa": True} if attn.num_heads != attn.num_kv_heads else {}
            out = forge_llama.attention_function(query, keys, values, attn.num_heads, mask=self.mask, skip_reshape=True, **extra)
            x = x + attn.o_proj(out)
            x = x + layer.mlp(layer.post_attention_layernorm(x))
        if stack.norm is not None:
            x = stack.norm(x)
        return x[:, -1]


class Qwen35Step(Step):
    """Qwen3.5: the decoder's own two layer steps, with the running state of linear attention kept in place."""

    def __init__(self, decoder):
        super().__init__(decoder)
        self.rope = self.rope_table(decoder.rotary_dim, decoder.config.mrope_section, True)
        self.linear: dict[int, list] = {}  # layer index -> [convolution window, state matrix]

    def _is_linear(self, layer) -> bool:
        return layer.layer_type == "linear_attention"

    def caches(self):
        return [(state[0], state[1]) for layer, state in zip(self.decoder.stack.layers, self.decoder.state) if not self._is_linear(layer)]

    def before(self):
        saved = {}
        for index, (layer, state) in enumerate(zip(self.decoder.stack.layers, self.decoder.state)):
            if self._is_linear(layer):
                if state[1] is None:
                    raise Unsuited("linear attention has no running state yet")
                # The state matrix is float32 from the first single step on (the decay is). After a request fed in
                # one piece it may still be in the stack's dtype; a buffer of that dtype would round every step.
                saved[index] = [state[0].clone(), state[1].to(torch.float32, copy=True)]
        return saved

    def rewind(self, saved):
        self.linear = saved

    def adopt(self):
        for index, pair in self.linear.items():
            self.decoder.state[index] = pair

    def compute(self):
        decoder = self.decoder
        stack = decoder.stack
        x = decoder.embed(self.tokens).to(decoder.dtype)
        rope = self.rope.index_select(2, self.position)
        for index, (layer, state) in enumerate(zip(stack.layers, decoder.state)):
            hidden = layer.input_layernorm(x)
            if self._is_linear(layer):
                window, matrix = self.linear[index]
                moved = [window, matrix]
                hidden = decoder._linear(layer.linear_attn, moved, hidden, None)
                # the decoder's step leaves new tensors in the list; a recording needs them where they were
                window.copy_(moved[0])
                matrix.copy_(moved[1])
            else:
                hidden = decoder._attention(layer.self_attn, state, hidden, self.mask, rope, 0, slot=self.position)
            x = x + hidden
            x = x + layer.mlp(layer.post_attention_layernorm(x))
        if stack.norm is not None:
            x = stack.norm(x)
        return x[:, -1]


def build(decoder) -> Step:
    from .generator import Decoder
    from .qwen35 import Qwen35Decoder

    reason = unsuited(decoder)
    if reason:
        raise Unsuited(reason)
    if type(decoder) is Qwen35Decoder:
        return Qwen35Step(decoder)
    if type(decoder) is Decoder:
        return LlamaStep(decoder)
    raise Unsuited(f"no recorded step for {type(decoder).__name__}")


class Stepper:
    """Feeds one token per row and returns the logits: replayed where that works, the decoder's own way otherwise.

    The request must already be in the decoder. `replayed` tells afterwards which way the tokens were written,
    and `reason` why not when they were not.
    """

    def __init__(self, decoder, head, enabled: bool = True):
        self.decoder, self.head = decoder, head
        self.step: Optional[Step] = None
        self.pending = enabled
        self.reason: Optional[str] = None if enabled else "switched off"

    @property
    def replayed(self) -> bool:
        return self.step is not None

    def _gave_up(self, problem: BaseException):
        self.reason = str(problem) if isinstance(problem, Unsuited) else f"{type(problem).__name__}: {str(problem)[:200]}"
        _say(self.reason)

    def feed(self, tokens: list[int]) -> torch.Tensor:
        if self.step is not None:
            return self.head(self.step.feed(tokens))
        rows = [[token] for token in tokens]
        if not self.pending:
            return self.head(self.decoder.feed_rows(rows))

        self.pending = False
        try:
            step = build(self.decoder)
            step.load(tokens)
            saved = step.before()
        except Exception as problem:  # noqa: BLE001 - nothing has been touched: write as before
            self._gave_up(problem)
            return self.head(self.decoder.feed_rows(rows))

        hidden = self.decoder.feed_rows(rows)
        try:
            step.record(hidden, saved)
        except Exception as problem:  # noqa: BLE001 - the decoder's own step stands
            self._gave_up(problem)
            return self.head(hidden)
        self.step = step
        return self.head(step.hidden)

    def release(self):
        self.step = None
