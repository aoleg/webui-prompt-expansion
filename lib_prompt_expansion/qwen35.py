"""Token-by-token decoding for Qwen3.5 stacks (the MOSS-Image text encoder).

Three of every four Qwen3.5 layers are "linear attention" (a gated delta rule): they keep a running state matrix
and a three-token convolution window, not a key/value cache. The fourth is ordinary attention with a gate.
Forge's layer classes run a whole sequence at once and keep nothing, so this module repeats their forward steps
with that state, using the layers' own weights and Forge's own functions.

Rows of different lengths are left-padded. A pad token must leave no trace: for ordinary attention it is hidden as
a key; for linear attention its convolution input is zeroed and its state update is switched off.
"""

from typing import Optional

import torch
import torch.nn.functional as F
from backend.nn.llm import llama as forge_llama


class Qwen35Decoder:
    """The Qwen3.5 counterpart of generator.Decoder; same interface."""

    def __init__(self, transformer, capacity: int, dtype: Optional[torch.dtype] = None, rows: int = 1, pads: Optional[list[int]] = None):
        from .generator import encoder_device, execution_dtype

        stack = transformer.model
        self.stack = stack
        self.config = stack.config
        self.embed = transformer.get_input_embeddings()
        self.device = encoder_device(transformer)
        self.dtype = dtype or execution_dtype(self.device)
        self.capacity = capacity
        self.rows = rows
        self.length = 0
        self.rotary_dim = int(self.config.head_dim * self.config.partial_rotary_factor)
        self.low = torch.finfo(self.dtype).min / 4

        self.state: list = []
        for layer in stack.layers:
            if layer.layer_type == "linear_attention":
                net = layer.linear_attn
                window = net.conv_kernel_size - 1
                conv_dim = net.key_dim * 2 + net.value_dim
                # [convolution window of earlier inputs, running state matrix]
                self.state.append([torch.zeros((rows, conv_dim, window), device=self.device, dtype=self.dtype), None])
            else:
                shape = (rows, self.config.num_key_value_heads, capacity, self.config.head_dim)
                # zeros, not empty: see generator.Decoder
                self.state.append([torch.zeros(shape, device=self.device, dtype=self.dtype), torch.zeros(shape, device=self.device, dtype=self.dtype)])

        self.key_mask = None
        self.pads = None
        if pads is not None and any(pads):
            if len(pads) != rows:
                raise ValueError("one pad count per row is needed")
            self.pads = torch.tensor(pads, device=self.device)
            self.key_mask = torch.zeros((rows, 1, 1, capacity), device=self.device, dtype=self.dtype)
            for row, pad in enumerate(pads):
                self.key_mask[row, :, :, :pad] = self.low

    def _attention(self, attn, state, x, mask, freqs_cis, past, slot=None):
        """slot: for a replayed step, a one-element index tensor that holds the position. The key and value are
        then written through it and the whole cache is attended (the mask hides what is not written yet)."""
        rows, count, _ = x.shape
        gated = attn.q_proj(x).view(rows, count, attn.num_heads, attn.head_dim * 2)
        query, gate = gated[..., : attn.head_dim], gated[..., attn.head_dim :]
        gate = gate.reshape(rows, count, -1)
        query = attn.q_norm(query).transpose(1, 2)
        key = attn.k_norm(attn.k_proj(x).view(rows, count, attn.num_kv_heads, attn.head_dim)).transpose(1, 2)
        value = attn.v_proj(x).view(rows, count, attn.num_kv_heads, attn.head_dim).transpose(1, 2)
        query, key = forge_llama.apply_partial_rope(query, key, freqs_cis, attn.rotary_dim)

        keys, values = state
        extra = {"enable_gqa": True} if attn.num_heads != attn.num_kv_heads else {}
        if slot is not None:
            keys.index_copy_(2, slot, key.to(keys.dtype))
            values.index_copy_(2, slot, value.to(values.dtype))
            out = forge_llama.attention_function(query, keys, values, attn.num_heads, mask=mask, skip_reshape=True, **extra)
        else:
            keys[:, :, past : past + count] = key
            values[:, :, past : past + count] = value
            out = forge_llama.attention_function(query, keys[:, :, : past + count], values[:, :, : past + count], attn.num_heads, mask=mask, skip_reshape=True, **extra)
        return attn.o_proj(out * gate.sigmoid())

    def _linear(self, net, state, x, real):
        rows, count, _ = x.shape
        projected = net.in_proj_qkv(x)
        z = net.in_proj_z(x)
        beta = net.in_proj_b(x).sigmoid()
        decay = -net.A_log.to(x.device, torch.float32).exp() * F.softplus(net.in_proj_a(x).float() + net.dt_bias.to(x.device, torch.float32))
        if real is not None:  # a pad token: no convolution input, no update, no decay
            projected = projected * real
            beta = beta * real
            decay = decay * real

        window = net.conv_kernel_size - 1
        joined = torch.cat([state[0], projected.transpose(1, 2).to(state[0].dtype)], dim=2)
        state[0] = joined[:, :, -window:].clone()
        # the layer's convolution pads both ends; output i reads inputs i-3..i, so the new tokens are at window..
        mixed = F.silu(net.conv1d(joined)[:, :, window : window + count]).transpose(1, 2)

        query, key, value = mixed.split([net.key_dim, net.key_dim, net.value_dim], dim=-1)
        query = query.reshape(rows, count, net.num_key_heads, net.key_head_dim)
        key = key.reshape(rows, count, net.num_key_heads, net.key_head_dim)
        value = value.reshape(rows, count, net.num_value_heads, net.value_head_dim)
        if net.num_value_heads != net.num_key_heads:
            repeat = net.num_value_heads // net.num_key_heads
            query = query.repeat_interleave(repeat, dim=2)
            key = key.repeat_interleave(repeat, dim=2)

        if count == 1 and state[1] is not None:
            # one step of the gated delta rule: decay the state, correct it towards the new value, read it out
            q = F.normalize(query[:, 0].float(), dim=-1) * (net.key_head_dim**-0.5)
            k = F.normalize(key[:, 0].float(), dim=-1)
            matrix = state[1] * decay[:, 0].exp()[..., None, None]
            remembered = (k.unsqueeze(-2) @ matrix).squeeze(-2)
            delta = (value[:, 0].float() - remembered) * beta[:, 0].float()[..., None]
            matrix = matrix + k.unsqueeze(-1) * delta.unsqueeze(-2)
            state[1] = matrix
            core = (q.unsqueeze(-2) @ matrix).squeeze(-2).unsqueeze(1).to(x.dtype)
        else:
            core, state[1] = forge_llama.torch_chunk_gated_delta_rule(query, key, value, g=decay, beta=beta, initial_state=state[1], output_final_state=True)

        core = net.norm(core.reshape(-1, net.value_head_dim)) * F.silu(z.reshape(-1, net.value_head_dim))
        return net.out_proj(core.reshape(rows, count, -1))

    def feed_rows(self, tokens: list[list[int]]) -> torch.Tensor:
        """Run one equal-length list of tokens per row after everything fed before. Returns (rows, hidden)."""
        if len(tokens) != self.rows:
            raise ValueError(f"{self.rows} rows expected, {len(tokens)} given")
        past, count = self.length, len(tokens[0])
        if count == 0 or any(len(row) != count for row in tokens):
            raise ValueError("every row needs the same number of tokens, at least one")
        if past + count > self.capacity:
            raise ValueError(f"the cache holds {self.capacity} tokens, {past + count} needed")

        config = self.config
        x = self.embed(torch.tensor(tokens, device=self.device, dtype=torch.long)).to(self.dtype)
        position_ids = torch.arange(past, past + count, device=x.device).unsqueeze(0)
        freqs_cis = forge_llama.rope_matrix(forge_llama.precompute_freqs_cis(self.rotary_dim, position_ids, config.rope_theta, config.rope_scale, config.mrope_section, interleaved_mrope=True, device=x.device))

        mask = None
        if count > 1:
            mask = torch.full((count, past + count), self.low, dtype=x.dtype, device=x.device).triu_(past + 1)
        real = None
        if self.key_mask is not None:
            keys = self.key_mask[:, :, :, : past + count].expand(self.rows, 1, count, past + count)
            mask = keys if mask is None else keys + mask
            positions = torch.arange(past, past + count, device=x.device)
            if bool((positions[None, :] < self.pads[:, None]).any()):
                real = (positions[None, :] >= self.pads[:, None]).to(x.dtype).unsqueeze(-1)

        for layer, state in zip(self.stack.layers, self.state):
            hidden = layer.input_layernorm(x)
            if layer.layer_type == "linear_attention":
                hidden = self._linear(layer.linear_attn, state, hidden, real)
            else:
                hidden = self._attention(layer.self_attn, state, hidden, mask, freqs_cis, past)
            x = x + hidden
            x = x + layer.mlp(layer.post_attention_layernorm(x))

        self.length = past + count
        if self.stack.norm is not None:
            x = self.stack.norm(x)
        return x[:, -1]

    def feed(self, ids: list[int]) -> torch.Tensor:
        if self.rows != 1:
            raise ValueError("feed() is for one row; use feed_rows()")
        return self.feed_rows([list(ids)])[0]
