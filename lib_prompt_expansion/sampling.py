"""Next-token sampling: penalties, temperature, top k, min p, top p.

The set of filters and their order follow ComfyUI's text generation (comfy/text_encoders/llama.py, GPL-3.0).
Sampling runs on the CPU in float32 with its own generator, so a seed gives the same choice on every device
for the same logits.
"""

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class SamplingOptions:
    do_sample: bool = True
    temperature: float = 0.6
    top_k: int = 20  # 0 = no limit
    top_p: float = 0.8  # 1 = no limit
    min_p: float = 0.0  # 0 = off
    repetition_penalty: float = 1.05  # 1 = off
    presence_penalty: float = 0.0  # 0 = off

    def key(self) -> tuple:
        """What the result depends on. With sampling off only the penalties matter."""
        penalties = (self.repetition_penalty, self.presence_penalty)
        if not self.do_sample or self.temperature <= 0.0:
            return (False, *penalties)
        return (True, self.temperature, self.top_k, self.top_p, self.min_p, *penalties)


class TokenSampler:
    """Picks one token id from a vector of logits. `history` holds the ids written so far."""

    def __init__(self, options: SamplingOptions, seed: int = 0):
        self.options = options
        self.generator = torch.Generator(device="cpu").manual_seed(int(seed) % (2**63))

    def __call__(self, logits: torch.Tensor, history: list[int]) -> int:
        options = self.options
        logits = logits.detach().to(device="cpu", dtype=torch.float32).clone()
        if not torch.isfinite(logits).any():
            raise FloatingPointError("the text encoder produced no finite logits")
        logits = torch.nan_to_num(logits, nan=-torch.inf, posinf=torch.inf, neginf=-torch.inf)

        # Penalties apply to the text written so far only. The request is left out on purpose:
        # penalising its tokens would push the rewrite away from the user's own words.
        if history and (options.repetition_penalty != 1.0 or options.presence_penalty != 0.0):
            used = torch.tensor(sorted(set(history)), dtype=torch.long)
            values = logits[used]
            if options.repetition_penalty != 1.0:
                values = torch.where(values < 0, values * options.repetition_penalty, values / options.repetition_penalty)
            if options.presence_penalty != 0.0:
                values = values - options.presence_penalty
            logits[used] = values

        if self.greedy:
            return int(torch.argmax(logits))

        # Candidates in descending order. With top k set this is a short list, so the later filters cost nothing.
        if 0 < options.top_k < logits.numel():
            values, candidates = torch.topk(logits, options.top_k)
        else:
            values, candidates = torch.sort(logits, descending=True)
        return self.pick(values, candidates)

    @property
    def greedy(self) -> bool:
        return not self.options.do_sample or self.options.temperature <= 0.0

    def pick(self, values: torch.Tensor, candidates: torch.Tensor) -> int:
        """Sample among candidates whose logits `values` are in descending order (penalties and top k already applied)."""
        options = self.options
        values = values.to(device="cpu", dtype=torch.float32)
        if not bool(torch.isfinite(values[0])):
            raise FloatingPointError("the text encoder produced no finite logits")

        if options.temperature != 1.0:
            values = values / options.temperature

        if options.min_p > 0.0:
            probabilities = torch.softmax(values, dim=-1)
            values = values.masked_fill(probabilities < options.min_p * probabilities[0], -torch.inf)

        if options.top_p < 1.0:
            probabilities = torch.softmax(values, dim=-1)
            # drop a token when the ones before it already reach top_p; the most likely token always stays
            drop = (torch.cumsum(probabilities, dim=-1) - probabilities) >= options.top_p
            drop[0] = False
            values = values.masked_fill(drop, -torch.inf)

        choice = torch.multinomial(torch.softmax(values, dim=-1), num_samples=1, generator=self.generator)
        return int(candidates[choice])


class BatchSampler:
    """One TokenSampler per row of a batched pass, with the heavy part done once for all rows on the device.

    Penalties and the candidate selection (top k) run on the device for the whole batch; only the short candidate
    lists go to the CPU, where each row samples with its own seeded generator. For one row this picks the same
    token as TokenSampler.
    """

    def __init__(self, options: SamplingOptions, seeds: list[int], vocab_size: int, device):
        self.options = options
        self.rows = [TokenSampler(options, seed) for seed in seeds]
        self.penalised = options.repetition_penalty != 1.0 or options.presence_penalty != 0.0
        # which tokens each row has written so far (the request is left out on purpose, see TokenSampler)
        self.used = torch.zeros((len(seeds), vocab_size), dtype=torch.bool, device=device) if self.penalised else None

    def record(self, tokens: list[int], active: list[bool]):
        """Note the token each active row has just written."""
        if self.used is None:
            return
        rows = [row for row, on in enumerate(active) if on]
        if rows:
            self.used[rows, [tokens[row] for row in rows]] = True

    def __call__(self, logits: torch.Tensor, active: list[bool]) -> list[int]:
        """logits: (rows, vocabulary). Returns one token per row; the value for an inactive row is meaningless."""
        options = self.options
        logits = torch.nan_to_num(logits.detach().float(), nan=-torch.inf, posinf=torch.inf, neginf=-torch.inf)
        if self.used is not None:
            penalised = logits
            if options.repetition_penalty != 1.0:
                penalised = torch.where(logits < 0, logits * options.repetition_penalty, logits / options.repetition_penalty)
            if options.presence_penalty != 0.0:
                penalised = penalised - options.presence_penalty
            logits = torch.where(self.used, penalised, logits)

        if self.rows[0].greedy:
            values, tokens = logits.max(dim=-1)
            if not bool(torch.isfinite(values).all()):
                raise FloatingPointError("the text encoder produced no finite logits")
            return tokens.tolist()

        count = options.top_k if 0 < options.top_k < logits.shape[-1] else logits.shape[-1]
        values, candidates = torch.topk(logits, count, dim=-1)
        values, candidates = values.cpu(), candidates.cpu()
        return [self.rows[row].pick(values[row], candidates[row]) if on else 0 for row, on in enumerate(active)]
