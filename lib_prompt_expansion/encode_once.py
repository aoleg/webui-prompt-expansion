"""Encode every prompt of a job in one go.

Forge encodes the prompts of one batch at a time, just before it samples that batch, and remembers only the last
result. A job whose images all have the same prompt therefore encodes once. After an expansion every image has a
text of its own, so the text encoder is needed again for every batch of a batch count; where the diffusion model,
the encoder and the VAE do not fit into video memory together, that means loading the encoder and reloading the
diffusion model for every image.

This module fills Forge's own cache ahead of time. `p.setup_conds` is wrapped for the job:

- at the first batch Forge encodes as always. Right after, while the encoder is still loaded and every extension
  has set up its state for a batch, the prompts of all later batches are encoded through the same Forge function
  (`p.get_conds_with_caching`), each into a cache entry of its own;
- at a later batch the entry of that batch's prompts is put into Forge's cache just before Forge looks there. Forge
  compares it with its own key (prompts, steps, size, extra networks, model, emphasis setting ...) and uses it only
  if everything matches; otherwise it encodes as always.

Nothing is assumed about what an encode does: it runs through whatever Forge and other extensions have installed.
The risk is state that differs between batches without being part of Forge's key, so the step stands down where
such state is known: prompt weights (NegPiP sets itself up per batch from them), an input or reference image, and
hires fix. It also stands down for the rest of a job if another extension clears Forge's cache.
"""

from typing import Any, Callable, Optional

LIMIT_BYTES = 1 << 30  # stored conditioning above this is not worth the memory: the rest is encoded as always
TAG = "[Prompt Expansion]"


def _size(value: Any, depth: int = 0) -> int:
    """Bytes of the tensors inside a conditioning object, whatever its shape."""
    if depth > 8 or value is None:
        return 0
    if hasattr(value, "element_size") and hasattr(value, "nelement"):
        return int(value.element_size() * value.nelement())
    if isinstance(value, dict):
        return sum(_size(item, depth + 1) for item in value.values())
    if isinstance(value, (list, tuple)):
        return sum(_size(item, depth + 1) for item in value)
    inner = getattr(value, "__dict__", None)
    return sum(_size(item, depth + 1) for item in inner.values()) if isinstance(inner, dict) else 0


def unsuitable(p, weighted: bool = False) -> Optional[str]:
    """Why this job's prompts are encoded one batch at a time, or None when they can be encoded in one go."""
    batches, size = int(getattr(p, "n_iter", 1) or 1), int(getattr(p, "batch_size", 1) or 1)
    prompts = list(getattr(p, "all_prompts", None) or [])
    if batches < 2 or len({tuple(prompts[n * size : (n + 1) * size]) for n in range(batches)}) < 2:
        return ""  # nothing to gain, nothing to say
    if weighted:
        return "the prompts carry weights"
    if getattr(p, "init_images", None):
        return "the job has an input image"
    if getattr(p, "enable_hr", False):
        return "hires fix is on"
    model = getattr(p, "sd_model", None)
    try:
        if getattr(model, "ref_latents", None) or getattr(model, "ini_latent", None) is not None:
            return "the model has a reference image"
        images = getattr(model, "condition_images", None)
        if callable(images) and len(images() or []) > 0:
            return "the model has a reference image"
    except Exception:  # noqa: BLE001 - a model that cannot say is treated as having one
        return "the model's reference images could not be checked"
    return None


class EncodeOnce:
    """Wraps p.setup_conds for one job. See the module text."""

    def __init__(self, p, should_stop: Optional[Callable[[], bool]] = None, announce: Optional[Callable[[Optional[str]], None]] = None, log: Callable[[str], None] = print):
        self.p, self.should_stop, self.announce, self.log = p, should_stop, announce, log
        self.store: dict[tuple, list] = {}
        self.first_done = False
        self.encoded = self.used = 0
        self.original = p.setup_conds
        p.setup_conds = self.setup_conds

    def setup_conds(self):
        p = self.p
        if getattr(p, "is_hr_pass", False):
            return self.original()
        if not self.first_done:
            result = self.original()
            self.first_done = True
            try:
                # the first batch's own entry is kept too: Forge's one slot is about to be reused, and a later
                # batch may have the same prompts again
                cache = getattr(p, "cached_c", None)
                if isinstance(cache, list) and len(cache) >= 2 and cache[0] is not None:
                    self.store[tuple(p.prompts)] = list(cache)
                self._encode_the_rest()
            except Exception as problem:  # noqa: BLE001 - this is an optimisation; the job goes on without it
                self.store.clear()
                self.log(f"{TAG} The prompts could not be encoded in one go ({type(problem).__name__}: {problem}); they are encoded one image at a time.")
            finally:
                if self.announce is not None:
                    self.announce(None)
            return result

        entry = self.store.get(tuple(p.prompts)) if self.store else None
        if entry is not None:
            cache = getattr(p, "cached_c", None)
            if not isinstance(cache, list) or len(cache) < 2 or cache[0] is None:
                # Forge never empties its cache between batches: another extension did, and knows why
                self.store.clear()
            else:
                cache[: len(entry)] = entry[: len(cache)]
                self.used += 1
        return self.original()

    def _encode_the_rest(self):
        from modules import extra_networks, prompt_parser

        p = self.p
        size, batches = int(p.batch_size), int(p.n_iter)
        first, networks = tuple(p.prompts), p.extra_network_data
        model = p.sd_model
        kept = dict(getattr(model, "extra_generation_params", None) or {})
        waiting = []
        for n in range(batches):
            prompts = list(p.all_prompts[n * size : (n + 1) * size])
            if not prompts:
                break
            texts, data = extra_networks.parse_prompts(prompts)
            key = tuple(texts)
            # another set of LoRAs changes the text encoder itself: such a batch is left to Forge
            if key != first and key not in self.store and data == networks and (key, data) not in [(k, d) for k, _, d in waiting]:
                waiting.append((key, texts, data))
        total = 0
        try:
            for number, (key, texts, data) in enumerate(waiting, start=1):
                if self.should_stop is not None and self.should_stop():
                    break
                if self.announce is not None:
                    self.announce(f"Prompt expansion: encoding prompt {number} of {len(waiting)}")
                required = prompt_parser.SdConditioning(texts, width=p.width, height=p.height, distilled_cfg_scale=p.distilled_cfg_scale)
                entry = [None, None, None]
                p.get_conds_with_caching(prompt_parser.get_multicond_learned_conditioning, required, p.firstpass_steps, [entry], data)
                self.store[key] = entry
                self.encoded += 1
                total += _size(entry[1])
                if total > LIMIT_BYTES:
                    break
        finally:
            # what the other batches' encodes reported belongs to them; Forge reapplies it from each entry
            if hasattr(model, "extra_generation_params") and isinstance(model.extra_generation_params, dict):
                model.extra_generation_params.clear()
                model.extra_generation_params.update(kept)
        if self.encoded:
            self.log(f"{TAG} {self.encoded + 1} prompts encoded in one go; the text encoder is not needed again in this job.")

    def release(self):
        """Forget the stored conditioning (at the end of the job)."""
        self.store.clear()
        if getattr(self.p, "setup_conds", None) == self.setup_conds:
            try:
                del self.p.setup_conds
            except AttributeError:
                pass


def install(p, weighted: bool = False, should_stop=None, announce=None, log: Callable[[str], None] = print) -> Optional[EncodeOnce]:
    """Arrange for the job's prompts to be encoded in one go, or say in the console why not. Never raises."""
    try:
        reason = unsuitable(p, weighted)
        if reason is None:
            return EncodeOnce(p, should_stop, announce, log)
        if reason:
            log(f"{TAG} The prompts are encoded one image at a time: {reason}.")
    except Exception as problem:  # noqa: BLE001
        log(f"{TAG} The prompts are encoded one image at a time ({type(problem).__name__}: {problem}).")
    return None
