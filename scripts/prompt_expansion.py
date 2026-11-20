"""Prompt Expansion: the image model's own text encoder rewrites every prompt of a job before it is encoded."""

import os

import gradio as gr
from lib_prompt_expansion import encode_once
from lib_prompt_expansion import fetch as head_fetch
from lib_prompt_expansion import presets as preset_files
from lib_prompt_expansion import substitute
from lib_prompt_expansion.adapters import status
from lib_prompt_expansion.sampling import SamplingOptions

from modules import scripts, shared
from modules.ui import random_symbol, refresh_symbol, reuse_symbol
from modules.ui_components import InputAccordion, ToolButton

TITLE = "Prompt Expansion"


def status_line() -> str:
    """Whether the loaded checkpoint's text encoder can write. Never loads a model."""
    from modules import sd_models

    model = sd_models.model_data.sd_model
    if model is None or type(model).__name__ == "FakeInitialModel":
        return "No checkpoint is loaded yet. The text encoder is checked when you generate."
    return status(model)[1] + pruna_hint()


STOCK_ENCODER_INFO = (
    "For P-Image-OSS and MiniMax-H3-Image: builds a text encoder from the stock Qwen3-VL 4B and Pruna's text connector. "
    "It writes far better text than Pruna's retrained encoder. The first click only says what it would do."
)


STOCK_ENCODER_BUTTON, STOCK_ENCODER_AGAIN = "Build stock encoder for P-Image-OSS", "Click again to build"
STOCK_ENCODER_PRECISIONS = {"int8 with ConvRot (best)": "int8_convrot", "bf16": "bf16", "a copy of a stock file on disk": "copy"}


def pruna_hint() -> str:
    """A word in the status line when the loaded text encoder is Pruna's retrained one. Reads file headers only."""
    try:
        from lib_prompt_expansion import hybrid
        from lib_prompt_expansion.headfile import loaded_files

        for path in loaded_files():
            if hybrid.kind(path) == "connector":
                return "" if hybrid.built_here(path) else " Pruna's own encoder writes poor text; *Build stock encoder* under Advanced makes one that writes well."
    except Exception:  # noqa: BLE001 - a hint is not worth an error in the panel
        pass
    return ""


class PromptExpansionScript(scripts.Script):
    warned_order = False

    def title(self):
        return TITLE

    def show(self, is_img2img):
        return scripts.AlwaysVisible

    def ui(self, is_img2img):
        presets = preset_files.load()
        names = [preset.name for preset in presets]
        first = presets[0] if presets else None

        # No gr.Group in here: sd-webui-tabs-extension hides a panel that contains one.
        with InputAccordion(False, label=TITLE, elem_id=self.elem_id("enable")) as enabled:
            state_text = gr.Markdown(status_line(), elem_id=self.elem_id("status"))
            with gr.Row():
                preset = gr.Dropdown(label="Preset", choices=names, value=first.name if first else None, elem_id=self.elem_id("preset"), scale=8)
                refresh = ToolButton(value=refresh_symbol, elem_id=self.elem_id("refresh"), tooltip="Reload the preset files, check the loaded checkpoint, and let a failed download be tried again")
            instruction = gr.Textbox(
                label="Expansion prompt",
                info="The instruction given to the text encoder; your prompt follows as the request. {words} is replaced by the length. Edit freely.",
                lines=6,
                value=first.text if first else "",
                elem_id=self.elem_id("instruction"),
            )
            begin = gr.Checkbox(
                label="Start with my prompt",
                info="The expansion begins with your prompt, word for word, and continues it. Each preset sets this; off for presets that rewrite or translate.",
                value=bool(first.begin) if first else True,
                elem_id=self.elem_id("begin"),
            )
            with gr.Row():
                words = gr.Slider(label="Length (words)", info="0 = the preset's own length", minimum=0, maximum=400, step=10, value=0, elem_id=self.elem_id("words"), scale=6)
                seed = gr.Number(label="Expansion seed", info="-1 = random; else the first prompt's seed, +1 for each next one", value=-1, precision=0, elem_id=self.elem_id("seed"), scale=4)
                random = ToolButton(value=random_symbol, elem_id=self.elem_id("random_seed"), tooltip="Random expansion seed")
                reuse = ToolButton(value=reuse_symbol, elem_id=self.elem_id("reuse_seed"), tooltip="Reuse the start seed of the last job")
            with gr.Accordion("Advanced", open=False, elem_id=self.elem_id("advanced")):
                with gr.Row():
                    do_sample = gr.Checkbox(label="Sampling", info="Off: always the most likely word, one text for equal prompts", value=True, elem_id=self.elem_id("sampling"))
                    temperature = gr.Slider(label="Temperature", minimum=0.1, maximum=1.5, step=0.05, value=0.6, elem_id=self.elem_id("temperature"))
                    top_p = gr.Slider(label="Top p", minimum=0.1, maximum=1.0, step=0.05, value=0.8, elem_id=self.elem_id("top_p"))
                with gr.Row():
                    max_tokens = gr.Slider(label="Max new tokens", info="Upper limit for one text", minimum=64, maximum=2048, step=32, value=512, elem_id=self.elem_id("max_tokens"))
                    rows = gr.Slider(label="Prompts per pass", info="How many prompts are written side by side", minimum=1, maximum=32, step=1, value=10, elem_id=self.elem_id("rows"))
                    thinking = gr.Checkbox(label="Thinking", info="Reason before writing (Qwen3-4B, and Qwen3-8B with its own output head; the other encoders ignore it). Much slower; set Max new tokens to 1500 or more", value=False, elem_id=self.elem_id("thinking"))
                encode_all = gr.Checkbox(
                    label="Encode all prompts at once",
                    info="With a batch count above one: the text encoder is loaded once for the job, not once for every image. Stands down for weights, input images and hires fix.",
                    value=True,
                    elem_id=self.elem_id("encode_all"),
                )
                # P-Image-OSS and MiniMax-H3-Image: Pruna's encoder writes poorly, the stock model behind its connector well
                with gr.Row():
                    build_stock = gr.Button(value=STOCK_ENCODER_BUTTON, elem_id=self.elem_id("build_stock"), scale=1)
                    build_as = gr.Dropdown(
                        label="Build as",
                        info="int8 with ConvRot is made from Qwen's full-precision files (8.9 GB to download unless on disk); a copy takes a stock file from your folders as it is",
                        choices=list(STOCK_ENCODER_PRECISIONS),
                        value=next(iter(STOCK_ENCODER_PRECISIONS)),
                        elem_id=self.elem_id("build_stock_as"),
                        scale=1,
                    )
                    build_text = gr.Markdown(STOCK_ENCODER_INFO, elem_id=self.elem_id("build_stock_text"))
                build_armed = gr.State("")

        def build_stock_encoder(armed, chosen):
            """First click: say what would be done. Second click on the same plan: do it."""
            from lib_prompt_expansion import hybrid
            from lib_prompt_expansion.headfile import library_files, loaded_files

            loaded = loaded_files()
            armed, text = hybrid.press(armed, head_fetch.target_folder(loaded), loaded + library_files(), precision=STOCK_ENCODER_PRECISIONS.get(chosen))
            return armed, text, gr.update(value=STOCK_ENCODER_AGAIN if armed else STOCK_ENCODER_BUTTON)

        build_stock.click(fn=build_stock_encoder, inputs=[build_armed, build_as], outputs=[build_armed, build_text, build_stock])
        # another choice is another plan: the button asks anew
        build_as.change(fn=lambda: ("", STOCK_ENCODER_INFO, gr.update(value=STOCK_ENCODER_BUTTON)), inputs=[], outputs=[build_armed, build_text, build_stock], show_progress=False)

        def pick(name):
            found = preset_files.find(preset_files.load(), name)
            return (gr.skip(), gr.skip()) if found is None else (found.text, bool(found.begin))

        def reload(current):
            head_fetch.forget_failures()  # a download that failed is tried again by the next job
            fresh = [item.name for item in preset_files.load()]
            return gr.update(choices=fresh, value=current if current in fresh else (fresh[0] if fresh else None)), status_line()

        preset.change(fn=pick, inputs=[preset], outputs=[instruction, begin], show_progress=False)
        refresh.click(fn=reload, inputs=[preset], outputs=[preset, state_text], show_progress=False)
        enabled.change(fn=status_line, inputs=[], outputs=[state_text], show_progress=False)
        random.click(fn=lambda: -1, inputs=[], outputs=[seed], show_progress=False)
        reuse.click(fn=lambda: -1 if substitute.last_start_seed is None else substitute.last_start_seed, inputs=[], outputs=[seed], show_progress=False)

        return [enabled, preset, instruction, words, seed, do_sample, temperature, top_p, max_tokens, rows, thinking, begin, encode_all]

    def scripts_after_me(self, p) -> set[str]:
        """Names ("Class@file.py") of the scripts whose process step runs after this one."""
        try:
            order = [callback.callback for callback in p.scripts.ordered_callbacks("process")]
            mine = order.index(self)
        except Exception:  # noqa: BLE001 - only used for two precautions
            return set()
        return {f"{type(script).__name__}@{os.path.basename(script.filename or '')}" for script in order[mine + 1 :]}

    def process(self, p, enabled=False, preset=None, instruction="", words=0, seed=-1, do_sample=True, temperature=0.6, top_p=0.8, max_tokens=512, rows=10, thinking=False, begin=False, encode_all=True, *_):
        if not enabled:
            return
        instruction = (instruction or "").strip()
        if not instruction:
            print(f"[{TITLE}] The expansion prompt is empty; the prompts are used as typed.")
            return

        presets = preset_files.load()
        chosen = preset_files.find(presets, preset)
        settings = substitute.Settings(
            instruction=instruction,
            preset_name=preset_files.name_of(presets, instruction),
            words=preset_files.effective_words(instruction, int(words or 0), chosen.words if chosen else 0),
            seed=-1 if seed is None else int(seed),
            sampling=SamplingOptions(do_sample=bool(do_sample), temperature=float(temperature), top_p=float(top_p)),
            max_new_tokens=int(max_tokens),
            rows_per_pass=int(rows),
            thinking=bool(thinking),
            begin=bool(begin),
        )

        later = self.scripts_after_me(p)
        comments_follow = "ScriptComments@comments.py" in later and bool(getattr(shared.opts, "enable_prompt_comments", False))
        if any(name.endswith("@dynamic_prompting.py") for name in later) and not PromptExpansionScript.warned_order:
            PromptExpansionScript.warned_order = True
            print(f"[{TITLE}] Dynamic Prompts runs after this extension, so wildcards are expanded as text. Check Settings > Callbacks, or the folder name (webui-prompt-expansion).")

        # with Emphasis set to "None" brackets are plain characters, for Forge and for weight extensions alike
        weights = getattr(shared.opts, "emphasis", "Original") != "None"
        report = substitute.expand(p, settings, state=shared.state, comments_follow=comments_follow, weights=weights)
        print(f"[{TITLE}] {report.line()}")

        if encode_all and report.expanded:

            def stopped():
                state = shared.state
                return bool(getattr(state, "interrupted", False) or getattr(state, "skipped", False) or getattr(state, "stopping_generation", False))

            def announce(line):
                shared.state.textinfo = line

            p.prompt_expansion_encode_once = encode_once.install(p, weighted=report.weighted, should_stop=stopped, announce=announce)

    def postprocess(self, p, processed, *args):
        helper = getattr(p, "prompt_expansion_encode_once", None)
        if helper is not None:
            helper.release()
            p.prompt_expansion_encode_once = None
