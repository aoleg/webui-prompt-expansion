"""Prompt Expansion: the image model's own text encoder, or a writer model of its own, rewrites every prompt of a job
before it is encoded."""

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

# Settings > Prompt Expansion. Both are speed-ups that stand down by themselves; the switches are for the day
# one of them misbehaves on somebody's machine.
SPEED_OPTIONS = {
    "prompt_expansion_replay": ("Write from a recorded decoding step", "much faster on a CUDA GPU, with the same numbers; stands down by itself where it cannot be used"),
    "prompt_expansion_int8_kernel": ("Write through the int8 kernel when the text encoder is an int8 ConvRot file", "faster again; the wording comes out slightly different, the conditioning is not affected"),
}


# The preset chosen last time, kept in Forge's settings (a section without a page, so it is not shown). The panel
# opens with that preset and its text; the expansion prompt itself is never saved, so an edited or stale text
# cannot hide the preset that is selected.
LAST_PRESET = "prompt_expansion_last_preset"


# The writer: the checkpoint's own text encoder, or a model file of its own from Forge's text encoder folders
# (Molmo2-8B, recognised by its tensors). The choice is kept like the preset, by the name it has in the list.
LAST_WRITER = "prompt_expansion_last_writer"
OWN_ENCODER = "This checkpoint's text encoder"


def register_settings():
    section = ("prompt_expansion", TITLE)
    shared.options_templates.update(shared.options_section(section, {name: shared.OptionInfo(True, label).info(info) for name, (label, info) in SPEED_OPTIONS.items()}))
    hidden = {LAST_PRESET: shared.OptionInfo("", "Last chosen Prompt Expansion preset"), LAST_WRITER: shared.OptionInfo("", "Last chosen Prompt Expansion writer")}
    shared.options_templates.update(shared.options_section((None, TITLE), hidden))


def remembered_preset(names: list):
    """The preset to open with: the one chosen last time if it is still offered, else the first. Read at every page load."""
    presets = [preset for preset in preset_files.load() if preset.name in names]
    if not presets:
        return None
    return preset_files.find(presets, getattr(shared.opts, LAST_PRESET, "")) or presets[0]


def remember(option, name):
    """Keep a choice for the next start. Quietly does nothing with frozen settings."""
    if not name or getattr(shared.opts, option, "") == name:
        return
    try:
        if shared.opts.set(option, name):
            shared.opts.save(shared.config_filename)
    except Exception:  # noqa: BLE001 - --freeze-settings and the like; remembering is a convenience
        pass


def remember_preset(name):
    remember(LAST_PRESET, name)


if getattr(shared, "options_templates", None) is not None:
    register_settings()


def writer_files() -> tuple[dict, list]:
    """({name in the list: file} of the writer models in Forge's text encoder folders, the Molmo2 files there that
    cannot write). Reads file headers only, each file once."""
    try:
        from lib_prompt_expansion import molmo
        from lib_prompt_expansion.headfile import library_files

        usable, refused = molmo.files(library_files())
        return {found.choice: found.path for found in usable}, refused
    except Exception:  # noqa: BLE001 - a list is not worth an error in the panel
        return {}, []


def remembered_writer(names: list) -> str:
    """The writer to open with: the one chosen last time if it is still offered, else the checkpoint's text encoder."""
    last = getattr(shared.opts, LAST_WRITER, "")
    return last if last in names else OWN_ENCODER


def status_line(writer=None) -> str:
    """Whether the writer can write: the one named, or the one chosen last time. Never loads a model."""
    from modules import sd_models

    files, refused = writer_files()
    passed_over = "".join(f" *Not offered as a writer: {found.name}, {found.brief}.*" for found in refused)
    path = files.get(remembered_writer(list(files)) if writer is None else writer)
    if path is not None:
        return status(None, path)[1] + passed_over
    model = sd_models.model_data.sd_model
    if model is None or type(model).__name__ == "FakeInitialModel":
        return "No checkpoint is loaded yet. The text encoder is checked when you generate." + passed_over
    return status(model)[1] + pruna_hint() + passed_over


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

        # Filled from the remembered preset whenever the page loads, not from ui-config.json.
        def opening_name():
            chosen = remembered_preset(names)
            return chosen.name if chosen else None

        def opening_text():
            chosen = remembered_preset(names)
            return chosen.text if chosen else ""

        def opening_begin():
            chosen = remembered_preset(names)
            return bool(chosen.begin) if chosen else True

        writers = [OWN_ENCODER, *writer_files()[0]]

        # No gr.Group in here: sd-webui-tabs-extension hides a panel that contains one.
        with InputAccordion(False, label=TITLE, elem_id=self.elem_id("enable")) as enabled:
            # read again at every page load, like the writer below it: the line speaks of the writer that is chosen
            state_text = gr.Markdown(value=status_line, elem_id=self.elem_id("status"))
            writer = gr.Dropdown(
                label="Writer",
                info="What writes the expansion: the text encoder of the loaded checkpoint, or a model of its own from the text encoder folders (Molmo2-8B), which works with every checkpoint",
                choices=writers,
                value=lambda: remembered_writer(writers),
                elem_id=self.elem_id("writer"),
            )
            with gr.Row():
                preset = gr.Dropdown(label="Preset", choices=names, value=opening_name, elem_id=self.elem_id("preset"), scale=8)
                refresh = ToolButton(value=refresh_symbol, elem_id=self.elem_id("refresh"), tooltip="Reload the preset files, check the loaded checkpoint, and let a failed download be tried again")
            instruction = gr.Textbox(
                label="Expansion prompt",
                info="The instruction given to the text encoder; your prompt follows as the request. {words} is replaced by the length. Edit freely.",
                lines=6,
                value=opening_text,
                elem_id=self.elem_id("instruction"),
            )
            begin = gr.Checkbox(
                label="Start with my prompt",
                info="The expansion begins with your prompt, word for word, and continues it. Each preset sets this; off for presets that rewrite or translate.",
                value=opening_begin,
                elem_id=self.elem_id("begin"),
            )
            for component in (writer, preset, instruction, begin):
                component.do_not_save_to_config = True
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
            if found is None:
                return gr.skip(), gr.skip()
            remember_preset(found.name)
            return found.text, bool(found.begin)

        def reload(current, writing):
            head_fetch.forget_failures()  # a download that failed is tried again by the next job
            fresh = [item.name for item in preset_files.load()]
            offered = [OWN_ENCODER, *writer_files()[0]]
            writing = writing if writing in offered else OWN_ENCODER
            return gr.update(choices=fresh, value=current if current in fresh else (fresh[0] if fresh else None)), gr.update(choices=offered, value=writing), status_line(writing)

        def pick_writer(name):
            remember(LAST_WRITER, name)
            return status_line(name)

        preset.change(fn=pick, inputs=[preset], outputs=[instruction, begin], show_progress=False)
        writer.change(fn=pick_writer, inputs=[writer], outputs=[state_text], show_progress=False)
        refresh.click(fn=reload, inputs=[preset, writer], outputs=[preset, writer, state_text], show_progress=False)
        enabled.change(fn=status_line, inputs=[writer], outputs=[state_text], show_progress=False)
        random.click(fn=lambda: -1, inputs=[], outputs=[seed], show_progress=False)
        reuse.click(fn=lambda: -1 if substitute.last_start_seed is None else substitute.last_start_seed, inputs=[], outputs=[seed], show_progress=False)

        # the writer comes last: a call through the API that does not know it gets the checkpoint's text encoder
        return [enabled, preset, instruction, words, seed, do_sample, temperature, top_p, max_tokens, rows, thinking, begin, encode_all, writer]

    def scripts_after_me(self, p) -> set[str]:
        """Names ("Class@file.py") of the scripts whose process step runs after this one."""
        try:
            order = [callback.callback for callback in p.scripts.ordered_callbacks("process")]
            mine = order.index(self)
        except Exception:  # noqa: BLE001 - only used for two precautions
            return set()
        return {f"{type(script).__name__}@{os.path.basename(script.filename or '')}" for script in order[mine + 1 :]}

    def process(self, p, enabled=False, preset=None, instruction="", words=0, seed=-1, do_sample=True, temperature=0.6, top_p=0.8, max_tokens=512, rows=10, thinking=False, begin=False, encode_all=True, writer=None, *_):
        if not enabled:
            return
        instruction = (instruction or "").strip()
        if not instruction:
            print(f"[{TITLE}] The expansion prompt is empty; the prompts are used as typed.")
            return
        writer_file = ""
        if writer and writer != OWN_ENCODER:
            writer_file = writer_files()[0].get(writer)
            if not writer_file:
                print(f"[{TITLE}] The writer \"{writer}\" is not in the text encoder folders any more; the prompts are used as typed. Choose another writer in the panel.")
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
            replay=bool(getattr(shared.opts, "prompt_expansion_replay", True)),
            int8_kernel=bool(getattr(shared.opts, "prompt_expansion_int8_kernel", True)),
            writer_file=writer_file,
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
