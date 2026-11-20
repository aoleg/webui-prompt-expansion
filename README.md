# Prompt Expansion for Forge Neo

Newer image models work best with long, concrete prompts: every shop with a name on its sign, every bus with a route, the light coming from somewhere. This extension writes that prompt for you, and it uses the model you already have loaded to do it.

Models such as Z-Image, Krea 2, FLUX.2 Klein, Qwen-Image 2.1 and MOSS-Image use a 4B to 9B chat LLM as their text encoder. Forge Neo only ever asks that LLM to read. This extension also asks it to write: at Generate, the encoder expands each prompt, and the expanded text is what gets encoded. There is no second model to download and nothing extra to keep in memory between runs.

```
a bus stop on a rainy street at night
```

becomes, with the MOSS-Image encoder and the "Legible world" preset:

```
A bus stop on a rainy street at night features a shelter with a red roof where water drips from the edge onto the wet pavement. A single passenger stands under the awning, holding a clear umbrella that reflects the neon glow of nearby signs. Above the shelter, a blue sign reads "Route 42" while a smaller white panel below it displays the destination "Central Station". To the left, a yellow taxi is parked with its license plate reading "NYC-884", and its headlights cut through the mist. Further down the street, a shop window displays a poster with the text "Open Late" in bright green letters against a dark background. The asphalt glistens under the streetlights, creating long reflections of the orange traffic signals and the distant city skyline. ...
```

## Supported text encoders

| Image model | Text encoder | Notes |
|---|---|---|
| Z-Image, FLUX.2 Klein 4B | Qwen3-4B | |
| Krea 2 | Qwen3-VL 4B | |
| Qwen-Image 2.1 | Qwen3-VL 8B | |
| MOSS-Image | Qwen3.5 9B | |
| P-H3-Image, and MiniMax-H3-Image run with the P-H3-Image encoder | Qwen3-VL 4B | Through the [MiniMax H3 extension](https://github.com/aoleg/pruna-P-H3-Image-forge-neo) |
| FLUX.2 Klein 9B | Qwen3-8B | Needs an output head from a second file, see below |

All six were run on real encoder files through Forge Neo's own loader.

**FLUX.2 Klein 9B.** Qwen3-8B keeps its output table (`lm_head`) apart from its input table, and the usual fp8 encoder file for Klein 9B was saved without it. The extension then borrows a head of the same size from another text encoder file in your text encoder folders. The Qwen-Image 2.1 encoder (Qwen3-VL 8B) has one, and Qwen3-VL 8B is built on Qwen3-8B, so the two fit: with that file on disk, Klein 9B writes well. The panel names the file it borrows from. Without such a file the extension says so and leaves your prompts as typed. An encoder file that contains its own `lm_head` is used directly.

**MiniMax-H3-Image with its own 32B encoder** cannot be supported: that encoder is cut off after 50 of its 64 layers and has no output side at all.

Flux.1, Chroma, Wan, SD 1.5 and SDXL cannot be supported either: T5 and CLIP are encoders only and cannot write text. With such a checkpoint the extension does nothing and says so in its panel.

Quantised encoder files work: scaled fp8, fp8 mixed and int8 files were tested. GGUF files were not.

## Install

Clone into Forge Neo's `extensions` folder and restart the UI:

```
git clone https://github.com/aoleg/webui-prompt-expansion
```

Keep the folder name `webui-prompt-expansion`. Forge names an extension after its folder, and the rule that runs this extension after Dynamic Prompts and after Forge's comment remover is written for that name.

There are no extra Python packages.

## Use

Open the **Prompt Expansion** panel in txt2img or img2img, tick **Enable**, and generate as usual.

- **Preset**: a ready-made instruction for the encoder. Choosing one fills the box below.
- **Expansion prompt**: the instruction itself. Edit it freely; what is in the box is what is sent. Your own prompt follows it as the request. `{words}` is replaced by the length.
- **Length (words)**: how long the result should be. 0 uses the preset's own length. It is a guide, not a count.
- **Expansion seed**: -1 picks a new random seed for every job. Any other value is the seed of the first prompt; each next prompt of the job gets the next number.
- **Advanced**: sampling on or off, temperature, top p, the token limit, how many prompts are written side by side, and thinking.

Every image of a batch gets its own expansion, so one typed prompt gives a varied batch. The console shows one line per job, for example `[Prompt Expansion] 8 of 8 prompts expanded by Qwen3-VL 4B in 9.7 s, seeds from 1234`.

## Presets

| Preset | What it asks for |
|---|---|
| Faithful detail | Your prompt, continued with the look of each subject, materials, setting, framing and light. Adds no lettering. |
| Legible world | The same, and every surface that would carry writing in real life gets short, plausible text: a shop name, a bus route and destination, a license plate. |
| Photographic | A real photograph: camera distance and angle, lens, depth of field, light source, time of day. |
| Cinematic still | A single frame from a film: shot size, staging, light sources, colour palette. |
| Illustration | A drawn or painted artwork. Uses the medium you name, or picks one that suits the subject. |
| Attribute caption | A careful caption: colour, shape, size, texture, quantity and position of every object. |
| Tags to prose | Turns a comma-separated tag list into sentences without adding anything. |
| Translate and expand | Takes a prompt in any language and writes the expansion in English. Text you put in quotes stays in its original script. |
| Minimal polish | Fixes grammar and order, adds nothing, keeps the length. |

Presets are plain text files in `presets/`. To add your own, put a `.txt` file in `presets/user/` (create the folder). The file name is the preset's name; an optional first line `# words: 150` sets its default length. A user preset with the name of a shipped one replaces it.

Writing your own instruction for a 4B model: keep it under about 150 words, say what to do rather than what to avoid, and give no examples. A small model copies an example into every answer; a list such as "for example anime, watercolour" turned photo prompts into watercolours in testing.

## How it behaves

- **Batches.** All prompts of a job are written together, in passes of up to "Prompts per pass" (default 10). A pass takes about as long as a single prompt, so a batch of 8 costs the time of one. If your batch is larger than the pass size, raise the pass size rather than wait for a second pass.
- **Dynamic Prompts and wildcards** are resolved first; the encoder only sees the final text of each image.
- **LoRA and other `<name:value>` tags** are taken out before the encoder sees the prompt and put back at the end.
- **The negative prompt** is never touched.
- **Hires fix** uses the expanded prompt, unless you typed a separate hires prompt.
- **Infotext.** The prompt saved with the image is the expanded one, so pasting it back reproduces the image with the extension off. Three keys are added: `Prompt expansion` (the preset), `Expansion seed` and `Expansion source` (what you typed).
- **Nothing can break a generation.** If the encoder refuses, returns nothing, or the run is interrupted, the image is generated from the prompt as typed.

The answer is cleaned before use: introductions and Markdown are removed, a text cut off by the token limit is trimmed to its last full sentence, and a sentence that drifts into another writing system is dropped. A small model sometimes locks onto one word and repeats it; that is detected, and the text before it is kept.

Three more guards keep a weak answer from costing you the picture:

- **Your prompt comes first.** The presets tell the encoder to begin with your prompt word for word. If the answer has lost any of its content words or its quoted lettering, your prompt is put in front of the answer. This applies to any expansion prompt that contains the words "begin with the request".
- **Length.** An answer longer than 1.5 times the length asked for is cut back to it, at a sentence end.
- **Self-talk.** If the encoder breaks off and starts commenting on its own writing ("Wait, let me try again"), the text is cut there.

A few things in a written paragraph would be misread by the web UI's own prompt syntax, so they are changed: an upper-case `AND` becomes `&` inside quoted lettering and `and` elsewhere; square brackets become round ones.

## Speed and memory

Measured on an RTX 5090, encoder only, ten prompts in one pass, about 150 words each:

| Encoder | File | Time for 10 prompts | Extra VRAM while writing |
|---|---|---|---|
| Qwen3-4B | bf16 | 6.5 s | about 2.5 GB |
| Qwen3-VL 4B | scaled fp8 | 10 s | about 3 GB |
| Qwen3-VL 4B (P-H3-Image) | int8 | 10 s | about 3 GB |
| Qwen3-VL 8B | int8 | 9 s | about 2.5 GB |
| Qwen3-8B | fp8 mixed, borrowed head | 12 s | about 3.6 GB |
| Qwen3.5 9B | int8 | 20 s | about 3.8 GB |

The extra memory (the output table and the cache of the pass) is released when the job's prompts are written, before the image model starts. A single prompt takes nearly as long as ten.

This is slower than llama.cpp. The encoder runs as ordinary PyTorch modules, one small operation at a time, and the time goes into that overhead rather than into arithmetic. Writing several prompts in one pass is what makes it practical.

## What to expect from each encoder

- **Qwen3.5 9B (MOSS-Image)** follows the instruction most closely and writes the best text. It is the slowest. Its reasoning mode is switched off: it reasons at great length and did not finish within 1,500 tokens in testing.
- **Qwen3-VL 8B (Qwen-Image 2.1)** is close behind: always on length, always ends cleanly, almost no non-visible detail.
- **Qwen3-VL 4B (Krea 2)** writes well and stays on the rules.
- **Qwen3-8B (FLUX.2 Klein 9B)** keeps every subject and all lettering, but writes through a borrowed output head. About one text in ten does not come to an end by itself and is cut back to the length.
- **Qwen3-4B (Z-Image, Klein 4B)** keeps your subjects and lettering, but it adds sounds, smells and "atmosphere" in about a third of its texts whatever the instruction says, sometimes invents a sign, and now and then runs into a loop (the text before the loop is kept).
- **The P-H3-Image encoder** is the weakest writer. It is a Qwen3-VL 4B whose layers differ from the stock model: it paraphrases your prompt away in most answers (your prompt is then put in front, see above), and in about one text in ten it breaks off and starts talking to itself (the text is cut there). Expect plainer expansions and the occasional prompt left as typed.

No refusals were seen with ordinary prompts. A stock model may still decline some; then the prompt is used as typed.

## Limits

- The expansion is not reproducible across batch sizes. The same job with the same expansion seed gives the same text; the same prompt in a batch of 4 and a batch of 8 may not. The infotext holds the text that was used.
- Prompt-editing syntax in your own prompt (`[a:b:0.5]`, `(word:1.2)`, `BREAK`) is given to the encoder as plain text and will usually not survive the rewrite.
- Reference images are not shown to the encoder. In img2img and edit workflows the expansion is written from the text alone.
- Qwen-Image (Qwen2.5-VL), ERNIE-Image (Ministral 3), Lumina 2 and Anima are not supported.

## Credits

The idea of using the image model's own text encoder for prompt expansion comes from [ComfyUI-SingleEncoderEnhancer](https://github.com/Magirad/ComfyUI-SingleEncoderEnhancer); the sentence trimming is adapted from it. The sampling filters follow ComfyUI's text generation. Both are GPL-3.0, and so is this extension. See [LICENSE](LICENSE).
