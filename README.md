# Prompt Expansion for Forge Neo

Newer image models work best with long, concrete prompts: every shop with a name on its sign, every bus with a route, the light coming from somewhere. This extension writes that prompt for you, and it uses the model you already have loaded to do it.

Models such as Z-Image, Krea 2, FLUX.2 Klein, Qwen-Image 2.1 and MOSS-Image use a 4B to 9B chat LLM as their text encoder. Forge Neo only ever asks that LLM to read. This extension also asks it to write: at Generate, the encoder expands each prompt, and the expanded text is what gets encoded. There is no second model to download and nothing extra to keep in memory between runs.

```
a bus stop on a rainy street at night
```

becomes, with the MOSS-Image encoder and the "Legible world" preset:

```
a bus stop on a rainy street at night, featuring a translucent yellow plastic shelter with water droplets beading on its surface, positioned on the left side of the frame. A single streetlamp casts a dim, cool blue light from above, reflecting off the wet asphalt and puddles scattered across the road. In the background, blurred red taillights of distant cars streak horizontally due to motion blur. A silver sedan is parked nearby, displaying a license plate reading "742-XJN" in black characters on a white background. Above the shelter, a rectangular digital display shows the route number "42" and destination "DOWNTOWN" in bright green LED text against a dark screen. The camera is positioned at eye level, capturing a medium shot that includes the empty bench inside the shelter and the rain-slicked pavement stretching into the foggy distance under an overcast sky.
```

## Examples

The pictures below were made in Forge Neo by the extension's author with one short prompt each. The lathe pictures are consecutive seeds, not a selection. The expanded tram is the best of four tries made while the wording of the preset was being tuned. All pictures were made with an earlier wording of the presets, which allowed more ornament than the present one (the "warm, golden tones" in the Krea 2 texts below).

### Krea 2: the same seeds without and with expansion

![Krea 2, four seeds as typed and the same four seeds expanded](img/krea2_lathe.jpg)

Prompt as typed:

> A close-up of a working man in a helmet working at a lathe, shot on Kodachrome.

Expanded prompt of the first image in the lower row (preset *Faithful detail*; each image gets a text of its own):

> A close-up of a working man in a helmet working at a lathe, shot on Kodachrome. A close-up of a working man in a helmet, sweat glistening on his brow as he grips the handle of a lathe with calloused hands, his focused gaze fixed on the spinning metal rod; the helmet’s visor reflects faint industrial light, its surface slightly dusty from prolonged use. The lathe hums softly beneath him, its metallic body gleaming under warm, golden Kodachrome tones that saturate every curve and shadow. His work clothes are coarse denim, frayed at the elbows, stained with grease and grit. Behind him, a dimly lit workshop holds shelves of tools and coiled wires, bathed in soft ambient glow. The camera hovers just above eye level, capturing the motion blur of the rotating spindle and the rhythmic clink of metal against steel, all rendered in Kodachrome’s rich, saturated hues — deep reds, amber yellows, and cool grays — with sharp detail and a slight vignette framing the scene.

What changes. The typed prompt gives four flash-lit snapshots: a man, a machine part, and a nearly black or plain background; one helmet carries garbled lettering. With the expanded prompt every picture has a workshop around the man (tool racks, shelves, a lamp, a window), a visor or face shield on the helmet, grease on the denim and on the hands, a brass workpiece and metal chips, and a named light source. The four pictures also differ more from one another: a three-quarter view of the whole lathe, a profile, a tight close-up of the hands, a view along the spindle. The price is a shared look: every text speaks of warm, golden Kodachrome tones, so all four are warm and golden, and one gained a slide-film border.

### MOSS-Image: a new text for every image

![MOSS-Image, nine seeds, each with its own expanded prompt](img/moss_lathe.jpg)

The same typed prompt. Expanded prompt of the first image (preset *Attribute caption*):

> A close-up of a working man in a helmet working at a lathe, shot on Kodachrome. A close-up captures a working man wearing a yellow hard helmet, his face partially obscured by safety goggles as he operates a large industrial lathe. The machine itself is painted a matte grey with visible silver metallic gears and black rubber belts connecting its components. His hands grip a lever on the lathe's control panel, which features circular dials and rectangular switches arranged in a grid pattern. Sparks fly from the contact point between the spinning metal workpiece and the cutting tool, illuminating the immediate area with bright orange light against the darker machinery. The man wears a blue work shirt with rolled-up sleeves, revealing forearms dusted with fine grey particles. Behind him, a blurred wall of pale green panels provides a neutral backdrop to the mechanical activity. The entire scene is rendered in the rich, saturated tones characteristic of Kodachrome film, emphasizing deep reds in the shadows and vibrant highlights on the metal surfaces.

What you get. Each picture follows its own text closely: the yellow helmet, the sparks and the control panel with round dials in the first image, the blue wall with the stack of crates in the third, the white helmet with ventilation holes in the sixth are all in the texts. The set is varied in framing, colour and setting, and every image is clean and evenly lit. The look is that of a present-day industrial photograph rather than of 1950s Kodachrome, and in two of the nine images the close-up turned into a wider shot, because a caption that lists the things in the background asks for them to be shown.

### Lettering: the Legible world preset

![Krea 2, a tram as typed and with the Legible world preset](img/krea2_tram.jpg)

Prompt as typed:

> A classic red-and-yellow tram from the 1950s on tracks on a town square.

Expanded prompt of the right image (preset *Legible world*):

> A classic red-and-yellow tram from the 1950s glides along tracks embedded in a sunlit town square, its polished metal body gleaming under overhead lamps. The tram’s front window reflects the pale blue sky and scattered clouds, while its side panels bear faded lettering: “TOWNLINE TRAMWAY” in bold black capitals above “ROUTE 3 TO CENTRAL STATION.” Its wheels click softly against the rails as it moves past a stone bench and a lamppost with a brass globe. A small sign beside the tram reads “OPEN DAILY 8 AM–6 PM” in green letters on white background. Behind it, a café awning displays “COFFEE & CROISSANTS” in cursive script. The tram’s destination board flickers with “CENTRAL STATION” in bright yellow, and its license plate reads “TRM-5472.”

What changes. Left to itself, the model knows that a tram has a route number and a destination board but not what they say: the board reads "ARPE", the number is "06", and the lettering along the side is a squiggle. With the expanded prompt every piece of writing has been decided beforehand, in capitals, and the model draws it: the company name on the roof board, the destination, the plate "TRM-5472", the kerb sign and the café awning. One board lost a letter ("CENTRAL TATION"); short texts in capitals are what the image models get right most often, which is why the preset asks for them.

## Supported text encoders

| Image model | Text encoder | Notes |
|---|---|---|
| Z-Image, FLUX.2 Klein 4B | Qwen3-4B | |
| Krea 2 | Qwen3-VL 4B | |
| Qwen-Image 2.1 | Qwen3-VL 8B | |
| MOSS-Image | Qwen3.5 9B | |
| P-H3-Image, and MiniMax-H3-Image run with the P-H3-Image encoder | Qwen3-VL 4B | Through the [MiniMax H3 extension](https://github.com/aoleg/pruna-P-H3-Image-forge-neo) |
| FLUX.2 Klein 9B | Qwen3-8B | Downloads its output head (1.24 GB) at first use, see below |

All six were run on real encoder files through Forge Neo's own loader.

**FLUX.2 Klein 9B.** Qwen3-8B keeps its output table (`lm_head`) apart from its input table, and the usual fp8 and fp4 encoder files for Klein 9B were saved without it. Qwen publishes that table as a file of its own, and the extension downloads it for you: the first time you generate with Klein 9B and the panel switched on, it fetches `model-00005-of-00005.safetensors` (1.24 GB, Apache 2.0) from [Qwen/Qwen3-8B](https://huggingface.co/Qwen/Qwen3-8B) into the text encoder folder your Klein 9B encoder was loaded from, as `qwen_3_8b_lm_head.safetensors`. The file is checked against its SHA-256, and from then on it is found on disk. This is the only time the extension goes online. The encoder in the Klein 9B files is the unmodified Qwen3-8B, so this is its true head.

The console shows the download and its progress; Interrupt stops it, and the next job goes on where it stopped. If it fails (no connection, a full disk, a folder that cannot be written), the console says why and how to get the file by hand, the job goes on, and the download is tried again after a restart or when you press the panel's refresh button. To do it by hand, download [the file](https://huggingface.co/Qwen/Qwen3-8B/resolve/main/model-00005-of-00005.safetensors) and put it into a text encoder folder under any name; it is recognised by its content. A mirror set in `HF_ENDPOINT` is used.

While the true head is not there, the extension borrows a head of the same size from another text encoder file in your folders, if there is one. The Qwen-Image 2.1 encoder (Qwen3-VL 8B) has one, and Qwen3-VL 8B is built on Qwen3-8B, so it works, but clearly worse: about one text in ten does not come to an end, and thinking is not available. The panel names the file it borrows from. With no head at all your prompts are used as typed. An encoder file that contains its own `lm_head` (the full-precision 16 GB file does) is used directly, and nothing is downloaded.

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
- **Start with my prompt**: the expansion begins with your prompt, word for word, and continues it. Each preset sets this; it is off for the presets that rewrite or translate.
- **Length (words)**: how long the result should be. 0 uses the preset's own length. It is a guide, not a count.
- **Expansion seed**: -1 picks a new random seed for every job. Any other value is the seed of the first prompt; each next prompt of the job gets the next number.
- **Advanced**: sampling on or off, temperature, top p, the token limit, how many prompts are written side by side, and thinking.

Every image of a batch gets its own expansion, so one typed prompt gives a varied batch. The console shows one line per job, for example `[Prompt Expansion] 8 of 8 prompts expanded by Qwen3-VL 4B in 9.7 s, seeds from 1234`.

## Presets

| Preset | What it asks for |
|---|---|
| Faithful detail | Your prompt, continued with the look of each subject, materials, setting, framing and light. Adds no lettering. |
| Legible world | The same, and things that carry writing in real life get short, plausible text in capitals: a name, a route number with its destination, a plate. At most four pieces; in the language and script of the place. |
| Photographic | A real photograph: camera distance and angle, lens, depth of field, light source, time of day. |
| Cinematic still | A single frame from a film: shot size, staging, light sources, colour palette. |
| Illustration | A drawn or painted artwork. Uses the medium you name, or picks one that suits the subject. |
| Attribute caption | A careful caption: colour, shape, size, texture, quantity and position of every object. |
| Tags to prose | Turns a comma-separated tag list into sentences without adding anything. |
| Translate and expand | Takes a prompt in any language and writes the expansion in English. Text you put in quotes stays in its original script. |
| Minimal polish | Fixes grammar and order, adds nothing, keeps the length. |

The first six continue your prompt: the result starts with it, word for word. The presets do not ask the encoder for that. The extension puts your prompt at the start of the encoder's answer and the encoder goes on from there, which keeps your words exactly and stops it from saying them a second time.

All presets ask for plain statements of what is in the picture, the way a precise user types a prompt: what a thing is, its colour, its material, where it is. No story, no comparisons, no sounds or smells, no mood.

Presets are plain text files in `presets/`. To add your own, put a `.txt` file in `presets/user/` (create the folder). The file name is the preset's name. Two optional first lines: `# words: 150` sets its default length, and `# begin: request` ticks **Start with my prompt**. A user preset with the name of a shipped one replaces it.

Writing your own instruction for a 4B to 9B model, as measured on these encoders:

- **Four short parts** work best: who writes ("You turn a short image request into one text-to-image prompt, written the way a precise, experienced user types it."), what to write and how long, what to use and what never to do, and how to answer ("Answer with the prompt only").
- **Say what the text never does** in one sentence ("it never tells a story, never compares one thing to another, and never mentions sounds, smells, feelings, mood or atmosphere"). In this frame that worked; the same list as a loose "leave out…" did not.
- **Give no examples of content.** A small model copies them: "for example anime, watercolour" turned photo prompts into watercolours, and "a bus gets a route number and destination" put a passing bus into more than half of the street scenes that had none. Name the kind of text or detail you want, not a thing that carries it.
- **One number per sentence.** "At most six pieces of text, each of four words or fewer" gets the two limits mixed up.
- **Do not ask it to begin with your prompt**; tick **Start with my prompt** instead.

## How it behaves

- **Batches.** All prompts of a job are written together, in passes of up to "Prompts per pass" (default 10). A pass takes about as long as a single prompt, so a batch of 8 costs the time of one. If your batch is larger than the pass size, raise the pass size rather than wait for a second pass.
- **Dynamic Prompts and wildcards** are resolved first; the encoder only sees the final text of each image.
- **LoRA and other `<name:value>` tags** are taken out before the encoder sees the prompt and put back at the end.
- **Weights** such as `(red car:1.4)`, `((fog))`, `[crowd]` and `(dress:-1)` are taken out before the encoder sees the prompt and marked again in the expanded text: a phrase the text kept is wrapped again where it first occurs, a phrase whose words moved apart gets each word wrapped, and a phrase the text lost is added at the end. A phrase with a weight of zero or less (NegPiP's "less of this") is never shown to the encoder, which would only write more about it, and goes to the end as typed. When your prompt uses a weight with a number, brackets the encoder writes itself are escaped (`\(aside\)`) so that they are not read as weights. Forge Neo itself treats brackets as plain characters for the models listed above; weights act through an extension such as [NegPiP](https://github.com/aoleg/neo-krea2-negpip). With Forge's Emphasis setting on "None" the extension leaves them alone as well.
- **Sounds and smells** are removed for certain when the expansion prompt says the text never mentions them (all presets that describe a picture do): the sentence goes, or only the clause if the rest of the sentence shows something. Quoted lettering and your own words are never removed.
- **The negative prompt** is never touched.
- **Hires fix** uses the expanded prompt, unless you typed a separate hires prompt.
- **Infotext.** The prompt saved with the image is the expanded one, so pasting it back reproduces the image with the extension off. Three keys are added: `Prompt expansion` (the preset), `Expansion seed` and `Expansion source` (what you typed).
- **Nothing can break a generation.** If the encoder refuses, returns nothing, or the run is interrupted, the image is generated from the prompt as typed.

The answer is cleaned before use: introductions and Markdown are removed, a text cut off by the token limit is trimmed to its last full sentence, and a sentence that drifts into another writing system is dropped. A small model sometimes locks onto one word and repeats it; that is detected, and the text before it is kept.

Three more guards keep a weak answer from costing you the picture:

- **Your prompt comes first.** With **Start with my prompt** the encoder's answer is begun with your prompt, so it cannot be lost. Should it be missing all the same (with thinking on, the answer starts with the reasoning), it is put in front when the answer has lost any of its content words or its quoted lettering. An answer that ends right after your prompt is not accepted: the encoder must add something.
- **Length.** An answer longer than 1.5 times the length asked for is cut back to it, at a sentence end, or at a clause when the text is one long sentence.
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
| Qwen3-8B | fp8 mixed, with its own head | 12 s | about 3.6 GB |
| Qwen3.5 9B | int8 | 20 s | about 3.8 GB |

The extra memory (the output table and the cache of the pass) is released when the job's prompts are written, before the image model starts. A single prompt takes nearly as long as ten.

This is slower than llama.cpp. The encoder runs as ordinary PyTorch modules, one small operation at a time, and the time goes into that overhead rather than into arithmetic. Writing several prompts in one pass is what makes it practical.

## What to expect from each encoder

With the shipped presets all six write plain, usable text. In the last calibration (nine presets, twenty prompts each) the number of texts with a sound or smell was 0 or 1 of 180 on every encoder, and texts with a mood word were 0 to 9.

- **Qwen3.5 9B (MOSS-Image)** follows the instruction most closely and writes the best text. It is the slowest. Its reasoning mode is switched off: it reasons at great length and did not finish within 1,500 tokens in testing. With *Legible world* it finds something to write on in nearly every scene, also in a still life.
- **Qwen3-VL 8B (Qwen-Image 2.1)** is close behind: on length, always ends cleanly.
- **Qwen3-VL 4B (Krea 2)** writes well and stays on the rules. It likes to continue your prompt as one long chain of clauses rather than in separate sentences.
- **Qwen3-8B (FLUX.2 Klein 9B)**, with its own output head, is as dependable as Qwen3-VL 8B. Thinking works on it. Through a borrowed head (see above) about one text in ten does not come to an end by itself and is cut back to the length.
- **Qwen3-4B (Z-Image, Klein 4B)** keeps your subjects and lettering and now stays plain, but it is the least obedient: with *Legible world* it writes only about a third of its lettering in capitals and adds text to scenes that have nothing to write on, and now and then it runs into a loop (the text before the loop is kept).
- **The P-H3-Image encoder** is the weakest writer. It is a Qwen3-VL 4B whose layers differ from the stock model: in a few texts of a hundred it breaks off and starts talking to itself (the text is cut there, and when too little is left the prompt is used as typed).

No refusals were seen with ordinary prompts. A stock model may still decline some; then the prompt is used as typed.

## Limits

- The expansion is not reproducible across batch sizes. The same job with the same expansion seed gives the same text; the same prompt in a batch of 4 and a batch of 8 may not. The infotext holds the text that was used.
- Prompt editing (`[a:b:0.5]`, `[a|b]`) and `BREAK` in your own prompt are given to the encoder as plain text and will usually not survive the rewrite. A negative phrase keeps the encoder from being told about a thing; it cannot stop the encoder from writing about it unasked.
- Reference images are not shown to the encoder. In img2img and edit workflows the expansion is written from the text alone.
- Qwen-Image (Qwen2.5-VL), ERNIE-Image (Ministral 3), Lumina 2 and Anima are not supported.

## Credits

The idea of using the image model's own text encoder for prompt expansion comes from [ComfyUI-SingleEncoderEnhancer](https://github.com/Magirad/ComfyUI-SingleEncoderEnhancer); the sentence trimming is adapted from it. The sampling filters follow ComfyUI's text generation. Both are GPL-3.0, and so is this extension. See [LICENSE](LICENSE).
