"""A text encoder for P-Image-OSS (formerly P-H3-Image) and MiniMax-H3-Image that writes well: the stock Qwen3-VL 4B
with Pruna's text connector.

Pruna's encoder file holds a Qwen3-VL 4B that was retrained for the image model, and a small text connector (six
tensors, 262 MB) that turns five of its hidden states into the conditioning. The retrained model writes poor text
(invented words, facts wrong). The stock model behind the same connector writes well, and the pictures made with it
are good: its conditioning stays close to Pruna's (0.97 to 0.99 cosine per token).

This module builds that file. No model is loaded.

- The language model and the vision tower come from Qwen's own repository (two files, 8.9 GB, Apache 2.0) or from a
  stock Qwen3-VL 4B file that is already on disk (the Krea 2 encoder is one).
- The connector comes from a Pruna encoder file on disk, or its 262 MB are downloaded from Pruna's repository.
- The result is int8 with ConvRot, the format of Pruna's own int8 file (5.1 GB): the full-precision weights are
  rotated and quantised row by row here, with the quantiser Forge itself decodes with (comfy-kitchen). That needs
  the full-precision model, so Qwen's files unless a bf16 stock file is on disk. Without torch, or on request, the
  tensors are copied as they are stored instead: bf16 from Qwen's files (9.1 GB), or whatever a stock file on disk
  holds (fp8 from the Krea 2 encoder), with nothing to download.

Licences: the Qwen3-VL-4B parts are under the Apache License 2.0. The text connector is part of P-Image-OSS by Pruna
AI and under the MiniMax H3 Community License Agreement, like the image model itself. The file this builds is for
your own use under those terms; it is not something to pass on as Apache-licensed.

From a command line, without the web UI (run in the extension's folder, with Forge's Python):

    python -m lib_prompt_expansion.hybrid <text encoder folder> [--precision int8_convrot|bf16|copy] [--stock FILE]
                                          [--connector FILE] [--plan]
"""

import hashlib
import json
import os
import shutil
import struct
from dataclasses import dataclass, field
from typing import Callable, Iterable, Optional

from .fetch import CHUNK, TAG, FetchError, HeadDownload, Stopped, _explain, download

CONNECTOR = "text_connector."
CONNECTOR_KEYS = ("input_mean", "input_std", "output_mean", "output_std", "proj.weight", "sink_embedding")
CONNECTOR_BYTES = 262_307_840
CONNECTOR_SHA256 = "5b32c68d101f14062a9c84a041e2d3da150f6cf1c6656092a5fdddcd22ad97bb"
LANGUAGE, VISUAL = "model.language_model.", "model.visual."
HIDDEN, VOCAB, LAYERS = 2560, 151936, 36

# int8 with ConvRot, as in Pruna's int8 file: the weights of these layers are rotated in groups of 256 inputs and
# stored as int8 with one scale per row; everything else (norms, the vision tower, the connector) stays as it is
INT8, BF16, COPY = "int8_convrot", "bf16", "copy"
GROUP = 256
QUANTISED = ("self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj", "self_attn.o_proj", "mlp.gate_proj", "mlp.up_proj", "mlp.down_proj")
QUANT_NOTE = json.dumps({"format": "int8_tensorwise", "convrot": True, "convrot_groupsize": GROUP}).encode("utf-8")
FULL = {"BF16": 2, "F16": 2, "F32": 4}  # full-precision storage that can be quantised, and its bytes per value

# Qwen's own release of the model, pinned to one revision
QWEN = tuple(
    HeadDownload(repo="Qwen/Qwen3-VL-4B-Instruct", revision="ebb281ec70b05090aa6165b016eac8ec08e71b17", filename=name, size=size, sha256=sha, save_as=f"qwen3-vl-4b-instruct.{name}.download", licence="Apache 2.0")
    for name, size, sha in (
        ("model-00001-of-00002.safetensors", 4_967_229_296, "30a01a0556622645a3cce87b655bbbbbc1f170c196099f1b666c93202c3339a9"),
        ("model-00002-of-00002.safetensors", 3_908_490_048, "046296a2a387efb43b0c997d5833c789604d168834f6e0d3064bf7bb13d002a6"),
    )
)
# Pruna's encoder file; only its connector is read from it
PRUNA = HeadDownload(
    repo="PrunaAI/P-Image-OSS-Comfy",
    revision="177745a635feb2f0adf3db3693297cb63fea3dda",
    filename="text_encoders/p-image-oss_text_encoder_int8_convrot.safetensors",
    size=5_120_870_504,
    sha256="",
    save_as="p-image-oss_text_connector.download",
    licence="MiniMax H3 Community License Agreement",
)
LICENCE_NOTE = (
    "Language model and vision tower: Qwen3-VL-4B by the Qwen team, Apache License 2.0. "
    "Text connector: part of P-Image-OSS by Pruna AI, based on MiniMax-H3 by MiniMax, MiniMax H3 Community License Agreement."
)


class BuildError(Exception):
    """The file was not built; the text is written for the user."""


# --- reading safetensors files --------------------------------------------------------------------------------


def read_header(path: str) -> tuple[dict, dict, int]:
    """(tensors, metadata, where the data starts) of a .safetensors file."""
    with open(path, "rb") as handle:
        (size,) = struct.unpack("<Q", handle.read(8))
        if size > 100_000_000:
            raise ValueError("not a safetensors header")
        tensors = json.loads(handle.read(size))
    metadata = tensors.pop("__metadata__", None) or {}
    return tensors, metadata, 8 + size


def kind(path: str) -> Optional[str]:
    """What a file can give: "connector" (Pruna's encoder or a file built here), "stock" (a Qwen3-VL 4B without a
    connector), or None. Only the header is read."""
    try:
        if not str(path).lower().endswith((".safetensors", ".sft")):
            return None
        tensors, _, _ = read_header(path)
    except (OSError, ValueError, struct.error):
        return None
    if tensors.get(CONNECTOR + "proj.weight", {}).get("shape") == [2 * HIDDEN, 5 * HIDDEN] and all(CONNECTOR + key in tensors for key in CONNECTOR_KEYS):
        return "connector"
    prefix = _language_prefix(tensors)
    if prefix is None:
        return None
    whole = (
        tensors.get(prefix + "embed_tokens.weight", {}).get("shape") == [VOCAB, HIDDEN]
        and prefix + f"layers.{LAYERS - 1}.self_attn.q_proj.weight" in tensors
        and prefix + f"layers.{LAYERS}.self_attn.q_proj.weight" not in tensors
        and prefix + "norm.weight" in tensors
        and any(key.startswith(VISUAL) for key in tensors)
    )
    return "stock" if whole else None


def _language_prefix(tensors: dict) -> Optional[str]:
    if any(key.startswith(LANGUAGE) for key in tensors):
        return LANGUAGE  # as Qwen and Pruna store it
    if "model.layers.0.self_attn.q_proj.weight" in tensors:
        return "model."  # as Forge's and ComfyUI's own files store it
    return None


def _stored(path: str) -> str:
    """How a stock file stores its language model: "bf16", "fp16", "fp32", "fp8", "int8", "4bit", or "mixed".

    Every linear layer of every block is looked at. A file that keeps a few layers whole and the rest in 4 bits (an
    nvfp4 "full" file does) is "mixed", not full precision.
    """
    tensors, _, _ = read_header(path)
    prefix = _language_prefix(tensors)
    kinds = {tensors.get(f"{prefix}layers.{layer}.{part}.weight", {}).get("dtype", "") for layer in range(LAYERS) for part in QUANTISED}
    if len(kinds) != 1:
        return "mixed"
    dtype = kinds.pop()
    return {"BF16": "bf16", "F16": "fp16", "F32": "fp32", "F8_E4M3": "fp8", "F8_E5M2": "fp8", "I8": "int8", "U8": "4bit"}.get(dtype, dtype.lower() or "unknown")


def is_full_precision(path: str) -> bool:
    """Can int8 with ConvRot be made from this stock file? Only if every weight to be quantised is stored whole."""
    tensors, _, _ = read_header(path)
    prefix = _language_prefix(tensors)
    return _stored(path) in ("bf16", "fp16", "fp32") and tensors.get(prefix + "embed_tokens.weight", {}).get("dtype") in FULL


# --- the plan -------------------------------------------------------------------------------------------------


@dataclass
class Plan:
    folder: str
    stock: Optional[str]  # a local file, or None = download Qwen's
    connector: Optional[str]  # a local file, or None = download from Pruna's repository
    precision: str
    target: str
    downloads: int = 0  # bytes still to download
    quantise: bool = False  # the language model is quantised to int8 with ConvRot here
    stored: str = "bf16"  # how the stock source stores the model
    notes: list[str] = field(default_factory=list)

    @property
    def exists(self) -> bool:
        return os.path.isfile(self.target)

    def key(self) -> str:
        return "|".join((self.target, str(self.stock), str(self.connector), str(self.quantise)))

    def describe(self) -> str:
        """What will be done, as a few short lines of Markdown."""
        name = os.path.basename(self.target)
        if self.exists:
            return f"**Already built:** `{name}` in `{self.folder}`. Select it under VAE / Text Encoder in place of Pruna's encoder."
        if self.stock:
            stock = f"`{os.path.basename(self.stock)}` ({self.stored})"
        else:
            stock = f"download from {QWEN[0].repo} ({sum(part.size for part in QWEN) / 1e9:.1f} GB, Apache 2.0)"
        stock += ", then quantise to int8 with ConvRot" if self.quantise else ", copied as it is"
        if self.connector:
            connector = f"from `{os.path.basename(self.connector)}`"
        else:
            connector = f"download from {PRUNA.repo} ({CONNECTOR_BYTES / 1e9:.2f} GB, {PRUNA.licence})"
        return f"**Will build** `{name}`\n\n- Stock Qwen3-VL 4B: {stock}\n- Pruna's connector: {connector}\n- Into: `{self.folder}`"


def can_quantise() -> bool:
    try:
        import torch  # noqa: F401
    except Exception:  # noqa: BLE001 - a broken install counts as none
        return False
    return True


def make_plan(folder: str, files: Iterable[str] = (), stock: Optional[str] = None, connector: Optional[str] = None, precision: Optional[str] = None) -> Plan:
    """What would be built in `folder`, and from what. files: the encoder files to look through for sources.

    precision: "int8_convrot" (quantised here from the full-precision model; the default where torch is at hand),
    "bf16" (Qwen's weights as they are), or "copy" (whatever a stock file on disk holds, nothing downloaded if there
    is one). stock / connector: a file to use, whatever is found.
    """
    files = [path for path in files if path]
    precision = precision or (INT8 if can_quantise() else COPY)
    if precision not in (INT8, BF16, COPY):
        raise BuildError(f"{precision} is not a precision this can build")
    if precision == INT8 and not can_quantise():
        raise BuildError("int8 with ConvRot needs torch, which this Python does not have")
    if connector is None:
        # Pruna's own file before a file built here (their connectors are the same bytes; the name reads better)
        candidates = sorted((path for path in files if kind(path) == "connector"), key=lambda path: bool(read_header(path)[1].get("built_by")))
        connector = candidates[0] if candidates else None
    elif kind(connector) != "connector":
        raise BuildError(f"{connector} has no text connector")
    if stock is not None and kind(stock) != "stock":
        raise BuildError(f"{stock} is not a stock Qwen3-VL 4B file")
    if stock is None:
        found = [path for path in files if kind(path) == "stock"]
        if precision == INT8:
            # only the full-precision model can be quantised; a quantised or mixed file on disk is passed over
            # and Qwen's own files are downloaded instead
            found = [path for path in found if is_full_precision(path)]
        elif precision == BF16:
            found = [path for path in found if is_full_precision(path) and _stored(path) == "bf16"]
        # the best stored form first, then the smallest file
        rank = {"bf16": 0, "fp16": 0, "fp32": 0, "fp8": 1, "int8": 1}
        stock = min(found, key=lambda path: (rank.get(_stored(path), 2), os.path.getsize(path))) if found else None
    elif precision == INT8 and not is_full_precision(stock):
        raise BuildError(f"{os.path.basename(stock)} is not a full-precision file (it is {_stored(stock)}); int8 with ConvRot is made from the full-precision model. Leave the file out and Qwen's own files are downloaded")
    label = INT8 if precision == INT8 else (_stored(stock) if stock is not None else "bf16")
    target = os.path.join(folder, f"p-image-oss_text_encoder_stock_qwen3vl_4b_{label}.safetensors")
    downloads = (0 if stock else sum(part.size for part in QWEN)) + (0 if connector else CONNECTOR_BYTES)
    return Plan(folder, stock, connector, label, target, downloads, quantise=precision == INT8, stored=_stored(stock) if stock is not None else "bf16")


# --- building -------------------------------------------------------------------------------------------------


def _stock_tensors(path: str) -> list[tuple]:
    """(new name, entry, file, start in the file) for every tensor of a stock file, in the P-Image-OSS layout."""
    tensors, _, base = read_header(path)
    prefix = _language_prefix(tensors)
    out = []
    for name, entry in sorted(tensors.items(), key=lambda item: item[1]["data_offsets"][0]):
        if name.startswith(VISUAL):
            new = name
        elif name.startswith(prefix):
            new = LANGUAGE + name[len(prefix) :]
        else:
            continue  # an output head or anything else the encoder has no use for
        out.append((new, entry, path, base + entry["data_offsets"][0]))
    return out


def _connector_tensors(path: str) -> list[tuple]:
    tensors, _, base = read_header(path)
    names = sorted((name for name in tensors if name.startswith(CONNECTOR)), key=lambda name: tensors[name]["data_offsets"][0])
    return [(name, tensors[name], path, base + tensors[name]["data_offsets"][0]) for name in names]


def _fetch_connector(folder: str, should_stop, log, endpoint, progress) -> str:
    """Download only the connector's bytes from Pruna's encoder file and save them as a small safetensors file."""
    import requests

    url = PRUNA.url(endpoint)
    host = url.split("/")[2]
    target = os.path.join(folder, PRUNA.save_as)
    part = target + ".part"
    headers = {"User-Agent": "webui-prompt-expansion"}
    try:
        if os.path.isfile(target) and kind_of_connector_file(target):
            return target
        os.makedirs(folder, exist_ok=True)

        def ranged(start, end):
            with requests.get(url, headers={**headers, "Range": f"bytes={start}-{end}"}, timeout=(15, 60)) as response:
                response.raise_for_status()
                if response.status_code != 206:
                    raise FetchError(f"{host} does not send parts of a file")
                return response.content

        (size,) = struct.unpack("<Q", ranged(0, 7))
        if size > 10_000_000:
            raise FetchError(f"what {host} sent is not the expected file")
        remote = json.loads(ranged(8, 8 + size - 1))
        names = sorted((name for name in remote if name.startswith(CONNECTOR)), key=lambda name: remote[name]["data_offsets"][0])
        first, last = remote[names[0]]["data_offsets"][0], remote[names[-1]]["data_offsets"][1]
        if [name[len(CONNECTOR) :] for name in sorted(names)] != sorted(CONNECTOR_KEYS) or last - first != CONNECTOR_BYTES:
            raise FetchError(f"the file at {host} does not hold the expected text connector")
        start = 8 + size + first

        digest, have = hashlib.sha256(), 0
        if os.path.exists(part) and os.path.getsize(part) <= CONNECTOR_BYTES:
            with open(part, "rb") as handle:
                while block := handle.read(CHUNK):
                    digest.update(block)
                    have += len(block)
        elif os.path.exists(part):
            os.remove(part)
        if have < CONNECTOR_BYTES:
            log(f"{TAG} Downloading Pruna's text connector ({CONNECTOR_BYTES / 1e9:.2f} GB, {PRUNA.licence}) from {PRUNA.repo}")
            with requests.get(url, headers={**headers, "Range": f"bytes={start + have}-{start + CONNECTOR_BYTES - 1}"}, stream=True, timeout=(15, 60)) as response:
                response.raise_for_status()
                if response.status_code != 206:
                    raise FetchError(f"{host} does not send parts of a file")
                bar = _bar(CONNECTOR_BYTES, have, "text connector") if progress else None
                try:
                    with open(part, "ab" if have else "wb") as handle:
                        for block in response.iter_content(CHUNK):
                            if should_stop is not None and should_stop():
                                raise Stopped("interrupted")
                            handle.write(block)
                            digest.update(block)
                            have += len(block)
                            if bar is not None:
                                bar.update(len(block))
                finally:
                    if bar is not None:
                        bar.close()
        if have != CONNECTOR_BYTES or digest.hexdigest() != CONNECTOR_SHA256:
            os.remove(part)
            raise FetchError(f"what {host} sent is not the expected text connector (size or SHA-256 differs); it was deleted")

        # wrap the bytes as a safetensors file of their own, so that they are read like any other source
        offset, table = 0, {}
        for name in names:
            entry = remote[name]
            length = entry["data_offsets"][1] - entry["data_offsets"][0]
            table[name] = {"dtype": entry["dtype"], "shape": entry["shape"], "data_offsets": [offset, offset + length]}
            offset += length
        head = _header_bytes(table, {"format": "pt", "license": PRUNA.licence, "source": f"{PRUNA.repo}@{PRUNA.revision}"})
        with open(target + ".tmp", "wb") as out, open(part, "rb") as body:
            out.write(head)
            shutil.copyfileobj(body, out, CHUNK)
        os.replace(target + ".tmp", target)
        os.remove(part)
        return target
    except FetchError:
        raise
    except Exception as problem:  # noqa: BLE001 - every failure becomes one sentence
        raise FetchError(_explain(problem, host)) from problem


def kind_of_connector_file(path: str) -> bool:
    try:
        tensors, _, base = read_header(path)
    except (OSError, ValueError, struct.error):
        return False
    return sorted(tensors) == sorted(CONNECTOR + key for key in CONNECTOR_KEYS) and os.path.getsize(path) == base + CONNECTOR_BYTES


def _header_bytes(table: dict, metadata: dict) -> bytes:
    text = json.dumps({"__metadata__": metadata, **table}, separators=(",", ":")).encode("utf-8")
    text += b" " * (-len(text) % 8)
    return struct.pack("<Q", len(text)) + text


def _bar(total: int, initial: int, label: str):
    try:
        from tqdm import tqdm

        return tqdm(total=total, initial=initial, unit="B", unit_scale=True, unit_divisor=1000, desc=label, leave=False)
    except Exception:  # noqa: BLE001 - a progress bar is not worth a failed build
        return None


def is_quantised_here(name: str, entry: dict) -> bool:
    """Is this one of the weights that an int8 ConvRot file stores as int8? (Pruna's file: the same ones.)"""
    if not name.startswith(LANGUAGE) or entry["dtype"] not in FULL or len(entry["shape"]) != 2 or entry["shape"][1] % GROUP:
        return False
    rest = name[len(LANGUAGE) :]
    return rest == "embed_tokens.weight" or (rest.startswith("layers.") and rest.endswith(".weight") and rest.split(".", 2)[2][: -len(".weight")] in QUANTISED)


def _hadamard(torch, size: int, device):
    """The normalised regular Hadamard matrix ConvRot uses (as comfy_kitchen.tensor.int8_utils builds it)."""
    base = torch.tensor([[1, 1, 1, -1], [1, 1, -1, 1], [1, -1, 1, 1], [-1, 1, 1, 1]], dtype=torch.float32, device=device)
    matrix = base
    while matrix.shape[0] < size:
        matrix = torch.kron(matrix, base)
    return matrix / (size**0.5)


def quantise_convrot(raw: bytes, dtype: str, shape: list) -> tuple[bytes, bytes]:
    """(int8 codes, one float32 scale per row) of a full-precision weight: rotated in groups, then rounded row-wise.

    The same arithmetic as comfy-kitchen's quantize_int8_convrot_weight, which is used when it is installed: rows of
    the rotated weight are divided by their largest magnitude over 127 and rounded to the nearest integer.
    """
    import torch

    kinds = {"BF16": torch.bfloat16, "F16": torch.float16, "F32": torch.float32}
    rows, columns = shape

    def run(device):
        weight = torch.frombuffer(bytearray(raw), dtype=kinds[dtype]).reshape(rows, columns).to(device=device, dtype=torch.float32)
        try:
            from comfy_kitchen.backends.eager.quantization import quantize_int8_convrot_weight

            codes, scale = quantize_int8_convrot_weight(weight, GROUP)
        except ImportError:
            rotated = torch.matmul(weight.reshape(rows, columns // GROUP, GROUP), _hadamard(torch, GROUP, device).T).reshape(rows, columns)
            scale = (rotated.abs().amax(dim=-1, keepdim=True) / 127.0).clamp(min=1e-30)
            codes = torch.round(rotated / scale).clamp(-127, 127)
        return codes.to(torch.int8).cpu().numpy().tobytes(), scale.reshape(rows, 1).to(torch.float32).cpu().numpy().tobytes()

    with torch.inference_mode():
        if torch.cuda.is_available():
            try:
                return run(torch.device("cuda"))
            except RuntimeError:  # most likely no memory left beside the loaded models
                torch.cuda.empty_cache()
        return run(torch.device("cpu"))


def _with_quantising(parts: list[tuple]) -> list[tuple]:
    """The same tensors, with every weight that is to be quantised replaced by its three stored tensors."""
    out = []
    stems = {name[: -len("weight")] for name, entry, _, _ in parts if is_quantised_here(name, entry)}
    for name, entry, source, start in parts:
        if not is_quantised_here(name, entry):
            if name.rsplit(".", 1)[0] + "." in stems:
                continue  # storage data of that weight's old form: it gets its own below
            out.append((name, entry, source, start))
            continue
        rows, columns = entry["shape"]
        stem = name[: -len("weight")]
        origin = (entry, source, start)
        out.append((name, {"dtype": "I8", "shape": [rows, columns], "data_offsets": [0, rows * columns]}, origin, "codes"))
        out.append((stem + "weight_scale", {"dtype": "F32", "shape": [rows, 1], "data_offsets": [0, rows * 4]}, origin, "scale"))
        out.append((stem + "comfy_quant", {"dtype": "U8", "shape": [len(QUANT_NOTE)], "data_offsets": [0, len(QUANT_NOTE)]}, origin, "note"))
    return out


def assemble(parts: list[tuple], target: str, metadata: dict, should_stop: Optional[Callable[[], bool]] = None, progress: bool = True) -> int:
    """Write the tensors into one safetensors file.

    A part is (name, entry, file, start in that file): its bytes are copied. Or, from _with_quantising,
    (name, entry of what is written, (source entry, file, start), "codes" | "scale" | "note"): the source weight is
    quantised when its codes are written, and its scales and its format note follow.
    """
    names = [name for name, *_ in parts]
    if len(set(names)) != len(names):
        raise BuildError("a tensor would be written twice")
    offset, table = 0, {}
    for name, entry, _, _ in parts:
        length = entry["data_offsets"][1] - entry["data_offsets"][0]
        table[name] = {"dtype": entry["dtype"], "shape": entry["shape"], "data_offsets": [offset, offset + length]}
        offset += length
    head = _header_bytes(table, metadata)
    free = shutil.disk_usage(os.path.dirname(target)).free
    if free < len(head) + offset + (64 << 20):
        raise BuildError(f"not enough free space in {os.path.dirname(target)}: {free / 1e9:.1f} GB free, {(len(head) + offset) / 1e9:.1f} GB needed")
    part = target + ".part"
    bar = _bar(offset, 0, os.path.basename(target)) if progress else None
    handles: dict[str, object] = {}
    scales = b""
    try:
        with open(part, "wb") as out:
            out.write(head)
            for _, entry, source, start in parts:
                if isinstance(source, tuple):  # a weight that is quantised here
                    if should_stop is not None and should_stop():
                        raise Stopped("interrupted")
                    if start == "codes":
                        origin, path, begin = source
                        handle = handles.get(path)
                        if handle is None:
                            handle = handles[path] = open(path, "rb")
                        handle.seek(begin)
                        raw = handle.read(origin["data_offsets"][1] - origin["data_offsets"][0])
                        if len(raw) != origin["shape"][0] * origin["shape"][1] * FULL[origin["dtype"]]:
                            raise BuildError(f"{os.path.basename(path)} ends before its last tensor")
                        block, scales = quantise_convrot(raw, origin["dtype"], origin["shape"])
                    else:
                        block = scales if start == "scale" else QUANT_NOTE
                    if len(block) != entry["data_offsets"][1] - entry["data_offsets"][0]:
                        raise BuildError("a quantised tensor has an unexpected size")
                    out.write(block)
                    if bar is not None:
                        bar.update(len(block))
                    continue
                handle = handles.get(source)
                if handle is None:
                    handle = handles[source] = open(source, "rb")
                handle.seek(start)
                left = entry["data_offsets"][1] - entry["data_offsets"][0]
                while left:
                    if should_stop is not None and should_stop():
                        raise Stopped("interrupted")
                    block = handle.read(min(left, 8 * CHUNK))
                    if not block:
                        raise BuildError(f"{os.path.basename(source)} ends before its last tensor")
                    out.write(block)
                    left -= len(block)
                    if bar is not None:
                        bar.update(len(block))
    except BaseException:
        for handle in handles.values():
            handle.close()
        handles.clear()
        if os.path.exists(part):
            os.remove(part)
        raise
    finally:
        for handle in handles.values():
            handle.close()
        if bar is not None:
            bar.close()
    os.replace(part, target)
    return len(head) + offset


def check(target: str) -> None:
    """The built file has everything the P-Image-OSS encoder needs. Raises BuildError."""
    tensors, _, base = read_header(target)
    wanted = [CONNECTOR + key for key in CONNECTOR_KEYS] + [LANGUAGE + "embed_tokens.weight", LANGUAGE + "norm.weight"]
    wanted += [LANGUAGE + f"layers.{layer}.{name}.weight" for layer in range(LAYERS) for name in ("self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj", "self_attn.o_proj", "mlp.gate_proj", "mlp.up_proj", "mlp.down_proj", "input_layernorm", "post_attention_layernorm")]
    missing = [name for name in wanted if name not in tensors]
    end = max(entry["data_offsets"][1] for entry in tensors.values())
    if missing or not any(name.startswith(VISUAL) for name in tensors) or os.path.getsize(target) != base + end:
        raise BuildError(f"the built file is not complete ({len(missing)} tensors missing)")


def build(
    plan: Plan,
    *,
    should_stop: Optional[Callable[[], bool]] = None,
    log: Callable[[str], None] = print,
    endpoint: Optional[str] = None,
    progress: bool = True,
    keep_downloads: bool = False,
) -> str:
    """Download what is missing and build the file. Returns its path. Raises BuildError with the reason."""
    if plan.exists:
        return plan.target
    downloaded: list[str] = []
    try:
        os.makedirs(plan.folder, exist_ok=True)
        connector = plan.connector
        if connector is None:
            connector = _fetch_connector(plan.folder, should_stop, log, endpoint, progress)
            downloaded.append(connector)
        connector_parts = _connector_tensors(connector)
        if sum(entry["data_offsets"][1] - entry["data_offsets"][0] for _, entry, _, _ in connector_parts) != CONNECTOR_BYTES:
            raise BuildError(f"the text connector in {os.path.basename(connector)} has an unexpected size")

        if plan.stock is not None:
            stock_parts = _stock_tensors(plan.stock)
            stock_name = os.path.basename(plan.stock)
        else:
            stock_parts = []
            for part in QWEN:
                path = os.path.join(plan.folder, part.save_as)
                if not os.path.isfile(path):
                    log(f"{TAG} Downloading {part.filename} of {part.repo} ({part.size / 1e9:.2f} GB, {part.licence})")
                    download(part, path, should_stop=should_stop, endpoint=endpoint, progress=progress)
                downloaded.append(path)
                stock_parts += _stock_tensors(path)
            stock_name = f"{QWEN[0].repo}@{QWEN[0].revision[:12]}"

        metadata = {"format": "pt", "built_by": "webui-prompt-expansion", "language_model": stock_name, "text_connector": os.path.basename(connector) if plan.connector else f"{PRUNA.repo}@{PRUNA.revision[:12]}", "license": LICENCE_NOTE}
        if plan.quantise:
            count = sum(is_quantised_here(name, entry) for name, entry, _, _ in stock_parts)
            if count != 7 * LAYERS + 1:
                raise BuildError(f"{stock_name} is not stored in full precision, so it cannot be quantised here")
            stock_parts = _with_quantising(stock_parts)
            metadata["quantization"] = f"int8 with ConvRot (group {GROUP}), quantised by webui-prompt-expansion from {stock_name}"
            log(f"{TAG} Quantising {count} weights to int8 with ConvRot and building {os.path.basename(plan.target)}")
        else:
            log(f"{TAG} Building {os.path.basename(plan.target)}: {len(stock_parts)} tensors of the stock model, {len(connector_parts)} of the connector")
        size = assemble(stock_parts + connector_parts, plan.target, metadata, should_stop, progress)
        check(plan.target)
        log(f"{TAG} Built {plan.target} ({size / 1e9:.2f} GB). Select it under VAE / Text Encoder in place of Pruna's encoder.")
        if not keep_downloads:
            for path in downloaded:
                try:
                    os.remove(path)
                except OSError:
                    pass
        return plan.target
    except Stopped as stop:
        raise BuildError("interrupted; what was downloaded is kept, and the next try goes on from there") from stop
    except FetchError as problem:
        raise BuildError(f"a download failed: {problem}") from problem
    except OSError as problem:
        raise BuildError(f"a file could not be read or written ({problem.strerror or problem})") from problem


def press(armed: str, folder: Optional[str], files: Iterable[str], **options) -> tuple[str, str]:
    """The panel's button. The first press says what would be done and arms it; a second press on the same plan does it.

    Returns (what is armed now, the text to show). While something is armed the button reads "Click again to build".
    Never raises.
    """
    try:
        if not folder:
            return "", "Forge's text encoder folder is not known."
        plan = make_plan(folder, files, precision=options.pop("precision", None))
        if plan.exists:
            return "", plan.describe()
        if armed != plan.key():
            return plan.key(), plan.describe() + "\n\nProgress is shown in the console."
        target = build(plan, **options)
        return "", f"**Built** `{os.path.basename(target)}` in `{folder}`. Refresh the VAE / Text Encoder list and select it in place of Pruna's encoder."
    except BuildError as problem:
        return "", f"**Not built.** {str(problem)[:1].upper()}{str(problem)[1:]}."
    except Exception as problem:  # noqa: BLE001 - a button must answer, whatever happened
        return "", f"**Not built.** {type(problem).__name__}: {problem}"


def main(arguments: Optional[list[str]] = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(prog="python -m lib_prompt_expansion.hybrid", description="Build a P-Image-OSS text encoder from the stock Qwen3-VL 4B and Pruna's text connector.")
    parser.add_argument("folder", help="the text encoder folder: sources are looked for here, the file is built here")
    parser.add_argument("--stock", help="a stock Qwen3-VL 4B file to use")
    parser.add_argument("--connector", help="a Pruna encoder file to take the connector from")
    parser.add_argument("--precision", choices=(INT8, BF16, COPY), help="int8_convrot: quantised here from the full-precision model (the default); bf16: Qwen's weights as they are; copy: a stock file from disk as it is stored")
    parser.add_argument("--keep-downloads", action="store_true", help="keep the downloaded source files")
    parser.add_argument("--plan", action="store_true", help="only say what would be done")
    options = parser.parse_args(arguments)
    folder = os.path.abspath(options.folder)
    files = sorted(os.path.join(folder, name) for name in os.listdir(folder)) if os.path.isdir(folder) else []
    try:
        plan = make_plan(folder, files, stock=options.stock, connector=options.connector, precision=options.precision)
        print(plan.describe())
        if options.plan or plan.exists:
            return 0
        build(plan, keep_downloads=options.keep_downloads)
        return 0
    except BuildError as problem:
        print(f"{TAG} Not built: {problem}.")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
