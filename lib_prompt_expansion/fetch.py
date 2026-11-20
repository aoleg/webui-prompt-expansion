"""Download an encoder's own output head when its file was saved without one.

Qwen publishes the head of Qwen3-8B as one shard of its repository: a file that holds that tensor and nothing else.
When FLUX.2 Klein 9B is first used for writing and the head is not on disk, that shard is downloaded once into
Forge's text encoder folder, checked against its SHA-256, and found there from then on.

Nothing here raises into a generation: a failed download is explained in the console and the job goes on (with a
borrowed head when one is at hand, else with the prompts as typed).
"""

import hashlib
import os
import shutil
import threading
from dataclasses import dataclass
from typing import Callable, Iterable, Optional

CHUNK = 1 << 20
TAG = "[Prompt Expansion]"


class FetchError(Exception):
    """A download that did not happen; the text is written for the console."""


class Stopped(FetchError):
    """The user interrupted the job while the download ran."""


@dataclass(frozen=True)
class HeadDownload:
    repo: str  # the Hugging Face repository
    revision: str  # the commit the file is taken from, so that it is always the same file
    filename: str  # its name in the repository
    size: int
    sha256: str
    save_as: str  # its name in the text encoder folder
    licence: str

    def url(self, endpoint: Optional[str] = None) -> str:
        endpoint = (endpoint or os.environ.get("HF_ENDPOINT") or "https://huggingface.co").rstrip("/")
        return f"{endpoint}/{self.repo}/resolve/{self.revision}/{self.filename}"

    def describe(self) -> str:
        return f"{self.filename} of {self.repo} on Hugging Face"

    def gigabytes(self) -> str:
        return f"{self.size / 1e9:.2f} GB"


_lock = threading.Lock()
_failed: dict[str, str] = {}  # save_as -> reason; a failed download is not tried again in this session


def forget_failures():
    """Called by the panel's refresh button: the next job tries a failed download again."""
    _failed.clear()


def failure(source: "HeadDownload") -> Optional[str]:
    """Why this download failed in this session, or None."""
    return _failed.get(source.save_as)


def configured_folders() -> list[str]:
    """Forge's text encoder folders: those given on the command line first, then models/text_encoder."""
    folders = []
    try:
        from modules import paths, shared

        folders += [str(folder) for folder in (getattr(shared.cmd_opts, "text_encoder_dirs", None) or [])]
        folders.append(os.path.join(paths.models_path, "text_encoder"))
    except Exception:  # noqa: BLE001 - outside Forge there is nothing to look at
        pass
    return [os.path.abspath(folder) for folder in folders]


def _inside(path: str, folder: str) -> bool:
    try:
        return os.path.commonpath([os.path.abspath(path), folder]) == folder
    except ValueError:  # another drive
        return False


def target_folder(loaded: Iterable[str], folders: Optional[Iterable[str]] = None) -> Optional[str]:
    """Where a head is saved: the text encoder folder the loaded encoder came from, else the first one configured."""
    folders = [os.path.abspath(folder) for folder in (configured_folders() if folders is None else folders)]
    for path in loaded:
        for folder in folders:
            if path and _inside(path, folder):
                return folder
    return folders[0] if folders else None


def target_path(source: HeadDownload, loaded: Iterable[str], folders: Optional[Iterable[str]] = None) -> Optional[str]:
    folder = target_folder(loaded, folders)
    return None if folder is None else os.path.join(folder, source.save_as)


def _explain(problem: Exception, host: str) -> str:
    """One readable sentence for a network or disk error."""
    import requests

    if isinstance(problem, FetchError):
        return str(problem)
    if isinstance(problem, requests.exceptions.HTTPError):
        code = getattr(problem.response, "status_code", 0)
        meaning = {401: "the server asks for a login", 403: "the server refused access", 404: "the file is not there any more", 429: "the server is limiting downloads right now"}.get(code, "the server reported an error" if code >= 500 else "the request was refused")
        return f"{host} answered {code} ({meaning})"
    if isinstance(problem, requests.exceptions.SSLError):
        return f"no secure connection to {host} ({type(problem).__name__})"
    if isinstance(problem, requests.exceptions.Timeout):
        return f"{host} did not answer in time"
    if isinstance(problem, requests.exceptions.ConnectionError):
        return f"no connection to {host} (offline, blocked, or the address does not resolve)"
    if isinstance(problem, requests.exceptions.RequestException):
        return f"the download from {host} broke off ({type(problem).__name__})"
    if isinstance(problem, PermissionError):
        return f"no permission to write there ({problem.strerror or problem})"
    if isinstance(problem, OSError):
        return f"the file could not be written ({problem.strerror or problem})"
    return f"{type(problem).__name__}: {problem}"


def download(source: HeadDownload, target: str, *, should_stop: Optional[Callable[[], bool]] = None, endpoint: Optional[str] = None, progress: bool = True) -> str:
    """Download `source` to `target`, resuming a part file, and check size and SHA-256. Raises FetchError."""
    import requests

    url = source.url(endpoint)
    host = url.split("/")[2]
    part = target + ".part"
    try:
        if os.path.exists(target):
            raise FetchError(f"{target} exists but is not the expected file; delete or rename it")
        os.makedirs(os.path.dirname(target), exist_ok=True)

        digest, have = hashlib.sha256(), 0
        if os.path.exists(part):
            if os.path.getsize(part) > source.size:
                os.remove(part)
            else:
                with open(part, "rb") as handle:
                    while block := handle.read(CHUNK):
                        digest.update(block)
                        have += len(block)
        free = shutil.disk_usage(os.path.dirname(target)).free
        if free < source.size - have + (64 << 20):
            raise FetchError(f"not enough free space in {os.path.dirname(target)}: {free / 1e9:.2f} GB free, {(source.size - have) / 1e9:.2f} GB needed")

        if have < source.size:
            headers = {"User-Agent": "webui-prompt-expansion"}
            if have:
                headers["Range"] = f"bytes={have}-"
            with requests.get(url, headers=headers, stream=True, timeout=(15, 60), allow_redirects=True) as response:
                if have and response.status_code == 416:  # the part file is already the whole file or longer
                    response.close()
                    os.remove(part)
                    return download(source, target, should_stop=should_stop, endpoint=endpoint, progress=progress)
                response.raise_for_status()
                if have and response.status_code != 206:  # the server sent the whole file again
                    digest, have = hashlib.sha256(), 0
                bar = None
                if progress:
                    try:
                        from tqdm import tqdm

                        bar = tqdm(total=source.size, initial=have, unit="B", unit_scale=True, unit_divisor=1000, desc=source.save_as, leave=False)
                    except Exception:  # noqa: BLE001 - a progress bar is not worth a failed download
                        bar = None
                try:
                    with open(part, "ab" if have else "wb") as handle:
                        for block in response.iter_content(CHUNK):
                            if should_stop is not None and should_stop():
                                raise Stopped("the job was interrupted")
                            handle.write(block)
                            digest.update(block)
                            have += len(block)
                            if bar is not None:
                                bar.update(len(block))
                            if have > source.size:
                                break
                finally:
                    if bar is not None:
                        bar.close()

        if have < source.size:
            raise FetchError(f"the connection to {host} closed after {have / 1e9:.2f} of {source.gigabytes()}; what was downloaded is kept and the download resumes next time")
        if have != source.size or digest.hexdigest() != source.sha256:
            os.remove(part)
            raise FetchError(f"what {host} sent is not the expected file (size or SHA-256 differs); it was deleted")
        os.replace(part, target)
        return target
    except FetchError:
        raise
    except Exception as problem:  # noqa: BLE001 - every failure becomes one sentence for the console
        raise FetchError(_explain(problem, host)) from problem


def existing(source: HeadDownload, loaded: Iterable[str], folders: Optional[Iterable[str]] = None) -> list[str]:
    """The downloaded head, when it is on disk. Forge's own file list only learns of it at its next refresh."""
    path = target_path(source, list(loaded), folders)
    return [path] if path and os.path.isfile(path) else []


def ensure(
    source: HeadDownload,
    loaded: Iterable[str],
    *,
    encoder: str = "the text encoder",
    fallback: str = "",
    should_stop: Optional[Callable[[], bool]] = None,
    announce: Optional[Callable[[str], None]] = None,
    log: Callable[[str], None] = print,
    folders: Optional[Iterable[str]] = None,
    endpoint: Optional[str] = None,
    progress: bool = True,
) -> Optional[str]:
    """The path of the downloaded head, downloading it first when needed; None when that failed. Never raises.

    encoder: its name for the console. fallback: what happens to the job if the download fails, as a sentence.
    announce: called with one line for the UI's progress text while the download runs.
    """
    with _lock:
        try:
            target = target_path(source, list(loaded), folders)
            if target is None:
                raise FetchError("Forge's text encoder folder is not known")
            if source.save_as in _failed:
                return None
            if os.path.exists(target + ".part"):
                log(f"{TAG} Going on with the download of the output head of {encoder}: {source.describe()} ({source.gigabytes()}) to {target}")
            else:
                log(f"{TAG} {encoder} was saved without its output head. Downloading it once: {source.describe()} ({source.gigabytes()}, {source.licence}) to {target}")
            if announce is not None:
                announce(f"Prompt Expansion: downloading the output head ({source.gigabytes()})")
            download(source, target, should_stop=should_stop, endpoint=endpoint, progress=progress)
            log(f"{TAG} The output head is saved and its SHA-256 is verified. It is used from now on.")
            return target
        except Stopped:
            log(f"{TAG} The download of the output head was interrupted. What was downloaded is kept, and the next job goes on from there.")
            return None
        except Exception as problem:  # noqa: BLE001 - a download must never stop a generation
            reason = str(problem) if isinstance(problem, FetchError) else f"{type(problem).__name__}: {problem}"
            _failed[source.save_as] = reason
            where = os.path.dirname(target) if "target" in locals() and target else "a text encoder folder"
            log(f"{TAG} The output head could not be downloaded: {reason}.")
            if fallback:
                log(f"{TAG} {fallback}")
            log(f"{TAG} To get it by hand, download {source.url(endpoint)} and put it into {where}. The download is tried again after a restart of Forge, or when you press the refresh button of the Prompt Expansion panel.")
            return None
