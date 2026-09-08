"""Paths, config loading and the child-process environment.

Everything the app touches lives under ROOT so the folder stays portable
(CLAUDE.md section 16: nothing is written outside the app directory).
"""

from __future__ import annotations

import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

BIN = ROOT / "bin"
MODELS = ROOT / "models"
PROMPTS = ROOT / "prompts"
WEB = ROOT / "web"
TEMP = ROOT / "temp"
OUTPUT = ROOT / "output"

# whisper.cpp and llama.cpp ship *different* builds of ggml.dll, ggml-base.dll
# and ggml-cuda.dll. They cannot share a directory, so each engine keeps its own
# (see BUILD_NOTES.md). Windows searches the executable's own directory first,
# so each picks up the right DLLs; the shared CUDA runtime sits in bin\.
#
# There is now a second dimension: one folder per backend. CUDA where the
# machine has an NVIDIA card, Vulkan for AMD and Intel (and as a fallback
# anywhere), CPU as the last resort.
#
#     bin\llama-cuda    bin\llama-vulkan    bin\llama-cpu
#     bin\whisper-cuda  bin\whisper-vulkan  bin\whisper-cpu
FFMPEG = BIN / "ffmpeg.exe"

# Older installs had bin\llama\ and bin\whisper\ with no backend suffix.
LEGACY_DIRS = {"llama": BIN / "llama", "whisper": BIN / "whisper"}


def engine_dir(engine: str, backend: str) -> Path:
    """Where this engine's binaries live for a given backend."""
    candidate = BIN / ("%s-%s" % (engine, backend))
    if candidate.is_dir():
        return candidate
    legacy = LEGACY_DIRS.get(engine)
    if legacy is not None and legacy.is_dir():
        return legacy
    return candidate


def llama_server_path(backend: str) -> Path:
    return engine_dir("llama", backend) / "llama-server.exe"


def whisper_cli_path(backend: str) -> Path:
    return engine_dir("whisper", backend) / "whisper-cli.exe"


_backend_cache: dict = {}


def backend(engine: str = "llama") -> str:
    """The backend this engine will actually use, honouring config.json.

    Per engine, not per machine: whisper and llama are not shipped in step.

    Memoised. It reads config.json and stats the binary folders, and the ETA
    asks for it on every progress event; config.json is read once at startup
    anyway, so nothing here can change without a restart.
    """
    if engine in _backend_cache:
        return _backend_cache[engine]
    from . import hardware

    try:
        requested = str(load_config().get("gpu", {}).get("backend", "auto"))
    except RuntimeError:
        requested = "auto"
    _backend_cache[engine] = hardware.resolve_backend(engine, requested)
    return _backend_cache[engine]


class _BackendPath:
    """`config.WHISPER_CLI` used to be a constant; it is now a lookup.

    Kept as an attribute-compatible object so every existing `str(...)`,
    `.exists()` and `.parent` call site keeps working, rather than touching
    every caller to thread a backend through.
    """

    def __init__(self, engine: str, resolver):
        self._engine = engine
        self._resolver = resolver

    def _p(self) -> Path:
        return self._resolver(backend(self._engine))

    def __getattr__(self, name):
        return getattr(self._p(), name)

    def __fspath__(self) -> str:
        return str(self._p())

    def __str__(self) -> str:
        return str(self._p())

    def __truediv__(self, other):
        return self._p() / other


WHISPER_CLI = _BackendPath("whisper", whisper_cli_path)
LLAMA_SERVER = _BackendPath("llama", llama_server_path)

CONFIG_PATH = ROOT / "config.json"


def meeting_dir(meeting_name: str) -> Path:
    """Everything one meeting produced, in one folder.

    A run makes up to five artefacts plus a folder of voice clips. Flat in
    `output\\` that interleaves with every other meeting and there is nothing
    sensible to hand to somebody. The filenames keep the meeting prefix so a
    document still identifies itself once copied out of the folder.
    """
    return OUTPUT / Path(meeting_name).name


def write_atomic(path: Path, text: str) -> None:
    """Write a file so the previous version survives a failure mid-write.

    Documents are overwritten in place when they are regenerated. A cancel is
    already safe -- nothing is written until the model has finished -- but a
    crash, a full disk or a power cut during the write itself would leave a
    truncated file where a good one used to be. Write beside it and rename:
    os.replace is atomic on Windows within a volume, so a reader sees either
    the old file or the new one and never half of either.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".partial")
    try:
        tmp.write_text(text, encoding="utf-8")
        os.replace(tmp, path)
    except OSError:
        try:
            tmp.unlink()
        except OSError:
            pass
        raise


def migrate_flat_output() -> int:
    """Move pre-folder outputs into per-meeting folders. Returns how many moved.

    Run once at startup. Without it, everything produced before this change
    would simply vanish from the history list, which reads as data loss.
    """
    import shutil

    if not OUTPUT.exists():
        return 0
    suffixes = ("_transcript_tagged.md", "_transcript.md", "_summary.md", "_minutes.md")
    names = set()
    for path in OUTPUT.glob("*.md"):
        if not path.is_file():
            continue
        for suffix in suffixes:
            if path.name.endswith(suffix):
                # A document with no transcript beside it still belongs in a
                # folder of its own; it is nobody's job to tidy the loose ones.
                names.add(path.name[: -len(suffix)])
                break

    moved = 0
    for name in sorted(names):
        if not name:
            continue
        dest_dir = meeting_dir(name)
        dest_dir.mkdir(parents=True, exist_ok=True)
        for suffix in suffixes:
            src = OUTPUT / (name + suffix)
            if src.is_file() and not (dest_dir / src.name).exists():
                shutil.move(str(src), str(dest_dir / src.name))
                moved += 1
        samples = OUTPUT / ("%s_speaker_samples" % name)
        target = dest_dir / "speaker_samples"
        if samples.is_dir() and not target.exists():
            shutil.move(str(samples), str(target))
            moved += 1
    return moved


# Files that must exist before the app can do anything useful. Checked at
# startup so the failure is a plain sentence instead of a crash mid-job.
REQUIRED_FILES = [
    FFMPEG,
    MODELS / "ggml-large-v3-turbo.bin",
    MODELS / "ggml-silero-v5.1.2.bin",
    MODELS / "segmentation-3.0.onnx",
    MODELS / "speaker-embedding.onnx",
]

_DEFAULTS = {
    "llm": {
        "model": "auto",
        "ctx_size": 32768,
        # "auto" = detected per machine (hardware.placement): the FFN split on
        # a dedicated GPU, the processor on unified memory. An explicit number
        # or regex overrides; "" means omit the flag and let llama.cpp fit it.
        "gpu_layers": "auto",
        "cpu_ffn_regex": "auto",
        "cache_type_k": "q8_0",
        "cache_type_v": "q8_0",
        "threads": 10,
        "port": 8080,
        "startup_timeout_s": 180,
        "idle_timeout_s": 300,
    },
    "chunking": {
        "target_tokens": 10000,
        "overlap_tokens": 400,
        "max_map_output_tokens": 1500,
        "group_reduce_threshold": 8,
        "group_size": 5,
        "max_group_output_tokens": 2500,
    },
    "thinking": {
        "map": False,
        "group_reduce": False,
        "reduce": True,
        "reduce_effort": "medium",
    },
    "pipeline": {"concurrent_diarization": True},
    # "auto" detects; "cuda" / "vulkan" / "cpu" pin it.
    "gpu": {"backend": "auto"},
    "whisper": {
        "model": "models/ggml-large-v3-turbo.bin",
        "vad_model": "models/ggml-silero-v5.1.2.bin",
        "language": "en",
        "threads": 5,
        "max_context": 0,
        "dtw": True,
    },
    "estimate": {
        "transcript_tokens_per_audio_minute": 206,
        "fixed_overhead_s": 60,
        "buffer_fraction": 0,
    },
    "diarization": {
        "enabled": True,
        "threads": 5,
        "num_speakers": 0,
        "cluster_threshold": 0.70,
        "min_duration_on": 0.3,
        "min_duration_off": 0.5,
        "max_speakers": 20,
        "min_cluster_speech_s": 30,
        "min_cluster_speech_fraction": 0.005,
        "reassign_max_distance": 0.85,
        "merge_centroid_distance": 0.35,
        "min_embed_duration": 1.0,
        "min_coverage_fraction": 0.80,
        "embedding_cache": True,
    },
    "server": {"port": 8000},
}


def _merge(base: dict, over: dict) -> dict:
    out = dict(base)
    for k, v in over.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _merge(out[k], v)
        else:
            out[k] = v
    return out


def load_config() -> dict:
    """Read config.json, falling back to defaults for anything absent.

    A malformed config.json must not take the app down silently, but it also
    must not be papered over -- the operator edits this file by hand.
    """
    cfg = _DEFAULTS
    if CONFIG_PATH.exists():
        try:
            with open(CONFIG_PATH, "r", encoding="utf-8") as fh:
                cfg = _merge(_DEFAULTS, json.load(fh))
        except (json.JSONDecodeError, OSError) as exc:
            raise RuntimeError(
                "config.json could not be read (%s). Fix or delete it." % exc
            ) from exc
    return cfg


def resolve(rel: str) -> Path:
    """Resolve a config path (always written relative to the app folder)."""
    p = Path(rel)
    return p if p.is_absolute() else (ROOT / p)


def child_env() -> dict:
    """Environment for spawned binaries.

    bin\\ must be on PATH so ggml-cuda.dll can find cublas64_12.dll and
    cudart64_12.dll, which are shared between the whisper and llama builds.
    Without this both engines silently fall back to CPU -- measured at roughly
    10x slower for whisper (BUILD_NOTES.md).
    """
    env = dict(os.environ)
    env["PATH"] = str(BIN) + os.pathsep + env.get("PATH", "")

    # AMD on Vulkan crashes hard in ggml's KHR_coopmat path -- see
    # hardware.needs_coopmat_workaround for the measurement. ggml tests this
    # variable for existence, not value, so "1" and "0" both disable it.
    #
    # Guarded on `probed()`: detection spawns children of its own, and those
    # children ask for this environment. Consulting detection here before it
    # has run would recurse forever. Enumeration itself is safe with matrix
    # cores enabled -- it is only inference that dies.
    from . import hardware

    if hardware.probed() and hardware.needs_coopmat_workaround():
        env["GGML_VK_DISABLE_COOPMAT"] = "1"
    return env


def missing_files() -> list[Path]:
    """Everything that must be present before a job can succeed.

    The LLM weights are resolved from config.json rather than hard-coded, and
    are checked here rather than at the LLM stage: without this, a missing
    16.5GB download only surfaces an hour into a job, after transcription and
    diarization have already run.
    """
    missing = [p for p in REQUIRED_FILES if not p.exists()]
    # The engines are resolved per backend, so check the pair this machine
    # will actually launch rather than a hard-coded folder.
    missing += [p for p in (whisper_cli_path(backend("whisper")),
                            llama_server_path(backend("llama")))
                if not p.exists()]
    try:
        from . import hardware

        requested = load_config()["llm"]["model"]
        if requested in ("auto", "", None):
            # Both are shipped, and the UI lets the user pick either, so both
            # must be present -- not just whichever this card would default to.
            wanted = [hardware.model_path(m["key"]) for m in hardware.MODELS]
        else:
            wanted = [resolve(requested)]
        missing += [p for p in wanted if not p.exists()]
    except Exception:  # noqa: BLE001 - a broken config is reported elsewhere
        pass
    return missing
