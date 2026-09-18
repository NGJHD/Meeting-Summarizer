"""GPU detection and LLM model selection.

Which quantisation to run is a property of the machine, not a preference, so
it is detected rather than configured. UD-IQ4_XS is the High Quality choice:
13.27 GiB of weights, which is small enough to sit entirely on a 16 GB card
alongside the KV cache and its own multi-token-prediction draft context. Five
runs each against Q4_K_M, Q4_K_S and a second IQ4_XS build found no quality
difference that reproduces, and a 3.4-3.6x speed advantage that does
(BUILD_NOTES §9z). Below the threshold, IQ3_XXS at 10.9 GB runs where the
larger file would be thrashing over PCIe.

The user can still override the choice in the UI. The detection picks the
default; it does not overrule anybody.
"""

from __future__ import annotations

import re
import subprocess
import threading

from . import config

CREATE_NO_WINDOW = getattr(subprocess, "CREATE_NO_WINDOW", 0)

# 15 GB, not 16: cards advertised as 16 GB report anywhere from 15.8 GB down
# once the driver has taken its share, and a threshold that a 16 GB card fails
# would be worse than useless.
#
# Deliberately unchanged when High Quality moved from Q4_K_M (15.33 GiB) to
# UD-IQ4_XS (13.27 GiB). The smaller file would clear a lower bar, but lowering
# it would hand High Quality to cards that have never run it here, and the
# point of the switch was to make the *same* machines faster rather than to
# widen who gets the large model. Lower it only with measurements from a
# 12-14 GB card in hand.
HIGH_MIN_VRAM_MB = 15000

# Headroom for the KV cache at ctx_size 32768 with q8_0 keys and values, plus
# llama.cpp's compute buffers.
#
# Deliberately *not* generous. llama.cpp loads weights through mmap, so a model
# that does not quite fit is paged rather than refused -- `/health` returns 200
# with only 907 MiB resident, and VRAM fills as inference runs. Allocation
# therefore does not hard-fail, which makes over-offloading the worse mistake
# of the two: it guarantees CPU execution for layers that would have fitted,
# while under-offloading merely lets the pager sort it out.
#
# Measured: IQ3_XXS (10.9 GB) ran on this 10 GB card with only 8 layers
# offloaded, peaking at 9.8 GB. This is an estimate and a starting point --
# set `llm.cpu_ffn_regex` explicitly in config.json to override it.
#
# Confirmed in GiB on the 16 GB card: IQ3_XXS is 10.18 GiB of weights and sat
# at 11.69 GiB resident with everything on the GPU, so 1.5 GiB it is.
OVERHEAD_GIB = 1.5

# Qwen3.8-27B has 64 transformer blocks, 0-63.
NUM_LAYERS = 64
# Roughly the share of a block's weights that the FFN tensors account for, and
# therefore what moving one block to the CPU actually frees on the GPU.
FFN_FRACTION = 0.67

MODELS = [
    {
        "key": "iq4_xs",
        "file": "Qwen3.8-27B-UD-IQ4_XS.gguf",
        "label": "High Quality: Qwen3.8-27B-UD-IQ4_XS",
        "size_gb": 14.3,
        "min_vram_mb": HIGH_MIN_VRAM_MB,
        # Where it comes from, pinned to the same immutable revision
        # DOWNLOAD_MODELS.bat uses. Here so the app can offer the download
        # itself: an install updating from 1.0.x keeps its Q4_K_M and would
        # otherwise never receive this one without being told to re-run a
        # batch file nobody knows about (BUILD_NOTES 9an).
        "url": "https://huggingface.co/unsloth/Qwen3.8-27B-GGUF/resolve/"
               "4ca720788d1e01f1bff70c033e0d0028fd02e502/"
               "Qwen3.8-27B-UD-IQ4_XS.gguf?download=true",
        "min_size": 14_000_000_000,
        # Multi-token prediction, from the layer inside the file itself
        # (blk.64.nextn.*). Applied only when the model fits without offload --
        # see llm.LlamaServer.spec_type.
        "spec": "draft-mtp",
    },
    {
        "key": "iq3_xxs",
        "file": "Qwen3.8-27B-UD-IQ3_XXS.gguf",
        "label": "Low Quality: Qwen3.8-27B-UD-IQ3_XXS",
        "size_gb": 10.9,
        "min_vram_mb": 0,
        "url": "https://huggingface.co/unsloth/Qwen3.8-27B-GGUF/resolve/"
               "4ca720788d1e01f1bff70c033e0d0028fd02e502/"
               "Qwen3.8-27B-UD-IQ3_XXS.gguf?download=true",
        "min_size": 10_000_000_000,
        "spec": "draft-mtp",
    },
]

# Models an older install may still have on disk. The update payload carries
# no `models\` (BUILD_NOTES 9q), so a copy updating from 1.0.x keeps the
# Q4_K_M it downloaded and never receives UD-IQ4_XS until DOWNLOAD_MODELS.bat
# is run again. Requiring a file the updater cannot deliver would stop those
# installs from starting at all, so Q4_K_M stays selectable while it is
# present -- it is a perfectly good model, just slower (BUILD_NOTES 9z).
LEGACY_MODELS = [
    {
        "key": "q4_k_m",
        "file": "Qwen3.8-27B-UD-Q4_K_M.gguf",
        "label": "High Quality (older): Qwen3.8-27B-UD-Q4_K_M",
        "size_gb": 16.5,
        "min_vram_mb": HIGH_MIN_VRAM_MB,
        "spec": "draft-mtp",
    },
]

ALL_MODELS = MODELS + LEGACY_MODELS
BY_KEY = {m["key"]: m for m in ALL_MODELS}


def _by_preference(models: list) -> list:
    """Most capable first, which is the order the dropdown and the default use.

    `min_vram_mb` descending: it is what the original two-entry list encoded by
    hand, and appending a legacy entry broke it -- an install holding both
    Q4_K_M and IQ3_XXS was offered Low Quality on a 16 GB card because the
    legacy entry sorted last. Stable, so a shipped model still precedes a
    legacy one of the same tier.
    """
    return sorted(models, key=lambda m: -m["min_vram_mb"])


def available_models() -> list:
    """Shipped models, plus any legacy one actually on disk."""
    return _by_preference([m for m in ALL_MODELS
                           if m in MODELS or model_path(m["key"]).exists()])


def any_model_present() -> bool:
    """Is there at least one language model to run?"""
    return any(model_path(m["key"]).exists() for m in ALL_MODELS)


# Which inference backend to run. CUDA is fastest where it exists; Vulkan is
# the cross-vendor answer -- one binary for AMD, Intel and NVIDIA, integrated
# and discrete, needing nothing installed beyond a current display driver. CPU
# is the last resort and is only usable for very short recordings.
BACKENDS = ("cuda", "vulkan", "cpu")

_probe_cache: dict = {}
_probe_lock = threading.Lock()


def _run(cmd: list, timeout: int = 20) -> str:
    try:
        proc = subprocess.run(
            cmd, capture_output=True, text=True, timeout=timeout,
            env=config.child_env(), creationflags=CREATE_NO_WINDOW,
        )
    except (OSError, subprocess.SubprocessError):
        return ""
    return (proc.stdout or "") + (proc.stderr or "")


def _nvidia_vram_mb() -> int:
    for line in _run(
        ["nvidia-smi", "--query-gpu=memory.total",
         "--format=csv,noheader,nounits"], timeout=15).splitlines():
        m = re.search(r"\d+", line)
        if m:
            return int(m.group())
    return 0


def _vulkan_devices() -> list:
    """Ask the Vulkan build what it can see. Returns [(name, total_mib), ...].

    This is the reliable cross-vendor VRAM read. `Win32_VideoController`'s
    AdapterRAM is a 32-bit field that caps at 4 GB, so it reports 4095 MB for a
    16 GB card; llama-server prints the real figure, and it is the same code
    that will do the allocating.

        Vulkan0: NVIDIA GeForce RTX 3080 (10051 MiB, 9283 MiB free)
    """
    exe = config.llama_server_path("vulkan")
    if not exe.exists():
        return []
    out = _run([str(exe), "--list-devices"], timeout=60)
    found = []
    for line in out.splitlines():
        m = re.search(r"^\s*(\w+\d*):\s*(.+?)\s*\((\d+)\s*MiB", line)
        if m:
            found.append({"id": m.group(1), "name": m.group(2).strip(),
                          "vram_mb": int(m.group(3))})
    return found


def _vulkan_banner() -> dict:
    """`uma` and the matrix-core extension, from ggml's own device banner.

        ggml_vulkan: 0 = AMD Radeon 780M Graphics (AMD proprietary driver)
                       | uma: 1 | fp16: 1 | ... | matrix cores: KHR_coopmat

    `--list-devices` reports memory but not these. llama-bench prints the full
    banner and then exits immediately when the model path does not exist, so
    this costs well under a second and loads nothing.
    """
    exe = config.engine_dir("llama", "vulkan") / "llama-bench.exe"
    if not exe.exists():
        return {}
    out = _run([str(exe), "-m", "__probe_no_such_model__.gguf",
                "-p", "1", "-n", "1", "-r", "1"], timeout=60)
    banner = {}
    for line in out.splitlines():
        m = re.search(r"ggml_vulkan:\s*(\d+)\s*=\s*(.+?)\s*\|\s*uma:\s*(\d)", line)
        if not m:
            continue
        cores = re.search(r"matrix cores:\s*(\S+)", line)
        banner[int(m.group(1))] = {
            "name": m.group(2).strip(),
            "uma": m.group(3) == "1",
            "matrix_cores": cores.group(1) if cores else "",
        }
    return banner


def detect_gpu() -> dict:
    """What is in this machine and which backend to use.

    Cached: it shells out to nvidia-smi and llama.cpp, and it is asked for on
    every page load.
    """
    if _probe_cache:
        return dict(_probe_cache)

    # Serialised. Two threads arriving together would otherwise each spawn the
    # whole probe -- and on a machine with no nvidia-smi that is two subprocess
    # calls with 60-second timeouts apiece.
    with _probe_lock:
        if _probe_cache:
            return dict(_probe_cache)
        return _probe()


def _probe() -> dict:
    vram = _nvidia_vram_mb()
    if vram > 0:
        info = {"vendor": "nvidia", "backend": "cuda", "vram_mb": vram,
                "device": "NVIDIA GPU", "uma": False, "matrix_cores": "",
                "device_id": "", "device_index": 0}
    else:
        devices = _vulkan_devices()
        if devices:
            banner = _vulkan_banner()
            for i, dev in enumerate(devices):
                dev.update(banner.get(i, {"uma": False, "matrix_cores": ""}))
            chosen = _pick_device(devices)
            lower = chosen["name"].lower()
            vendor = ("amd" if any(k in lower for k in ("amd", "radeon", "gfx"))
                      else "intel" if "intel" in lower
                      else "nvidia" if "nvidia" in lower
                      else "other")
            info = {"vendor": vendor, "backend": "vulkan",
                    "vram_mb": chosen["vram_mb"], "device": chosen["name"],
                    "uma": bool(chosen.get("uma")),
                    "matrix_cores": chosen.get("matrix_cores", ""),
                    "device_id": chosen["id"] if len(devices) > 1 else "",
                    "device_index": devices.index(chosen)}
        else:
            info = {"vendor": "none", "backend": "cpu", "vram_mb": 0,
                    "device": "no GPU detected", "uma": False,
                    "matrix_cores": "", "device_id": "", "device_index": 0}

    _probe_cache.update(info)
    return dict(info)


def _pick_device(devices: list) -> dict:
    """Choose between several Vulkan devices.

    A discrete GPU beats an integrated one, always -- never mind what each
    claims to have. A hybrid laptop whose NVIDIA driver is missing or broken
    falls through to the Vulkan path with both adapters visible, and there the
    integrated one advertises a share of system RAM: 31.7 GB on a 48 GB
    machine, which would beat a 16 GB discrete card on size and lose to it on
    every measurement that matters.

    Among devices of the same kind, the largest wins.
    """
    discrete = [d for d in devices if not d.get("uma")]
    return max(discrete or devices, key=lambda d: d["vram_mb"])


def probed() -> bool:
    """True once detect_gpu has run. Lets child_env avoid triggering it.

    The probe spawns children itself, and those children ask for the child
    environment -- consulting detection from inside child_env() without this
    guard is an infinite recursion.
    """
    return bool(_probe_cache)


def needs_coopmat_workaround() -> bool:
    """AMD on Vulkan hard-crashes in the KHR_coopmat path.

    Measured on a Radeon 780M with the AMD proprietary Windows driver:
    whisper-cli dies at the first encoder call with exit 0xC0000409 -- a
    __fastfail, no message, nothing in the log. It is an uncaught vulkan-hpp
    exception, so there is no error to catch and no way to degrade gracefully.
    With GGML_VK_DISABLE_COOPMAT set, the identical command completes in 17s.

    Scoped to AMD deliberately. NVIDIA takes the separate NV_coopmat2 path
    which this flag does not touch, and Intel's Vulkan path works as shipped --
    turning matrix cores off there would cost performance to fix nothing.
    """
    if not _probe_cache:
        return False
    return (_probe_cache.get("vendor") == "amd"
            and _probe_cache.get("backend") == "vulkan")


def _first_available(preferred: str, engine: str) -> str:
    """Fall back down the chain rather than launching something absent.

    Resolved **per engine**, because the two are not shipped in step. If a
    Vulkan whisper build is ever missing from a folder, a non-NVIDIA machine
    should still run the LLM on Vulkan and let transcription fall back to the
    CPU build rather than failing outright.
    """
    resolver = (config.llama_server_path if engine == "llama"
                else config.whisper_cli_path)
    order = [preferred] + [b for b in BACKENDS if b != preferred]
    # All backends ship, so "the CUDA binaries are present" says nothing about
    # whether this machine has an NVIDIA card. Falling back to them on an AMD
    # box would load ggml-cuda.dll, find no device and quietly run on CPU
    # anyway -- slower to start and far harder to diagnose than just choosing
    # the CPU build outright.
    if _probe_cache.get("vendor", "nvidia") != "nvidia":
        order = [b for b in order if b != "cuda"]
    for backend in order:
        if resolver(backend).exists():
            return backend
    return "cpu"


def resolve_backend(engine: str, requested: str = "") -> str:
    """Honour an explicit `gpu.backend` in config.json; otherwise detect.

    On unified memory the **language model** takes the CPU build outright, not
    the Vulkan build with the layers switched off. Those are not the same
    thing: with the Vulkan backend registered the scheduler still routes
    prefill to the integrated GPU, and on an Intel Arc Xe-LPG -- which has no
    matrix units at all -- that is catastrophic. Measured on the same model,
    converted into a real 4-hour job:

                              prefill   generation   4h job
        Intel  Vulkan -ngl 0   1.61       1.36        14.8 h
        Intel  CPU build       9.45       1.60         4.5 h   <- 3.3x
        AMD    Vulkan -ngl 0  33.80       3.15         1.8 h
        AMD    CPU build      26.72       3.73         1.8 h   <- a wash

    Never worse, and on Intel the difference between a working evening and an
    overnight job. **Transcription is unaffected** and stays on Vulkan: whisper
    is a much smaller model and the integrated GPU handles it well.
    """
    detected = detect_gpu()["backend"]        # also primes the vendor cache
    if requested in BACKENDS:
        preferred = requested                 # the operator has pinned it
    else:
        preferred = detected
        if engine == "llama" and detect_gpu().get("uma"):
            preferred = "cpu"
    return _first_available(preferred, engine)


def detect_vram_mb() -> int:
    """Total VRAM of the GPU this machine will use, in MiB. 0 if unknown."""
    return detect_gpu()["vram_mb"]


def placement(key: str, gpu: dict) -> tuple[str, str]:
    """(--n-gpu-layers, --override-tensor) for this model on this machine.

    Two regimes, both measured rather than reasoned about.

    **Dedicated GPU: keep the FFN split.** Section 11.1's approach -- everything
    nominally on the GPU, then push the upper blocks' FFN tensors into system
    RAM -- beats letting llama.cpp fit whole layers, because it keeps attention
    (the bandwidth-sensitive half) resident. Measured on a 10 GB card with a
    10.9 GB model, same VRAM occupied either way:

        llama.cpp auto-fit          2.63 tok/s
        -ngl 99 + FFN override      6.23 tok/s

    2.4x, so the hand-computed regex stays. It is only ever wrong about *how
    much* to offload, and the VRAM figure it works from is trustworthy here.

    **Unified memory: do not offload at all.** The VRAM figure is a fiction --
    a Radeon 780M advertises 18 GB of a 32 GB machine and refuses to allocate
    past about 4 -- so both auto-fit and our own arithmetic size against a
    number that does not exist, and llama-server dies with
    `vk::Queue::submit: ErrorOutOfDeviceMemory`. Even where it fits there is
    nothing to win, because an integrated GPU shares the CPU's memory bus:

        -ngl 0     prompt 33.8 tok/s   generation 3.15 tok/s
        -ngl 15    prompt 21.6 tok/s   generation 3.05 tok/s

    Generation unchanged, prompt processing a third slower. So: the processor.

    Confirmed on Intel too, where it is even more pronounced -- more layers on
    the GPU buy 12% on the prompt and cost 2.8x on generation:

        -ngl 0     prompt 1.61 tok/s   generation 1.36 tok/s
        -ngl 15    prompt 1.67 tok/s   generation 0.87 tok/s
        -ngl 99    prompt 1.80 tok/s   generation 0.48 tok/s

    Offloading would only pay if a call's prompt were more than 21x its output.
    Ours are 6.7x (map) and 1.8x (reduce), so the processor wins every call.

    Two vendors, opposite hardware, same answer: on unified memory the GPU has
    no bandwidth advantage to offer and the split costs synchronisation.

    Note what `-ngl 0` actually does here, because the name misleads. The
    weights live in system RAM, but the Vulkan backend stays registered and the
    scheduler still sends prefill -- the big compute-bound matmuls -- to the
    GPU. Measured on a dedicated card, same model, prefill only:

        Vulkan build, -ngl 0    40.4 tok/s
        CPU-only build          16.8 tok/s

    So this is a hybrid, not a retreat to the processor: generation on the CPU
    where the memory bus decides it, prefill on the GPU where compute does.
    That is why it wins on both counts.
    """
    if gpu.get("uma"):
        return "0", ""
    return "99", offload_regex(key, gpu["vram_mb"])


def choose_key(vram_mb: int) -> str:
    """The model this machine should use by default.

    A unified-memory GPU never gets the large one, whatever it advertises. The
    figure is a share of system RAM rather than a budget: a Radeon 780M reports
    18 GB on a 32 GB machine, which clears the 15 GB threshold and would select
    the 16.5 GB model -- on a device that will not allocate 4. It also runs on
    the CPU there (see gpu_layers_for), where the smaller model is roughly
    twice as fast and leaves the machine usable.
    """
    # Only files that are actually here. An install updated from 1.0.x has
    # Q4_K_M and no UD-IQ4_XS, and defaulting to a model it cannot load would
    # fail at the LLM stage, an hour in.
    here = _by_preference([m for m in ALL_MODELS if model_path(m["key"]).exists()])
    if not here:
        return MODELS[-1]["key"]          # nothing on disk; report the default
    if detect_gpu().get("uma"):
        return here[-1]["key"]
    # Unlike placement(), this applies to every unified-memory device, not just
    # AMD: whatever a shared-memory GPU advertises, a large model in RAM the
    # operating system is also using is the wrong choice on any of them.
    for model in here:
        if vram_mb >= model["min_vram_mb"]:
            return model["key"]
    return here[-1]["key"]


def resolve_key(requested: str, vram_mb: int | None = None) -> str:
    """Honour an explicit choice; fall back to detection for 'auto' or junk."""
    if requested in BY_KEY:
        return requested
    if vram_mb is None:
        vram_mb = detect_vram_mb()
    return choose_key(vram_mb)


def model_path(key: str):
    return config.MODELS / BY_KEY[key]["file"]


def model_gib(key: str) -> float:
    """The weights' real size in GiB, measured from the file where possible.

    The table's `size_gb` figures are decimal gigabytes -- bytes / 1e9, which
    is how model files are advertised -- while `vram_mb / 1024` is GiB. Mixing
    them overstated every model by 7% and offloaded more than it had to: on a
    16311 MiB card Q4_K_M was pushed to twelve FFN blocks where six is enough,
    and the whole measured cost of the High Quality default in BUILD_NOTES 9v
    is the offload. Stat the file so the number is neither unit-confused nor
    stale when a file is replaced.
    """
    path = model_path(key) if key in BY_KEY else None
    if path is not None and path.exists():
        return path.stat().st_size / float(2 ** 30)
    return BY_KEY[key]["size_gb"] * 1e9 / float(2 ** 30)


def offload_regex(key: str, vram_mb: int) -> str:
    """How much of the FFN stack has to live in system RAM on this card.

    Section 11.1 hard-codes a regex for layers 40-63, which was written for a
    16 GB card and is simply wrong on any other. Computing it from the measured
    VRAM and the actual file size is the same idea, applied honestly: keep
    everything on the GPU that fits, and push down only the excess.
    """
    if vram_mb <= 0:
        # No GPU information. Assume the worst rather than failing to allocate.
        return _regex_for(40)

    # Both sides in GiB. See model_gib: they used not to be.
    available = vram_mb / 1024.0
    size = model_gib(key)
    deficit = (size + OVERHEAD_GIB) - available
    if deficit <= 0:
        return ""

    per_layer = size * FFN_FRACTION / NUM_LAYERS
    layers = min(NUM_LAYERS, int(deficit / per_layer) + 1)
    return _regex_for(NUM_LAYERS - layers)


def _regex_for(first_layer: int) -> str:
    """Offload the FFN tensors of `first_layer`..63.

    Written as an explicit alternation rather than a character-class range.
    Ranges like `[4-6][0-9]` silently include layers that do not exist and
    exclude ones that do, and the failure is a quiet performance loss.
    """
    first = max(0, min(first_layer, NUM_LAYERS - 1))
    numbers = "|".join(str(n) for n in range(first, NUM_LAYERS))
    return r"blk\.(%s)\.ffn_.*=CPU" % numbers


def describe(vram_mb: int) -> dict:
    """Everything the UI needs to show and explain the model choice."""
    recommended = choose_key(vram_mb)
    gpu = detect_gpu()
    external, port = config.external_llm()
    return {
        "vram_mb": vram_mb,
        # What this card suggests, which is what the note under the dropdown
        # explains. Kept separate from `selected`: the two differ exactly when
        # the user has chosen a port, and the note should still say what the
        # detection found rather than fall silent.
        "recommended": recommended,
        "selected": "external" if external else recommended,
        "external": {"enabled": external, "port": port},
        "vendor": gpu["vendor"],
        "device": gpu["device"],
        # Reported per engine: they are not shipped in step, so a machine can
        # legitimately run the language model on Vulkan and transcribe on CPU.
        "llm_backend": config.backend("llama"),
        "whisper_backend": config.backend("whisper"),
        "models": [
            {
                "key": m["key"],
                "label": m["label"],
                "available": model_path(m["key"]).exists(),
                "size_gb": m["size_gb"],
            }
            for m in available_models()
        ],
    }
