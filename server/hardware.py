"""GPU detection and LLM model selection.

Which quantisation to run is a property of the machine, not a preference, so
it is detected rather than configured. Q4_K_M is better -- it was the only
variant to get the election-rate table right (BUILD_NOTES §9h) -- but its
16.5 GB of weights plus a 32k KV cache needs a card that can hold most of it.
Below that, IQ3_XXS at 10.9 GB scores the same on term recall and runs roughly
twice as fast when the larger model would be thrashing over PCIe.

The user can still override the choice in the UI. The detection picks the
default; it does not overrule anybody.
"""

from __future__ import annotations

import re
import subprocess

from . import config

CREATE_NO_WINDOW = getattr(subprocess, "CREATE_NO_WINDOW", 0)

# 15 GB, not 16: cards advertised as 16 GB report anywhere from 15.8 GB down
# once the driver has taken its share, and a threshold that a 16 GB card fails
# would be worse than useless.
Q4_MIN_VRAM_MB = 15000

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
OVERHEAD_GB = 1.5

# Qwen3.8-27B has 64 transformer blocks, 0-63.
NUM_LAYERS = 64
# Roughly the share of a block's weights that the FFN tensors account for, and
# therefore what moving one block to the CPU actually frees on the GPU.
FFN_FRACTION = 0.67

MODELS = [
    {
        "key": "q4_k_m",
        "file": "Qwen3.8-27B-UD-Q4_K_M.gguf",
        "label": "High Quality: Qwen3.8-27B-UD-Q4_K_M",
        "size_gb": 16.5,
        "min_vram_mb": Q4_MIN_VRAM_MB,
    },
    {
        "key": "iq3_xxs",
        "file": "Qwen3.8-27B-UD-IQ3_XXS.gguf",
        "label": "Low Quality: Qwen3.8-27B-UD-IQ3_XXS",
        "size_gb": 10.9,
        "min_vram_mb": 0,
    },
]

BY_KEY = {m["key"]: m for m in MODELS}


def detect_vram_mb() -> int:
    """Total VRAM of the first GPU in MiB, or 0 if it cannot be determined."""
    try:
        proc = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.total", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=15,
            env=config.child_env(), creationflags=CREATE_NO_WINDOW,
        )
    except (OSError, subprocess.SubprocessError):
        return 0
    for line in (proc.stdout or "").splitlines():
        m = re.search(r"\d+", line)
        if m:
            return int(m.group())
    return 0


def choose_key(vram_mb: int) -> str:
    """The model this machine should use by default."""
    for model in MODELS:
        if vram_mb >= model["min_vram_mb"]:
            return model["key"]
    return MODELS[-1]["key"]


def resolve_key(requested: str, vram_mb: int | None = None) -> str:
    """Honour an explicit choice; fall back to detection for 'auto' or junk."""
    if requested in BY_KEY:
        return requested
    if vram_mb is None:
        vram_mb = detect_vram_mb()
    return choose_key(vram_mb)


def model_path(key: str):
    return config.MODELS / BY_KEY[key]["file"]


def offload_regex(key: str, vram_mb: int) -> str:
    """How much of the FFN stack has to live in system RAM on this card.

    Section 11.1 hard-codes a regex for layers 40-63, which was written for a
    16 GB card and is simply wrong on any other. Computing it from the measured
    VRAM and the actual file size is the same idea, applied honestly: keep
    everything on the GPU that fits, and push down only the excess.
    """
    model = BY_KEY[key]
    if vram_mb <= 0:
        # No GPU information. Assume the worst rather than failing to allocate.
        return _regex_for(40)

    available_gb = vram_mb / 1024.0
    needed_gb = model["size_gb"] + OVERHEAD_GB
    deficit_gb = needed_gb - available_gb
    if deficit_gb <= 0:
        return ""

    per_layer_gb = model["size_gb"] * FFN_FRACTION / NUM_LAYERS
    layers = min(NUM_LAYERS, int(deficit_gb / per_layer_gb) + 1)
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
    return {
        "vram_mb": vram_mb,
        "recommended": recommended,
        "models": [
            {
                "key": m["key"],
                "label": m["label"],
                "available": model_path(m["key"]).exists(),
                "size_gb": m["size_gb"],
            }
            for m in MODELS
        ],
    }
