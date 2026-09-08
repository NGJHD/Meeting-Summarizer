# Third-party notices

The code in this repository is MIT (see `LICENSE`). Everything below is somebody else's
work, kept under its own licence; an assembled folder is a collection of separately
licensed works rather than a single one.

**Some of it we now redistribute directly.** The `-full` release zip contains the Python
runtime, ffmpeg, every inference binary and three of the models. That is a distribution
in the licensing sense, so the obligations attached to those components — LGPL v3 for
ffmpeg, CC-BY-4.0 for the speaker-embedding model — are ours to meet, and this file is
how they are met. It must travel with the zip.

The rest is fetched by `DOWNLOAD_MODELS.bat` from its publisher and never passes through
us: the two language models and the speech recognition model, which are too large to be
release assets.

Versions are pinned in `DOWNLOAD_MODELS.bat`, so this list stays true until somebody
deliberately changes a pin. The two speaker models are the exception — upstream publishes
them on a floating release tag rather than an immutable revision — which is part of why
they are now shipped in the zip instead.

---

## The one that constrains redistribution: ffmpeg

| | |
|---|---|
| **Component** | FFmpeg `n9.0.1-27-g9b0578816c`, win64 **LGPL** build by [BtbN](https://github.com/BtbN/FFmpeg-Builds) |
| **Licence** | **LGPL v3** (`--enable-version3`, *not* `--enable-gpl`) |
| **Source** | https://github.com/FFmpeg/FFmpeg — build recipe at https://github.com/BtbN/FFmpeg-Builds |

The LGPL build is chosen **deliberately over the GPL one**. Both decode everything this
app accepts and both keep `libmp3lame` for the speaker clips, so there is nothing to gain
from the GPL variant and a great deal of licensing friction to avoid: a GPLv3 ffmpeg in
the folder would place GPLv3 obligations on anyone passing that folder on.

Under LGPL v3 you may redistribute the folder provided you keep this notice, do not
restrict the recipient's rights over ffmpeg itself, and can point them at the source
above. The app invokes `ffmpeg.exe` as a separate process and links nothing, so the MIT
licence on this project's own code is unaffected either way.

If you rebuild or replace `bin/ffmpeg.exe`, check `ffmpeg -version` for `--enable-gpl`
before shipping it.

---

## Inference engines

| Component | Version | Licence |
|---|---|---|
| [whisper.cpp](https://github.com/ggml-org/whisper.cpp) — `bin/whisper-*` | `b4938` | MIT |
| [llama.cpp](https://github.com/ggml-org/llama.cpp) — `bin/llama-*` | `b10852` | MIT |
| ggml (bundled with both) | — | MIT |
| [sherpa-onnx](https://github.com/k2-fsa/sherpa-onnx) — diarization | 1.13.7 | Apache-2.0 |
| [ONNX Runtime](https://github.com/microsoft/onnxruntime) | 1.29.0 | MIT |

`bin/whisper-vulkan/` is built from whisper.cpp `b4938` by this project and published on
its releases; the recipe is in `BUILD_NOTES.md` §9o. It carries `MSVCP140.dll` and
`VCRUNTIME140*.dll` from the Microsoft Visual C++ redistributable, redistributable under
the Visual Studio licence terms.

`bin/cudart64_12.dll`, `cublas64_12.dll` and `cublasLt64_12.dll` are the NVIDIA CUDA
runtime, redistributable under the [CUDA Toolkit EULA](https://docs.nvidia.com/cuda/eula/).
They are only used by the CUDA builds.

---

## Models

Marked **(shipped)** where the file is inside the `-full` zip rather than downloaded.

| Model | Used for | Licence |
|---|---|---|
| [Whisper large-v3-turbo](https://huggingface.co/ggerganov/whisper.cpp) (ggml) | Transcription | MIT |
| [Silero VAD v5.1.2](https://huggingface.co/ggml-org/whisper-vad) (ggml) **(shipped)** | Speech detection | MIT |
| [pyannote segmentation-3.0](https://huggingface.co/pyannote/segmentation-3.0) (ONNX re-export) **(shipped)** | Speaker segmentation | MIT |
| [NVIDIA TitaNet-Large](https://huggingface.co/nvidia/speakerverification_en_titanet_large) (ONNX re-export) **(shipped)** | Speaker embedding | **CC-BY-4.0** |
| [Qwen3.8-27B](https://huggingface.co/unsloth/Qwen3.8-27B-GGUF) — Q4_K_M and IQ3_XXS | Summaries and minutes | Apache-2.0 |

The ONNX re-exports of the two speaker models are distributed by the sherpa-onnx project.

**CC-BY-4.0 is the one with a positive obligation:** TitaNet-Large requires attribution to
NVIDIA. This file satisfies it, so keep it with the folder.

---

## Python runtime

`runtime/` is CPython **3.12.14** from
[python-build-standalone](https://github.com/astral-sh/python-build-standalone), under the
**PSF Licence**. Its own `runtime/LICENSE.txt` travels with it and covers CPython and the
libraries CPython itself bundles.

Vendored packages, as each declares itself:

| Package | Licence |
|---|---|
| fastapi, pydantic, pydantic_core, anyio, h11, annotated_types, annotated_doc, typing_inspection, pip | MIT |
| onnxruntime | MIT |
| starlette, uvicorn, click, idna | BSD-3-Clause |
| numpy | BSD-3-Clause AND 0BSD AND MIT AND Zlib AND CC0-1.0 |
| packaging | Apache-2.0 OR BSD-2-Clause |
| python_multipart, flatbuffers | Apache-2.0 |
| sherpa_onnx, sherpa_onnx_core | Apache-2.0 |
| protobuf | BSD-3-Clause |
| typing_extensions | PSF-2.0 |

There is deliberately no PyTorch, no transformers and no HuggingFace client — see
`CLAUDE.md` §0.

---

## If you redistribute the assembled folder

1. Keep this file and `LICENSE` in it.
2. Keep `runtime/LICENSE.txt`.
3. Be able to point recipients at the FFmpeg source (LGPL v3) — the link above is enough.
4. Attribution to NVIDIA for TitaNet-Large is required (CC-BY-4.0); this file provides it.

The same four apply to publishing the `-full` zip, which is why they are worth getting
right: that zip is a redistribution of all of the above.

Nothing here forbids commercial use or requires you to open your own changes, provided
`bin/ffmpeg.exe` remains the LGPL build.
