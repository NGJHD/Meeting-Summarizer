# Meeting Summariser

<img width="895" height="468" alt="Meeting Summarizer" src="https://github.com/user-attachments/assets/25474576-1057-42e0-8076-e8a600d38cfa" />

Turns a long meeting recording into Markdown minutes or a summary. **Entirely offline** - 
nothing leaves the machine.

Built for 3–5 hour recordings on Windows 11. The end user unzips a folder and
double-clicks `run.bat`; there is no installer, no Python to set up, and no configuration
to edit.

Output is a meeting summary, minutes, or both, plus the full transcript. After a run you can play
the three longest things each speaker said, type their names, and have the document
tagged with the speaker names.

## Using it

1. Download **`Meeting-Summariser-vX.Y.Z-full.zip`** from the latest release and unzip it
2. Double click on `DOWNLOAD_MODELS.bat`

The full zip carries the app, the Python runtime and every inference binary, so step 2
only has to fetch the models. It needs internet access; nothing after it does.

After that, you can just copy the entire folder (about 30GB with the models) to other machines and it should work.

Then, double-click `run.bat` and the UI will open ready for you to load your meeting's MP3 recording.

---

# For Developers

## Folder Structure

```
Audio file
 └─> ffmpeg                    → 16kHz mono WAV
      ├─> whisper.cpp (GPU)    → words with timestamps
      └─> sherpa-onnx (CPU)    → speaker turns
           └─> merge           → speaker-attributed transcript
                └─> chunk      → ~10k-token windows
                     └─> map / group reduce / reduce (Qwen3.8-27B)
                          └─> output\<meeting>\<meeting>_summary.md
```
## Design constraints

Three decisions shape everything else:

- **No PyTorch, anywhere.** All inference happens in prebuilt native binaries called over
  subprocess or HTTP. That is why diarization is sherpa-onnx and onnxruntime rather than
  pyannote.audio.
- **Nothing outside the app folder.** No `%APPDATA%`, no registry. Copy the folder to
  another drive and it works unchanged; delete it and nothing is left behind.
- **No internet at runtime.** No CDN, no fonts, no telemetry. The server binds to
  `127.0.0.1` only. The one exception is the *Check for updates* button, which runs only
  when pressed.

## Hardware

Windows 11, at least 32GB RAM if there is no GPU

If more than 16GB VRAM is detected, the app will auto select Qwen3.8-27B-UD-Q4_K_M model.
Otherwise it would be the Qwen3.8-27B-UD-IQ3_XXS.gguf.

If no discrete GPU is detected, CPU will be used instead of the iGPU.

All of this is detected at startup; there is nothing to configure. `bin\` carries one
folder per backend and the right pair is chosen per engine.

## Layout

| Path | |
|---|---|
| `server/` | FastAPI orchestrator — stages, jobs, SSE progress, cancellation |
| `web/` | Frontend. Plain HTML/CSS/JS, no build step, no npm |
| `prompts/` | The map / group-reduce / reduce prompts, as plain text |
| `config.json` | Every tunable. There is deliberately no settings screen |
| `run.bat` | Preflight checks, port selection, starts the server, opens the browser |
| `DOWNLOAD_MODELS.bat` | Fetches ~30 GB of models and every per-backend binary. Resumable, size-verified, safe to re-run |

`bin/`, `models/` and `runtime/` are not in the repository — they are the shipped payload,
fetched by `DOWNLOAD_MODELS.bat` or copied with the release.

## Releases

Each release carries two zips, and they are not variants of the same thing.

| Asset | Size | For |
|---|---|---|
| `Meeting-Summariser-vX.Y.Z-full.zip` | ~1.1 GB | **A first install.** Everything except the models: the app, the Python runtime, ffmpeg and every inference binary — the exact set that was tested |
| `Meeting-Summariser-vX.Y.Z.zip` | ~190 KB | The update payload — what *Check for updates* downloads |

The models are never release assets: two of them are individually larger than GitHub's
2 GB per-asset limit, so `DOWNLOAD_MODELS.bat` is the only way to get those.

*About → Check for updates* takes the small zip and replaces the code in place, leaving
`bin/`, `models/`, `runtime/` and your `config.json` alone. It deliberately never takes
the full bundle — that would turn a 190 KB update into 1.1 GB, and it would overwrite
`runtime\python.exe`, the interpreter the running app is executing from.

If you cloned the source rather than taking the full zip, `DOWNLOAD_MODELS.bat` fetches
the runtime, ffmpeg and the binaries too. One of them is this project's own:
whisper.cpp publishes no Vulkan build for Windows, so `bin\whisper-vulkan\` is built
from source and hosted on this repository's releases rather than expecting anyone to
install a C++ toolchain. If you would rather build it yourself, the recipe and its
verification against the CUDA binary are in `BUILD_NOTES.md`.

## Documentation

- **`CLAUDE.md`** — the specification. Every design decision, and why.
- **`BUILD_NOTES.md`** — what was actually measured: every flag that turned out not to
  exist, every trap found the hard way, and the numbers behind each choice.

## Licence

**MIT** — see [`LICENSE`](LICENSE). That covers the source here: `server/`, `web/`,
`prompts/`, the batch files and the documentation.

It does not cover what `DOWNLOAD_MODELS.bat` fetches into `bin/`, `models/` and
`runtime/`. Those keep their own licences and are listed in
[`THIRD_PARTY_NOTICES.md`](THIRD_PARTY_NOTICES.md) — all permissive, with two things
worth knowing if you pass the assembled folder on:

- **ffmpeg is the LGPL v3 build, chosen deliberately over the GPL one.** It does
  everything this app needs, and avoids placing GPL obligations on anyone you give the
  folder to.
- **The speaker-embedding model is CC-BY-4.0**, which requires attribution to NVIDIA.
  `THIRD_PARTY_NOTICES.md` provides it, so keep that file with the folder.
