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

1. Download and unzip the release
2. Double click on `DOWNLOAD_MODELS.bat`

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

A release zip contains the **source tree only**, not the 30 GB of models and binaries. The
in-app *Check for updates* button downloads it and replaces the code in place, leaving
`bin/`, `models/`, `runtime/` and your `config.json` untouched.

For a first install: clone or download the source, then run `DOWNLOAD_MODELS.bat` once on
a machine with internet access and copy the whole folder to the offline machine.

`DOWNLOAD_MODELS.bat` fetches everything, including `bin\whisper-vulkan\`. That one is
the odd case: whisper.cpp publishes no Vulkan build for Windows, so it is built from
source and hosted on this repository's releases rather than expecting anyone to install a
C++ toolchain. However if you want to build the whisper vulkan yourself, the build recipe 
and its verification against the CUDA binary are in `BUILD_NOTES.md`.

## Documentation

- **`CLAUDE.md`** — the specification. Every design decision, and why.
- **`BUILD_NOTES.md`** — what was actually measured: every flag that turned out not to
  exist, every trap found the hard way, and the numbers behind each choice.

## Licence

Not yet chosen. The bundled binaries and models carry their own licences.
