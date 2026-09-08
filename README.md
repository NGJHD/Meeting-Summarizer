# Meeting Summariser

Turns a long meeting recording into Markdown minutes or a summary. **Entirely offline** —
it runs with the network adapter disabled, and nothing leaves the machine.

Built for 3–5 hour recordings on a single Windows 11 machine with an NVIDIA GPU. The end
user unzips a folder and double-clicks `run.bat`; there is no installer, no Python to
set up, and no configuration to edit.

---

## What it does

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

Output is a summary, minutes, or both, plus the full transcript. After a run you can play
the three longest things each speaker said, type their names, and have every document
rewritten — instantly, because it is a substitution, not another model call.

## Notable constraints

Three decisions shape everything else:

- **No PyTorch, anywhere.** All inference happens in prebuilt native binaries called over
  subprocess or HTTP. That is why diarization is sherpa-onnx and onnxruntime rather than
  pyannote.audio.
- **Nothing outside the app folder.** No `%APPDATA%`, no registry. Copy the folder to
  another drive and it works unchanged; delete it and nothing is left behind.
- **No internet at runtime.** No CDN, no fonts, no telemetry. The server binds to
  `127.0.0.1` only.

## Layout

| Path | |
|---|---|
| `server/` | FastAPI orchestrator — stages, jobs, SSE progress, cancellation |
| `web/` | Frontend. Plain HTML/CSS/JS, no build step, no npm |
| `prompts/` | The map / group-reduce / reduce prompts, as plain text |
| `config.json` | Every tunable. There is deliberately no settings screen |
| `run.bat` | Preflight checks, port selection, starts the server, opens the browser |
| `DOWNLOAD_MODELS.bat` | Fetches the ~29 GB of models. Resumable, size-verified |

`bin/`, `models/` and `runtime/` are not in the repository — they are the shipped payload,
fetched by `DOWNLOAD_MODELS.bat` or copied with the release zip.

## Documentation

- **`CLAUDE.md`** — the specification. Every design decision, and why.
- **`BUILD_NOTES.md`** — what was actually measured, every flag that turned out not to
  exist, and every trap found the hard way.
- **`AMD_INTEL_BUILD.md`** — assessment of what it would take to run on AMD or Intel
  graphics.

## Requirements

Windows 11, NVIDIA GPU (16 GB VRAM for the high-quality model, 8 GB for the low-quality
one), 32 GB system RAM, and a current NVIDIA display driver. That driver is the only
external dependency.

## Licence

Not yet chosen. The bundled binaries and models carry their own licences.
