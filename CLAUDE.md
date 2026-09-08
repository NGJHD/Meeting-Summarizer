# CLAUDE.md — Offline Meeting Transcription, Minutes & Summariser

You are building a **fully offline Windows desktop application** that turns a long meeting
recording into Markdown minutes or a Markdown summary.

Read this entire file before writing any code. Every design decision here was made
deliberately. If you think one is wrong, say so explicitly in your response — do not
silently substitute your own approach.

---

## 0. NON-NEGOTIABLE CONSTRAINTS

1. **No internet access at runtime.** Not for models, not for fonts, not for CDN scripts,
   not for telemetry. The app must work with the network adapter physically disabled.

   **One documented exception:** the *Check for updates* button in the About overlay
   (§13.12). It is the only outbound request the app ever makes, it happens only on a
   button press, and the whole pipeline still runs with the adapter disabled. Nothing
   checks on startup, on a timer, or in the background.
2. **No installation steps.** The end user unzips a folder and double-clicks `run.bat`.
   No Python installer, no CUDA toolkit, no pip.
3. **No PyTorch.** Anywhere. This is the constraint that shapes the whole architecture.
   All heavy inference happens in pre-built native binaries called via subprocess or HTTP.
4. **Non-technical end user.** They will not read a README, edit a config file, or
   understand an error message containing a stack trace.
5. **Single machine target.** Windows 11, NVIDIA GPU with 16GB VRAM, 32GB system RAM.
   The only external dependency you may assume is an installed display driver.

   **AMD and Intel are supported too**, via Vulkan — one cross-vendor binary, integrated
   or discrete, needing nothing installed beyond a current driver. The backend is
   detected at startup and resolved **per engine**, because whisper.cpp and llama.cpp
   are not shipped in step: see §2.1.

### Scope

**Two output modes: Minutes and Summary**, chosen by the user before processing starts.
Only one is produced per run.

The modes diverge **only at the final reduce step**. Extraction (`map`) and consolidation
(`group_reduce`) are mechanical and identical for both — the same facts, decisions,
speaker evidence and timestamps are needed either way. What differs is the final
synthesis: minutes are a structured record of decisions and owners; a summary is prose
organised by theme. This means one shared extraction path and two final prompts, not two
parallel pipelines.

**Recording length: 3–4 hours typical, 5 hours the realistic maximum.**

The operator has confirmed no meeting will reach 8 hours. Earlier revisions of this
document, and DIARIZATION_FIX.md part A4, required designing for 8; that requirement is
withdrawn as a *test target*, but **not** as a design constraint.

Every duration figure here is a scaling reference, not a limit. Nothing may hard-code an
assumption of any particular length — not buffer sizes, not chunk counts, not the reduce
strategy, not disk space checks, not the UI's time estimate, not timeout values. If any
part of the implementation would behave differently on a 9-hour recording than on a
4-hour one, that is still a defect: the cost of being length-agnostic is zero, and every
mechanism that provides it (streaming uploads, block-wise audio reads, the recursive
reduce in section 10.3) is already built and costs nothing to keep.

What changes is only where to spend testing effort: **verify at 4–5 hours.** A 5-hour
recording is roughly 7 chunks, which still takes the two-tier path; the three-tier path
is exercised deliberately by lowering `group_reduce_threshold` (section 17, step 9)
rather than by finding an enormous file.

Reference scaling (~150 spoken words/minute):

| Duration | 16kHz mono WAV | Transcript tokens | Chunks |
|---|---|---|---|
| 2h | 230 MB | ~24k | 3 |
| 4h | 460 MB | ~48k | 6 |
| 6h | 690 MB | ~72k | 8 |
| 8h | 920 MB | ~96k | 11 |

---

## 1. VERIFY BEFORE YOU CODE

The command-line flags in this document are drawn from recent llama.cpp, whisper.cpp and
sherpa-onnx releases. **These projects rename and remove flags frequently.**

Before writing the orchestration code, run each of the following against the exact
binaries present in `bin/` and read the output:

```
bin\llama-server.exe --help
bin\whisper-cli.exe --help
```

For every flag this document specifies, confirm it exists with that name in that build.
If a flag is absent or renamed, find the equivalent in the help output and **use it** —
then add a line to `BUILD_NOTES.md` recording what you changed and why. Do not silently
drop a flag. Do not guess.

If a binary or model file is missing from `bin/` or `models/`, stop and report which one,
rather than writing code that assumes it.

---

## 2. ARCHITECTURE

```
Audio file
 └─> ffmpeg                    → 16kHz mono WAV
      ├─> whisper.cpp (GPU)    → segments + word-level timestamps (JSON)
      └─> sherpa-onnx (CPU)    → speaker turns: (start, end, speaker_id)
           └─> MERGE           → speaker-attributed, timestamped transcript
                └─> CHUNK      → ~10k-token windows
                     └─> MAP   → per-chunk notes           (N calls)
                          └─> [GROUP REDUCE if N > 8]      (see §10.3)
                               └─> FINAL REDUCE            (1 call)
                                    └─> output\<name>_summary.md
```

### 2.1 Backends

Every inference binary ships once per backend, and the app picks at startup:

```
bin\llama-cuda\    bin\llama-vulkan\    bin\llama-cpubin\whisper-cuda\  bin\whisper-vulkan\  bin\whisper-cpu```

- **CUDA** where there is an NVIDIA card. Fastest, and what the numbers in
  `BUILD_NOTES.md` were measured on.
- **Vulkan** for AMD and Intel, integrated or discrete. One binary, no vendor runtime.
- **CPU** as the last resort — usable only for very short recordings.

Detection: `nvidia-smi` first; failing that, `llama-server --list-devices` from the
Vulkan build, which is the **only reliable cross-vendor VRAM read**.
`Win32_VideoController.AdapterRAM` is a 32-bit field that caps at 4 GB and reports
4095 MB for a 16 GB card, so it must not be used.

**The choice is per engine, not per machine.** whisper.cpp publishes no Vulkan binary
for Windows at all, so `bin\whisper-vulkan\` is **built from source** and ships inside
the folder rather than being downloaded — the recipe is in `BUILD_NOTES.md` §9o. If it
is ever missing the app still works: a non-NVIDIA machine falls back to the CPU whisper
build and keeps the LLM on Vulkan, which is 80–95% of the wall time.

The CUDA binaries are **never** used as a fallback on a machine with no NVIDIA card:
they would load `ggml-cuda.dll`, find no device and quietly run on CPU anyway — slower
to start and far harder to diagnose than choosing the CPU build outright.

`gpu.backend` in `config.json` is `"auto"`; `"cuda"` / `"vulkan"` / `"cpu"` pin it.

**Language: Python.** A standalone relocatable CPython runtime with vendored pure-Python
wheels. Frontend is plain HTML/CSS/JS with no build step and no npm.

**Backend: FastAPI + uvicorn**, bound to `127.0.0.1` only. Never `0.0.0.0`.

The Python layer is an **orchestrator only**. It shells out to binaries and makes HTTP
calls. It performs no inference itself.

---

## 3. DIRECTORY LAYOUT

```
MeetingSummariser\
  run.bat
  config.json
  BUILD_NOTES.md              <- you write this; records flag verification + deviations
  runtime\                    <- standalone CPython (python-build-standalone)
    python.exe
    Lib\site-packages\        <- fastapi, uvicorn, starlette, pydantic, anyio, httpx...
  bin\
    ffmpeg.exe
    cudart64_12.dll ...       <- CUDA runtime, shared by both NVIDIA builds
    llama-cuda\ llama-vulkan\ llama-cpu\      <- one folder per backend (§2.1)
    whisper-cuda\ whisper-vulkan\ whisper-cpu\   <- vulkan is a local build
    sherpa-onnx-*.dll / .pyd  <- ONNX Runtime diarization, CPU only, vendor-neutral
  models\
    ggml-large-v3-turbo.bin
    ggml-silero-v5.1.2.bin
    segmentation-3.0.onnx
    speaker-embedding.onnx
    Qwen3.8-27B-UD-Q4_K_M.gguf   <- High Quality, used at 15GB VRAM and above
    Qwen3.8-27B-UD-IQ3_XXS.gguf  <- Low Quality, used below that
  prompts\
    map.txt
    group_reduce.txt
    reduce_summary.txt
    reduce_minutes.txt
  web\
    index.html
    app.js
    style.css
  server\
    main.py                   <- FastAPI app, routes, SSE, streaming upload
    open_browser.py           <- waits for the server, then opens the browser
    calibration.py            <- per-machine stage timings; estimate + bar weights
    speakers.py               <- voice samples, speaker renaming, note cache
    pipeline.py               <- stage orchestration
    audio.py                  <- ffmpeg
    transcribe.py             <- whisper.cpp wrapper
    diarize.py                <- sherpa-onnx wrapper
    merge.py                  <- transcript + speaker merge
    chunker.py                <- tokenizer-aware chunking
    llm.py                    <- llama-server lifecycle + chat calls
    reduce.py                 <- map / group reduce / final reduce
    jobs.py                   <- job state, cancellation, cleanup
    version.py                <- app name, author, version, repo. Nothing else
    updater.py                <- the Check-for-updates button (UPDATE_BUTTON.md)
  temp\                       <- created at runtime, always emptied
  output\
    <meeting name>\           <- one folder per meeting, never loose files
      <meeting name>_transcript.md
      <meeting name>_transcript_tagged.md
      <meeting name>_summary.md
      <meeting name>_minutes.md
      speaker_samples\
        samples.json
        spk00_1.mp3 ...
```

Filenames keep the meeting prefix inside the folder, so a document still identifies
itself once it has been copied out of it.

---

## 4. CONFIGURATION

All tunables live in `config.json`. **There is no settings screen in the UI.** The
operator edits this file in Notepad; the end user never sees it.

```json
{
  "llm": {
    "model": "models/Qwen3.8-27B-UD-Q4_K_M.gguf",
    "ctx_size": 32768,
    "gpu_layers": 99,
    "cpu_ffn_regex": "blk\\.(6[0-3]|5[0-9]|4[0-9])\\.ffn_.*=CPU",
    "cache_type_k": "q8_0",
    "cache_type_v": "q8_0",
    "port": 8080,
    "startup_timeout_s": 180
  },
  "chunking": {
    "target_tokens": 10000,
    "overlap_tokens": 400,
    "max_map_output_tokens": 1500,
    "group_reduce_threshold": 8,
    "group_size": 5,
    "max_group_output_tokens": 2500
  },
  "thinking": {
    "map": false,
    "group_reduce": false,
    "reduce": true
  },
  "pipeline": {
    "concurrent_diarization": true
  },
  "gpu": { "backend": "auto" },
  "estimate": {
    "transcript_tokens_per_audio_minute": 206,
    "fixed_overhead_s": 60,
    "buffer_fraction": 0.15
  },
  "whisper": {
    "model": "models/ggml-large-v3-turbo.bin",
    "vad_model": "models/ggml-silero-v5.1.2.bin",
    "language": "en",
    "threads": 6
  },
  "diarization": {
    "enabled": true,
    "threads": 6,
    "num_speakers": 0,
    "cluster_threshold": 0.5,
    "min_duration_on": 0.3,
    "min_duration_off": 0.5
  },
  "server": { "port": 8000 }
}
```

`num_speakers: 0` means auto-detect. If a run produces obviously wrong speaker counts,
the operator sets it explicitly.

---

## 5. UPLOAD HANDLING

An 8-hour MP3 can be 250–400MB.

**Stream the upload to disk in chunks.** Do not `await file.read()` into memory — that
will spike RAM and can fail outright on large files. Write to `temp\upload.<ext>` in
1MB blocks as it arrives, and report upload progress to the UI.

Accept MP3, WAV, M4A, MP4, and MKV — ffmpeg handles them all and the user should not have
to convert anything first.

Before accepting, check free disk space on the app's drive. Require at least
**3× the uploaded file size** free, to cover the upload plus the converted WAV plus
headroom. If insufficient, refuse with a plain message stating how much is needed.

---

## 6. STAGE 1 — AUDIO CONVERSION

```
bin\ffmpeg.exe -y -i "temp\upload.<ext>" -ar 16000 -ac 1 -c:a pcm_s16le "temp\audio.wav"
```

- The WAV is roughly **115MB per hour** of audio. Do not assume any particular size.
- Parse ffmpeg's stderr `time=` output to drive a progress percentage for this stage.
- Extract the total duration from ffmpeg's output and store it on the job. **Every
  downstream time estimate is derived from this number**, not from a constant.
- Reject files ffmpeg cannot decode with: *"That file couldn't be read as audio. Try a
  different recording."* — never surface ffmpeg's raw stderr.

---

## 7. STAGE 2 — TRANSCRIPTION (GPU)

```
bin\whisper-cli.exe ^
  -m models\ggml-large-v3-turbo.bin ^
  -f temp\audio.wav ^
  -l en ^
  -oj -of temp\transcript ^
  -pp ^
  --vad --vad-model models\ggml-silero-v5.1.2.bin ^
  --dtw large.v3.turbo ^
  -t 8
```

**Why each flag matters:**

- `--vad` — **mandatory.** Whisper hallucinates repeated text during silence, and a long
  recording contains a great deal of it. Without VAD you will get fabricated paragraphs.
  The longer the recording, the worse this gets.
- `--dtw large.v3.turbo` — produces token-level timestamps, **required by the merge
  stage**. Without them you cannot assign words to speakers accurately. The preset string
  must match the model; verify accepted values in `--help`.
- `-pp` — prints progress to stdout. Parse it to drive the progress bar. Without it the
  UI shows nothing for 10–40 minutes and the user will assume a crash.
- `-oj` — JSON output including per-token offsets.

Expected runtime: roughly **3–5 minutes per hour of audio**.

### Concurrency with Stage 3

**Stage 3 does not depend on this stage.** Diarization reads `temp\audio.wav` directly and
produces speaker turns from the audio waveform alone; it never sees the transcript. Only
Stage 4 (merge) depends on both. So the two may run in parallel — transcription is
GPU-bound, diarization is CPU-only.

Run them concurrently by default. At 8 hours the saving is substantial: sequential is
roughly 40 min transcribe + 30 min diarize, concurrent is roughly 45 min total.

**Constraints when running concurrently:**

- **Budget threads explicitly.** Both processes will otherwise each claim ~8 threads and
  contend. `whisper.threads` and `diarization.threads` in `config.json` default to 6 each;
  their sum must not exceed the machine's logical core count minus 2.
- **Failures are independent.** If diarization fails, transcription must continue to
  completion and the pipeline degrades to unattributed output. If transcription fails,
  cancel diarization and fail the job — there is nothing to merge.
- **Cancel must terminate both**, and must not leave one running when the other dies.
- **Progress:** drive the shared 55% stage allocation from transcription only. Its
  progress output is fine-grained; diarization's is coarse and would make the bar jump.
  Report diarization separately in the text log.

**`pipeline.concurrent_diarization: false` runs them sequentially** (transcribe, then
diarize). Implement this path too. It is slower but strictly simpler, and it is the first
thing to try when debugging an attribution or resource problem — it removes an entire
class of interaction from the picture.

---

## 8. STAGE 3 — DIARIZATION (CPU)

Speaker attribution is required for both output modes. In Summary mode it distinguishes
one person's objection from a room-wide one; in Minutes mode it is what makes action
item ownership possible at all.

It also gives the model dialogue structure to reason over across a very long transcript.

Run **three separate stages**, not sherpa's one-call `OfflineSpeakerDiarization`:

1. **Segment** — pyannote segmentation-3.0 (`models/segmentation-3.0.onnx`) under
   onnxruntime, with powerset decoding → per-frame local speaker activity.
2. **Embed** — sherpa-onnx `SpeakerEmbeddingExtractor`
   (`models/speaker-embedding.onnx`) → one vector per (window, local speaker).
3. **Cluster** — sherpa-onnx `FastClustering`, then prune (§8.1).

`OfflineSpeakerDiarization` does all three in a single call, so every change to a
clustering parameter re-pays segmentation and embedding — about 40 minutes on a 3.5-hour
recording. Stages 1 and 2 are deterministic; **cache their output** keyed on the audio.
Re-clustering then costs seconds, which is what makes the cap in §8.1 possible at all.

**Do not use pyannote.audio, whisperX, or anything importing torch.** That is the entire
reason these components were chosen. onnxruntime is acceptable — it is a plain wheel with
no torch in its tree.

**All three stages must run in a child process.** sherpa-onnx holds the CPython GIL for
the whole of its native calls, which starves every other thread in the server; see
BUILD_NOTES.md §3.7 for the deadlock this caused.

Input is a float32 array in `[-1, 1]` at 16kHz. Read `temp\audio.wav` with the stdlib
`wave` module and normalise. Read it **in blocks** — do not load a 900MB WAV fully into a
Python list.

Output: a list of `(start_seconds, end_seconds, speaker_id_int)`.

**Scaling warning:** the cost is dominated by **per-segment embedding extraction**, which
is roughly linear in the number of speech segments and therefore in recording duration.
Measured: ~12 minutes per hour of audio, of which clustering itself is milliseconds.

An earlier revision of this section blamed superlinear *clustering* cost. That was wrong
and it pointed tuning in the wrong direction; see BUILD_NOTES.md §6.

Raising `min_duration_on` is **withdrawn as the recommended mitigation**. It discards
segments from the timeline before anything knows whether they matter, and it eats real
short interjections — including the one-word "no, we agreed Q3" that changes what a
decision was. Those words then vanish from the transcript entirely and no downstream
stage can recover them. Prune *after* clustering instead (§8.1): that keeps the audio and
the words and discards only the label.

### 8.1 Cluster pruning and automatic degradation

Automatic speaker counting is unreliable on long recordings: a fixed distance threshold
that is correct at 15 minutes fragments badly at 3 hours. Threshold tuning cannot fix
this — no value is right for both. Prune *after* clustering instead, which is
duration-independent by construction:

1. Rank clusters by total speech duration.
2. Keep clusters above `min_cluster_speech_s`, defaulting to
   `max(30, 0.005 x total speech seconds)`.
3. Cap survivors at `max_speakers`, keeping the longest.
4. Reassign each discarded cluster's segments to the nearest surviving centroid by
   cosine distance, but only within `reassign_max_distance`. Otherwise mark `UNKNOWN`.
5. Renumber survivors `SPEAKER_00 …` by first appearance time.
6. At merge (§9), `UNKNOWN` inherits the previous turn's speaker — the same rule already
   used for words falling in diarization gaps.

**Degrade automatically.** After pruning, drop attribution for the whole run if the
survivor count still exceeds `max_speakers`, or if survivors cover less than
`min_coverage_fraction` of total speech. Emit the unattributed transcript, log why, and
put a line in the output document: *"Speaker identification was unreliable for this
recording and has been omitted."*

`SPEAKER_183` is worse than no label at all: the reader cannot tell clustering debris
from a real participant and will act on it either way. Encode that judgement here rather
than leaving it to whoever reads the output.

**The user may also just tell us.** §13 has an optional participant count. When supplied
it is passed as the cluster count and distance thresholding is skipped entirely — by far
the most reliable path. The cap in step 3 still applies as a guard.

**If diarization fails for any reason, log it and continue with an empty speaker list.**
The merge stage must degrade gracefully to an unattributed transcript rather than failing
the job. The same applies when `diarization.enabled` is `false` — the pipeline must run
end to end with the stage skipped entirely.

---

## 9. STAGE 4 — MERGE (the stage most likely to produce garbage)

Whisper segments and diarization segments are two independent timelines that do not line
up. A single Whisper segment frequently spans a speaker change. Naive segment-level
merging is the most common source of wrong attribution in this kind of pipeline.

**Algorithm — operate at word level, not segment level:**

1. Flatten the Whisper JSON into a list of words, each with `(text, t_start, t_end)`.
2. For each word, compute its midpoint `(t_start + t_end) / 2`.
3. Find the diarization segment containing that midpoint. If none contains it (the word
   falls in a gap), inherit the speaker of the **previous** word.
4. Group runs of consecutive same-speaker words into turns.
5. Discard turns shorter than 3 words that sit between two turns of the same other
   speaker — these are almost always clustering noise, not real interjections.

**Output format fed to the LLM:**

```
[SPEAKER_01] (00:14:22) So the vendor contract — we agreed to defer that to Q3.
[SPEAKER_03] (00:14:31) Only if legal signs off on the indemnity clause first.
```

Timestamps as `HH:MM:SS` of the turn start. **Carry them through to the final document.**
On a multi-hour recording they are what lets a reader jump back to the source, and they
partially recover what anonymous speaker labels cost you.

Write the merged transcript to `output\<name>_transcript.md` and keep it. It is a useful
deliverable in its own right and it is what you debug against when a summary is wrong.

---

## 10. STAGE 5 — CHUNKING AND REDUCE STRATEGY

### 10.1 Chunking

Do **not** attempt a single-pass call over the full transcript. The KV cache for a 50k–
100k token transcript will not fit alongside the model weights at this quantisation, and
recall degrades badly in the middle of very long contexts.

**Count tokens accurately** using llama-server's `/tokenize` endpoint. Do not estimate
with a characters-per-token heuristic; it drifts badly on transcript text full of names
and disfluencies.

Rules:
- Target `target_tokens` per chunk, never split mid-turn — always break on a turn boundary.
- Carry `overlap_tokens` of the previous chunk's tail into the next chunk, so a point
  discussed across a boundary is not lost.
- Record each chunk's start and end timestamp; pass them into the map prompt.

### 10.2 Two-tier reduce (chunk count ≤ `group_reduce_threshold`)

```
MAP over N chunks  →  N sets of notes  →  FINAL REDUCE  →  document
```

### 10.3 Three-tier reduce (chunk count > `group_reduce_threshold`) — REQUIRED

**This is not optional and it is the reason this section exists.** With 11 chunks, a
two-tier reduce feeds ~16.5k tokens of notes plus up to 8k of output plus thinking tokens
into a 32k context. It does not fail cleanly — it silently drops material from the middle
of the meeting, which is the failure mode least likely to be noticed.

```
MAP over N chunks
  → partition notes into consecutive groups of `group_size`, in chronological order
  → GROUP REDUCE each group  →  a section summary (~`max_group_output_tokens`)
  → FINAL REDUCE over the section summaries  →  document
```

An 8-hour meeting: 11 chunks → 3 groups → 3 section summaries (~7.5k tokens) → one final
call. Comfortable.

**Generalise this.** If the number of section summaries ever exceeds
`group_reduce_threshold`, apply the grouping step again recursively. It should not happen
below 40 hours of audio, but the code must not assume exactly three tiers.

Before every LLM call, assert that the assembled prompt fits within
`ctx_size - max_output_tokens - 2000`. If it does not, log the overflow and split further
rather than sending it.

---

## 11. STAGE 6 — THE LLM

### 11.1 Server lifecycle

**Whisper and the LLM must never be resident in VRAM at the same time.** Sequence
strictly: transcribe → whisper process exits → *then* start llama-server.

Launch:

```
bin\llama-server.exe ^
  -m models\Qwen3.8-27B-UD-Q4_K_M.gguf ^
  --ctx-size 32768 ^
  --n-gpu-layers 99 ^
  --override-tensor "blk\.(6[0-3]|5[0-9]|4[0-9])\.ffn_.*=CPU" ^
  --flash-attn on ^
  --cache-type-k q8_0 --cache-type-v q8_0 ^
  --jinja ^
  --threads 8 ^
  --batch-size 512 --ubatch-size 512 ^
  --parallel 1 ^
  --host 127.0.0.1 --port 8080 ^
  --no-webui
```

**Why `--override-tensor`:** a 27B model at Q4_K_M does not fit in 16GB. This pushes the
FFN tensors of the upper layers into system RAM while keeping attention on the GPU. It is
preferred over dropping to a 3-bit quant because summaries live or die on getting names,
figures and dates exactly right, and sub-4-bit quants are where models begin mangling
proper nouns and digits.

**The regex is a starting point, not a tuned value.** After the first successful run,
report peak VRAM. If there is headroom, move layers back to GPU; if it OOMs, move more to
CPU. It is in `config.json` so this needs no code change.

`--jinja` is required — without it llama.cpp uses a generic chat template and the
per-request `chat_template_kwargs` below has nothing to pass through to.

Poll `GET /health` until ready or `startup_timeout_s` elapses. First load off a cold disk
takes 60–120 seconds. Surface this as *"Loading language model (up to 2 minutes on first
run)…"* so it does not look hung.

Shut the server down when the job finishes. Do not leave 14GB of VRAM allocated.

### 11.2 Sampling parameters

The model is a hybrid thinking model with **extra-high reasoning effort enabled by
default**. Left alone it will burn thousands of reasoning tokens on every map call. On an
8-hour recording that is 11 map calls, and it will turn a 40-minute job into a 2-hour one.

**Map and group-reduce calls — thinking OFF:**
```json
{
  "temperature": 0.7,
  "top_p": 0.8,
  "top_k": 20,
  "presence_penalty": 1.5,
  "repeat_penalty": 1.0,
  "chat_template_kwargs": { "enable_thinking": false }
}
```

**Final reduce call — thinking ON:**
```json
{
  "temperature": 1.0,
  "top_p": 0.95,
  "top_k": 20,
  "min_p": 0.0,
  "presence_penalty": 0.0,
  "repeat_penalty": 1.0,
  "max_tokens": 8000,
  "chat_template_kwargs": { "enable_thinking": true, "reasoning_effort": "medium" }
}
```

The split is deliberate. Map and group-reduce are mechanical consolidation — reasoning
buys little and costs a great deal across many calls. The final reduce does cross-section
inference (resolving a speaker mentioned late against a name introduced early,
recognising that two differently-transcribed phrases refer to one thing), which is where
thinking earns its cost. All three are switchable via `config.json.thinking`.

**Note on `--reasoning-budget`:** setting it at server level may disable thinking
globally, which would break the final reduce. It is deliberately omitted from the launch
command above in favour of per-request control. **Verify which mechanism the shipped
build actually honours** — send one request with `enable_thinking: false` and confirm no
`<think>` block comes back — and record the finding in `BUILD_NOTES.md`.

**Always strip `<think>...</think>` blocks from responses before use, even when thinking
is disabled.** Belt and braces.

### 11.3 Robustness

- Retry a failed call twice with backoff before failing the job.
- If a map call returns empty or unparseable output after retries, insert a placeholder
  note for that chunk and continue. One bad chunk out of eleven must not destroy a job
  that has already consumed 40 minutes.
- Log every request's token counts and wall time to `temp\job.log`.

---

## 12. PROMPTS

Store as plain `.txt` in `prompts\` so they can be tuned without touching code. Load at
job start. Substitute placeholders with `str.replace`, not f-strings — the prompts contain
literal braces.

Placeholders: `{{TRANSCRIPT}}`, `{{NOTES}}`, `{{CHUNK_INDEX}}`, `{{CHUNK_TOTAL}}`,
`{{TIME_START}}`, `{{TIME_END}}`, `{{MEETING_NAME}}`, `{{DURATION}}`.

### Shared rules — include verbatim in ALL prompts

```
- Base all factual content strictly on the transcript. Never infer decisions, numbers,
  deadlines, or outcomes that were not stated.
- Speaker labels (SPEAKER_00, SPEAKER_01, ...) are anonymous voice clusters produced by
  automatic diarization. They are NOT names and carry no identity information.
- You MAY map a speaker label to a real name when the transcript contains sufficient
  evidence — for example, someone addresses them by name, they introduce themselves, or
  another participant refers back to something that speaker said and attributes it. When
  you do this, state it as an inference: "SPEAKER_02 (likely Sarah)".
- Never assert a name-to-speaker mapping as established fact without transcript evidence.
- The transcript is machine-generated and contains recognition errors. If a term appears
  in several inconsistent spellings, treat them as the same thing and use the most
  plausible form.
- If something is ambiguous, contested, or left unresolved, flag it explicitly rather
  than smoothing it over.
- Preserve (HH:MM:SS) timestamps against significant points.
```

### `map.txt`

Extract from chunk `{{CHUNK_INDEX}}` of `{{CHUNK_TOTAL}}`, covering `{{TIME_START}}` to
`{{TIME_END}}`. Return Markdown under fixed headings:

- **Themes** — what was actually being discussed, in substance
- **Positions** — who argued what, and where participants disagreed
- **Outcomes** — anything concluded, decided, or explicitly deferred, with timestamps
- **Loose Ends** — questions raised and not answered
- **Speaker Evidence** — anything bearing on which real person a speaker label might be

Instruct it to be **exhaustive rather than concise**. This is raw material, not a
deliverable — over-inclusion at this stage is cheap, and omission is unrecoverable.

### `group_reduce.txt`

Consolidate `{{NOTES}}` from several consecutive chunks into one section summary covering
`{{TIME_START}}`–`{{TIME_END}}`. Deduplicate points repeated across chunk overlaps. Keep
the same five headings. Preserve timestamps and speaker evidence — **do not drop speaker
evidence at this tier**, the final reduce needs it to resolve identities across the whole
meeting.

### `reduce_summary.txt`

Produce the final document from `{{NOTES}}`:

```
# Summary — {{MEETING_NAME}}
*Duration: {{DURATION}}*

## Executive Summary        <- 3–5 sentences, no bullets
## Key Themes               <- organised by topic, NOT chronologically; flowing prose
## Decisions and Outcomes   <- with timestamps
## Points of Disagreement
## Unresolved Questions
## Participants             <- speaker labels, inferred names, confidence
```

Explicitly instruct: organise by theme, not by time. A chronological retelling of a
6-hour meeting is nearly as long as the transcript and nearly as useless.

### `reduce_minutes.txt`

Same `{{NOTES}}` input, different synthesis. This is a record, not an essay — terse,
scannable, and complete on decisions and ownership:

```
# Minutes — {{MEETING_NAME}}
*Duration: {{DURATION}}*

## Participants             <- speaker labels, inferred names, confidence
## Decisions                <- one line each, with (HH:MM:SS)
## Action Items             <- Markdown checkboxes, grouped by owner
## Discussion Record        <- by topic; 2–4 bullets each, not prose
## Open Questions
## Attribution Notes        <- how each speaker label was resolved, and how confidently
```

Action item format: `- [ ] **Owner** — task — deadline — (HH:MM:SS)`. Use `Unassigned`
where no owner was stated. **Never invent an owner or a deadline.** An action item with
`Unassigned` and no deadline is a correct and useful output; a fabricated owner is a
serious failure, because the reader will act on it.

Instruct it explicitly not to write prose paragraphs in this mode. The most common
failure here is the model producing a narrative summary under minutes headings.

---

## 13. USER INTERFACE

Single page, no framework, no bundler, no CDN references. Everything served locally.

**Elements:**
1. Drop zone — accepts drag-and-drop, and opens a file picker on click. Shows filename,
   size and detected duration once selected.
2. Mode dropdown — `Summary` / `Minutes` / `Summary & Minutes`. Default
   **`Summary & Minutes`**, at the operator's request.

   That third mode was also added at the operator's request, overriding the
   prohibition that used to sit in §16. It is not a second pipeline:
   everything up to the final reduce is identical for the two modes, so it
   runs the same consolidated notes through a second final reduce and writes
   two files. One extra call. Its value is still `both` on the wire.
3. Model dropdown — `High Quality: Qwen3.8-27B-UD-Q4_K_M` /
   `Low Quality: Qwen3.8-27B-UD-IQ3_XXS`, with a tooltip saying Low Quality suits
   machines with about 8GB of video memory and High Quality about 16GB or more.

   The default is **detected**, not configured: VRAM is read at startup and
   Q4_K_M chosen at 15GB or above (15, not 16 — cards sold as 16GB report as
   little as 15.8GB once the driver has taken its share). The user can override
   the default; detection picks it, it does not overrule anyone. The time
   estimate updates with the choice, since the two differ by roughly 2x.

   Both models ship. `llm.model` and `llm.cpu_ffn_regex` default to `"auto"`;
   an explicit value in `config.json` still wins.
4. Participant count — optional numeric input, *"How many people spoke? (optional —
   leave blank if unsure)"*. Blank means auto-cluster then prune (§8.1); a number is
   passed straight to the clusterer and distance thresholding is skipped, which is much
   the most reliable path on a long recording.

   This is the **only** UI control beyond the two above. §16's ban on settings screens
   covers inference parameters; it does not cover metadata about the recording, and
   "how many people were in this meeting" is something the user knows and the clusterer
   cannot reliably work out.
5. Process button — disabled until a file is selected.
6. Cancel button — visible only while a job runs.
7. Progress area — current stage, percentage, elapsed time, live ETA, scrolling log.
8. Result area — a button that opens **this meeting's own folder**, plus the rendered
   Markdown inline, with a tab per document (`Summary` / `Minutes` / `Transcript`).

   There is one transcript tab, not two. Once names have been applied the named
   transcript is the one worth reading, so it takes the slot; the raw labelled file
   stays on disk, which is the whole point of never overwriting it.
9. Speaker naming panel — appears with the results. One row per detected speaker showing
   the **three longest things that speaker said**, each with a play button, plus a name
   box.

   Hearing a voice is how you identify someone; reading a line is not. Diarization
   labels are anonymous and the reduce stage can only infer a name where the transcript
   happens to contain evidence, so on many recordings the labels stay anonymous no
   matter how good the model is. The person who was in the meeting knows in seconds.

   Applying names rewrites the transcript into `<name>_transcript_tagged.md` — **never
   over the raw one**, which is the only record of what diarization actually decided —
   and rewrites the documents by substitution. It is **instant** (measured at 38ms), not
   a model call: typing a name must not cost minutes. Each document is kept as the model
   wrote it, labels intact, so a corrected spelling on a second pass still has something
   to replace.

   A speaker label in the rendered transcript is clickable; it jumps to that speaker's
   row here and focuses the name box.
10. Rebuild buttons on the finished page — *"Write minutes"* / *"Rewrite the summary"*.
   This is where a model call belongs: getting the other document after a single-mode
   run, or having a document re-written now that the speakers have names. One final
   reduce over the cached notes, not the recording again.
11. History list — every meeting still in `output\` with a `*_transcript.md`. The files
   are the record; a meeting deleted from the folder disappears from the list. Opening
   one restores the finished page with its documents, transcript and naming panel, name
   fields pre-filled from what was applied last time.

12. About overlay — reachable from the header **in every state the app can be in**,
   including mid-job. Shows built-by, version and the repository, and holds the
   *Check for updates* button.

   The update flow is `UPDATE_BUTTON.md` §1, with the download, unpack and verify done
   in Python before anything is replaced, so the batch script only has to wait, copy and
   restart. Every failure path leaves the installed app exactly as it was, and
   `config.json` is never overwritten — `load_config` merges new keys in from
   `_DEFAULTS`, so a stale config loses nothing.

**Reattaching.** Closing the tab must not orphan a job. On load the page asks
`/api/current` and, if something is running, jumps straight to the progress view and
attaches to its event stream — which is also the only way back to the Cancel button
after a reload. Elapsed time comes from the server, not from when the tab opened.

Cancelling means different things for the two kinds of job: a cancelled run has nothing
behind it and returns to the upload form, a cancelled rebuild still has a finished
meeting behind it and returns to that.

**The time remaining must be counted from the work left, never extrapolated from the
bar.** `elapsed / percent` looks reasonable and lies: the bar's position inside an LLM
call is a deliberately asymptotic curve, not a measurement, and dividing by it reported
"about 6 seconds left" with twenty minutes still to run. What remains is a known number
of LLM calls of known kinds, plus whatever pipeline stages have not started. Price the
calls from measured tokens-per-call and tokens-per-second — the live rate of the call in
flight, the stored median for the rest — and price not-yet-planned stages from seconds
per audio-hour. Both live in `calibration.json`.

**Progress inside a single LLM call** must not be `tokens / max_tokens`. `max_tokens` is
a cap, not an expectation — the reduce cap is 8000 and a real document is 1500–2500 — so
that formula crawls to a third and then jumps. Take the greater of the token fraction and
`1 - e^(-elapsed/expected)`, where `expected` is the measured cost of that stage on this
machine. The asymptote is the point: however wrong the expectation is, the bar cannot
stall and cannot overshoot.

The mode selects which final reduce prompt is loaded. It changes nothing else in the
pipeline. Include the mode in the output filename: `<name>_minutes.md` /
`<name>_summary.md`; `Both` writes both.

**Set expectations from the actual duration**, computed once ffmpeg reports it. Do not
hard-code a sentence about 4 hours, and do not use a minutes-per-audio-hour constant
either — that shape is wrong for the LLM stages and it showed: a 90-second recording was
quoted at 2 minutes and took 10. Writing a document costs roughly the same whether the
meeting ran twenty minutes or two hours, because the model still produces a whole
document, so a purely proportional figure collapses on short recordings.

Estimate from tokens, which is what the work actually is:

```
transcript tokens = audio_minutes x estimate.transcript_tokens_per_audio_minute
map calls         = transcript tokens / (target_tokens - overlap_tokens)
+ group reduces if that exceeds group_reduce_threshold, + one or two documents
seconds           = sum over calls of (measured tokens per call / measured tokens per second)
                  + (convert + transcribe + merge) x audio_hours
                  + estimate.fixed_overhead_s, all x (1 + estimate.buffer_fraction)
```

`transcript_tokens_per_audio_minute` is **206**, measured with the shipped tokenizer over
a real 3h25m transcript (a 34-minute one gives 180; the longer figure is the shape of
recording this is for). Transcription, conversion and the merge genuinely are
proportional to length, so those stay on seconds-per-audio-hour.

Everything else is measured on the machine it is running on and stored in
`calibration.json`. Two numbers per stage:

- **tokens per second** — from the **most recent run only**. It is the most
  hardware-dependent number in the pipeline, so after a card is changed every older
  sample describes a machine that no longer exists. One completed run is enough to
  notice an upgrade.
- **tokens per call** — a median over recordings of comparable length (within 3x),
  falling back to all of them. Document length is a property of the meeting, not the
  hardware.

**Before the first completed run on a machine, show no number at all.** There is nothing
honest to say: the LLM stage varies by several times across cards, and a guess that is
five times out is worse than admitting ignorance. Say so plainly and note that later
recordings will be estimated properly.

Display, once measured: *"This recording is 6h 12m. Processing will take roughly 55
minutes. You can leave this window open and come back."*

**Progress via Server-Sent Events** on `GET /events/{job_id}`. Stage weights:

| Stage | Share |
|---|---|
| Upload | 2% |
| Convert | 3% |
| Transcribe + diarize | 55% |
| Merge | 2% |
| Map | 28% |
| Group reduce | 4% |
| Final reduce | 6% |

Within the map stage, report `chunk i of N` — on a long recording this is the only signal
the user has that anything is happening.

**Never show a raw exception to the user.** Map failures to plain sentences; write the
full traceback to `temp\job.log` and offer a "Copy diagnostic info" button.

Visual design: clean, calm, high-contrast, system font stack, generous whitespace. This is
a utility, not a showcase. Nothing requiring an external asset.

---

## 14. CANCELLATION AND CLEANUP

This is a v1 requirement. A user who cannot stop an hour-long job will force-quit the
window, leaving orphaned processes holding VRAM until reboot.

Cancel means different things per stage:

**During ffmpeg or whisper-cli** — these are child processes. Spawn with
`creationflags=subprocess.CREATE_NEW_PROCESS_GROUP`, retain the PID, and on cancel run
`taskkill /F /T /PID <pid>`. The `/T` matters; it kills the process tree.

**During diarization** — check a cancellation flag between sherpa-onnx windows.

**During map / reduce** — llama-server is a persistent server, **not** a per-job process.
Do not kill it. Abort the in-flight HTTP request, set the flag, and have the loop check it
before dispatching the next chunk. Worst case the user waits out one chunk.

**Always, on cancel:** delete `temp\audio.wav` and `temp\upload.*`. On an 8-hour recording
that is over a gigabyte. A user who cancels three times will fill their disk and have no
idea why.

Register the identical cleanup path on:
- the cancel endpoint
- FastAPI shutdown
- `atexit`
- a `SIGTERM`/`SIGBREAK` handler

so force-quitting the console window also cleans up. Also sweep `temp\` on startup, in
case a previous run died badly.

---

## 15. `run.bat`

```bat
@echo off
setlocal
cd /d "%~dp0"

set "PYTHONDONTWRITEBYTECODE=1"
set "PATH=%~dp0bin;%PATH%"

if not exist "temp" mkdir temp
if not exist "output" mkdir output

echo Starting Meeting Summariser...
start "" http://127.0.0.1:8000
"%~dp0runtime\python.exe" -m uvicorn server.main:app --host 127.0.0.1 --port 8000
```

Requirements:
- Check the port is free before binding; if not, try the next few and open the browser at
  whichever succeeded.
- If `runtime\python.exe` or any model file is missing, print a plain message naming the
  missing file and `pause` so the window does not vanish.
- The browser opens before the server is ready, so `index.html` must retry the initial
  connection rather than showing an error.
- Raise uvicorn's request body size limit and timeouts so a 400MB upload is not rejected.

---

## 16. THINGS NOT TO DO

- Do not add a settings screen, model picker, chunk-size slider, or temperature control.
  Those live in `config.json`.
- Do not add user accounts, a job queue, or multi-file batching.
- Do not add a fourth output mode. `Summary` / `Minutes` / `Both` is the full set;
  `Both` was added deliberately (see §13) and costs one extra reduce call, not a
  second pipeline.
- Do not use `localStorage` or `sessionStorage`.
- Do not add any dependency that pulls in torch, transformers, or the HuggingFace hub
  client. If you find yourself needing one, stop and report it instead.
- Do not reference any external URL from the frontend — no Google Fonts, no CDN, no
  favicon fetch. The repository link in the About overlay is a plain `<a>` the user may
  click; it loads nothing, so the page still renders identically with no network. The
  update check itself goes through the server, never from the page.
- Do not bind the server to anything except `127.0.0.1`.
- Do not write to `%APPDATA%`, `%USERPROFILE%` or the registry. Everything stays inside
  the app folder so the whole thing is portable and deletable.
- Do not read whole audio files or whole uploads into memory. Stream everything.
- Do not delete the merged transcript after producing the summary. Keep it.
- Do not hard-code any assumption that a recording is 4 hours long.

---

## 17. BUILD ORDER

Work in this sequence and confirm each stage before moving on:

1. Verify all binary flags per section 1. Write `BUILD_NOTES.md`.
2. FastAPI skeleton + static frontend + SSE plumbing + streaming upload, with a fake
   30-second job that emits progress. Get the UI, progress bar and cancel button working
   against the fake.
3. ffmpeg conversion with real progress parsing and duration extraction.
4. whisper.cpp transcription with real progress parsing.
5. sherpa-onnx diarization.
6. Merge — then **eyeball `output\*_transcript.md` against the audio before going
   further.** Do not proceed until turn boundaries look right. Every downstream stage
   inherits this stage's errors, and a summary built on wrong attribution reads perfectly
   fluently while being confidently wrong.
7. llama-server lifecycle, `/tokenize`-based chunking.
8. Two-tier map/reduce. Verify on a short recording.
9. Three-tier group reduce. **Verify by forcing `group_reduce_threshold` down to 2** on a
   short recording, so you exercise the path without waiting for an 8-hour file.
10. Cancellation and cleanup across all stages.
11. Error message polish.

---

## 18. ACCEPTANCE TESTS

The build is done when all of these pass:

1. A 5-minute two-speaker recording produces correct output end to end in **both** modes.
   Minutes mode must produce checkbox action items, not prose.
2. A 4-hour recording completes without OOM.
3. **A 7–8 hour recording completes without OOM and without dropping content from the
   middle of the meeting.** Verify by checking that a distinctive topic discussed around
   the 4-hour mark appears in the output.
4. Forcing `group_reduce_threshold: 2` on a short file exercises the three-tier path and
   produces sane output.
5. Cancel during transcription: process tree dies, temp files deleted, VRAM released, and
   a new job starts immediately afterwards.
6. Cancel during the map loop: stops within one chunk, llama-server stays alive, a new job
   starts cleanly.
7. Force-quit the console mid-job: no orphaned `whisper-cli.exe` or `llama-server.exe`
   remains in Task Manager.
8. **With the network adapter disabled**, a full job runs successfully.
9. Copy the entire folder to a different drive letter and run it: works unchanged.
10. Feed it a corrupt/non-audio file: plain-language error, no traceback in the UI, app
    remains usable.
11. Setting `diarization.enabled: false` runs the full pipeline with unattributed output.
12. Setting `pipeline.concurrent_diarization: false` produces the same merged transcript
    as the concurrent path on the same input.
13. Killing the diarization process mid-run does not abort transcription; the job finishes
    with unattributed output.
14. Names, figures and dates in the output match the transcript. Spot-check ten.
15. In minutes mode, every action item owner appears by name or label in the transcript.
    No invented owners, no invented deadlines.

---

## 19. REPORT BACK ON COMPLETION

- Any flag from this document that did not exist in the shipped binaries, and what you
  used instead.
- Which mechanism actually controls thinking in this build (§11.2).
- Peak VRAM during the LLM stage, and your recommendation for tuning `cpu_ffn_regex`.
- Measured tokens/second during map calls.
- Wall time broken down by stage, for both a 4-hour and an 8-hour recording.
- Diarization time at 4h vs 8h, so the scaling concern in §8 can be assessed.
- Final unzipped folder size.
