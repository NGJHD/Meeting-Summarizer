# BUILD_NOTES.md

Flag verification, deviations from `CLAUDE.md`, and measurements.
Everything here was checked against the binaries actually present in `bin\`.

**Complete through step 11** of CLAUDE.md section 17: convert → transcribe + diarize →
merge → chunk → map → group reduce → final reduce, with cancellation and error handling.
Verified end to end on a 3h25m recording in both modes.

CLAUDE.md has been amended per DIARIZATION_FIX.md part A (§0 duration wording, the §9/§10
cross-references, the §8 opening) and parts D2/D3/D4 (§8 rewritten as three stages, new
§8.1 on pruning and degradation, §13 participant count).

---

## 1. Component versions

| Component | Version / build | Source |
|---|---|---|
| llama.cpp | `b10797`, `win-cuda-12.4-x64` | ggml-org/llama.cpp releases |
| whisper.cpp | `b4938` (2026-08-20), `cublas-12.4.0-bin-x64` | ggml-org/whisper.cpp releases |
| ffmpeg | `n9.0-latest-win64-gpl` | BtbN/FFmpeg-Builds |
| CPython | 3.12.14 (`20260901`, install_only) | astral-sh/python-build-standalone |
| sherpa-onnx | 1.13.7 (`cp312-win_amd64` wheel) | PyPI |
| LLM | `Qwen3.8-27B-UD-Q4_K_M.gguf` 16.46 GB **and** `-IQ3_XXS.gguf` 10.93 GB, arch `qwen35`; picked by VRAM (§9j) | unsloth/Qwen3.8-27B-GGUF |
| ASR | `ggml-large-v3-turbo.bin` (fp16), 1.62 GB | ggerganov/whisper.cpp |
| VAD | `ggml-silero-v5.1.2.bin` | ggml-org/whisper-vad |
| Segmentation | `sherpa-onnx-pyannote-segmentation-3-0` → `segmentation-3.0.onnx` | k2-fsa/sherpa-onnx |
| Speaker embedding | `nemo_en_titanet_large.onnx` → `speaker-embedding.onnx` | k2-fsa/sherpa-onnx |

CUDA **12.4** was chosen over 13.3 deliberately: it works with a wider range of installed
display drivers, and the display driver is the only external dependency we are permitted
to assume (section 0.5).

---

## 2. Flag verification

### `whisper-cli.exe --help` — every flag in section 7 exists

| Section 7 flag | Status |
|---|---|
| `-m`, `-f`, `-l`, `-of`, `-t` | present |
| `-pp` | present |
| `--vad`, `--vad-model` | present |
| `--dtw large.v3.turbo` | present; preset string accepted verbatim |
| `-oj` | present — **but not used, see below** |

`--dtw` preset values were probed directly: `large.v3.turbo` is **accepted**,
`large-v3-turbo` (hyphens) is **rejected**. The document's spelling is correct.

### `llama-server.exe --help` — every flag in section 11.1 exists

`--ctx-size`, `--n-gpu-layers`, `--override-tensor`, `--flash-attn`, `--cache-type-k`,
`--cache-type-v`, `--jinja`, `--threads`, `--batch-size`, `--ubatch-size`, `--parallel`,
`--host`, `--port`, `--no-webui` — all present with the documented syntax.
`--flash-attn` takes `[on|off|auto]` exactly as written. Nothing needed substituting.

---

## 3. Deviations from CLAUDE.md

### 3.1 `-oj` → `-ojf` (required)

Section 7 specifies `-oj`. Plain `-oj` emits **segment-level JSON only**. The merge stage
in section 9 needs per-word timing, which requires `-ojf` (`--output-json-full`).
Changed to `-ojf`.

Note that even `-ojf` gives **token**-level, not word-level, detail: whisper emits
sub-word pieces (`" Pen"` + `"cil"`). `transcribe.parse_json` reassembles words by
treating a leading space as a word boundary.

### 3.2 `--dtw` silently does nothing without `-nfa` (required)

With the documented flag set, every token came back with `t_dtw: -1` — DTW produced no
alignment at all, silently. The cause is that **this build defaults `--flash-attn` to
`true`**, and DTW needs the cross-attention weights that flash attention does not expose.

Adding `-nfa` populates `t_dtw` correctly. Measured cost of `-nfa` on 1062 s of audio
(RTX 3080): **29 s → 38 s, about +31%**. Kept, because section 7 makes DTW a hard
requirement of the merge stage. Switchable via `whisper.dtw` in `config.json`.

This is the failure mode section 7 warns about generally: it does not error, it just
quietly produces worse output.

### 3.3 Token timestamps are in VAD-compressed time (required workaround)

**This is the most consequential finding of the build.** With `--vad` enabled,
whisper.cpp remaps *segment* timestamps back onto the original audio timeline but leaves
*token* timestamps in the compressed timeline where the silence has been removed.

Observed on a test clip:

| | segment offset (original) | first token offset | `vad_start` from log |
|---|---|---|---|
| seg 1 | 4530 ms | 1840 ms | 1.79 s |
| seg 2 | 9390 ms | 4080 ms | 4.07 s |
| seg 3 | 12220 ms | 6440 ms | 6.41 s |

Diarization works on the original audio, so comparing raw token times against speaker
turns would misattribute **every word**, by an error that grows with the amount of
silence removed — worst on exactly the long recordings this app targets. It would not
have failed loudly; it would have produced a fluent, confidently wrong transcript.

Fix: whisper-cli logs the mapping it used, one line per retained speech region:

```
whisper_vad: vad_segment_info: orig_start: 4.48, orig_end: 6.56, vad_start: 1.79, vad_end: 3.87
```

`transcribe.VadTimeline` parses these and maps every token time back, interpolating
linearly within a region and clamping to the nearest edge for a token that lands in
removed silence. Verified: feeding the region table the observed `vad_start` values
reproduces whisper's own segment offsets to within 0.02 s.

### 3.4 `-mc 0` added — whisper loses all punctuation without it (required)

whisper conditions each decoding window on the text of the previous one. On this
recording it drifted into an unpunctuated lowercase style within the first minute, and
the carried context then locked that style in **for the entire 3 hours 25 minutes**.

Measured over the same first 15 minutes of audio:

| `--max-context` | capital letters | full stops | commas |
|---|---|---|---|
| default (carry context) | **0** | 10 | **0** |
| `-mc 0` | 179 | 71 | 84 |

The first utterance is mis-cased either way — it begins mid-sentence — but with `-mc 0`
the model recovers immediately, and without it never does.

This is not cosmetic. The reduce stages depend on sentence boundaries to quote accurately
and on capitalised proper nouns to attribute speaker names, and section 12's shared rules
ask the model to infer identities from exactly that evidence. `-mc 0` is now the default,
exposed as `whisper.max_context`.

It is also the standard mitigation for whisper's long-form failure mode generally: with
context carried, one hallucinated or malformed window propagates into its successors.

### 3.5 Progress and VAD info go to **stderr**, not stdout

Section 7 says `-pp` "prints progress to stdout". It does not — both the progress lines
and the `vad_segment_info` lines go to **stderr**. Confirmed by stream-splitting.
Both parsers read stderr.

### 3.6 `bin\` is split per engine (required)

Section 3 shows `whisper-cli.exe` and `llama-server.exe` flat in `bin\`. They cannot
share a directory: both builds ship `ggml.dll`, `ggml-base.dll` and `ggml-cuda.dll`, and
the files **differ** (`ggml.dll` is 86016 bytes in the llama build, 66560 in the whisper
build). Flattening them means one overwrites the other and at least one engine loads a
mismatched DLL.

Layout used:

```
bin\
  ffmpeg.exe
  cublas64_12.dll  cublasLt64_12.dll  cudart64_12.dll   <- identical in both, shared
  llama\    llama-server.exe, llama-tokenize.exe + its ggml DLLs
  whisper\  whisper-cli.exe + its ggml DLLs
```

Windows searches an executable's own directory first, so each engine gets its own ggml.
The three CUDA runtime DLLs are byte-identical between the builds, so one shared copy
sits in `bin\`, saving about 575 MB.

**Consequence:** `bin\` must be on `PATH` for the child processes, or `ggml-cuda.dll`
cannot resolve `cudart64_12.dll` and **silently falls back to CPU**. Measured: 14.5 s vs
1.4 s on a 16 s clip, roughly 10x. `config.child_env()` sets this for every spawn, and
`run.bat` sets it too.

### 3.7 Diarization runs in a child process — sherpa-onnx holds the GIL (required)

**This is the finding that actually broke the build, and it is worth reading in full
before changing anything in this area.**

Running stages 2 and 3 concurrently as section 7 asks, with diarization on a thread,
**deadlocks**. Observed on the 3h25m file: whisper-cli's CPU time froze at 135.8 s and
stayed there — six consecutive samples over a minute, not a single tick — while the
diarization thread ran at full tilt. The GPU sat at ~10%.

The chain:

1. `sherpa_onnx.OfflineSpeakerDiarization.process()` holds the CPython GIL for its
   **entire** duration. Measured directly: during a 166 s call, a competing Python thread
   ticking every 50 ms managed **2 ticks instead of ~3,300**, with a single 166.31 s gap.
2. So the Python thread draining whisper-cli's stderr pipe cannot run.
3. A Windows pipe buffers ~64 KB. whisper-cli's stderr for this file is **693 KB** across
   2,498 `vad_segment_info` lines.
4. whisper-cli blocks on write and never finishes.

The pipe fills within seconds of diarization starting, so the two stages that section 7
says should run in parallel instead stop each other dead. Neither process errors; the job
simply hangs forever.

Two independent fixes, both kept because each closes the failure on its own:

- **`server/diarize_worker.py`** runs diarization in a separate process, so no amount of
  native GIL-holding can affect the server. It also makes cancellation a process kill
  rather than a cooperative flag, and it is why `min_duration_on`-style tuning can never
  wedge the UI.
- **whisper's stderr goes to a file, not a pipe** (`transcribe._follow_progress` tails
  it). Pipe backpressure disappears entirely: a slow or starved reader now costs only
  progress resolution, never the transcription.

The general lesson for the remaining stages: any pybind11 extension may hold the GIL, so
never put a native call and a pipe reader in the same process and assume both run.

### 3.7a Diarization split into segment / embed / cluster, with a cache

Per DIARIZATION_FIX.md part D2, `OfflineSpeakerDiarization` is no longer used. It does
segmentation, embedding and clustering in one call, so every change to a clustering
parameter re-pays the whole cost — which is why BUILD_NOTES previously (wrongly)
concluded a `max_speakers` guard was not viable. It is trivially viable once the
expensive stages are cached.

| Stage | Implementation | Cost on 3h25m |
|---|---|---|
| Segment | `segmentation-3.0.onnx` under **onnxruntime**, powerset decoding | ~2.5 min |
| Embed | sherpa `SpeakerEmbeddingExtractor`, one vector per (window, local speaker) | ~31 min |
| Cluster | sherpa `FastClustering` + own pruning | ~7 s |

Both expensive stages are cached to `temp\embeddings-<key>-seg.npz` and `-emb.npz`,
keyed on file size plus the head and tail of the audio and on the model paths. Cached
separately, so losing the long embedding stage to a cancel does not also discard
segmentation.

Notes on the implementation:

- **onnxruntime was added as a dependency.** sherpa bundles `onnxruntime.dll` but does
  not expose the segmentation model to Python on its own; only the combined
  `OfflineSpeakerDiarization` uses it. The PyPI wheel is a plain compiled wheel with no
  torch in its tree, and it coexists with sherpa's bundled copy in one process.
- The algorithm follows the reference `speaker-diarization-onnx.py` shipped inside the
  sherpa segmentation model tarball (kept at `tools\ref\`), including the powerset
  decoding and the frame-count reconstruction.
- **Clustering is sherpa's `FastClustering`, not hand-rolled AHC.** Part D2 suggested
  writing our own; measured, `FastClustering` does 16,000 vectors in 6.5 s, and an
  average-linkage AHC in numpy over the same set needs a 16k x 16k distance matrix
  (~1 GB). The cap and pruning that part D2 actually wants are post-hoc operations on
  the labels and embeddings, so they work identically on top of it.
- The per-frame gather in the embed stage was originally a Python loop over
  windows x speakers x frames — about 22 million iterations, which dominated the stage.
  Vectorising it with `np.diff` run-boundary detection cut embedding by roughly 30%.

### 3.8 Speaker embedding model: TitaNet-large, threshold 0.7

Section 8 names `speaker-embedding.onnx` without specifying which model, and section 4
sets `cluster_threshold: 0.5`. On a known two-speaker clip, `wespeaker_en_voxceleb_CAM++_LM`
never resolved to two speakers at any threshold:

| Embedding model | th 0.5 | th 0.6 | th 0.7 | th 0.8 |
|---|---|---|---|---|
| wespeaker CAM++_LM | 5 | 5 | 4 | 4 |
| **nemo_en_titanet_large** | **2** | **2** | 3 | **2** |
| 3dspeaker campplus zh_en adv | 3 | 3 | 3 | 2 |
| wespeaker resnet34_LM | 3 | 2 | 1 | 1 |

TitaNet-large is used (97 MB). With it, the stable plateau on the same clip runs
0.6–0.8, so `cluster_threshold` is set to **0.7**, the middle of that plateau, rather
than the document's 0.5 which sat on a cliff edge.

To swap models, replace `models\speaker-embedding.onnx` — no code change.

### 3.9 Overlapping diarization segments (merge correctness)

pyannote emits overlapping segments for concurrent speech, e.g.
`33.04–42.98 spk 5` overlapping `33.93–34.29 spk 0`. A plain "last segment whose start
precedes this word" lookup returns the short overlapping segment for a word at 35.0 s,
finds it does not contain the word, and falls through to inheriting the previous
speaker — when the correct answer was available. `merge.assign_speakers` scans back
across all segments that could still contain the instant and takes the latest-starting
genuine match.

### 3.10 Turns also break on a 2-second pause

Section 9 groups words into turns purely by speaker. That is right when diarization
works, but when it is disabled or fails, every word carries the same speaker and the
whole recording becomes **one turn** — 27,547 words of it on the test file. Section 10
requires the chunker to split only on turn boundaries, so a single turn that size cannot
be chunked at all, and the LLM stages would have nothing to work with.

`merge.TURN_GAP_S` (2.0 s) also ends a turn when the same speaker pauses for longer than
that. It does not invent a speaker change, it keeps turns to a chunkable size, and it
reads better in the transcript regardless.

### 3.11 Sparse cluster ids are renumbered

sherpa returns raw cluster indices, which are not contiguous: a two-speaker clip came
back as clusters 0, 2 and 6. Labels are renumbered by order of first appearance, so a
reader does not see `SPEAKER_06` in a three-person meeting and assume four people were
lost.

### 3.12 Additions to `config.json`

Section 4's config is reproduced as written, plus:

- `whisper.dtw` (bool) — turn DTW and its `-nfa` cost off if needed.
- `llm.threads` — section 11.1 hard-codes `--threads 8`; on the 12-core target that
  competes with nothing else at that point in the pipeline, so it is exposed as a value.
- `thinking.reduce_effort` — see section 5 below.

Section 7's command line shows `-t 8` while section 4's config says `threads: 6`. The
config value wins. Both whisper and diarization default to **5** here, not 6: the target
machine has **12 logical processors**, and section 7's own rule caps their sum at
cores − 2 = 10.

### 3.13 `server\config.py` added

Not in section 3's file list. Holds paths, config loading and `child_env()`, which are
needed by every other module and would otherwise be duplicated.

---

## 4. Runtime and packaging notes

- **Windows job object.** `atexit` and signal handlers do not run when a console window
  is force-quit, which would strand `whisper-cli.exe` or `llama-server.exe` holding VRAM
  until reboot (acceptance test 7). `jobs.install_kill_on_close()` creates a job object
  with `KILL_ON_JOB_CLOSE` and assigns every spawned child to it; the OS then terminates
  them unconditionally when this process dies for any reason. `taskkill /F /T` remains
  the path for an ordinary cancel.
- **SIGINT is deliberately not handled.** uvicorn installs its own handler and uses it to
  run a clean shutdown; overriding it leaves Ctrl+C unable to stop the server. `SIGTERM`
  and `SIGBREAK` are handled.
- **Uploads** are streamed from the raw request body, not multipart, so nothing buffers
  the file a second time. uvicorn imposes no body-size limit of its own.
- **`pypdf`** is pulled in transitively by fastapi 0.141.1. It is unused and harmless.
- Nothing in the dependency tree imports torch, transformers, or the HuggingFace hub.

---

## 5. Thinking control (section 11.2) — resolved from the chat template

The GGUF's embedded chat template settles this without needing a server experiment:

```jinja
{%- if enable_thinking is undefined or enable_thinking is true %}
    {%- set resolved_reasoning_effort = reasoning_effort|default('xhigh') %}
    {%- if resolved_reasoning_effort == 'high' %}
        {%- set resolved_reasoning_effort = 'xhigh' %}
    {%- if resolved_reasoning_effort not in ('xhigh', 'medium', 'low') %}
        {{- raise_exception('Unexpected reasoning effort ...') }}
...
{%- if enable_thinking is defined and enable_thinking is false %}
    {{- '<think>\n\n</think>\n\n' }}
```

- **`chat_template_kwargs.enable_thinking` is the mechanism**, exactly as section 11.2
  assumes. Setting it `false` makes the template emit a pre-closed `<think></think>`
  block, so the model has nothing to reason into.
- **`reasoning_effort` is real and supported** on this model, and does default to
  `xhigh` — confirming section 11.2's claim about reasoning cost. Accepted values are
  **`xhigh`, `medium`, `low` only**. `high` is silently remapped to `xhigh`.
- Any other value raises a Jinja exception, which surfaces as **HTTP 500** from
  llama-server. The value must therefore be validated before dispatch, not passed
  through from config unchecked. Exposed as `thinking.reduce_effort`.
- `--reasoning-budget` exists at server level and defaults to `-1` (unrestricted).
  Section 11.1 advises leaving it unset on the grounds that per-request control
  governs. **That was measured and is wrong** -- see section 9t. Per-request control
  over the *budget* does not exist in this build: `enable_thinking` and
  `reasoning_effort` are per-request, the budget is not. The flag is now set.

**Now verified against a running server.** The round-trip check section 11.2 asks for:

| request | `reasoning_content` | completion tokens | `content` |
|---|---|---|---|
| `enable_thinking: false` | 0 chars | 2 | `ACK` |
| `enable_thinking: true`, effort `low` | 73 chars | 22 | `ACK` |
| effort `turbo` (invalid) | — | — | **HTTP 500** from the Jinja template |

So `chat_template_kwargs.enable_thinking` is confirmed as the mechanism, the effort
values are confirmed as `xhigh` / `medium` / `low` only, and an invalid value really does
fail the request rather than being ignored — `llm.VALID_EFFORT` clamps before dispatch.

One difference from what section 11.2 assumes: with `--jinja`, llama.cpp returns the
reasoning in a **separate `reasoning_content` field**, not inline `<think>` tags in
`content`. `content` is therefore already clean. `llm.strip_thinking` is kept anyway as
the belt-and-braces section 11.2 asks for, and it also handles a truncated stream that
leaves an unbalanced `</think>`.

---

## 6. Measurements

Dev machine: **RTX 3080 10 GB**, Ryzen 9 3900X (12C/24T), 64 GB RAM.
Target machine: **17.9 GB VRAM**, **12 logical processors**. VRAM- and thread-dependent
figures will differ and are re-measured on the target.

### Transcription throughput (whisper large-v3-turbo q5_0, CUDA, `-t 5`)

| Configuration | 1062 s of audio | Ratio |
|---|---|---|
| `-fa` (flash attn on, **no DTW**) | 29 s | 36.6x realtime |
| `-nfa` (flash attn off, DTW works) | 38 s | 27.9x realtime |
| CPU only (no `bin\` on PATH) | ~10x slower | — |

### Full pipeline — 3 h 25 m 40 s council planning meeting (189 MB MP3)

Concurrent path, whisper 5 threads / diarization 5 threads:

| Stage | Wall time | Notes |
|---|---|---|
| Convert (ffmpeg) | 10.2 s | 395 MB WAV out |
| Transcribe | 6 m 48 s | 30x realtime; 27,547 words; 2,498 VAD regions |
| Diarize | 45 m 36 s | segmentation ~6.5 m, embedding+clustering ~39 m |
| Merge | < 0.1 s | 816 turns |
| **Total** | **45 m 48 s** | 4.5x realtime |

**Diarization dominates completely** — it is roughly 6.7x the cost of
transcription, at about **12.3 minutes per hour of audio**. Section 13's estimate
(transcribe 4, diarize 1) has the ratio inverted. `MINUTES_PER_AUDIO_HOUR` in
`server/main.py` is set to 13 to match the measurement, and must rise when the LLM
stages land.

### Diarization scales poorly with threads

Same file, diarization alone:

| Threads | Wall time | Speedup |
|---|---|---|
| 5 | 45 m 36 s | — |
| 10 | 36 m 05 s | 1.26x for 2x the threads |

Sublinear — segmentation barely moved (~6 min either way) and only the embedding phase
improved. It is closer to memory-bandwidth bound than core bound.

Consequence for `pipeline.concurrent_diarization`: at the shipped 5/5 thread split,
concurrent (45.8 min) is the right default and matches section 7. But an operator who
sets both thread counts to 10 and runs **sequentially** would see roughly 6.8 min
(whisper) + 36.1 min (diarize) = **~43 min**, about 6% faster than concurrent, and
section 7 itself notes the sequential path is strictly simpler to debug.

The margin is small enough that the default is left as section 7 specifies. What is
*not* true is section 7's stated reasoning — it assumes transcription and diarization are
comparable in cost ("40 min transcribe + 30 min diarize" at 8 h). Measured, diarization
is about 6.7x transcription.

### Merge correctness on the long file

Verified on the 3h25m output: 816 turns, timestamps running 00:00:00 to 03:25:13 —
**99.8% of the true 12,340 s duration, with zero out-of-order turns**. This is the check
that confirms the VAD remap of section 3.3 is right. Had token times been left in
compressed time, the final turn would have landed far short of the true duration.

---

## 7. Automatic speaker counting on a long recording — RESOLVED

> **Status: fixed.** This section records the original diagnosis and is kept because the
> reasoning in it was partly wrong and the correction matters. The fix and the final
> numbers are in §7a. Summary: 249 speakers became 16, coverage 100%.

On the 3h25m meeting, `num_speakers: 0` (auto) with `cluster_threshold: 0.7` returned
**249 speakers** across 2,229 segments. The true figure is perhaps 10-15. The transcript
is still correctly *segmented* — turn boundaries land in the right places and the text is
right — but the labels are close to meaningless, and feeding `SPEAKER_183` to the reduce
stages is worse than feeding nothing, because section 12 asks the model to reason about
speaker identity from those labels.

The same setting is correct on short audio: the two-speaker regression clip resolves to
exactly `SPEAKER_00` and `SPEAKER_01`.

Threshold sweeps, same 15-minute slice of the real meeting (TitaNet-large):

| threshold | speakers | segments |
|---|---|---|
| 0.70 | 8 | 117 |
| 1.00 | 4 | 115 |
| 1.30 | 1 | 113 |
| 1.60 | 1 | 113 |

The usable band is narrow, and it moves with recording length: 0.70 gives a plausible 8
on a 15-minute slice and an absurd 249 on the full 3h25m of the same meeting. That is
inherent to agglomerative clustering against a fixed distance threshold — as more
material arrives (different speakers, mic distances, room noise) the embedding cloud
spreads and a fixed cut-off fragments it.

Raising the threshold helps but does not solve it. Re-run on the **full** file:

| threshold | speakers (full 3h25m) |
|---|---|
| 0.70 | 249 |
| 1.00 | 87 |

87 is still roughly six times the plausible count, and 1.30 already collapses a
15-minute slice to a single speaker — so there is no value that is both safe on short
audio and correct on long. Tuning the threshold is not the fix.

**What this section originally concluded, and where it was wrong.**

It recommended setting `diarization.num_speakers` by hand and rejected a `max_speakers`
guard as unaffordable, on the grounds that "sherpa exposes no way to re-cluster without
recomputing embeddings, so this would mean paying the ~40-minute cost twice."

That was correct about `OfflineSpeakerDiarization` and wrong as a conclusion. The right
response was to stop using it as a single call: once segmentation and embedding are
cached (§3.7a), re-clustering costs about six seconds and the guard is trivial. Reasoning
from the constraints of one API to "not viable" skipped the step of asking whether the
API was the right one.

It also proposed raising `min_duration_on` as a mitigation. That is withdrawn — it
discards short interjections from the transcript entirely, and pruning after clustering
keeps the words and drops only the label.

See §7a for the fix and the measurements.

## 7a. Diarization results after the part D fixes

All figures from the 3h25m council planning meeting.

### Part H item 1 — cluster duration table (threshold 0.70)

282 raw clusters over 15,285 embeddings.

| rank | cluster | speech s | segments | mean s |
|---|---|---|---|---|
| 1 | 11 | 32,864 | 3,984 | 8.25 |
| 2 | 1 | 6,266 | 1,224 | 5.12 |
| 3 | 10 | 5,961 | 766 | 7.78 |
| 4 | 235 | 5,836 | 637 | 9.16 |
| 5 | 15 | 4,639 | 699 | 6.64 |
| … | … | … | … | … |
| 282 | 251 | 1.0 | 1 | 1.00 |

- **top-15 coverage: 75.9%**
- speech held by clusters under 5 s: **0.13%** across 48 clusters

**Verdict on part C: both hypotheses are partly right, and neither reading in part D1
quite fits.** There is a singleton tail (48 clusters holding 0.13%), but top-15 at 75.9%
is short of the 85% that would mean noise alone, and there are substantial clusters down
to rank ~50. So genuine fragmentation is real.

Part D1 predicted that under fragmentation "pruning will discard real speech". **It does
not**, because of the reassignment step: 4,288 embeddings from discarded clusters were
absorbed into surviving centroids and only **9** fell outside `reassign_max_distance`.
Coverage after pruning is **99.99%**. Pruning plus reassignment works under both
hypotheses; discarding alone would not have.

### Part H item 2 — threshold sweep from the cache (28 points, ~6 s each)

| threshold | raw clusters | top15% | kept | coverage% |
|---|---|---|---|---|
| 0.50 | 679 | 62.1 | 20 | 99.8 |
| 0.70 | 282 | 75.9 | 20 | 100.0 |
| 0.90 | 101 | 85.2 | 20 | 100.0 |
| 1.00 | 52 | 90.3 | 20 | 100.0 |
| 1.10 | 28 | 94.8 | 19 | 100.0 |
| 1.20 | 13 | 100.0 | 12 | 100.0 |
| 1.40 | 2 | 100.0 | 2 | 100.0 |

**Coverage is ~100% at every threshold from 0.65 up.** The result has become essentially
threshold-independent, which is exactly what part D3 predicted of a duration-independent
pruning rule and what threshold tuning could never achieve. `cluster_threshold` stays at
0.70; the plateau argument in §3.8 holds, and the sweep shows the choice barely matters
any more.

### An addition to part D3: merging near-identical centroids

Pruning alone left 20 clusters, and the transcript showed why that was still wrong: one
speaker held two or three large clusters and the labels alternated **mid-sentence**.
Pruning cannot fix that — the fragments are all large enough to survive.

The centroid distances make the fix obvious. Of the 190 pairs among 20 survivors:

- six pairs at cosine distance **0.032 – 0.150** (all the same voice)
- nothing at all between 0.150 and **0.566**
- median 0.871

A merge pass over surviving centroids therefore has a very wide safe band. Sweeping
`merge_centroid_distance` from 0.25 to 0.50 gives **16 clusters at every value**. Default
set to **0.35**, the middle of the gap.

This works where per-embedding thresholding failed for a simple reason: a centroid is an
average over hundreds of vectors, so the decision is far better conditioned than the same
decision made on one noisy 5-second embedding.

### Part H item 3 — final result at the shipped defaults

| | before | after |
|---|---|---|
| speakers | **249** | **16** |
| coverage | — | **100%** |
| merge turns | 816 | 511 |

Turns per label now read like a real meeting: chair 166, planning officer 124, then
twelve public speakers between 5 and 32 turns each. Spot-checking the transcript, the
mid-sentence label flapping is gone and one speaker self-identifies by name, which is
exactly the evidence section 12 asks the reduce stages to use.

**16 speakers with 100% coverage clears the gate in part H** ("a plausible speaker count
with coverage above 0.80").

### Part H item 4 — segments per hour, and threads actually in effect

- 12,332 analysis windows over 12,340 s = **3,600 windows per hour of audio**
  (the window shift is 1 s, so this is one per second by construction).
- 15,285 embeddings = **4,460 per hour of audio**; 1,479 more were skipped by
  `min_embed_duration`.
- 2,073 raw diarization segments before merge = **605 per hour**.
- ORT threads in effect: `intra_op_num_threads` = `diarization.threads`,
  `inter_op_num_threads` = 1, set explicitly in `SegmentationModel`. sherpa's embedding
  extractor gets the same `num_threads`. Shipped default **5**; the runs above used 10.

### Part H item 5 — time after duration-gated embedding

| stage | 10 threads | share |
|---|---|---|
| load audio | 0.7 s | — |
| segment (onnxruntime) | 111.8 s | 5% |
| embed (15,285 vectors) | 2,010.1 s | 94% |
| cluster + prune + merge | 6.5 s | <1% |
| reconstruct segments | 3.0 s | <1% |
| **total** | **2,132 s (35.5 min)** | |

**10.4 minutes per hour of audio**, against 12.3 before — the duration gate and the
vectorised gather together took roughly 15% off. Against sherpa's single-call
`OfflineSpeakerDiarization` at the same thread count (2,165 s) the restructure is
performance-neutral, which is the point: the same cost now buys a cache.

Part E's prediction is confirmed exactly. The embedding duration histogram:

| bucket | count | seconds | share |
|---|---|---|---|
| 1–2 s | 1,081 | 1,608 | 1.6% |
| 2–5 s | 3,292 | 11,530 | 11.3% |
| 5–10 s | 10,029 | 80,247 | 78.5% |
| 10–30 s | 883 | 8,830 | 8.6% |

Mean 6.69 s, median 7.64 s, **nothing below 1 s**. These are not sub-second fragments, so
duration gating removed 1,479 embeddings (8.8%) for a similar share of the time — a
modest win, kept for cluster quality, exactly as part E said to expect.

**Note on units.** `total_speech_seconds` (102,215 s) far exceeds the 12,340 s of audio
because windows overlap 10:1 (10 s window, 1 s shift), so each second of speech is
embedded about eight times. Absolute per-cluster seconds are inflated by that factor;
*ratios* — coverage, `min_cluster_speech_fraction` — are unaffected, which is why the
fraction rule is the one that governs in practice.

### Part E — execution provider

Not pursued, and it should not be without more thought. TitaNet's ONNX takes **80-dim mel
features, not a waveform** (`audio_signal` `[batch, 80, frames]`), so sherpa is doing
NeMo's feature frontend internally. Driving the model from our own onnxruntime session —
which is what a CUDA provider or batching would require — means reimplementing that
frontend (per-feature normalisation, 25 ms / 10 ms hann) and getting it bit-comparable.
That is a real correctness risk for a stage that currently works, and it would also mean
shipping onnxruntime-gpu's CUDA DLLs. Section 0.2 outranks throughput.

## 8. Step 7 — llama-server and chunking

Verified end to end on the 3h25m transcript:

- llama-server starts and reports healthy in **9–10 s** (mmap, warm page cache).
- `/tokenize` drives the chunker: **511 turns → 5 chunks**, sized 9,959 / 9,891 / 9,840 /
  9,954 / 6,288 tokens against a 10,000 target, with correct overlap (chunk 1 ends
  00:45:15, chunk 2 starts 00:44:04) and full coverage to 03:25:40.
- Chunking cost 14.1 s including model load and 511 `/tokenize` calls.
- 5 chunks is below `group_reduce_threshold` (8), so this recording takes the two-tier
  reduce. An 8-hour recording lands at ~12 chunks and takes the three-tier path, matching
  section 10's table.

llama-server is started **after** whisper has exited and shut down again as soon as
chunking finishes, so the two never hold VRAM together (section 11.1).

## 9. Peak VRAM and `cpu_ffn_regex`

**On the 10 GB dev card, the regex from section 11.1 peaks at 9,885 MiB of 10,240** — it
only just fits, which is coincidence: the regex was written for a 16 GB card.

Generation ran at roughly **5 tokens/second** on this card (78 tokens; 3.2 tok/s if the
9 s model load is counted in).

### Measured generation throughput, and why it dominates everything

A real map call on the 3h25m meeting: **10,741 prompt + 1,500 completion tokens in
821 seconds.** That is about **1.8 tokens/second**.

Almost all of that is generation, not prompt processing: 1,500 tokens at 1.8 tok/s is
833 s, which accounts for the whole call. Prompt ingestion of 10.7k tokens is
comparatively cheap. So the bottleneck is token *generation*, and generation speed on
this configuration is set almost entirely by how much of the model is executing on the
CPU.

That makes `cpu_ffn_regex` by far the most consequential setting in `config.json`. The
LLM stage is one call per chunk plus one reduce — 6 calls for this recording, ~13 for an
8-hour one — and every one of them scales directly with it.

**Recommendation for the 17.9 GB target: move layers back onto the GPU.** With ~8 GB more
VRAM than this machine had, offloading the FFN tensors of layers 40–63 is far more than
necessary. Start by narrowing the range to layers 56–63:

```
"cpu_ffn_regex": "blk\\.(5[6-9]|6[0-3])\\.ffn_.*=CPU"
```

and widen it again only if llama-server fails to allocate. Every layer moved back to the
GPU is a direct throughput win, and the map stage makes one call per chunk, so this is
the highest-value knob in the file. **Measure peak VRAM on the target before settling
it** — this figure does not transfer.

## 9a. Steps 8-11 — map, reduce, cancellation, error polish

### Two-tier map/reduce (step 8)

Verified in **minutes** mode on the two-speaker clip. The output has all six required
headings, is bulleted throughout with no narrative paragraphs, and correctly declines to
invent: "No decisions were recorded", "No action items were assigned".

Two behaviours worth recording, because they are what the thinking-on final reduce is
supposed to buy and it is rarely obvious that it does:

- The map stage's notes mislabelled the speaker, and **the final reduce caught and
  corrected it**: *"the cited evidence (SPEAKER_00 saying 'Hi, Sanjay') contradicts that
  label; the name belongs to the addressee (SPEAKER_01)."*
- It flagged `Generative Pre-Trend Transformer` as a recognition error and proposed the
  correct expansion — exactly what the shared rules in section 12 ask for.

### Three-tier and deeper (step 9)

Forced with `group_reduce_threshold: 2`, `group_size: 2`, `target_tokens: 60` on the same
clip. The log shows the recursion section 10.3 requires:

```
chunker: 5 turns -> 5 chunks
group reduce: tier 1, 5 notes -> 3 groups of up to 2
group reduce: tier 2, 3 notes -> 2 groups of up to 2
reduce: wrote ThreeTier Check_summary.md
```

Two tiers, not a hard-coded three. Content survived both: the final summary still carried
the GPT-3 example, the timestamps, and the unanswered fine-tuning question.

**A count threshold alone is not sufficient**, so `reduce._fit_for_final` measures the
real assembled final prompt with `/tokenize` and forces further tiers until it fits.
Section 10.3 says to "log the overflow and split further rather than sending it"; the
count check is a proxy for size, and notes can run long individually.

### Cancellation (step 10)

| Cancelled during | Result |
|---|---|
| Transcription | whisper-cli tree killed, diarization worker killed, temp swept, new job accepted immediately |
| Map loop | **1 second** to take effect; VRAM 9,866 → 671 MiB; no `llama-server.exe` left |

**Deviation from section 14, deliberate.** Section 14 says llama-server is "a persistent
server, not a per-job process. Do not kill it." In this build it *is* per-job — started
after whisper exits and stopped as soon as the document is written — because section 11.1
requires that 14GB of VRAM not stay allocated once a job is done, and a cancelled job is
a done job. The abort path section 14 asks for exists regardless (`LlamaServer.abort()`
closes the in-flight connection), which is why cancel lands in one second instead of
waiting out a chunk. Acceptance test 6's real requirement — "a new job starts cleanly" —
holds: the next job starts its own server.

### Streaming, and why the first full run failed

The first end-to-end attempt reached the final reduce and then **timed out at exactly
30 minutes**, three times in a row.

The bug was a single total-request timeout. There is no good value for one: the final
reduce may legitimately emit 8,000 tokens, which at the 1.8 tok/s this machine manages is
over 70 minutes, while a genuinely hung server should be caught in a couple of minutes.
Any constant is either too short for the slow-but-working case or useless for the hung
case. Worse, the retry logic then re-ran the whole generation at the same speed, so a
timeout cost three times the wait and could never succeed.

Fixed by switching the chat call to **streaming** (`"stream": true`, SSE parsed from the
response). The socket read now only blocks *between* tokens, so the timeout becomes an
**idle** timeout (`llm.idle_timeout_s`, default 300 s) and the distinction the constant
could not express falls out for free: a slow call runs as long as it needs, a dead one is
caught in five minutes.

Two related changes:

- **A timeout is no longer retried.** If the server produced nothing for five minutes,
  running the same generation again is not a recovery strategy. It fails immediately and
  keeps the transcript. Connection errors and empty completions are still retried twice.
- Token counts stream into the job log every 250 tokens with a live rate. A single map
  call is ~14 minutes on this hardware, and section 13's warning about silence reading as
  a crash applies inside a chunk as much as between chunks.

### Error handling (step 11)

Every user-facing string is a plain sentence; raw exception text only ever reaches the
`detail` field, which goes to `temp\job.log` and never to the UI. Audited across all
call sites of `job.fail()`.

Failures are graded rather than fatal wherever the work so far is still worth something:

- A map call that fails three times inserts a placeholder note for that chunk and
  continues (section 11.3) — one bad chunk of eleven must not destroy a 40-minute job.
- A group-reduce failure passes its input notes through unconsolidated.
- Any failure in the whole LLM stage keeps the transcript, sets `document_error`, and the
  UI shows *"…The full transcript below was produced successfully."*
- Diarization failure degrades to an unattributed transcript.

The "Copy diagnostic info" bundle now also carries the last 60 lines of
`llama-server.log`, since when the LLM stage fails the reason is almost always there and
almost never in ours.

## 9b. Acceptance tests (section 18)

| # | Test | Status |
|---|---|---|
| 1 | 5-min two-speaker, both modes | **pass** — minutes bulleted, not prose; also re-run in minutes mode on 30 min of the real meeting, see below |
| 2 | 4-hour completes without OOM | **pass at 3h25m** end to end, 2 h 12 m wall (no 4-hour file available) |
| 3 | 7–8 hour without OOM or dropped middle | **not run** — no file of that length |
| 4 | `group_reduce_threshold: 2` exercises three-tier | **pass** |
| 5 | Cancel during transcription | **pass** |
| 6 | Cancel during map loop | **pass** (see deviation above) |
| 7 | Force-quit console leaves no orphans | **implemented, not force-tested** — Windows job object with `KILL_ON_JOB_CLOSE` |
| 8 | Runs with the network adapter disabled | **pass by construction** — audited: no external URLs in the frontend, loopback-only HTTP, no writes outside the app folder. Not tested with the adapter physically off |
| 9 | Copy to a different drive letter | **not run** — only one drive on this machine. All paths derive from `config.ROOT`; nothing is absolute |
| 10 | Corrupt / non-audio file | **pass** — plain message, no traceback, app stays usable |
| 11 | `diarization.enabled: false` | **pass** — unattributed output end to end |
| 12 | Sequential path matches concurrent | **pass** — byte-identical transcripts |
| 13 | Killing diarization mid-run | **not run** — the code path is the same one exercised by test 5's cancel, which kills the worker while transcription continues |
| 14 | Names, figures, dates match the transcript | **pass** — 14 spot-checks, all grounded (§9c) |
| 15 | No invented owners or deadlines | **pass** — 15 checks, all grounded (§9c) |

## 9c. Full end-to-end run — 3h25m40s, summary mode

| Stage | Wall time | Share |
|---|---|---|
| Convert | 10.4 s | 0.1% |
| Transcribe + diarize (embeddings cached) | 355.4 s | 4.5% |
| Merge | <0.1 s | — |
| Chunk + map + reduce | 7,532.0 s | 95.4% |
| **Total** | **7,897.7 s (2 h 12 m)** | **1.56x realtime** |

Breaking the LLM stage down: 5 map calls at ~820 s each (68 min, 1,500 completion tokens
apiece at 1.8 tok/s), group reduce correctly skipped at 5 notes against a threshold of 8,
then a final reduce of 7,520 s producing a 3,877-word document.

**The LLM stage is 95% of the job on this hardware.** That is a property of the 10 GB
card, not of the design — see §9 on `cpu_ffn_regex`.

`pipeline.minutes_per_audio_hour` is set to the measured **38**. Section 13's figure of 9
is out by roughly four times: about 10 for diarization and 26 for the LLM stage. It sits
in `config.json` because it is a property of the machine and should be recalibrated on
the deployment laptop once the offload regex is tuned.

### Output quality

3,877 words, all six required sections, organised **by theme not chronologically** as
section 12 demands, with ten topic sub-headings covering the eleven agenda items.

**Test 14 — names, figures and dates.** Fourteen spot-checks, all present in the
transcript: the subdivision id `4189-26…`, `55.7` acres, Pine Ridge Road, the Bell cell
tower, the September 1 deferral, Heatherdale Road, Earthworks Landscaping, Oak Bank,
Councillor Kaczynski, the compost condition, Murdoch Road, Poplar Road, Ralph Street,
Dougal Road. **No fabrications found.**

Two behaviours worth recording, both of them section 12's shared rules working as
intended rather than by luck:

- Whisper spells one councillor's name inconsistently — 12 `Krasinski` against 14
  `Kaczynski`. The summary silently picked the majority form, which is exactly the
  instruction ("treat them as the same thing and use the most plausible form").
- Where it could not resolve a variant it said so instead of choosing:
  *"Mr. Jopling (transcript variants: Dropling, Jopling)"*.
- It also refused to overclaim on outcomes, adding: *"The working notes do not record
  explicit motion-and-vote language for most items… their outcomes are inferred… but this
  is not stated as confirmed fact."*

### Minutes mode on real content — 30-minute extract

The two-speaker clip has no action items, so minutes mode was re-run on 30 minutes of the
real meeting (00:45:00–01:15:00, four planning items). 2,247 s wall, one chunk.

Every heading present, **bulleted throughout with no prose paragraphs**, and six action
items in the required `- [ ] **Owner** — task — deadline — (HH:MM:SS)` form.

**Test 15 — no invented owners or deadlines.** Every owner, entity and date in the action
items is grounded in the transcript: Vitorik, Winnie Drives, Hadi, Adher, Kozelman,
Expert Exteriors, Earthworks, 150 Transport Road, Penner, surveyor, fire chief, building
inspector, business license, signage, and the single "90 days" deadline. **15 of 15.**

Crucially, it used the escape hatch rather than guessing: **two** items carry a named
owner, **four** are `Unassigned`, and five of six say "no deadline stated". That is
exactly the behaviour section 12 asks for — an unassigned item is a correct output, a
fabricated owner is a serious failure.

(My first pass at this check reported four names as missing. The grep patterns were
wrong, not the model — `Transport Rd` against a transcript saying `150 Transport Road`,
and so on. Re-checked with exact strings, everything is grounded.)

**Test 3 (partial) — no dropped middle.** Content from 01:32–02:24, the middle of the
recording, is present in detail (the Oak Bank variance, the 40 ft vs 50 ft house width,
the 13 affected lots). At 3h25m this takes the two-tier path; the eight-hour case that
would exercise three tiers on real audio remains untested for want of a file.

## 9d. Whisper model comparison — q5_0 stays

All four models run over the same 34m31s council recording, scored against the **official
published minutes** for that meeting. There is no verbatim reference transcript, so this
is not WER; it scores the thing that actually matters downstream — whether the proper
nouns, money and dates in the official record survive transcription. A model that is
fluent but renames the councillors is useless for minutes, and WER would barely notice.

| Model | Size | Time | Terms found |
|---|---|---|---|
| `ggml-large-v3-turbo-q5_0` | 574 MB | 57 s | **17 / 18** |
| `ggml-large-v3-turbo-q8_0` | 874 MB | 56 s | **17 / 18** |
| `ggml-large-v3-turbo` (f16) | 1.6 GB | 50 s | **17 / 18** |
| `ggml-large-v3` (non-turbo) | 3.0 GB | 128 s | 16 / 18 |

Four of the 22 terms I originally scored (CAO Draper, STARS, AMM, Seine Rat Roseau) are
absent from **every** transcript including all four models'. They are consent-agenda
items listed on paper and never read aloud, so they are not transcription failures and
are excluded from the denominator above.

**Conclusion: keep q5_0.** Larger quantisations buy nothing measurable, and full
`large-v3` is both worse and 2.5x slower — the extra 28 decoder layers do not help on
clear council-chamber audio and cost real time. q5_0 also saves 2.5 GB against f16 in a
folder that is already 19 GB.

Word-level agreement between the three turbo variants is only 85-87%, but that is
ordinary decoding variation on filler words and disfluencies, not a quality signal: the
terms that matter land identically.

`tools\asr_compare.py` reproduces this against any recording with published minutes.

### Repeated on the 4h22m planning meeting, with a better method

The 34-minute test above used a hand-written list of expected spellings, which turned out
to be too generous to itself: q5_0 rendered Giesbrecht as "geesebrick", which my list
missed entirely and scored as absent. The tool now extracts terms from the PDF
automatically — so the scoring cannot be tuned to flatter a model — and grades each into
three buckets rather than two.

4h21m51s planning meeting, **149 distinctive terms** taken from its official minutes:

| Model | Size | Time | exact | near | miss | exact % |
|---|---|---|---|---|---|---|
| `ggml-large-v3-turbo-q5_0` | 574 MB | 455 s | 119 | 24 | 6 | **79.9%** |
| `ggml-large-v3-turbo-q8_0` | 874 MB | 459 s | 118 | 24 | 7 | 79.2% |
| `ggml-large-v3-turbo` (f16) | 1.6 GB | **436 s** | 119 | 24 | 6 | **79.9%** |
| `ggml-large-v3` (non-turbo) | 3.0 GB | 1069 s | 117 | 26 | 6 | 78.5% |

*exact* = usable as written; *near* = recognisable but misspelt; *miss* = absent.

**q5_0 and f16 tie exactly.** The spread across the three turbo quantisations is one term
in 149 — noise. Full `large-v3` is again both slightly worse and, at 2.4x the time, much
slower: 28 extra decoder layers that do not help on council-chamber audio.

Five terms are missed by every model (Abdulrazzaq, Doolan, Hrycak, Joshua, Khalid) — the
attendance list, printed in the minutes but never read aloud.

**On "f16 fits entirely in VRAM":** true, but so does every option here. The largest is
3 GB against 8 GB, so all four fit comfortably and VRAM does not discriminate between
them. That argument belongs to the *LLM*, which is 6-16 GB and genuinely constrained.
f16 is 4% faster, but transcription is only ~4% of a whole job, so the saving is ~0.2%
overall — against 1 GB more in a folder that is already 19 GB.

**Decision: fp16 (`ggml-large-v3-turbo.bin`).** Chosen by the operator. It is the fastest
of the four and ties q5_0 exactly on accuracy; the cost is 1 GB more in the folder. There
was never a case for q8_0 (slowest of the three turbo quants, no better) or for
`large-v3`.

`config.json`, `server/config.py` (defaults and the required-files check) and
`DOWNLOAD_MODELS.bat` all point at it; the total download is now ~18.2 GB.

## 9e. Diarization boundaries — sentence snapping

Reported from a read of the transcript: speakers "flap" mid-sentence, with the change
landing one or two words late.

    [SPEAKER_00] ... any inquiries going up. My name
    [SPEAKER_03] is Candice Starr. I'm at 67085 Pine Ridge Road.

"My name" is Candice's. Four reported cases all had the same shape — the change fell
mid-sentence when a sentence boundary sat one or two words earlier.

The rule now applied (`merge.snap_to_sentences`): **a sentence should not span a speaker
change**, so a change falling mid-sentence is pulled to the nearest sentence boundary
within `SNAP_WINDOW` words, never crossing a neighbouring change.

Measured on the 34-minute council recording, as the share of speaker changes that begin a
sentence rather than cutting one in half:

| window | changes | clean | words moved |
|---|---|---|---|
| 0 (off) | 37 | 18.9% | 0 |
| 4 | 34 | 50.0% | 30 (0.7%) |
| **6** | **34** | **70.6%** | **68 (1.6%)** |
| 12 | 34 | 82.4% | 104 (2.5%) |

**6 chosen.** Beyond it the gain is small and the claim is large — moving a boundary a
dozen words is guessing, not correcting a lag. An asymmetric window biased backwards was
also tried, on the theory that the error is systematically late; it scored *worse*
(64.7% at 6-back/3-forward), so the lag is not purely one-directional.

Note the change count also drops 37 → 34: three spurious speaker changes disappear
entirely, not just move.

This only works where punctuation exists, which is a second reason `-mc 0` matters
(§3.4) — on the unpunctuated transcript we would have had before that fix, there is
nothing to snap to. The remaining ~30% are changes with no sentence boundary within
range; some of those are genuine interruptions.

**Validated against a second, independent metric.** "Changes that begin a sentence" is
the snap's own objective, so it could in principle be fitting its own score. A real
speaker change should also coincide with a *pause*, which punctuation does not enter
into:

| | median gap at a change | changes at a gap under 0.10 s |
|---|---|---|
| no snap | 0.00 s | 33 of 37 |
| snap | 0.06 s | 21 of 34 |

It moves boundaries toward real pauses as well, so the gain is not an artefact.

That raw median of **0.00 s** is worth noting on its own: before the snap, diarization
changes essentially never coincided with a gap between words.

A pause-aware variant was also built and measured — scoring each candidate position by
inter-word gap plus a bonus for punctuation. On this recording it was no better
(70.6% at a window of 6, identical) while moving 25% more words, so it was **not**
adopted. It may still be the right answer for poorly punctuated audio; see below.

**Where this helps least.** The snap needs punctuation to aim at, and whisper drifts in
and out of an unpunctuated style within a recording (§9i). Stretches with no punctuation
get no benefit — the originally reported "sherry petrasco i'm | 67040 pine ridge road"
case sits in one and is *not* fixed.

The "22% vs 76% of turns end with punctuation" figure first quoted here was a bad metric
and is withdrawn; see §9i. Turns end at speaker changes, so it measured this very defect
rather than punctuation quality.

`tools\merge_check.py` reproduces the measurement.

## 9f. Startup, "Both" mode, and the model downloader

**Startup (reported as hanging at "Starting up…").** Measured cold start to first healthy
response: **593 ms**, so the server was never the problem. The fault was in `run.bat`,
which opened the browser *before* starting uvicorn — the page always loaded against a
socket that was not listening and relied on the frontend's retry loop, so any real
failure was indistinguishable from a slow start.

Fixed at the source: `server\open_browser.py` polls `/api/health` and opens the browser
only once it answers, printing progress to the console meanwhile. The frontend keeps
retrying as a backstop but now shows elapsed seconds, a progress bar, and escalating
guidance at 10 s / 30 s / 90 s instead of a bare spinner.

**"Both" output mode.** Section 16 forbids "an option to produce both modes in one run".
The operator has asked for it explicitly, so it is built, and the objection the rule was
guarding against does not apply here: everything up to the final reduce — transcription,
diarization, chunking, map, group reduce — is identical for the two modes, so "Both" runs
the same notes through a second final reduce. One extra call, not a second pipeline.

**`DOWNLOAD_MODELS.bat`.** Fetches all five models with `curl.exe` and `tar.exe` (both
shipped with Windows 10 1803+). Resumes partial downloads (`curl -C -`), skips files that
are already present and large enough, and verifies size on completion. Tested by deleting
a model and re-running: restored byte-identical.

One cmd trap worth recording: a description containing parentheses — `"… (574 MB)"` —
breaks the script, because it is echoed inside a parenthesised `if` block and cmd treats
the `)` as the block terminator. The error surfaces far from the cause as
`- was unexpected at this time.` Descriptions must avoid brackets.

**Missing-model reporting.** `/api/health` already listed missing binaries and models,
but **not the 16.5 GB LLM** — a missing GGUF would only surface an hour into a job, after
transcription and diarization had run. `config.missing_files()` now resolves the LLM path
from `config.json` and checks it too, and the frontend names the missing files and points
at `DOWNLOAD_MODELS.bat`.

## 9g. Speaker naming, voice samples, and calibrated timing

### Speaker naming

After a run, the results screen lists each detected speaker with the **three longest
things they said**, each as a playable clip, plus a name box. Clips are cut with ffmpeg
at merge time — while the WAV still exists — to
`output\<name>_speaker_samples\spk<NN>_<n>.mp3`, capped at 18 s and 64 kbps mono. Sixteen
clips for seven speakers came to well under a megabyte.

Three clips rather than one because a single sample is often unrepresentative: the
longest turn may be someone reading a bylaw number aloud. Hearing three gives the voice,
the manner and the role.

Applying names:

1. rewrites the transcript into **`<name>_transcript_tagged.md`**, leaving the raw
   `<name>_transcript.md` untouched — that file is the only record of what diarization
   actually decided, and it is what you audit when an attribution looks wrong;
2. substitutes the names into the **cached map notes** and re-runs only the final
   reduce, so the names appear in the prose and against action-item owners.

The note cache is what makes this affordable: one model call instead of the 68-minute map
loop. It also independently protects a long job whose reduce stage fails.

**Known limit:** the job registry is in memory, so renaming is available until the app is
restarted or another recording is processed. The samples and notes survive on disk, but
there is currently no UI to reopen a past meeting. That is the natural next step if it
matters.

The clip endpoint takes both path segments from the URL, so it validates the filename
against `spk\d{2}_\d\.mp3` and checks the resolved path's parent is the samples folder.
Verified: traversal in either segment, an absolute path, and a non-clip filename all
return 404.

### Timing is now measured, not assumed

The old `pipeline.minutes_per_audio_hour: 38` was a constant measured on one machine, and
it was wrong in a way that could not be fixed by picking a better constant: the figure is
dominated by generation speed, which depends on how much of the model `cpu_ffn_regex`
pushes onto the CPU. A 10 GB card and a 17.9 GB card are not the same job.

`server/calibration.py` records seconds-per-audio-hour per stage after every completed
run into `calibration.json`, and later runs use the median of the last ten. Fed the
measured 3h25m run it predicts **131 minutes against an actual 132**.

The same data fixes a defect nobody had reported yet: **the progress bar was lying.**
Section 13's weights put transcription at 55% and the LLM stages at 38%; measured, the
split is 4.5% and 95%. The bar therefore raced to 62% and then crawled for an hour.
Weights are now derived from the same measured rates:

| stage | section 13 | measured here |
|---|---|---|
| transcribe + diarize | 55 | 4.3 |
| map | 28 | 49.8 |
| reduce | 6 | 41.1 |

There is also a **live ETA** in the progress panel, computed from the run's own elapsed
time against its own percentage. That needs no calibration at all and is the most honest
number available, because it already reflects this machine, this recording and these
settings.

## 9h. LLM quantisation — IQ2_XXS and IQ1_S

Same 34-minute council meeting through the real map/reduce path, minutes mode, scored
against the official minutes. The small quants fit entirely in 10 GB, so they run with
**no CPU offload at all** — which is the whole performance argument for them.

| Variant | Size | Offload (10 GB card) | map | reduce | Terms |
|---|---|---|---|---|---|
| Q4_K_M | 16.5 GB | layers 40-63 | 686 s | 1258 s | **30 / 49** |
| IQ3_XXS | 10.9 GB | layers 56-63 | 374 s | 574 s | **30 / 49** |
| IQ2_XXS | 7.3 GB | none | **49 s** | **129 s** | 25 / 49 |
| IQ1_S | 6.2 GB | none | 45 s | **FAILED** | — |

**IQ1_S is unusable.** The map call worked, but the final reduce returned an *empty
completion* three times in a row and the job failed. Thinking is enabled for the reduce;
at 1-bit the model could not produce a document at all.

**IQ2_XXS is 19x faster and measurably wrong.** The speed is startling — 32 minutes of
LLM work becomes 3 — but it fails in exactly the way section 11.1 predicts of sub-4-bit
quants, "mangling proper nouns and digits":

| | Q4_K_M | IQ2_XXS |
|---|---|---|
| Mayor | "Mayor Patrick Therrien" ✓ | *"No single name can be reliably assigned"* |
| Election rates | "$250 election day; $300 information officers; $50 per training session" ✓ | "$250 (election day information officers), $300 (election day training)" ✗ |
| Order E-21-140 | present | absent |

Ground truth is *Voting Officials $250 / Information Officers $300 / Training $50*.
IQ2_XXS keeps all three numbers and **attaches them to the wrong roles** — the most
damaging possible error in minutes, because it is fluent, specific and confidently
wrong. It also lost the Mayor's name, which Q4_K_M recovered correctly.

**IQ3_XXS is the interesting one.** It matches Q4_K_M's term count exactly, recovers
"Mayor Patrick Therrien" where IQ2 gave up, and runs about twice as fast here. But it
repeats IQ2's characteristic error on the most numerically dense item in the meeting:

> IQ3_XXS: "$250 (election-day information officers); $300 (election-day training)"

Ground truth is *Voting Officials $250 / Information Officers $300 / Training $50*. Both
sub-4-bit quants shuffle the labels; Q4_K_M does not. Since each variant ran its own map
stage, the error originates in extraction, not in the final write-up.

**Caveat on all of this: n = 1 per variant, with `temperature: 0.7` on the map calls.**
Runs are not deterministic, so a 30-vs-30 tie is a tie *within noise*, and the rate error
may or may not reproduce. These numbers separate IQ1_S (broken) and IQ2_XXS (visibly
degraded) from the rest confidently; they do **not** cleanly separate Q4_K_M from
IQ3_XXS. Several runs each would be needed for that.

**Recommendation: stay on Q4_K_M.**

The important caveat is that every timing here is distorted by the 10 GB card. What
matters is what fits on the **17.9 GB** target:

| Variant | weights | + 32k KV cache | fits in 17.9 GB? |
|---|---|---|---|
| Q4_K_M | 16.5 GB | ~19.5 GB | no — a few layers offloaded |
| **IQ3_XXS** | 10.9 GB | ~13.9 GB | **yes, whole, with headroom** |
| IQ2_XXS | 7.3 GB | ~10.3 GB | yes, whole |

So on the target the ordering changes: Q4_K_M keeps a small offload while IQ3_XXS runs
entirely on the GPU. IQ3_XXS could plausibly be several times faster *there* for output
that scored the same here.

**Re-measure both on the target before deciding.** If Q4_K_M lands anywhere near
10-15 tok/s with a light offload, keep it — the accuracy is not worth trading. If it
stays slow, IQ3_XXS is the sensible fallback, and worth several runs to check whether the
rate-attribution error reproduces. IQ2_XXS and IQ1_S are ruled out either way.

Reproduce with `tools\llm_compare.py`.

## 9i. Was the 3h25m transcript wrong? No — and I mis-measured it

Two concerns were raised: the transcript looked badly punctuated, and the run seemed
disproportionately slow. Neither holds up, and the first was my own measurement error.

### The punctuation figure was wrong

§9e reported "76% of turns end with punctuation on the council recording, only 22% on the
3h25m". That metric is confounded: a *turn* ends where the **speaker changes**, and a
speaker change landing mid-sentence is precisely the diarization defect §9e is about. The
3h25m file has 511 turns over 26,860 words (53 words/turn) against the council file's 34
turns over 4,237 (125 words/turn) — far more boundaries, so far more of them mid-sentence.
I was measuring the diarization problem and reporting it as a transcription problem.

Punctuation **density**, which is not confounded, says the transcripts are all alike:

| transcript | periods/100w | commas/100w | capitals/100w |
|---|---|---|---|
| 3h25m as shipped (q5_0) | 4.44 | 3.60 | 8.7 |
| 3h25m re-run (fp16) | **5.08** | 4.00 | 10.0 |
| council 34m | 4.72 | 4.30 | 10.6 |
| 4h22m (fp16) | 5.20 | 4.50 | 9.5 |
| GMT 3h43m (fp16) | 5.36 | 4.55 | 10.0 |

Nothing went wrong. Re-transcribing the same file with fp16 took **337 s** (against 408 s
for q5_0) and came out slightly *better* punctuated than the shipped version.

What is true: whisper drifts in and out of an unpunctuated style *within* a recording,
even with `-mc 0`. `-mc 0` prevents the permanent lock-in that §3.4 documents; it does not
prevent per-window drift. `--prompt` with `--carry-initial-prompt` was tried as a fix and
produced **byte-identical output**, so it was not adopted.

### The run was not disproportionately slow either

The comparison drawn was against "the 4h at 1258 s". That figure is the **34-minute
council** file's reduce from the quantisation test — the 4h22m recording has only ever
been transcribed here, never taken through the LLM stage.

Like for like:

There is also a labelling error in the earlier logs worth correcting: the "7,520 s"
recorded against the 3h25m reduce is the **whole LLM stage** (map + reduce), because the
timer started at the top of `produce_document`. The reduce alone was about 3,400 s.

Like for like, per hour of audio:

| | audio | map | reduce | LLM total | LLM s per audio-hour |
|---|---|---|---|---|---|
| council 34m | 0.58 h | 686 s | 1,258 s | 1,944 s | **3,381** |
| planning 3h25m | 3.43 h | 4,100 s | ~3,400 s | 7,500 s | **2,188** |

Per hour of audio the long recording was **more** efficient, not less — the short one
pays a fixed reduce cost over less material. Nothing about the 3h25m run was
disproportionate; it is simply what 1.8 tok/s looks like on a card offloading two dozen
layers, which §9j now avoids on the target.

## 9j. Model selection is detected, not configured

Both language models now ship, and which one runs is decided from the card rather than
written into a config file.

| VRAM | model | rationale |
|---|---|---|
| ≥ 15 GB | `Qwen3.8-27B-UD-Q4_K_M` (16.5 GB) | the only variant that got the election-rate table right (§9h) |
| < 15 GB | `Qwen3.8-27B-UD-IQ3_XXS` (10.9 GB) | same term recall, ~2x faster, and it fits |

**15 GB, not 16.** Cards sold as 16 GB report as little as 15.8 GB once the driver has
taken its share, and a threshold a 16 GB card fails would be worse than useless.

The UI dropdown ("Model", with a tooltip naming the 8 GB / 16 GB guidance) defaults to the
detection but does not enforce it — verified by forcing Q4_K_M on this 10 GB card, which
selected it and offloaded 47 layers.

### The offload regex is computed too

`cpu_ffn_regex` was a constant written for a 16 GB card, and it is wrong on every other
one. It is now derived from measured VRAM and the actual file size, keeping on the GPU
whatever fits and pushing down only the excess:

| card | model | layers to CPU |
|---|---|---|
| 8 GB | IQ3_XXS | 39 |
| 10 GB (dev) | IQ3_XXS | 22 |
| 15.8 GB | Q4_K_M | 13 |
| **17.9 GB (target)** | **Q4_K_M** | **1** |
| 24 GB | Q4_K_M | none |

The target therefore runs Q4_K_M essentially entirely on the GPU, against the 24 layers
the hard-coded regex would have offloaded — which is where the 1.8 tok/s in §9h came
from. Both `llm.model` and `llm.cpu_ffn_regex` default to `"auto"`; an explicit value in
`config.json` still overrides.

**On the overhead estimate.** Sizing this needs a figure for the KV cache and compute
buffers. Measuring it directly turned out to be impossible the obvious way: llama.cpp
mmaps its weights, so `/health` returns 200 with only **907 MiB** resident and VRAM fills
during inference instead. The useful consequence is that a model which does not quite fit
is *paged*, not refused — so over-offloading is the worse error, because it guarantees
CPU execution for layers that would have fitted. The headroom figure is deliberately
lean (1.5 GB) for that reason. It is an estimate; `cpu_ffn_regex` overrides it.

### Estimates are per model

Calibration records are keyed by model — a run on IQ3_XXS says nothing about how long
Q4_K_M will take — and the dropdown updates the estimate as you change it. For the 3h25m
recording: **131 min on Q4_K_M against 69 min on IQ3_XXS** (188 / 95 in "Both" mode).

## 9k. Reattach, history, instant renaming, and a bar that actually moves

Seven usability problems reported after the first real runs. All seven are fixed; two of
them were genuine design errors rather than polish.

### Closing the tab looked like losing the job

There was no way back into a running job. Reopening the page showed the upload form, the
Cancel button was unreachable, and the only way to stop an hour-long run was to kill the
console — which is exactly the force-quit the cancellation design exists to avoid.

`GET /api/current` now reports the job in flight and the page reattaches to its SSE
stream on load. The stream already replayed current status and the last 200 log lines to
a new listener, so a reattached tab is indistinguishable from one that never closed.

Elapsed time is taken from the server's own `elapsed` field on every status event rather
than counted from when the tab opened; otherwise a reload resets the clock to 0:00 while
the job is forty minutes in.

### Renaming speakers took minutes when it is a search and replace

It re-ran the final reduce. That was defensible — the model can write better prose when
it knows the names — but it is the wrong default: it made typing a name cost five
minutes, and the user reasonably stopped it and assumed something had hung.

Split into two actions:

- **Apply names** — pure substitution over the stored text. Measured at **38 ms**.
- **Rewrite the summary / Write minutes** — an explicit button on the finished page that
  re-runs one final-reduce call over the cached notes.

Substitution needs the model's own output, labels intact, or a second pass with a
corrected spelling would find nothing to replace. `samples.json` now keeps a pristine
copy of each document as generated, and the output files are rewritten from that copy
each time. Verified: SPEAKER_00 → Alice → Alicia leaves no stale "Alice" behind. For
meetings produced before this change there is no pristine copy, so the first rename
adopts what is on disk (it still carries the labels) and stores it.

### Cancel went to the wrong page

Cancelling a rewrite dropped the user on the upload form, which reads as "the meeting is
gone" — it is not, only the rewrite was abandoned. `Job.kind` now distinguishes `run`
from `generate`, `/api/current` reports it, and cancelling a `generate` returns to the
finished page. This survives a reload mid-rewrite, which is how it is most likely to be
hit.

### The transcript's speaker tags did nothing, and overlapped

Two separate faults. The naming panel's card class was `.spk`, which also matched the
`<span class="spk">` the transcript renderer emits for every turn — so 14px of card
padding and a border landed on each inline label, and the lines collided. Renamed to
`.spk-row`.

The tags now do something: clicking one scrolls to that speaker's row in the naming
panel, highlights it and focuses the name box. The transcript is where you notice you do
not know who someone is, so that is where the affordance belongs.

### History

`GET /api/history` lists everything in `output\` with a `*_transcript.md`; the files are
the record, so a meeting deleted from the folder simply disappears from the list. Opening
one restores the finished page — documents, transcript, naming panel — and the name
fields pre-fill from `samples.json`, or, if that sidecar has been deleted, by reading the
raw and tagged transcripts side by side and pairing turns on their timestamps.

### The progress bar stalled at the LLM, then jumped

The real fault, and the most interesting one. Progress inside a call was
`tokens_seen / max_tokens`. But `max_tokens` is a **cap**, not an expectation: the reduce
cap is 8000 and a real document is 1500–2500. So the bar crawled to about a third and
then leapt to 100% — precisely the reported symptom.

Progress is now the greater of two signals:

- tokens against the cap, as before; and
- elapsed against how long this stage actually takes **on this machine**, from
  `calibration.py`, approached asymptotically as `1 - e^(-t/expected)`: 63% at the
  expected time, 86% at twice it, never 100%.

The asymptote is the point. However wrong the expectation is, the bar cannot stall and
cannot overshoot; a bad estimate only changes the pace. Measured on a 34-minute meeting
regenerating its minutes:

```
0:42   4%   about 15m 58s left
2:42  17%   about 12m 57s left
5:43  34%   about 11m 17s left
7:59  44%   about 10m 16s left
10:58 55%   about  9m  3s left
```

Continuous motion and a monotonically falling ETA, against a bar that previously read
49% for ten minutes.

A rewrite job also used the full run's stage weights, so it opened at 49% — the width of
everything it was skipping — and looked stuck before it started. It now gives the reduce
stage the whole bar.

### Force-quitting the console does kill the model (acceptance test 7)

Tested the harshest case rather than the actual one: `taskkill /F` on the Python process,
which runs no `atexit` handler, no `SIGTERM` handler and no FastAPI shutdown. With
`llama-server.exe` resident and holding 10.9 GB:

```
> taskkill /F /PID 3504          (uvicorn)
> tasklist /FI "IMAGENAME eq llama-server.exe"
INFO: No tasks are running which match the specified criteria.
```

The Windows job object with `JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE` takes the children down
with the parent handle. Closing the console window is strictly gentler than this, so it
is covered.

### `tools\jscheck.py` was lying

It claimed to handle regex literals and did not — it only skipped comments — so
`/^\[([^\]]{1,60})\]/` reported a bracket mismatch that was not there. It now skips
regex literals properly, tracking character classes. (Node does exist on the development
machine, so `node --check` is the real check; jscheck is what ships.)

## 9l. An ETA built from the work left, and per-meeting output folders

### The ETA was extrapolating the progress bar, which is not a measurement

`remaining = elapsed x 100 / percent - elapsed`. Reasonable-looking, and wrong for a
specific reason: §9k had just replaced in-call progress with a deliberately asymptotic
curve, `1 - e^(-t/expected)`. That curve is a *display* device — it guarantees the bar
moves and never overshoots — and it is emphatically not a claim about completion.
Dividing by it produced "about 0m 6s left" with many minutes still to run.

Worse, the two are coupled in the wrong direction: as the asymptote flattens, the
implied total shrinks, so the ETA gets *more* confident exactly as it becomes less
informed.

Counting the remaining work instead. At any moment what is left is:

- **a known number of LLM calls, of known kinds.** `reduce.py` publishes the plan
  (`job.pending_calls`) as soon as chunking is done: N map calls, ceil(N/group_size)
  group reduces if the count is over threshold, one or two final reduces.
- **the call currently streaming**, priced from its own live token rate.
- **pipeline stages that have not started**, priced from seconds per audio-hour.

Each call is priced from two measured numbers per stage — tokens produced and tokens
per second — held in `calibration.json` per model and updated after every call
(`calibration.record_call`). Generation speed is the most machine-dependent number in
the pipeline, set by how much of the model `cpu_ffn_regex` pushed into system RAM, so it
is never assumed for longer than one call: once a call is 60 tokens and 5 seconds in,
its own rate replaces the stored median.

The handover between the two pricing schemes happens when the call plan appears. Before
that (during transcription, when all the LLM work is ahead and its size is unknown)
every stage is priced from its audio-hour rate, so the ETA does not have to leap upward
when the map loop starts.

### The bar had to share that measurement, or the two would disagree

With the ETA fixed, the bar was left pricing itself from seconds-per-audio-hour while
the ETA priced itself from tokens — so the bar read 46% with 90 seconds to run. Both now
take `expect_tokens` and `expect_seconds` from the same `call_profile`, and the token
fraction is measured against *observed tokens per call* rather than `max_tokens`. The
asymptote survives as a floor: it is what guarantees the bar cannot stall when the
expectation is wrong.

Two runs on a 34-minute meeting (regenerate minutes, IQ3_XXS, 10GB card). First, with
the ETA fixed but the bar still on the old pricing:

```
0:19   2%   about 13m 15s left
1:39  11%   about  8m 43s left      -> implies 10m 22s total
4:39  28%   about  5m 28s left      -> implies 10m 07s total
8:39  46%   about  1m 23s left      -> implies 10m 02s total
                                       actual 9m 20s
```

Then with both on the measured profile:

```
0:47  14%   about 8m 8s left        -> implies 8m 55s total
2:46  31%   about 6m 26s left
4:47  50%   about 4m 36s left
6:47  71%   about 2m 39s left
8:46  93%   about 0m 37s left
                                       actual 8m 48s
```

Within about 1% of the truth from 47 seconds in, and the bar tracks it. The stored
profile after these runs: reduce = 3994 tokens at 7.2 tok/s, which is 553s against a
553s call. (The token figure counts streamed events, reasoning included, so it exceeds
llama-server's reported completion count — the bar compares against the same quantity,
so it is self-consistent.)

It will still move as the live rate changes — reasoning tokens come out at 7/s here and
prose at 4/s — but it moves because the measurement changed, which is the opposite of
the old failure.

### `output\` is one folder per meeting

A run produces up to four documents plus a folder of voice clips. Flat, those interleave
with every other meeting and there is nothing to hand to anybody. Now:

```
output\<meeting>\<meeting>_transcript.md
                  <meeting>_transcript_tagged.md
                  <meeting>_summary.md
                  <meeting>_minutes.md
                  speaker_samples\samples.json + spk00_1.mp3 ...
```

Filenames keep the meeting prefix inside the folder so a document still identifies
itself once copied out. "Open output folder" opens the meeting's own folder.

Existing flat outputs are migrated at startup (`config.migrate_flat_output`) rather than
left behind — without it they would silently vanish from the history list, which reads
as data loss. It sweeps loose documents that never had a transcript too. Verified: four
meetings, thirteen files and two sample folders moved, history intact afterwards.

### Wording that was not true

"Takes a few minutes; the recording is not processed again." Regenerating a document is
the single heaviest step in the pipeline and on a five-hour recording it is not a few
minutes. The finished page now quotes the same measured figure the first screen's
estimate is built from — "About 14 minutes on this machine" — falling back to an honest
"can take a while" when there is nothing measured to quote.

Also: `Rewrite` → `Regenerate` / `Generate` on the rebuild buttons, `Duration:` →
`Meeting Duration:` in every generated document header, and the default output mode is
now `Summary & Minutes` (the label `Both` was too terse to be obvious; the wire value is
unchanged).

### One transcript tab, not two

`Transcript` and `Named transcript` side by side asks the reader to care about a
distinction that only matters to the pipeline. When a named transcript exists it takes
the tab; the raw labelled file stays on disk, which is the entire reason it is never
overwritten.

## 9m. The up-front estimate, and a link that is actually clickable

### "About 2 minutes" for a job that took 10

Reported against a 90-second clip. The estimate was `sum(seconds_per_audio_hour) x
audio_hours` — proportional to length in every stage. Transcription, conversion and the
merge really are proportional. **The LLM stages are not.** A 90-second clip still gets a
whole document written for it: one map call and two reduce calls, ~10 minutes on this
card, and no amount of shrinking the recording makes that smaller. Proportional
arithmetic therefore collapses on short recordings, and is optimistic on long ones for
the mirror-image reason.

Rebuilt around tokens, which is what the work is:

```
transcript tokens = audio_minutes x 206
map calls         = transcript tokens / (target_tokens - overlap_tokens)
+ group reduces above the threshold, + one or two documents
seconds           = per call: measured tokens / measured tokens-per-second
                  + (convert + transcribe + merge) x audio_hours
                  + 60s overhead, all x 1.15
```

**206 tokens per audio minute** is measured, not guessed: the shipped tokenizer over the
real 3h25m transcript gives 206 tok/min (42,367 tokens / 205.7 min); the 34-minute
transcript gives 180. The longer figure is the baseline because it is the shape of
recording this tool exists for. It sits in `config.json` under `estimate`.

Against known ground truth (IQ3_XXS, 10GB card):

| Recording | Old estimate | New estimate | Actual |
|---|---|---|---|
| 90s clip, both | ~2 min | 18 min | ~10 min (reported) |
| 34m32s, summary | — | 20 min | ~25 min (from its run record) |
| 3h25m, summary | 69 min | 75 min | 69 min |

Still not perfect on the very short clip — it now errs long rather than short, which is
the right direction to be wrong in.

### Before the first run, say nothing

There is no honest number for a machine that has never run the model: generation speed
is set by how much of the model `cpu_ffn_regex` pushed into system RAM, and that varies
by several times across cards. `estimate_seconds` returns `measured=False`, the API
returns `null` per model, and the page says so:

> *This is the first run on this computer with this model, so there is no reliable time
> estimate yet — it will be timed as it goes, and every recording after this one will be
> estimated up front.*

Verified per model: Q4_K_M has never run on this card, so it reports `null` while
IQ3_XXS reports 20 minutes for the same file, and the dropdown switches the wording
between them.

### Latest run wins, except where it should not

Both halves of the profile changed:

- **tokens per second** now comes from the most recent run alone, not a median of ten.
  Hardware is the dominant term, so after a card change every older sample describes a
  machine that no longer exists; one run is enough to notice. The same rule now applies
  to the per-stage seconds-per-audio-hour rates.
- **tokens per call** is a median over recordings within 3x of the target length,
  falling back to all. Pooling regardless of length was measurably wrong: a document for
  a one-minute clip runs ~1,700 tokens and one for a 34-minute meeting ~3,900, so the
  pooled median suited neither. Verified against synthetic records — a 0.02h target
  draws 1,675 tokens, a 3.4h target 6,100, and the rate reading stays on the latest run
  in both cases.

### Ctrl+click on the console URL does not work

Whether a console linkifies a URL depends on which terminal Windows happens to be using;
on the target machine it did not, and the printed instruction was simply wrong. Removed.

`run.bat` now writes `Open Meeting Summariser.url` beside itself on each launch, pointing
at whichever port it settled on, and the console says to double-click that or type the
address. An internet shortcut is clickable everywhere on Windows regardless of terminal,
and it stays inside the app folder like everything else.

## 9n. The progress view and the first screen were running different estimators

Reported: the first screen said **56 minutes**, the progress bar said **2h 21m** as soon
as the job started, and only once the LLM stage began did the number look sane. The job
came in at **68 minutes**.

Two separate faults, and only one of them was a bug.

### The bug: two estimators, one job

§9m replaced the up-front estimate with a token model. The live ETA has a fallback for
stages with no call plan yet — during transcription the LLM work is entirely ahead and
its size is unknown — and that fallback was still the old seconds-per-audio-hour model.
So the two views priced the same LLM work completely differently.

Reconstructed exactly, using the calibration state at the time:

```
(convert 3.09 + transcribe 832.73 + merge 2.67
 + map 378.92 + group_reduce 0.17 + reduce 1425.36 x 2) s/audio-hour x 2.084 h
= 8,478 s = 141 min
```

141 minutes, against the 2h 21m reported. `Job._stage_remaining` now prices only
convert/transcribe/merge from audio-hours and calls `calibration.llm_stage_seconds` for
the LLM stages — the same function the first screen uses. Verified end to end on a
55-second clip: first screen 23 min, live ETA at the start of the run 22 min. The one
minute is `fixed_overhead_s`, which the live view correctly does not add because the
model is already loading.

### Not a bug: 56 was simply under-informed

At the time that estimate was made, the only length-tagged LLM samples on the machine
came from a 90-second clip: one map call of 862 tokens and reduce calls of ~1,700. A real
meeting's map calls run 1,500 tokens and its documents 5,500–6,800. The length window
added in §9m had nothing to match against, so it fell back to "use everything".

With that 2h05 run now recorded, the same estimate is **69 minutes against an actual 68**.
Which is the design working — but note it is now being tested against its own training
data. The honest test is the next unseen recording.

### The buffer is gone

`estimate.buffer_fraction` was 0.15 and is now 0, at the operator's request. Padding an
estimate that is within 2% only makes it wrong. Current standing against the two recorded
runs: 2h05 both, 66 min estimated / 68 actual; 35 min summary, 22 / 25. Short recordings
still come out low, which points at `fixed_overhead_s` being under 60 s of real startup
cost rather than at anything proportional — not adjusted, because two points is not
enough to fit one.

### And a third fault the buffer removal exposed: 23 minutes for a 5-minute job

Verifying the ETA fix on a 55-second clip, the first screen said 23 minutes and the job
took 5. §9m's length matching used a nearest-length window with "use everything" as its
fallback, and with only long meetings on record a short one inherited their document
size -- 6,178 expected tokens against an actual 814.

Replaced with interpolation: one point per recorded meeting length, log-scale
interpolation between them, extrapolation outside clamped to half the smallest and twice
the largest ever observed. From the two lengths now on record the curve reads 814 tokens
for a 55-second clip, 4,285 for 35 minutes, 5,510 for 2h05, 6,343 for five hours.

Standing against every run on record, buffer removed:

| Run | Estimate | Actual |
|---|---|---|
| 55 s clip, Summary & Minutes | 6 min | 5 min |
| 35 min, Summary | 22 min | 25 min |
| 2h05, Summary & Minutes | 66 min | 68 min |

The middle row is the only length with no calls recorded at that length; it is being
interpolated across two decades and will correct itself when a meeting that size is next
processed.

The whole calculation, step by step with a worked example, is in §9m and §9n.

### Markdown tables were not rendered

`reduce_summary.txt` asks for a Participants section with speaker, inferred name and
confidence. The model answers with a Markdown table, quite reasonably. `renderMarkdown`
had no table support, so every row fell through to the plain-paragraph branch and the
section rendered as a stack of pipe-laden sentences.

Added: a header row followed by a `---|---` divider starts a table, alignment colons are
accepted and ignored, short rows are padded. It scrolls inside its own
`overflow-x: auto` box — an evidence column runs long and the page body must never scroll
sideways.

## 9o. Vulkan, and the About overlay that was always on screen

### The About overlay could not be closed, and opened itself

`<div id="about" class="overlay" hidden>` with `.overlay { display: flex }`. The browser's
`[hidden] { display: none }` is a **user-agent** rule; any author rule that sets `display`
beats it. So the sheet was permanently on screen, which produced all three reported
symptoms at once: visible on load, every field still a dash because it had never been
opened and populated, and `hidden = true` on close doing nothing.

Fixed with one global rule rather than a guard on this component:

```css
[hidden] { display: none !important; }
```

The whole UI is driven by toggling `hidden`, so it was luck that nothing had hit this
before — every other panel simply never sets `display`.

### AMD and Intel via Vulkan

Implemented across §9o-§9p; the assessment that preceded it (`AMD_INTEL_BUILD.md`)
has been removed now that the work is done and measured.

**Binaries.** One folder per backend, all of them shipped rather than chosen at download
time: the folder is meant to be copied to a different machine, and choosing wrong at
download time would only be discovered over there. Adds ~1.1 GB against 29 GB of models.

```
bin\llama-cuda\    bin\llama-vulkan\    bin\llama-cpubin\whisper-cuda\  bin\whisper-cpu```

**Detection.** `nvidia-smi`, then `llama-server --list-devices` from the Vulkan build:

```
Available devices:
  Vulkan0: NVIDIA GeForce RTX 3080 (10051 MiB, 9283 MiB free)
```

That is the reliable cross-vendor VRAM read, and the reason not to use
`Win32_VideoController.AdapterRAM` — a 32-bit field that caps at 4 GB and reports
4095 MB for a 16 GB card.

**Per engine, not per machine.** whisper.cpp has never published a Vulkan Windows
binary — checked every release back to v1.7.2, the assets are CPU, BLAS and cuBLAS only.
So the backend is resolved separately for each engine, and a non-NVIDIA machine runs the
LLM on Vulkan while transcription falls back to CPU. The LLM is 80–95% of wall time, so
that is most of the benefit for none of the build effort. Verified by simulating an AMD
probe: `llama -> vulkan, whisper -> cpu`.

The CUDA folders are **excluded from the fallback chain** on a machine with no NVIDIA
card. They ship everywhere, so "the files exist" says nothing; falling back to them would
load `ggml-cuda.dll`, find no device and quietly run on CPU anyway — slower to start and
much harder to diagnose.

**Verified on this machine**, which has an NVIDIA card and can therefore run both paths:

| Check | Result |
|---|---|
| `--list-devices` on the Vulkan build | enumerates the 3080, 10051 MiB |
| Flags we depend on (`--override-tensor`, `--flash-attn`, `--cache-type-k`, `--jinja`, `--no-webui`, `--n-gpu-layers`) | all present |
| `bin\llama-vulkan\` contents | `ggml-vulkan.dll`, **no** `ggml-cuda.dll` |
| whisper Vulkan build vs CUDA | identical text and identical DTW timestamps |
| Pin `gpu.backend: vulkan`, run a real reduce | model loaded in 27 s, 2,254-token prompt processed |
| Per-engine fallback with that pin | llama → vulkan, whisper → cuda |
| `DOWNLOAD_MODELS.bat` skip path | all 12 items skip correctly |
| `DOWNLOAD_MODELS.bat` download path | removed `bin\llama-cpu\`, re-run, downloaded and unpacked cleanly |

**A retracted measurement.** An earlier note here recorded Vulkan prefill at 2,254 tokens
in 138 s and called it slow. That run happened while ComfyUI was generating images on the
same card. It is withdrawn — it measured contention, not Vulkan. No clean LLM throughput
comparison has been made, because the card still shows 6.2 GB held by another process;
the whisper figures below were taken back to back under identical conditions on a 1.6 GB
model, so they are unaffected.

### The Vulkan whisper build

whisper.cpp has never shipped a Vulkan Windows binary, so this one is built from source.
It is **published on this repository's own releases** under the pre-release tag
`whisper-vulkan-b4938`, and `DOWNLOAD_MODELS.bat` fetches it by URL like any other binary
-- expecting an end user to install Visual Studio, CMake and the Vulkan SDK was never
realistic. The pre-release flag matters: `/releases/latest` skips those, so a binary asset
can never be mistaken for an app version by the updater.

Toolchain used to build it:

| | |
|---|---|
| CMake 4.4.3 | `winget install Kitware.CMake --scope user` — no admin needed |
| Vulkan SDK 1.4.357.0 | `winget install KhronosGroup.VulkanSDK` |
| MSVC 14.44 | `winget install Microsoft.VisualStudio.2022.BuildTools` with `--add Microsoft.VisualStudio.Workload.VCTools` |

The build, against the **same tag as the CUDA binary** (b4938) so the two backends accept
exactly the same flags:

```bat
call "...\BuildTools\VC\Auxiliary\Build\vcvars64.bat"
set "VULKAN_SDK=C:\VulkanSDK\1.4.357.0"
set "PATH=%VULKAN_SDK%\Bin;%PATH%"
cmake -B build-vulkan -DGGML_VULKAN=1 -DCMAKE_BUILD_TYPE=Release ^
      -DWHISPER_BUILD_TESTS=OFF -DWHISPER_BUILD_EXAMPLES=ON
cmake --build build-vulkan --config Release --target whisper-cli -j 10
```

Output goes to `build-vulkan\bin\Release\`. Copy `whisper-cli.exe`, `whisper.dll`,
`ggml*.dll` into `bin\whisper-vulkan\`, **plus `MSVCP140.dll`, `VCRUNTIME140.dll` and
`VCRUNTIME140_1.dll`** from the Build Tools redist folder. Upstream's prebuilt binaries
have the same dependency and simply assume the VC++ redistributable is installed; this
app promises "unzip and double-click", so it carries them. 56 MB in total.

### Verification against CUDA

The real risk was never whether it compiles. It was `--dtw`: BUILD_NOTES §3.2 records
that flag silently returning `t_dtw: -1` under flash-attention on CUDA, which would have
destroyed the word-level merge without any visible error. So the test is output equality,
not "it produced text".

Same 55-second clip, same flags the pipeline actually passes (`-ojf -pp -mc 0 --vad
--vad-model ... --dtw large.v3.turbo -nfa`), both backends warm:

| | CUDA | Vulkan |
|---|---|---|
| Wall time | 3,192 ms | 3,103 ms |
| Segments / tokens | 7 / 168 | 7 / 168 |
| Text | — | **identical** |
| `t_dtw` values | — | **identical, all 168** |
| `t_dtw >= 0` | 159 | 159 |

`ggml_vulkan: Found 1 Vulkan devices: NVIDIA GeForce RTX 3080 | uma: 0 | fp16: 1 |
matrix cores: NV_coopmat2`.

**A trap worth recording.** A first attempt measured CUDA at 52 s against Vulkan's 3 s.
That was not a backend difference — it was `bin\` missing from PATH, so `ggml-cuda.dll`
could not find `cublas64_12.dll` and whisper fell back to CPU without complaint. Exactly
the failure `config.child_env()` exists to prevent, reproduced by hand the moment the
binary was invoked outside the app. Any benchmark of the CUDA build from a plain shell
is measuring the CPU unless `bin\` is on PATH first.

With this in place, a machine with no NVIDIA card resolves **both** engines to Vulkan;
verified by simulating an AMD probe.

## 9p. The AMD desktop: two bugs, one wrong diagnosis, and a knob worth keeping

A Radeon 780M desktop failed at transcription. Working it out took four rounds of
diagnostics and produced two real bugs, one performance finding that reversed a change
mid-flight, and one wrong conclusion I had to withdraw.

### Wrong turn worth recording

The failure log ended with `whisper_backend_init_gpu: no GPU found`, and I concluded the
app had selected a non-Vulkan whisper binary. It had not. That line is printed by the
**VAD context**, which is CPU-only, and it appears on every successful GPU run too --
verified here at line 66 of a run whose line 30 reads `using Vulkan0 backend`. It was
simply the first line to survive the 60-line stderr tail.

Two lessons, both now fixed elsewhere: `transcribe.py` logged `" ".join(cmd[1:])`, which
drops the exe path, so the log could not say which binary ran; and `_tail_lines(..., 60)`
threw away the evidence that would have settled it in one round.

### Bug 1: KHR_coopmat hard-crashes on the AMD proprietary driver

whisper-cli dies at the first encoder call with exit `0xC0000409` -- `__fastfail`, no
message, no `GGML_ASSERT`. An uncaught vulkan-hpp exception reaching `std::terminate`.

Isolated by ladder. Step 1 -- no VAD, no DTW, flash attention on -- already crashed, so
none of our flags were implicated:

| Run | `matrix cores` | Result |
|---|---|---|
| bare minimum | `KHR_coopmat` | crash at first encode |
| VAD, no DTW | `KHR_coopmat` | crash at first encode |
| **full flag set**, `GGML_VK_DISABLE_COOPMAT=1` | `none` | **success, 17.5 s** |

The device differs from the development NVIDIA card in every dimension that selects a
shader path: `KHR_coopmat` against `NV_coopmat2`, subgroup 64 against 32, 32 KB of shared
memory against 48 KB, and `uma: 1`.

`config.child_env()` now sets the variable for AMD on Vulkan. ggml tests it with
`getenv`, so it is existence, not value -- setting it to `0` also disables coopmat.

### Bug 2: unified memory advertises VRAM it will not allocate

```
Vulkan0: AMD Radeon 780M Graphics (18065 MiB, 17161 MiB free)
```

18 GB of a 32 GB machine. The real ceiling is 15 of 64 layers -- roughly 3.5 GB, matching
the BIOS carve-out. Above that: silent exit at 16, `vk::Queue::submit:
ErrorOutOfDeviceMemory` at 32.

`-ngl auto` does not help, and this is the important part: llama.cpp's own fitting reads
the same reported free memory, believes the 18 GB, and fails identically. It is not
probing what can be allocated.

`GGML_VK_ALLOW_SYSMEM_FALLBACK` does not help either -- ggml takes a separate branch on
UMA that already prefers host-visible memory and never consults the flag.

So the advertised figure is untrustworthy on any UMA device, and `choose_key()` now
refuses the 16.5 GB model there regardless of what it claims. Without that, this machine
would have selected Q4_K_M -- 18065 clears the 15000 threshold -- with zero offload,
because the deficit computes negative.

### A change made, then reversed by measurement

Having found that `-ngl 99` defeats llama.cpp's auto-fit, I removed both it and
`--override-tensor` in favour of letting llama.cpp decide. Benchmarking that on the 10 GB
development card, same VRAM occupied either way:

| | generation |
|---|---|
| llama.cpp auto-fit | 2.63 tok/s |
| `-ngl 99` + FFN `--override-tensor` | **6.23 tok/s** |

2.4x in favour of the thing I had just deleted, because the FFN split keeps attention
resident and whole-layer placement does not. Restored from the previous commit. Section
11.1's approach was right; only its hard-coded regex was wrong, and that was already
computed per machine.

The UMA case is the mirror image, measured on the 780M:

```
-ngl 0     prompt 33.8 tok/s   generation 3.15 tok/s
-ngl 15    prompt 21.6 tok/s   generation 3.05 tok/s
```

Generation identical, prompt processing a third slower with the GPU involved. So
`placement()` has two regimes and no middle ground: FFN split on a dedicated card, the
processor on AMD unified memory.

Then benchmarked on the Intel iGPU laptop, which settled it the same way and more
sharply -- offloading buys 12% on the prompt and costs 2.8x on generation:

| `-ngl` | prompt | generation |
|---|---|---|
| 0 | 1.61 t/s | **1.36 t/s** |
| 15 | 1.67 t/s | 0.87 t/s |
| 99 | **1.80 t/s** | 0.48 t/s |

Offloading would only pay if a call's prompt exceeded 21x its output. Ours are 6.7x (map)
and 1.8x (reduce), so the processor wins every call on that machine too. Two vendors,
different drivers and different matrix hardware, same answer -- so the rule is now UMA-wide
rather than AMD-only. The coopmat workaround stays AMD-only: setting it on the Intel
laptop changed nothing at any `-ngl`, which is what "not needed here" looks like.

### `-ngl 0` is not "on the processor", and that cost Intel 3x

The Intel laptop was 21x slower than the Radeon 780M at *prefill* but only 2.3x slower at
generation. Paging was the first guess and it was wrong -- the machine has 48 GB. The CPU
backend variant was the second guess and also wrong: it loads `ggml-cpu-alderlake.dll`,
a proper AVX2 build.

The answer is that `-ngl 0` does not take the GPU out of the picture. The weights move to
system RAM but the Vulkan backend stays registered, and the scheduler still routes prefill
-- the compute-bound matmuls -- to the GPU. Confirmed on the development machine, prefill
only, same model:

    Vulkan build, -ngl 0    40.4 tok/s
    CPU-only build          16.8 tok/s

So on a dedicated card that routing is a 2.4x *gain*, which is why nobody noticed. On an
Intel Arc Xe-LPG -- an integrated GPU with no matrix units at all, reporting
`matrix cores: none` natively -- it is a disaster. Using the **CPU build** instead, so
there is no Vulkan backend for the scheduler to reach for:

| | prefill | generation | 4-hour job |
|---|---|---|---|
| Intel, Vulkan `-ngl 0` | 1.61 t/s | 1.36 t/s | **14.8 h** |
| Intel, CPU build | 9.45 t/s | 1.60 t/s | **4.5 h** |
| AMD, Vulkan `-ngl 0` | 33.80 t/s | 3.15 t/s | 1.8 h |
| AMD, CPU build | 26.72 t/s | 3.73 t/s | 1.8 h |

3.3x on Intel, a wash on AMD -- never worse. So on unified memory the language model now
takes the **CPU binary**, not the Vulkan binary with the layers switched off. Whisper is
unaffected and stays on Vulkan: it is a far smaller model and both integrated GPUs handle
it well.

### Thread count: let llama.cpp choose

`-t 6` was tested on the theory that Meteor Lake's E and LP-E cores were dragging the
batch down. They are not -- the default beat it on both machines:

| | default | `-t 6` |
|---|---|---|
| Intel (16 cores) | 9.45 t/s | 8.32 t/s |
| AMD (8 cores) | 26.72 t/s | 21.94 t/s |

`llm.threads` was 10, which is over-subscribed on the 8-core machine and under-subscribed
on the 16-core one -- worse than the default on both. It is now `"auto"`, which omits the
flag. The LLM stage runs after whisper has exited, so it may have every core; §7's thread
budgeting applies to the concurrent transcribe/diarize stage, not this one.

### Latency the investigation exposed

Timing the cancel path found that detection -- which now spawns up to two llama.cpp probes
with 60-second timeouts on a machine without nvidia-smi -- is reachable from
`jobs.remaining_seconds()`, which runs inside `_status()` **on the event loop**. A first
call arriving there would have stalled every request, Cancel included. Now primed on a
background thread at startup, and the probe itself is serialised behind a lock so
concurrent callers cannot each spawn it.

`calibration.json` was also being re-read from disk on every progress event -- every 25
generated tokens -- and is now cached against its mtime.

Measured after: **cancel during generation, POST to the SSE `cancelled` event reaching
the client, 0.37 s.**

### Calibration had to learn about backends

Records were keyed by model alone, so a CUDA run's tokens-per-second would have been
applied to a Vulkan or CPU run -- a difference of an order of magnitude. The key is now
`model@backend`; records written before this are never matched, which is correct, as they
describe an unknown machine.

## 9q. The updater, and the five traps it is built around

`UPDATE_BUTTON.md` used to carry this; it was a portable write-up from two earlier apps
and has been folded in here so the code has something real to point at.

### The shape

Nine steps, and every failure leaves the running install untouched:

```
  check GitHub /releases/latest  ->  compare tag with APP_VERSION
    -> not newer: say so, stop
    -> newer: show version and size, wait for the user
         -> folder writable?  no: explain, change nothing
         -> download the zip to %TEMP%, with a progress bar
         -> unpack, verify the app is in there and its version matches the tag
         -> write a .cmd, launch it, quit
              the script waits for the app to let go, robocopies, restarts
```

The app never overwrites itself, because Windows holds a running executable open. That is
the only reason step 9 needs a script at all.

Two simplifications fall out of this being a Python app in a plain folder rather than a
packaged exe: download, unpack and verify all happen **in Python before anything is
replaced**, so the batch script only waits, copies and restarts; and there is no exe
version to read, so verification greps `APP_VERSION` out of the downloaded
`server/version.py` without importing it.

### The release has to look like this

| | |
|---|---|
| **Tag** | `v<version>`, matching `server/version.py` exactly |
| **Asset** | Exactly one `.zip`. Nothing else is looked at, and two zips are refused rather than guessed at |
| **Publish** | A real published release. `/releases/latest` skips drafts and pre-releases |

The zip is the **source tree only** — `server/`, `web/`, `prompts/`, `run.bat`,
`config.json`, the documentation. Not `bin/`, `models/` or `runtime/`: those are 30 GB,
past GitHub's per-asset limit, and unchanged between versions anyway. robocopy without
`/MIR` leaves them alone, which is what makes a 400 KB update possible.

Anonymous GitHub API calls are rate limited to 60/hour per IP. A button nobody can press
that fast will never see it, so no token is needed — and one must not be shipped anyway.

### The five traps

Four of them only appear once the thing is packaged and running for real.

**1. Node/Python will not spawn a `.cmd` directly.** Since the CVE-2024-27980 fix, passing
a `.bat` or `.cmd` as the executable is refused. `cmd.exe` is the executable and the
script is its own argv entry:

```python
subprocess.Popen([os.environ.get("ComSpec") or "cmd.exe", "/c", str(script)], ...)
```

Never `shell=True` with an interpolated path — that is the vulnerability the fix exists
for.

**2. Do not wait with `tasklist | find`.** It works interactively and **hangs** when
launched detached from a dying parent: `find.exe` sits forever on its end of the pipe, the
copy never runs, and a stray console is left on the desktop. Wait on something that needs
no pipe. This app holds `temp\running.lock` for its lifetime and the script polls
`if not exist`, with a 60-iteration cap so a force-killed app cannot strand the update.

**3. Unzip without a dependency.** Windows 10 1803 and later ship bsdtar as
`%SystemRoot%\System32\tar.exe`, which reads zip. Always the absolute System32 path —
`tar` on PATH may be GNU tar, which cannot. Here the unpacking happens in Python's
`zipfile` instead, which also allows checking every member for a path that would escape
the destination before extracting anything.

**4. Copy with robocopy, not xcopy.**

```bat
robocopy "%READY%" "%TARGET%" /E /R:3 /W:2 /XF "config.json" /NFL /NDL /NJH /NJS /NP
if errorlevel 8 ( echo failed )
```

It retries a locked file instead of giving up; it skips files whose size and timestamp
already match; **exit codes 0–7 are all success**, so `if errorlevel 8` is the failure
test and a plain `if errorlevel 1` would report every successful copy as a failure. No
`/MIR` — mirroring would delete the user's models.

`/XF "config.json"` is this app's addition: the operator tunes that file by hand, and
`load_config()` merges any new keys in from `_DEFAULTS`, so a stale config loses nothing.

**5. Verify before trusting.** Check the unpacked folder actually contains the app
(`run.bat`, `server/main.py`, `server/version.py`, `web/index.html`) and that its version
matches the tag. "Cannot read it" is treated as unknown and allowed; "read it and it
disagrees" is a hard stop.

### Details that are easy to get wrong

- **Compare versions numerically.** `"1.10.0" > "1.9.0"` is false as strings. An
  unparseable tag must answer *not newer* — never offer an update you cannot reason about.
- **Bake paths into the script**, do not pass them as arguments. `set "TARGET=C:\Program
  Files\X"` has no quoting left to get wrong.
- **Only delete a staging folder you created.** The sweep checks for this app's own prefix
  before removing anything; pointed anywhere else it would take a real folder with it.
- **Sweep abandoned staging folders.** A machine that loses power mid-update leaves a whole
  unpacked copy in `%TEMP%`, and the `.cmd` cannot delete itself.
- **Check writability before the download**, not after. Finding out after 400 KB is cheap;
  after 250 MB it is rude.
- **Show bytes, not just a percentage.** "48.2 MB of 248.8 MB" tells the user whether it is
  stuck.
- **Restrict what the page may fetch.** The install endpoint refuses any URL that did not
  come from github.com, so a compromised page cannot point the downloader elsewhere.
- **A console window flashes** while the script runs — `DETACHED_PROCESS` beats
  `windowsHide`. It is given `title Updating Meeting Summariser` so it reads as
  intentional for the second it exists.

### What it deliberately is not

- **Not automatic.** Nothing checks on launch, nothing nags. An app that quietly replaces
  itself is an app that breaks in the middle of someone's work.
- **Not delta updates.** The whole zip every time — which is 400 KB here, so it does not matter.
- **Not signed.** Nothing verifies the download beyond HTTPS to github.com and the version
  check. That is the same trust as clicking the release link by hand, which is what this
  replaces.

## 9r. A fresh clone could not start at all

Reported after downloading a fresh copy:

```
  These files are missing from the application folder:
      runtime\python.exe
      bin\ffmpeg.exe
```

`bin\` and `runtime\` are gitignored -- they are 30 GB of payload -- and
`DOWNLOAD_MODELS.bat` fetched the models and the inference binaries but **nothing at all
fetched the Python runtime or ffmpeg**. They existed only because they were sitting on the
development machine. Same class of hole as the whisper-vulkan one, but a hard stop rather
than a silent slowdown: a first install was impossible.

Both are now fetched:

- **`runtime\`** -- CPython 3.12.14 with every wheel already vendored, published on this
  repository under the pre-release tag `runtime-cpython-3.12.14`. Hosting the exact tested
  environment beats a `pip install` list that would drift, and keeps `pip` out of the
  install path entirely.
- **`bin\ffmpeg.exe`** -- from BtbN's builds, pinned to `autobuild-2026-09-07-15-39`.

### The ffmpeg licence, changed on purpose

The build that was on the development machine was `--enable-gpl --enable-version3`: a
**GPLv3** ffmpeg. That places GPLv3 obligations on anyone the assembled folder is passed
to, which is a poor trade for a component we invoke as a subprocess and link nothing
against.

The downloader now fetches the **LGPL** variant (`--enable-version3` only). Verified it
loses nothing the pipeline uses:

| | |
|---|---|
| mp3 / aac / opus / vorbis / pcm decode | yes |
| `libmp3lame` (speaker clips) | yes |
| `pcm_s16le` (stage 1 output) | yes |
| `Duration:` banner that `audio.py` parses | yes |

Both real commands the app issues were run against it: mp3 to 16 kHz mono WAV, and an
8-second 64 kbps mono mp3 clip. Both exit 0.

### Tested as the user did

A fresh `git archive` of HEAD, with only `models\` junctioned in to avoid re-downloading
28 GB:

1. `run.bat` first -- reproduced the reported error exactly.
2. `DOWNLOAD_MODELS.bat` -- fetched the runtime, ffmpeg and all six binary sets.
3. `config.missing_files()` -- empty.
4. Server started, `/api/health` ok, backend detection correct.
5. A real 55-second job: convert, 117 words transcribed, diarization ran, merge wrote
   5 turns.

`run.bat`'s failure message was also wrong for this case -- it said the folder "may not
have copied completely", when the right advice for a fresh clone is to run the downloader.
It now says both.

### Versions are pinned, and now actually pinned

The binaries were already pinned to build numbers and release tags, but all four
HuggingFace model URLs used `resolve/main` -- a branch head. An upstream re-upload would
have handed a new machine different weights from the ones every measurement in this file
was taken against, and the size check would not have caught it. They are now pinned to
commit SHAs, verified byte-identical to the local copies:

| Model | Revision | Bytes |
|---|---|---|
| ggml-large-v3-turbo.bin | `5359861c` | 1,624,555,275 |
| ggml-silero-v5.1.2.bin | `9ffd54a1` | 885,098 |
| Qwen3.8-27B-UD-IQ3_XXS.gguf | `4ca72078` | 10,934,860,704 |
| Qwen3.8-27B-UD-Q4_K_M.gguf | `4ca72078` | 16,464,440,224 |

### Releasing: cut a version, do not clobber an asset

Replacing the v1.0.0 asset in place worked but the CDN served the old bytes from
`browser_download_url` for about 80 seconds -- and that is precisely the URL the updater
downloads from. The API already reported the new size while the CDN did not. So a content
change gets a new version number, which is why this went out as v1.0.1 rather than another
clobber.

## 9s. Two release assets, and why the updater must not take the big one

The instinct to ship the proven binaries rather than re-download them is right, and the
earlier reasoning against it was partly wrong. Measured:

| | zipped |
|---|---|
| `runtime\` + `bin\ffmpeg.exe` + `bin\whisper-vulkan\` | 159 MB |
| `runtime\` + all of `bin\` | **1.14 GB** |
| the same plus the three small models | **1.24 GB** |
| GitHub per-asset limit | 2 GB |

So size was never the obstacle for the binaries, and neither was licensing -- everything
in there is MIT, BSD, Apache, PSF or LGPL, all redistributable. **Only the models are
impossible:** `Qwen3.8-27B-UD-Q4_K_M.gguf` is 16.5 GB and `IQ3_XXS` is 10.9 GB, each on
its own over the 2 GB limit, so they can never be release assets whatever the packaging.

The three *small* models do fit, and were added once the bundle proved out: silero
(0.8 MB), the pyannote segmentation export (5.7 MB) and TitaNet (96.7 MB), which
compress to 96 MB and take the bundle to 1.24 GB. They earn their place twice. They are
the only two downloads that were never pinned to immutable bytes -- sherpa-onnx
publishes both speaker models on floating release tags, so the bytes could change
under us -- and the segmentation model is the single ugliest step in
`DOWNLOAD_MODELS.bat`, a tar.bz2 that has to be fetched, unpacked, copied, renamed and
swept up, with two distinct failure branches. Shipping them retires all of that.

Whisper large-v3-turbo was measured and rejected: 1549 MB raw, 1424 MB zipped (these are
float16 weights, so compression buys 8%), which would put the bundle at 2.56 GB.

A fresh extract of the bundle now reports three missing model files instead of six.

A release therefore carries two zips, built by `tools/make_release.py`:

- `Meeting-Summariser-vX.Y.Z-full.zip` (~1.24 GB) -- a first install. Unzip it and only
  the three large models remain to download.
- `Meeting-Summariser-vX.Y.Z.zip` (~190 KB) -- the update payload.

*(The full zip is ~1.47 GB from 1.2.0 onwards: the word aligner joined the bundle. See
section 9aj. The figures above are what was measured at the time.)*

### The updater must never take the full one

Two reasons, and the second is the real one:

1. A 190 KB update would become 1.24 GB, to ship a few changed `.py` files.
2. It would robocopy `runtime\python.exe` over the interpreter the running app is
   executing from. `run.bat` launches `runtime\python.exe -m uvicorn`, and Windows locks
   a running executable's image. A half-copied interpreter cannot start, so it cannot
   self-repair -- there is no recovery path from that.

There is a specific race that makes (2) worse than it looks. `updater._quit()` clears the
`running.lock` marker *before* `os._exit(0)`, so the batch script can begin copying while
the process is still exiting and holding its own image open. Harmless today because the
payload is only `.py`, `.js` and `.md`; fatal if the interpreter were in it.

`version.pick_asset` therefore skips any asset carrying `-full`, and **refuses rather
than guesses** when that leaves no candidate -- a release with only a full bundle yields
"nothing to install", not "install the 1.24 GB one". Exercised against seven asset layouts
including both orderings, the single-zip releases v1.0.0 and v1.0.1, and the ambiguous
two-zip case.

### The bundle caught a licensing error

Building it revealed that `bin\ffmpeg.exe` on the development machine was still the
**GPLv3** build. §9r had switched `DOWNLOAD_MODELS.bat` to the LGPL variant, but the local
copy was never replaced -- the LGPL build had only ever been tested in a scratch folder.
The full bundle would have shipped a GPLv3 ffmpeg alongside a `THIRD_PARTY_NOTICES.md`
claiming LGPL v3.

Replaced and re-verified from inside the finished zip: `--enable-gpl` absent,
`--enable-version3` present. Worth remembering that a licence file is a claim about an
artefact, and only checking the artefact tests it.

### It also stranded v1.0.0 and v1.0.1

The cost was not free, and it was not visible until tested. The picker that decides which
asset to download is **the installed copy's**, not the new release's. v1.0.0 and v1.0.1
shipped this:

```python
if len(zips) != 1:
    return zips[0] if len(zips) == 1 else None
```

-- refuse whenever a release carries more than one `.zip`. v1.0.2 carries two, so both of
those versions now answer *"The latest release has no download attached to it"* and can
never update themselves again. Confirmed by downloading the real v1.0.0 zip, running its
own `updater.check()` against the live release, and reading `status: error` back.

The population is two releases published the same day and held only by the operator, so
the accepted fix is to replace those installs by hand. Recorded because the general shape
recurs: **a change to the layout of a release is judged by every version that came
before it.** Today's picker is written to tolerate that -- it filters `-full` out and
only gives up when no single candidate remains -- so adding further assets is safe, but
renaming the update payload would not be.

Being several versions behind is otherwise a non-event: the app asks for `/releases/latest`
and installs it directly, with no chain to walk. A 1.0.0 install carrying the current
picker was pointed at v1.0.2 with 1.0.1 skipped; it downloaded, verified across the
skipped version, ran the shipped `apply-update.cmd`, and came out at 1.0.2 with the edited
`config.json`, `calibration.json` and `output\` intact.

Two things do not survive a hop, and both scale with how many versions are skipped.
`robocopy /E` never deletes, so a file removed in a skipped version stays on disk -- the
same property that protects the user's data. And the update payload carries no `bin\` or
`models\`, so a release needing a newer llama.cpp build or a renamed model updates the
code and leaves the binaries behind. That has to go in the release notes, because the
update button cannot deliver it.

---

## 9t. The reasoning budget

Thinking and the document come out of one allowance. Without a cap the reduce can spend
all of it reasoning and return `finish_reason: length` with **empty content**, which
`llm.chat` retries twice and then fails the job on — after transcription, diarization and
the whole map stage have been paid for. Reproduced at 400 tokens: 1700 chars of reasoning,
zero of content.

**The cap is a launch flag, and only a launch flag.** Build 10797 (`832fd6f17`):

```
--reasoning-budget N              -1 unrestricted, 0 immediate end, N>0 a token budget
--reasoning-budget-message MSG    injected before the end-of-thinking tag when N is hit
```

Section 11.2 warned it might disable thinking globally. It does not: with a budget set,
`enable_thinking: false` still returns 0 reasoning tokens and `true` still thinks. Map and
group-reduce are unaffected.

**`reasoning_control` in the request body is accepted and ignored.** Against a server
launched with `--reasoning-budget 60`, requests asking for 2000 came back at 298 / 318 /
276 / 261 chars — all the server's 60. It was wired up, measured doing nothing, and
removed. Do not re-add it.

**So an external server (§9u) is capped only if whoever started it said so.** Hence the
second half of the fix, which needs no cooperation: an empty completion with
`finish_reason: length` is retried **with thinking off** rather than repeated. Measured:
0 tokens → retry → 1587 chars of document.

`thinking.reduce_budget_tokens` sets it; 4000 is measured-correct (§9x).

---

## 9u. "Port": using a llama-server we did not start

Third entry in the Model dropdown. Reveals a port box (default **9931**, which is what
llama-server is moving its own default to) and sends every LLM call to `127.0.0.1:<port>`.
Only `/tokenize` and `/v1/chat/completions` go there; ffmpeg, whisper, diarization and
merge all stay local.

| | our server | a server on a port |
|---|---|---|
| start | spawn, poll `/health` for `startup_timeout_s` (180 s) | poll for 10 s, then fail plainly |
| stop | `taskkill /F /T` | **nothing** — it was running before this job |
| cancel | abort request, then kill | abort request only |
| placement, budget | ours to choose | not ours |

The short attach timeout is deliberate: ours has 16 GB to read off a cold disk, theirs is
either up or it is not.

**The context is theirs too.** `llm.ctx_size` describes the server *we* launch. The
overflow guard in `reduce.py` exists to force another group-reduce tier rather than send a
prompt that cannot fit, and measured against the wrong number it does not guard at all.
`LlamaServer.ctx_size` reads the running server's `/props`, and `chunker.build_chunks`
shrinks `target_tokens` to the window that actually exists. `_ctx_from_props` tries
`default_generation_settings.n_ctx`, then `.params.n_ctx`, then the top level, and returns
0 (falling back to ours) rather than guessing — llama.cpp has moved the field between
builds, and `/props` returns `{"error": ...}` while a model is still loading.

This is not hypothetical: a 4-hour reduce is a 7997-token prompt, and `7997 + 12000 + 2000`
does not fit a server started with `-c 16384`.

**Persistence.** Written to `config.json` on *change*, not on Process. Only
`llm.external.enabled` and `llm.external.port`, into a *raw* read of the file — writing the
merged config back would bake every current default in and freeze it against later changes
to `_DEFAULTS`. Picking High or Low clears the flag so detection runs again; the port
number survives either way.

`run.bat` skips its two `*.gguf` preflight checks when the flag is set — 25 GB of the
download that Port does not need. It asks `config.json` through Python rather than
pattern-matching, because `"enabled": true` also appears under `diarization`.

Timings are filed under `external:<port>`, not `<model>@<backend>`.

**Verified** against the operator's own `C:\llama` server (IQ2_S, `-c 16384`, llama.cpp
b10819 — *newer* than ours): full run in 2m58s, correct context detected, and **cancel left
their server running and healthy**. That build reports `reasoning_format: none`, so
thinking arrives as inline `<think>` tags rather than `reasoning_content` — the first time
§11.2's belt-and-braces `strip_thinking` has actually been needed.

---

## 9v. Quantisation comparison — method

Superseded on findings by §9z, which is the one to read. Kept for the method and for two
results that still stand.

Same 34-minute council meeting, cached transcript and diarization, minutes mode, scored
against the official published minutes. Each variant placed the way the shipped app would
place it. `tools\llm_compare.py` for single runs, `tools\llm_consistency.py` for repeats.

**MTP is worth roughly 50%.** The i1-IQ4_XS build is 24% *larger* than IQ3_XXS and 51%
faster (37.4 vs 24.7 tok/s). At equal settings a larger model is never faster, so the gain
is the speculative decoding. Confirmed by `common_speculative_init_result: creating MTP
draft context` in the log, and by its absence: without `--spec-type` the loader prints
fifteen `model has unused tensor blk.64.nextn.* -- ignoring` warnings. The tensors ship in
the file either way. MTP is lossless — draft-and-verify, so accepted tokens are the ones
the target model would have produced.

**Term recall measures coverage, not correctness.** It counts capitalised ground-truth
words appearing anywhere, so a document scores for "Information Officers" even in the
sentence that attaches the wrong figure to them. Worse, the denominator lies: of the
council meeting's 49 terms, **14 are not in the transcript at all** (the ASR wrote
"Kaczynski" for Kuczynski, "Fuel" for Fuhl; the rest were never spoken). The reachable
ceiling is 35, not 49 — and 119 of 149, not 149, on the planning meeting.

---

## 9w. Gigabytes against gibibytes

`offload_regex` compared decimal GB against GiB:

```python
available_gb = vram_mb / 1024.0        # GiB   -- 16311 MiB -> 15.93
needed_gb    = model["size_gb"] + 1.5  # GB    -- Q4_K_M    -> 18.00
```

That overstates every model by ~7%, spent entirely on offloading blocks that would have
fitted. `model_gib()` now stats the file, so the number is neither unit-confused nor stale
when a file is replaced.

| | advertised | real | offload before | after |
|---|---|---|---|---|
| Q4_K_M | 16.5 GB | 15.33 GiB | 12 blocks | **6** |
| Q4_K_S | 15.4 GB | 14.30 GiB | 7 blocks | **none** |

Worth 174s → 125s on map and 429s → 326s on reduce for Q4_K_M, identical output. The
offload *is* the cost of a large model, so the unit has to be right.

`OVERHEAD_GIB = 1.5` is confirmed in the right unit: IQ3_XXS is 10.18 GiB of weights and
sat at 11.69 GiB resident with everything on the GPU.

**Caveat.** The arithmetic sizes against *total* VRAM; the card reports 16310 MiB but only
~15172 MiB free once the desktop has its share, and llama.cpp says so
(`projected to use 15258 MiB vs. 15022 MiB of free device memory`). It works because
llama.cpp mmaps the weights and pages the excess, and it measured faster this way — but a
model at the edge will page if something else claims VRAM. Erring toward less offload is
the safer direction: over-offloading guarantees CPU execution for layers that would have
fitted.

---

## 9x. The reduce's token allowance

`REDUCE_MAX_TOKENS` is **12000**, not section 11.2's 8000. Measured on the cached 3h51m
transcript, thinking uncapped:

```
5 chunks -> 7298 tokens of notes -> one reduce (7997-token prompt)
thinking   ~3440 tokens        document 4158 tokens        total 7598 of 8000 = 95%
```

**4000 is the right thinking budget** — uncapped, the model asked for 3440 and stopped by
itself. The *document* is what scales with recording length (2100 tokens at 34 minutes,
4158 at 3h51m), so 8000 binds at five hours and what gets truncated is the document, which
is worse than the failure §9t prevents. Two of four variants later exceeded 8000 at 3h51m.

12000 costs nothing on our context (`7997 + 12000 + 2000 = 21997` against 32768) and only
makes `_fit_for_final` tier earlier against a small external one. Both the request and the
fit check read the same constant; if they disagree the tiering is decided against a number
the request does not use.

**Map calls never think** — every one logged `0 reasoning chars`. The budget applies to one
call per document.

Two traps worth keeping:

- **A label is not a model key.** A probe set `job.model_key = "budget-probe"` expecting the
  external path; `LlamaServer.__init__` reads any non-empty key other than `"external"` as
  an explicit choice of one of *our* models, so it took the local path and filed six calls
  under the real `q4_k_m@cuda` key, which had to be deleted from `calibration.json` by
  hand. `tools\llm_consistency.py` uses `compare-<tag>` plus `external.enabled: false`.
- **Cached transcripts need their VAD timeline.** `parse_json` with an empty `VadTimeline`
  leaves timestamps in VAD-compressed time, where silences are stripped and
  `merge.TURN_GAP_S` can rarely fire — six turns for a 4-hour meeting, and chunks of 19169
  tokens against a 10000 target. Production rebuilds the timeline from whisper's stderr.

---

## 9y. The two IQ4_XS builds are different files

Same architecture, same `file_type` 30, same MTP tensors — different importance matrix and
size. Do not treat a result for one as a result for the other.

| | size | imatrix | chunks |
|---|---|---|---|
| `Qwen3.8-27B-i1-IQ4_XS-GGUF-Smaller` | 12.61 GiB | mradermacher/ubergarm | 319 |
| **`Qwen3.8-27B-UD-IQ4_XS`** | **13.27 GiB** | Unsloth's own | **1251** |

**MTP costs about 700 MiB**, which is what makes it conditional (§9aa). UD-IQ4_XS at 32k
with `--spec-type draft-mtp`:

```
model 13061 MiB + KV 1088 + recurrent 598 + compute 240
MTP draft context: KV 128 MiB + compute 130 MiB
total 15688 MiB of 16311
```

The recurrent-state allocation grows from 150 MiB to 598 MiB when the draft context is
added; the rest is the draft's own KV and compute buffers.

The KV cache is small because Qwen3.8-27B is a **hybrid**: `full_attention_interval = 4`,
so only 16 of 65 layers keep a KV cache at all and the rest hold a fixed-size recurrent
state. 32k context costs ~1.24 GiB, not the 6-8 GiB a conventional 27B would need.

---

## 9z. Five runs each — what reproduces and what does not

`tools\llm_consistency.py`. Server started once per variant and reused for every repeat, so
repeats differ only in sampling.

**Council meeting, n = 5:**

| Variant | GiB | Offload | MTP | map s | reduce s | terms (range) | rate table | timestamps |
|---|---|---|---|---|---|---|---|---|
| Q4_K_M | 15.33 | 6 blocks | — | 116 | 295 | 31 (30–32) | **5/5** | 3 (0–14) |
| Q4_K_S | 14.30 | none | — | 65 | 170 | 30 (29–32) | **2/5** | 10 (6–13) |
| **UD_IQ4_XS** | 13.27 | none | yes | **34** | **79** | 29 (26–31) | **5/5** | 4 (0–12) |
| i1_IQ4_XS | 12.61 | none | yes | 36 | 100 | 31 (30–33) | **5/5** | 2 (0–8) |

**Reproduces:** speed, exactly — it is set by file size and offload, not sampling. Term
counts, within a 29–31 band that separates nobody. The mayor's name, 20/20 runs.

**Does not reproduce: timestamp counts.** Q4_K_M produced 0, 0, 1, 14, 0. Every variant
does this. Never judge a model on it.

**Q4_K_S is the one real failure** — the election-rate table right in only 2 of 5, where the
others managed 5 of 5. By eye: `"$250 (election-day info officers"` — §9h's characteristic
error (amounts shifted one role up) appearing in a **4-bit** model, repeatably.

**n = 1 is worthless here except for timing.** Single runs said i1_IQ4_XS failed the rate
table (5/5 over five runs), Q4_K_S passed it (2/5), and Q4_K_S lost the mayor's name (5/5).
All three reversed.

**3h51m planning meeting, n = 1 each** (149 terms, 119 reachable): terms 64–67 for all
four, no separation; the speed ratio holds at length (UD_IQ4_XS 375 s against Q4_K_M's
1350 s, 3.6x). *Measured under the old 8000-token cap — not comparable with the IQ2/IQ3
figures below.*

### IQ2_XXS vs IQ3_XXS, for a card too small for 4-bit

Council meeting n = 5, planning meeting n = 3:

| corpus | variant | map s | reduce s | terms (range) | words | facts |
|---|---|---|---|---|---|---|
| 34 min | IQ3_XXS | 53 | 151 | 31 (29–32) | 1127–1306 | 15/15 |
| 34 min | IQ2_XXS | 44 | 103 | 32 (28–35) | 752–1156 | 12/15 |
| **3h51m** | **IQ3_XXS** | 260 | 313 | **81 (68–89)** | 2521–3097 | 8/9 |
| **3h51m** | **IQ2_XXS** | 215 | 212 | **58 (56–60)** | 1676–1892 | 6/9 |

On a one-chunk meeting they are indistinguishable. **On a 4-hour meeting they separate
cleanly and the ranges do not overlap** — IQ2's best run (60) is below IQ3's worst (68).

The mechanism is omission, not error: IQ2 is slightly *more* term-dense (32.3 against 27.9
per 1000 words) and writes **38% less document** (1795 against 2888 words). It covers less
of the meeting. For minutes that is the failure that matters, because a decision left out
cannot be noticed by the reader.

**IQ2_XXS is not a substitute at the length this app is for.** It is fine under about an
hour. IQ2_XXS also has no `nextn` tensors (64 blocks) and so can never take MTP.

---

## 9aa. High Quality is UD-IQ4_XS; MTP is conditional

`Qwen3.8-27B-UD-Q4_K_M` (15.33 GiB) → `Qwen3.8-27B-UD-IQ4_XS` (13.27 GiB), key
`q4_k_m` → `iq4_xs`. Both 4-bit, no quality difference that reproduces (§9z), and the
smaller file is the only one that fits a 16 GB card whole — 3.4x on the short meeting,
3.6x at 3h51m. Low Quality stays IQ3_XXS.

`HIGH_MIN_VRAM_MB` stays **15000**. The smaller file would clear a lower bar, but the point
was to make the same machines faster, not to widen who gets the large model. Lower it with
measurements from a 12-14 GB card in hand.

The old key disappearing is safe: `resolve_key` falls back to detection for any key it does
not recognise, so a `config.json` pinned to `"q4_k_m"` gets the new default.

**MTP only when nothing is being offloaded.** `llm.spec_type` defaults to `"auto"`;
`LlamaServer.spec_type()` returns the model's `spec` field when the placement produced no
`--override-tensor`, and nothing when it did. The draft context costs ~700 MiB (§9y), and
buying that by pushing more FFN blocks onto the processor is a bad trade — the offload is
the largest cost there is (§9w), and MTP is worth ~50% while a heavy offload costs several
times that. `""` forces off, `"draft-mtp"` forces on.

Moved with it: `hardware.MODELS` and `HIGH_MIN_VRAM_MB`; `llm.spec_type()`;
`config.llm.spec_type` default; `calibration._LLM_RATES` and `DEFAULT_MODEL` (now 56 and 41
s per audio-hour, measured on the 5060 Ti with the model resident, replacing a figure taken
on a 10 GB card with 24 layers offloaded); `run.bat`; `DOWNLOAD_MODELS.bat` (URL checked,
not assumed: `content-length: 14252845984`, byte-identical to the file measured); README;
CLAUDE.md.

**Changing the model would have bricked every install in the field.** The update payload
carries no `models\` (§9q), so a copy updating from 1.0.x keeps Q4_K_M and never
receives UD-IQ4_XS. Both startup gates named the new file specifically, so `run.bat`
would have refused to start with *"These files are missing: Qwen3.8-27B-UD-IQ4_XS.gguf"*
until the user re-ran a 14 GB download. Caught before publishing, not after.

The fix is to require **at least one** language model rather than every one, in both
`run.bat` and `config.missing_files()`, and to keep Q4_K_M selectable while it is on
disk (`hardware.LEGACY_MODELS`). `choose_key` now picks among files that actually exist,
ordered by `min_vram_mb` descending -- appending the legacy entry broke that ordering and
offered Low Quality to a 16 GB card holding Q4_K_M, which `_by_preference` restores.

The general rule this is an instance of: **a release may not require a file the updater
cannot deliver.** Only `DOWNLOAD_MODELS.bat` ships models, and the update button does not
run it.

**A hardcoded key in the UI broke with the rename.** `app.js` read
`d.recommended === "q4_k_m"` to choose between "High Quality" and "Low Quality" in the note
under the dropdown, and silently told every machine that Low Quality was selected while
High Quality ran. It now takes the tier from the model's own label.

---

## 10. Still not measured

- Wall time by stage for a full 4-hour and 8-hour run including the LLM stages.
- An 8-hour recording end to end. Everything is measured at 3h51m or less.
- **Five runs of the 4-bit files on a long meeting.** §9z has n=5 on the 34-minute
  meeting and n=1 on the 3h51m one. The case for UD_IQ4_XS does not depend on it --
  it rests on a 3.4-3.6x speed advantage that reproduced at both lengths.
- **What MTP is worth on Q4_K_M and Q4_K_S.** Every file except IQ2_XXS carries the
  `blk.64.nextn.*` tensors; `llm.spec_type` turns it on without a code change. Q4_K_M has
  the least VRAM headroom and is the one most likely not to fit with the draft context.
- **IQ2_S (7.8 GiB)** as the middle option for an 8 GB card: ~16 FFN blocks offloaded
  against IQ3_XXS's 35, where IQ2_XXS needs 4. Untested.
- **The Port option against a server that is not llama.cpp.** §9u assumes llama-server:
  `/tokenize` for chunking, `/props` for the context, `chat_template_kwargs` for thinking.
  An OpenAI-compatible server without those needs a fallback that does not exist.
- Final unzipped folder size is now **~17 GB** (14.3 GB model, 1.8 GB binaries, 253 MB
  runtime). `tools\` adds 180 MB and should be deleted before shipping.

## 9ab. The participant count was being silently overridden

Reported from a real 2h12m five-person meeting with heavy cross-talk: diarization
produced three speakers. The operator had entered 5 in the UI.

CLAUDE.md section 13.4 calls the participant count "by far the most reliable path",
and section 8.1 says distance thresholding is skipped entirely when it is supplied.
Both were true of the clusterer. Neither was true of what happened afterwards.

`diarize_worker.main` runs `cluster -> prune -> merge_close_clusters`, and the merge
ran unconditionally. Measured from the embedding cache for that recording:

| `num_speakers` | after cluster | after prune | after merge |
|---|---|---|---|
| auto | 536 | 20 | 4 |
| **5** | **5** | **5** | **3** |

So the count was honoured and then thrown away two lines later. The fix is one
condition: skip the merge when an explicit count was given. There is no
fragmentation to repair when the clusterer was told the answer, so the merge can
only destroy it.

### Why 0.35 was safe there and not here

Section 9 chose `merge_centroid_distance: 0.35` because on the 3h25m council
recording the 190 centroid pairs "split into six at 0.03-0.15 and nothing else
below 0.57". That gap is the entire justification, and it is a property of the
recording, not of the constant. Both caches, same code path:

| | widest gap in 0.05-0.70 | pairs within +/-0.05 of 0.35 |
|---|---|---|
| council 3h25m | **0.416** (0.150 -> 0.566) | 0 |
| this meeting | **0.027** | 12 |

Cross-talk is what closes the gap: two voices inside one embedding window produce a
contaminated vector, and enough of them smear the centroids into a continuum. The
merge is greedy and recomputes centroids after each step, so folding two different
people together creates a hybrid that then attracts more -- which is the 20 -> 4
cascade above.

**The auto path is still not fixed by this, and cannot be.** Declining to merge
leaves 20 clusters, which section 8.1 rightly calls worse than no labels at all.
Automatic counting genuinely cannot resolve a recording with this much overlap.
That is what the participant count is for, and it now works.

### Ask for active speakers, not people in the room

The operator reported five people: four active, and one who spoke for under a
minute. Entering 5 is worse than entering 4, because the fifth is below what
clustering can resolve and the clusterer splits an active speaker to reach the
count. Per-speaker minutes from the same cache:

    n=4   39.9  32.1  29.8   7.3
    n=5   34.0  32.1  29.5   7.5  6.0     <- the 39.9 split, she was not found
    n=6   34.2  29.4  28.3   7.6  6.0  3.5

The smallest cluster at n=6 is 3.5 minutes, so a sub-minute participant is not
recoverable at any setting. Section 13.4's wording should ask how many people spoke
*substantially*, or the honest answer costs the user a real speaker.

## 9ac. The embedding cache missed on path form

Re-running the same recording through the app after building a cache by hand
re-ran the 22-minute embedding pass. `cache_key` hashes the *strings* in
`diarize_params.json`, and the app writes resolved absolute paths where a
hand-run passes the relative ones out of `config.json`. Same file, same bytes,
different hash.

It never affects a production run -- nobody processes the same audio twice -- but
it defeats exactly the workflow section 3.7a exists for, where the whole point is
that re-clustering costs seconds. The paths are resolved before hashing now, so
both forms agree.

## 9ad. Whisper's DTW word ends, and the silent voice sample

Reported: the clips in the speaker-naming panel do not match their text. One was
18 seconds of audio containing the single word "so".

The word *starts* are sound. The ends are not. Measured over the 2h12m meeting:

| | |
|---|---|
| consecutive word pairs that overlap | **32.5%** (median depth 0.12 s) |
| words claiming over 2 s | 4.8% |
| longest single word | **43.1 s** |
| midpoints landing past the next word's start | 10.9% |

### It does not move attribution

The obvious worry is section 9 step 2, which assigns each word by its midpoint.
Clamping every end to the next word's start and re-running the merge:

    as shipped     1159 changes, 46.5% clean
    ends clamped   1190 changes, 45.8% clean

Nothing, or slightly worse. The diarization segments are long next to a 0.12 s
overlap. **So the shipped attribution path is left alone** -- and note that
clamping to the next start is the wrong repair anyway, because it forces every
inter-word gap to zero and destroys the signal turn grouping runs on.

### What it does wreck is turn grouping

`merge.assign_speakers` breaks a turn on
`word.start - previous.end > TURN_GAP_S` (2 s). One inflated end swallows the
silence behind it, so a turn spans a gap it should have broken on -- in the
reported case a 17-second one, from a turn whose first word sat alone at
1810.7 s with the next at 1827.8 s. `speakers.build_samples` cuts its clip from
the turn's start, so the clip was that one word and then silence.

Capping word *duration* fixes it where clamping did not:

| cap | turns | clean | turns > 25 s | sparse-long turns |
|---|---|---|---|---|
| none | 909 | 46.5% | 59 | **10** |
| 3.0 s | 1020 | 46.6% | 32 | **0** |
| 2.0 s | 1053 | 46.7% | 31 | 0 |
| 1.0 s | 1118 | 46.7% | 21 | 0 |

**3.0 s adopted** (`transcribe.MAX_WORD_S`), applied in `parse_json` so
everything downstream inherits it. The 90th-percentile word is 1.15 s, so it
truncates only degenerate ends and leaves real words alone. Attribution is
unchanged, which is the point -- this is a turn-grouping fix, not an
attribution one.

### Two further sample defects, same report

- **Ranked by the wrong thing.** `build_samples` sorted by wall-clock span, so
  it preferentially picked the turns with the largest silences in them. It now
  ranks by speech actually inside the clip window, unioning the word intervals
  rather than summing them -- summing overlapping ends reported 175 s of speech
  inside an 18-second clip.
- **The text ran past the audio.** The clip is capped at `MAX_CLIP_S` (18 s)
  while the caption showed 400 characters of the whole turn, so on any long turn
  the reader saw far more than they heard. The caption is now cut to the words
  the clip contains.

## 9ae. The partition is not stable, and TitaNet may have it wrong here

> **Superseded by 9as-9au.** This section blames `diarization.threads`. That is a
> symptom: thread count is merely one of many things that perturbs the embeddings by
> 1e-07, and *any* such perturbation flipped the result because the clustering was
> being asked to hit a target count. The instability was real and is now fixed; the
> cause named here is not the cause.

With `num_speakers: 4` the same audio gave materially different splits between a
lab run (threads 6, share 28.9/32.6/5.3/33.2) and the production run (threads 5,
share 31.0/8.7/5.9/54.4). Same model, same threshold, same count. Embedding under
a different thread count perturbs the vectors enough to flip a borderline
clustering, so **the diarization result is not reproducible across
`diarization.threads`** -- worth knowing before trusting any single partition,
and a caveat on acceptance test 12.

The operator reports the production labels are wrong in a specific way: 01 and 02
are one person, and 03 is two. That implies a true share near 31/15/27/27.

    TitaNet n=4   28.9 / 32.6 /  5.3 / 33.2
    CAM++   n=4   28.8 / 27.4 / 27.4 / 16.4

Cross-tabulated per word, CAM++ pulls ~1200 words out of TitaNet's largest
cluster into its own third one. That is the direction the operator describes, and
the distribution is much the closer match -- but a distribution match is not an
identity check, so this is **not** adopted. `output\_compare_CAMPLUS_transcript.md`
is rendered for a listening check. Section 9's finding that CAM++ was no better
stands only for the *automatic* path, where neither model produces a usable
centroid gap (0.026 vs 0.035).

## 9af. wav2vec2 forced alignment measured end to end -- not adopted

`whisper.align_cmd` (empty by default) runs an out-of-process forced aligner
after transcription and replaces the word timings. The only implementation is a
torch venv outside the app, which constraint 3 forbids shipping; it exists to
decide whether an onnxruntime port is worth building. Any failure keeps the DTW
words, because a timing experiment must not fail a job that has already paid for
transcription.

Full 2h12m run, same audio, same embeddings, same clusters -- only the timing
method differs:

| | DTW (capped) | wav2vec2 |
|---|---|---|
| words aligned | -- | 15550 / 15550 (100%) |
| overlapping consecutive pairs | 32.5% | **0.2%** |
| turns | 1020 | 1106 |
| speaker changes | 904 | 916 |
| changes starting a sentence | 51.1% | **52.8%** |
| turns of 5 words or fewer | 31.1% | 31.9% |
| alignment cost | 0 | **362 s** (CPU) |

The timestamps really are far better. The transcript is **1.7 points** better on
the only quality measure available, against 5 points on a 10-minute slice --
the gain shrinks with length. Turn boundaries move in both directions: a long
turn now splits correctly on a real pause at 00:00:43, while "I think" is split
off from its own sentence elsewhere.

**The speaker partition is untouched** (28.9/32.6/5.3/33.2 against
28.7/32.5/5.4/33.3). It cannot be otherwise -- alignment does not see speaker
identity -- so this does nothing for the failure the operator actually reported,
which is two clusters being one person and one cluster being two.

**Not adopted.** 1.7 points does not buy torch, ~3 GB of wheels (CPU-only from
PyPI on Windows), a runtime model download and 6 minutes a run. Revisit only if
the clustering is fixed first and boundary precision becomes the limit.

### Two things the run established regardless

- **Editing `server/*.py` while uvicorn is running changes nothing.** Modules
  import once. `diarize_worker.py` is the exception -- a subprocess, re-imported
  per job -- which is why a fix to it appeared to work while four other files
  silently did not. A whole run was wasted on this.
- **The large external context does not reduce map calls.** `chunker.py` only
  shrinks `target_tokens` to fit a smaller window and never grows it, so a 2h12m
  meeting is 4 x 10000 tokens at 81920 exactly as at 32768. The 80k window buys
  reduce headroom, nothing else.

## 9ag. The summary is two sections, not six

At the operator's request, `reduce_summary.txt` drops **Decisions and Outcomes**,
**Points of Disagreement**, **Unresolved Questions** and **Participants**,
leaving Executive Summary and Key Themes. This is a deliberate deviation from
CLAUDE.md section 12, which specifies all six.

The reasoning is that it removes a duplication rather than information. In
`both` mode -- the default -- minutes already carry Decisions, Action Items,
Open Questions, Participants and Attribution Notes, in the terse scannable form
that suits a record. Repeating them as prose in the summary asked the reader to
read the same material twice in two registers. Key Themes is now told
explicitly to carry what was settled, where people differed and what was left
open, inside the theme each arose in, so nothing is dropped -- it moves from a
section of its own into the narrative.

**Known consequence:** a `summary`-only run now has no speaker-label key
anywhere, because Participants was the only place the labels were enumerated.
Renaming is unaffected (it substitutes labels wherever they appear, and they
still appear inline). The six-section original is in git:
`git show <pre-1.2.0-commit>:prompts/reduce_summary.txt`.

## 9ah. DTW against wav2vec2, matched full runs

Both runs: same audio, same cached embeddings, same clusters, same external
server on 9931, `num_speakers: 4`, `both` mode. Only the word timings differ.

| | DTW | wav2vec2 |
|---|---|---|
| wall clock | **731 s** (10.8x realtime) | 1080 s (7.3x) |
| alignment | -- | 362 s |
| turns | 1020 | 1106 |
| changes starting a sentence | 51.1% | **52.8%** |
| turns of 5 words or fewer | **31.1%** | 31.9% |
| word share | 28.9/32.6/5.3/33.2 | 28.7/32.5/5.4/33.3 |

The share is identical to within 0.2 points, as it must be -- alignment cannot
see speaker identity. Everything wav2vec2 changes is a boundary moving a word or
two, and reading the divergences it goes both ways: it correctly splits a long
turn on a real pause at 00:00:43, and it incorrectly splits "Maybe it can't" from
"be." and breaks "check out" across two speakers. Confirms 9af: **not adopted.**

### The documents vary far more than the timing method does

| | DTW | wav2vec2 |
|---|---|---|
| summary words | 3215 | 3848 |
| action items | 11 | 8 |
| `(HH:MM:SS)` in minutes | **85** | **1** |

The wav2vec2 run's minutes fell back to "(early §1)" and "(§3)" almost
throughout, which section 12 requires to be real timestamps. The matched DTW run
produced 85 of them from the same prompt, the same server and the same notes
structure.

**This is not attributed to the aligner.** The final reduce runs at
`temperature: 1.0` (section 11.2) and this is one sample per arm. What it does
establish, and what was not recorded anywhere before, is that **minutes output is
high-variance run to run** -- enough that a single run is not evidence about any
change upstream of it. Any future comparison of document quality needs several
runs per arm, or a lower temperature for the comparison.

## 9ai. Ground truth from the operator: CAM++ 6/6, TitaNet 4/6

> **Read with 9at and 9au.** Each arm here is a *single* clustering, and at the time
> a single TitaNet clustering was a coin flip between two partitions. Some of the
> 6/6-against-4/6 gap was therefore luck. With the count path fixed (9au), TitaNet
> scores 6/6 on these same clips. The conclusion drawn below -- that CAM++ is the
> better embedder -- is not established by this evidence.

The operator listened to the voice samples from two runs and identified two
clips as the wrong speaker. Both are the same failure: cluster 03 swallowing
speaker 02. Majority vote over each clip's window, same audio and segmentation,
`num_speakers: 4`:

| clip | truth | TitaNet | CAM++ |
|---|---|---|---|
| 00:23:50 | spk 02 | S3 (100%) **wrong** | S2 (95%) right |
| 01:18:04 | spk 02 | S3 (100%) **wrong** | S2 (100%) right |
| 00:17:13 | spk 02 | S2 right | S2 right |
| 00:53:22 | spk 02 | S2 right | S2 right |
| 01:44:41 | spk 02 | S2 right | S2 right |
| 01:14:50 | spk 03 | S3 right | S1 right |

**TitaNet 4/6, CAM++ 6/6.** CAM++ fixes both errors and matches TitaNet
everywhere TitaNet was right. Taken with the distribution argument in 9ae, and
with CAM++ being trained Chinese-English against TitaNet's English-only on a
Singlish recording, this is now the best-supported change available and it costs
no new dependency -- `diarization.embedding_model` selects it, and the
thresholds are bypassed when a count is supplied.

### The voice samples, and a fix that was reverted

The operator also reports the DTW run's clips contain heavy cross-talk while the
wav2vec2 run's are clean. The two runs pick almost entirely different clips (one
of twelve in common), because clip ranking scores speech density from word
intervals and DTW's inflated overlapping ends make two people at once look like
one dense speaker.

A penalty was written for this, scoring candidate windows by diarization overlap
with other speakers -- and **reverted**, because it cannot be shown to work:

- Measured other-speaker *words* inside each clip window: **0.00 s for both
  runs**. During cross-talk whisper transcribes one voice, so the second speaker
  leaves no words to count. The transcript cannot see what the operator hears.
- Measured diarization *overlap*: 0.0-0.6 s in an 18 s clip, noise-level, and it
  still moved two of four clips. Where the overlap is between two people inside
  one cluster -- which is exactly the 03-swallows-02 failure above -- it is
  same-label and invisible to the measure anyway.

So this stays an open problem with no cheap fix. It does revise 9af's verdict in
one respect: wav2vec2's real benefit here is **sample quality**, not transcript
accuracy, and that is a user-facing feature (section 13.9 -- hearing a voice is
how you identify someone). Still not adoptable at the cost of torch, but it
strengthens the case for an onnxruntime aligner if this ever gets revisited.

## 9aj. Forced alignment, measured then built

### The blind test that justified it

Six summaries, three per arm, from the *cached notes* of two finished runs of
the same meeting -- so nothing was re-transcribed and the only difference
between arms was which transcript produced the notes. Shuffled, unlabelled,
rated by the operator out of 5:

| | scores | mean |
|---|---|---|
| DTW timings | 2, 3, 2 | 2.33 |
| wav2vec2 timings | 5, 4, 5 | **4.67** |

No overlap between arms. With 3 against 3 that is p = 1/20 = 0.05, the strongest
result the sample size allows.

**The prediction this overturned was mine.** I argued alignment could not affect
the summary because all three runs share an identical 15550 words. True, and
beside the point: the map stage does not read words, it reads
*speaker-attributed turns*, and turn boundaries are computed from the timings.
Section 8 says diarization "gives the model dialogue structure to reason over";
that structure is exactly what better timings improve. The words being identical
is what makes the result clean -- nothing varied but structure.

### What shipped

`server/align.py`, under onnxruntime, no torch:

    emissions -> trellis -> Viterbi backtrace -> character spans -> word spans

torchaudio's algorithm reimplemented in numpy. `models/wav2vec2-align.onnx` is
WAV2VEC2_ASR_BASE_960H exported once by `tools/export_align_onnx.py`. torchaudio
publishes no .onnx, so like the Vulkan whisper build (section 2.1) it is ours to
distribute -- but unlike that one it **travels in the full zip**: 360 MB raw
deflates to 220 MB, taking the bundle from 1.24 GB to 1.47 GB, still clear of
GitHub's 2 GB per-asset limit. A first install therefore has it without a
further download.

`DOWNLOAD_MODELS.bat` fetches it too, for an install made from the source zip,
but through a new `:get_optional` -- a failure there prints a line and carries
on instead of setting `FAILED`. The app runs without the model; telling someone
their install is broken because an optional file did not arrive would be false.
(`:get_optional` saves and restores `FAILED` rather than clearing it, so it
cannot erase an earlier real failure.)

Validated word for word against the torch implementation over the same
recording -- the same algorithm on the same audio, so this is a regression test
rather than an eyeball (`tools/align_check.py`):

| | |
|---|---|
| words aligned | 15468 of 15550 (99.5%) |
| median delta vs torch | **0.000 s** |
| within 0.10 s | 94.3% of starts |
| wall clock | **274 s**, against torch's 362 s |

The 0.5% it skips are digit-only words -- "20" has no character in an alphabet
of 26 letters and an apostrophe. Those keep whisper's timings rather than being
dropped, because the caller indexes positionally and a missing word would shift
every speaker assignment after it.

Downstream, which is the acceptance test that matters:

| | turns | clean changes | overlapping word pairs |
|---|---|---|---|
| DTW | 1020 | 46.6% | **31.9%** |
| torch align | 1106 | 47.9% | 0.2% |
| **ONNX align** | 1114 | **47.3%** | **0.8%** |

### int8 rejected

Dynamic quantisation gives 90.8 MB against 360.3, and is worse on both axes:
**5.7x slower** (240 s against 42 s on a 20-minute slice) and less accurate
(clean changes 39.5% against 43.3%, only 20.4% of words within one frame of
fp32). Presumably the conv stack falls off its fast kernels. fp32 ships.

### Limits

- **English only.** The label set is 26 letters and an apostrophe. Any other
  `whisper.language` skips the stage and logs it -- the failure mode otherwise
  is not an error but confident nonsense, aligning to the wrong phonemes.
- **Not required.** Absent from `config.REQUIRED_FILES`: without the model the
  pipeline runs on whisper's DTW timings exactly as before.
- **Never fatal.** Every failure path -- missing model, bad segment, exception,
  cancellation -- returns the DTW timings. It runs after transcription has been
  paid for and must not be able to waste it.
- It costs ~275 s of CPU on a 2h12m recording, after whisper has released the
  GPU, so `align_threads` defaults to `whisper.threads` and section 7's thread
  budget is preserved.

## 9ak. An update that added a model did not bring the model

Reported immediately after 1.2.0: updating a 1.1.0 install from the button
succeeded and relaunched, and nothing anywhere said that a 360 MB model was now
expected. The app degraded silently to whisper's own word timings and logged it
to `temp\job.log`, which nobody reads. The user got a working application
producing quietly worse summaries -- the exact failure section 7.1's
"never fatal" rule was supposed to make *safe*, not *invisible*.

The update payload is the source zip and nothing else, by design (section 9q): a
230 KB update must not become a gigabyte, and the full bundle carries
`runtime\python.exe`, which cannot overwrite the interpreter executing it. So
`models\` was never in scope. That was correct while releases only changed code
and wrong the moment one added a model.

Telling the user to run `DOWNLOAD_MODELS.bat` afterwards is not a fix either.
They updated from a button inside the app; there is no reason for them to
suspect a second step exists, and a step nobody knows about has not happened.

**`updater.EXTRA_MODELS`** now lists what a version needs that an older install
cannot have. After the payload is verified and before the restart, anything
missing is downloaded into `models\`:

- **Size-checked, not just existence-checked.** A truncated file from an earlier
  attempt is worse than an absent one, because the app would load it.
- **Downloaded to `.partial`, then renamed.** A failure leaves nothing behind.
- **Never fails the update.** These models are optional by construction, so a
  failure warns and carries on rather than abandoning an update that is
  otherwise complete and already verified.
- **And it is visible.** `extra_failed` reaches the About panel. Silence here is
  the bug being fixed; replacing one silent degradation with another would miss
  the point.

Verified against the live release: both files downloaded, SHA-256 identical to
the originals, no `.partial` left behind, and the downloaded model aligned 217
of 217 words. A deliberately truncated file was correctly re-reported as
missing.

**Fresh installs are unaffected either way** -- the model travels inside
`-full.zip`, and `DOWNLOAD_MODELS.bat` fetches it for a source install. This
was only ever the upgrade path.

## 9al. A self-updater cannot install its own improvements

9ak added `updater.EXTRA_MODELS` so an update that adds a model brings the
model. It does not work for the people who need it, and the reason is
structural: **the update is carried out by the old version's code.** The batch
script only waits, copies and restarts; everything before it -- download,
verify, and the new model fetch -- runs in the Python that is already
installed.

So an install on 1.0.3, which is where most of them are, runs 1.0.3's
`install()`. That has never heard of `EXTRA_MODELS`. It lands on the new code
with the model absent, exactly as before. The fix in 9ak only covers 1.2.1 and
later upgrading to something later still, which is nobody yet.

A self-updater can only ever improve *the next* update. Anything that must be
true after upgrading from an arbitrary old version has to be checked by the new
version, at startup, on its own.

### What the new version does

`GET /api/components` is a purely local filesystem check -- nothing is fetched
and nothing is contacted -- reporting which `EXTRA_MODELS` are absent or
truncated. When any are, the front page carries a notice and one button, which
posts to `/api/components/fetch`.

Constraint 1 permits exactly this shape: the user pressed a button. Nothing
checks on startup, on a timer, or in the background -- the *check* is local, and
only the *download* leaves the machine, on an explicit press.

The notice is on the front page rather than in the About overlay deliberately.
It changes the quality of every summary produced, and the entire failure being
fixed is one that nobody could see.

### Grouped by capability, not by file

Two files make word alignment work, and the first version of the notice listed
both: *"The word alignment model and word alignment labels isn't installed"* --
wrong number, and wrong unit. `EXTRA_MODELS` entries now carry a `component`,
and the notice names what the user loses once, however many files it takes.

Found by looking at it in a browser. The static checks passed -- balanced
braces, every element id resolving -- and neither can see a sentence that does
not agree with itself.

### Verified

Against the live release, with the models moved aside to simulate an upgraded
1.0.3 install: the notice appears, the button downloads with a live byte count,
both files arrive SHA-256 identical, the notice disappears by itself, and
`/api/components` returns empty.

## 9am. The shipped config.json carried this machine's Port selection

Reported after downloading a fresh 1.2.2: the Model dropdown defaulted to
**Port**, on a machine with nothing listening on 9931.

`config.json` is tracked, because it is the default a new install gets. The app
also writes to it -- choosing Port in the dropdown sets `llm.external`
(CLAUDE.md section 13.3). So developing with Port selected dirties a tracked
file, and `git add -A` ships it. Commit 80d7e0e did exactly that, and 1.2.0
through 1.2.2 all carry `external.enabled: true`.

**Existing installs were never affected**: `version.PRESERVE` is
`("config.json",)`, so an update leaves it alone. Only a fresh install from
`-full.zip` saw it.

Two fixes, because the default being wrong and the failure being late are
separate problems:

### The build refuses to ship a dirty config

`make_release.check_shippable_config` fails the build when
`llm.external.enabled` is true. It fails rather than normalising: the working
tree and the zip should not disagree about what was shipped. Verified by
setting it true and watching the build refuse.

### The failure now happens before the work, not after it

The LLM is not contacted until after transcription and diarization, so that
nothing else is in VRAM when the model loads (section 11.1). The cost is that
"nothing answered on that port" surfaces fifteen minutes into a job that could
never finish -- which is what a fresh install would have hit.

`llm.preflight_external` does a TCP connect to `127.0.0.1` at job start, before
ffmpeg. Local, instant, and no part of constraint 1 is in play. It checks
reachability only; whether the server is *healthy* remains
`LlamaServer.start()`'s business at the point it attaches.

Verified across all three arms: external off does not probe at all, external on
with a dead port refuses immediately, external on with a listening port passes.

## 9an. The browser cached the frontend, so the fix was invisible

Reported: a fresh 1.0.3 updated to 1.2.3 and showed none of the new UI. The
missing-component notice appeared only after clearing the browser cache, which
nobody has any reason to do.

The frontend is served with no `Cache-Control` at all. That is not "do not
cache" -- it is *heuristic* caching, where the browser decides for itself how
long a 200 stays fresh and serves it without asking. The updater replaces
`app.js` and `index.html` underneath a tab that goes on running the old ones.

Every route now sends `Cache-Control: no-cache, must-revalidate`, which means
"ask before reusing", not "never store". The ETag `FileResponse` already sets
makes the usual answer cheap, and on 127.0.0.1 even a full re-send is nothing.

**This made 9al ineffective in exactly the case it was written for.** The
component check shipped in 1.2.2 and did work -- but the person most likely to
need it is upgrading from an old version, and they are also the person most
likely to have a stale `app.js` cached. A fix nobody can see is not a fix, which
is the second time that sentence applies in this file.

## 9ao. Offering the language model the machine should have

An install updating from 1.0.x keeps the `Q4_K_M` it downloaded and never
receives `UD-IQ4_XS`: the update payload carries no `models\`. It runs -- the
legacy entry in `hardware.LEGACY_MODELS` exists so it can -- about 3.4x slower
than the model it should be using (section 9z), and nothing says so.

The front-page notice now offers it, next to the word aligner.

**Not in `EXTRA_MODELS`.** Those are fetched automatically during an update,
which is correct for 360 MB and completely wrong for 14 GB: an update must
never silently become a download that size. `offered_language_model()` is only
ever *offered*, with the size stated, behind a button.

### It asks what the machine should have, not what it will run

The first version called `hardware.choose_key()`, which answers the wrong
question: it picks from what is **on disk**. An install carrying Q4_K_M is
offered Q4_K_M, nothing is missing, and the faster model is never mentioned --
precisely the case this exists for. Caught by simulating it: the IQ4_XS file was
moved aside and the endpoint reported nothing missing.

It now picks from `hardware.MODELS` -- the shipped set -- by VRAM, with the same
unified-memory rule `choose_key` applies. Only the recommended one is offered:
suggesting the 14.3 GB model to a machine that will run the 10.9 GB one is a
14 GB mistake, and offering both is a 25 GB one.

### Two wording bugs, both only visible in a browser

- The notice explained every missing thing as *"speaker attribution and summary
  quality are noticeably worse"*, which is true of the aligner and nonsense
  about a language model. Each component now carries its own `reason`.
- Sizes were rendered in GiB: **13.3 GB** beside a dropdown saying **14.3 GB**,
  which reads as two different files. Decimal throughout now, matching the
  registry and the documentation.

Neither is visible to a linter, and both were obvious on screen.

## 9ap. A header cannot fix a cache entry that is never revalidated

9an added `Cache-Control: no-cache` and it did not work. Reported again: 1.0.3
updated to 1.2.4 and still showed the old page until the cache was cleared by
hand.

The reason is the same shape as 9al. The stale entry was stored by the **old**
server, which sent no `Cache-Control` at all, so the browser gave it a freshness
window of its own choosing. Inside that window the browser does not ask -- it
serves the old copy and never discovers that the header changed. A response
header can only govern responses the browser actually requests.

What defeats an entry that is never revalidated is a **different URL**, because
that is a different cache key:

- `index.html` references `/app.js?v=__APP_VERSION__` and the same for the
  stylesheet; `main.index` substitutes `version.APP_VERSION` when serving.
- `open_browser` opens `http://127.0.0.1:<port>/?v=<version>`, so the *document*
  is a new key too -- without that the browser serves a stale page which
  references the old asset URLs, and the stamped assets are never reached.
- The `Open Meeting Summariser.url` shortcut written by run.bat carries it as
  well.

This works on the next update because the restart is performed by the new
run.bat, which runs the new `open_browser`. The `no-cache` header stays: it
stops the problem being created again for anyone whose cache is populated from
here on.

## 9aq. Two windows during an update, and a third that looked like a crash

- **The update script had a console.** It was spawned with `DETACHED_PROCESS`,
  under which `cmd.exe` allocates a console of its own, so a black window
  running the `ping` wait loop appeared beside the new app's window.
  `CREATE_NO_WINDOW` gives it a console with no window. The two flags are
  mutually exclusive, so this is a swap rather than an addition, with a
  fallback if the constant is unavailable -- a script that dies with its parent
  cannot copy the files.

- **The old console sat on `pause` saying "Meeting Summariser has stopped".**
  Correct after a crash, wrong after an update: the user saw the new window
  open while the old one claimed failure behind it. `_quit` now writes
  `temp\updating.flag` and run.bat closes quietly when it finds it.

**Both only take effect one update later.** The update is carried out by the
installed version's `updater.py` and its `run.bat`, so 1.2.4 -> 1.2.5 still
shows the old behaviour and 1.2.5 -> 1.2.6 is the first clean one. Same
constraint as 9al; worth stating rather than letting it look unfixed.

## 9ar. Offer every shipped model, not the one this machine would pick

9ao offered only the recommended model, reasoning that suggesting 14.3 GB to a
machine that will run 10.9 GB is a 14 GB mistake. That was wrong, and the
operator's reason is the one already encoded in `DOWNLOAD_MODELS.bat`, which
fetches both: **the folder is portable**. Section 16 requires that copying it to
another drive or machine works unchanged, so the machine that downloads is
routinely not the machine that runs -- fetched on a laptop, copied to an on-prem
box with a far better card. Offering only what the downloading machine needs
quietly strips the folder of the model the destination wanted.

`offered_language_models()` returns every shipped model that is absent. On an
install with neither, the notice reads 25.2 GB across 2 files, which is what
`DOWNLOAD_MODELS.bat` would have fetched anyway.

Still never in `EXTRA_MODELS`: 25 GB must not arrive automatically with an
update.

## 9as. The clustering decides by coin flip, and section 9ae named the wrong cause

Chasing why two runs of the same pipeline on the same audio produced different
speakers. Section 9ae blamed `diarization.threads`. That is a symptom.

Measured, each clustering call in its own process (which matters -- see below):

| stage | thread-dependent? |
|---|---|
| segmentation | **no** -- bitwise identical labels at 5 and 6 threads |
| embedding | yes, but only just: max delta **3.5e-07** |
| clustering | deterministic given its input |

So nothing here is a race and nothing is random. Every stage is reproducible.
What is not reproducible is the *composition*: a 1e-7 difference in the
embeddings changes which speakers come out.

### How often

Perturbing the embeddings by 3e-07 -- the magnitude a different thread count
actually produces -- twelve times, `num_speakers: 4`:

    38.7 / 30.1 / 24.6 / 6.6   x6      the balanced partition
    59.9 / 24.6 /  8.4 / 7.1   x5      one cluster swallowing two speakers
    54.8 / 31.9 / 11.8 / 1.5   x1      a third answer

**A coin flip.** There is no sense in which a thread count is correct: 1 and 6
won the toss on this recording and 5 lost it. Every thread count from 1 to 8
produces a different embedding hash -- there is no even/odd structure, no
grouping, nothing to pin.

The defect is that `FastClustering` has merge decisions on this recording so
finely balanced that the last decimal place decides them, and the outcome is
then amplified into an entirely different set of speakers.

### What this invalidates

- **Section 9ae** attributes the instability to thread count. It is one of many
  things that perturbs the input by 1e-7; any of them flips the result.
- **Section 9ai's CAM++ result** (6/6 against TitaNet's 4/6 on operator-verified
  clips) came from one clustering of each. At a ~50% flip rate some of that gap
  may be luck. The direction may well be real; it was reported as settled and
  it was not.
- **Section 9aj's blind summary test is unaffected.** Both arms used the same
  cached embeddings and therefore the same partition, so the 4.67-against-2.33
  result is a clean comparison of word timings.

### `FastClustering` is not stateless

The first `cluster()` call in a process returns a different partition from every
call after it, on identical input:

    call 1: 38.7 / 30.1 / 24.6 / 6.6
    calls 2-5: 59.9 / 24.6 / 8.4 / 7.1   (stable among themselves)

Across *fresh processes* the first call is perfectly reproducible -- six
processes, identical labels. The app is therefore safe: `diarize_worker` is a
subprocess that clusters exactly once.

`tools/diar_lab.py` was not. Its `sweep`, `table` and `pipeline` commands loop
over many calls in one process, so every row after the first described a regime
the app never enters. Two of the measurements reported earlier today came from
it. Fixed by giving each clustering its own process.

## 9at. Consensus does not help; a better embedder removes the problem

Two candidate fixes for the coin flip in 9as, both measured with
`tools/consensus_test.py` -- K clusterings under a 3e-07 perturbation, scored by
pair-counting agreement, which needs no correspondence between labellings.

### Consensus clustering: no

Take the partition that agrees most with the other K-1. Four independent trials,
K=11, TitaNet, `num_speakers: 4`:

| trial | vote | consensus pick |
|---|---|---|
| 1 | 6/11 balanced, 4/11 merged, 1/11 other | balanced |
| 2 | 8/11 merged, 3/11 balanced | merged |
| 3 | 6/11 merged, 5/11 balanced | merged |
| 4 | 6/11 merged, 5/11 balanced | merged |

**The majority is itself a coin flip.** With the split near 50/50 no K converges;
more sampling only picks the more likely of two arbitrary answers. Mean
agreement 0.85-0.90 -- the partitions concur about most pairs and differ on a
large minority, which is exactly what two near-equally-supported solutions look
like.

So the ambiguity is not noise to be averaged away. Under TitaNet, this recording
genuinely has no well-defined four-way split.

### A better embedder: yes, and it is not close

The same test on CAM++ embeddings, same audio, same segmentation, same
clustering code:

| | TitaNet | CAM++ |
|---|---|---|
| `num_speakers: 4`, 4 trials x 11 | coin flip | **44/44 identical**, agreement 1.0000 |
| auto, 2 trials x 7 | -- | **14/14 identical**, agreement 1.0000 |

CAM++ separates these speakers well enough that there is no knife-edge to fall
off, and its partition is four balanced speakers -- 30.4 / 25.2 / 24.6 / 19.8 --
against the operator's account of four active participants.

**This is the argument section 9ai should have made.** That one compared a
single TitaNet clustering against a single CAM++ clustering on six verified
clips and called 6/6 against 4/6 decisive. Given a ~50% flip rate, that gap was
partly luck. Stability under perturbation is the property worth testing, because
it says the solution is well-defined rather than that one draw happened to land
well.

### The tension to resolve

The operator listened and judged TitaNet more accurate, and CAM++ was reverted
on that basis. But that compared one TitaNet *draw* -- and there was a coin
flip's chance it was the other partition. The judgement was sound about what was
heard; it does not establish that TitaNet is better, because TitaNet has no
stable answer to be better.

CAM++ is reproducible, so it can be judged once and the verdict holds.

## 9au. Never ask FastClustering for a count

9as found the partition decided by a coin flip; 9at found consensus cannot fix
it and CAM++ does not suffer from it. The cause is narrower than either:

| path | TitaNet | CAM++ |
|---|---|---|
| `num_clusters=K` | **coin flip** | stable |
| threshold (auto) | stable, 13/14 identical | stable, 14/14 |

It is asking `FastClustering` for a **target count** that is unstable. Its
threshold path is well behaved for both embedders. And the count path is the one
CLAUDE.md section 13.4 calls "by far the most reliable", so the single most
trusted input was running through the single least reliable code path.

`merge_to_count` takes that step back: over-cluster on the stable threshold path,
then merge centroids down to the requested count in numpy, where the arithmetic
is ours and `argmin` breaks ties by lowest index the same way every run. It is
`merge_close_clusters` with a count as the stopping condition instead of a
distance, for the same reason -- a centroid averages hundreds of vectors and is
far better conditioned than any one of them.

Measured on TitaNet, 9 runs under the same 3e-07 perturbation:

    before   38.7/30.1/24.6/6.6  or  59.9/24.6/8.4/7.1   -- a coin flip
    after    8/9  32.6/27.8/27.0/12.6
             1/9  32.8/27.8/26.9/12.5   -- the same partition, a few segments moved

### It also fixes the errors found by ear

Against the six clips the operator verified (9ai), TitaNet through the new path:

| clip | truth | old TitaNet | new path |
|---|---|---|---|
| 00:23:50 | spk 02 | S3 **wrong** | S1 (96%) |
| 01:18:04 | spk 02 | S3 **wrong** | S1 (100%) |
| 00:17:13 | spk 02 | S2 right | S1 (100%) |
| 00:53:22 | spk 02 | S2 right | S1 (100%) |
| 01:44:41 | spk 02 | S2 right | S1 (100%) |
| 01:14:50 | spk 03 | S3 right | S2 (100%) |

**6/6**, with the shipped embedder. Share 30.3 / 11.5 / 28.7 / 29.5 -- four
speakers, no cluster swallowing the meeting.

CAM++ is therefore no longer needed to get a stable answer, though it remains
independently stable and is still selectable through
`diarization.embedding_model`.

### Confirmed on a second recording

The 3h25m council meeting -- clean audio, the one sections 9e and 9 tuned the
snap window and the 0.35 centroid threshold against -- is unstable too, just
less obviously:

| | old path | new path |
|---|---|---|
| 2h12m cross-talk | two partitions, ~50/50 | same partition, +/-0.2% |
| council 3h25m | two partitions, ~70/30 | same partition, +/-0.1% |

So this is not a property of difficult audio. A clean recording flips as well;
it simply flips less often, so nobody noticed.

**Which casts a shadow on how several constants here were tuned.** The snap
window of 6 (9e), `merge_centroid_distance` of 0.35 (section 9), the threshold
sweep in 7a -- each was measured on outcomes from that recording, on top of one
draw from a distribution that had two. The snap measurements used a fixed
transcript and are probably unaffected; the clustering ones may not be.
`tools/consensus_test.py` is the way to check any of them.

**Evidence limits.** Two recordings, nine perturbation trials each, six
ground-truth points on one of them.

---

# What this codebase keeps teaching

Five patterns cost most of a day between them, each more than once. They are
recorded together because none of them is visible from inside the section it
first appeared in.

## 1. A self-updater can only fix the *next* update

The update is carried out by the version being updated **from**: its
`updater.py` downloads and verifies, its `run.bat` restarts. Nothing in the new
version runs until it is already installed.

So anything about *how an update behaves* -- fetching a newly-added model
(9ak), the windows it opens (9aq), the console it leaves behind -- is fixed one
release later than it is written. Three separate fixes were shipped believing
they would help the person updating from an old version, and none of them
could.

What *does* work is the new version checking on its own behalf, once it is
running (9al). If a release adds something users must have, the new version has
to notice and say so; the update cannot be relied on to carry it.

## 2. A response header cannot reach a browser that never asks

`Cache-Control: no-cache` was added and did nothing, twice (9an, 9ap). The
stale copy had been stored by the *previous* version, which sent no header at
all, so the browser picked a freshness window of its own and inside it stopped
asking. Headers govern responses that are requested; they cannot govern an entry
that is never revalidated.

Only a different URL defeats that, because it is a different cache key. The
version is now in the asset URLs *and* the document URL -- the document matters,
because a stale page references the old asset paths and never reaches the
stamped ones.

## 3. Measure the thing that varies, not one draw of it

The deepest defect here (9as) went unnoticed for the life of the project because
nobody ran the same file twice and compared. Worse, several constants were
*tuned* on single draws from a distribution that had two very different outcomes
-- and two of the measurements made while hunting this very bug were themselves
single draws, one of which produced a confident retraction of a correct claim.

Before trusting a number from this pipeline, ask whether the quantity behind it
is stable. `tools/consensus_test.py` answers that: perturb the embeddings by the
amount real machine variation produces and report agreement between runs.

## 4. A library call is not a black box you can ignore

`sherpa_onnx.FastClustering` is not stateless: the first call in a process
returns a different partition from every call after it, on identical input. That
is undocumented, and it silently invalidated every sweep `tools/diar_lab.py`
produced, because the app clusters once per job and the tool clustered in a loop.

It also offers no seed and no tie-break control, which is precisely why the fix
in 9au was to take the final decision back into our own numpy, where `argmin`
breaks ties by lowest index every run. Prefer code whose determinism you can
read over a library call whose determinism you are assuming.

## 5. Look at the running application

Three defects shipped past a passing linter, a balanced-braces check and a
manifest of every element id, and were obvious within seconds on screen:

- *"The word alignment model and word alignment labels isn't installed"* --
  plural subject, singular verb, because two files were being listed where the
  user cares about one capability.
- Every missing component explained as *"speaker attribution is worse"*, which
  is true of the aligner and nonsense about a language model.
- **13.3 GB** rendered beside a dropdown saying **14.3 GB**, because one was
  GiB and the other decimal.

None of these is reachable by a static check. A screenshot is a test.

## 9av. `diarization.threads` is detected now, and it is safe to be

Embedding is the bulk of a run -- about 24 of the 28 minutes on a 2h12m
recording -- and `diarization.threads` was hard-coded to 5 on every machine,
while section 4 says everything machine-dependent is detected rather than
guessed. Section 7 also says the default is 6; `config.json` shipped 5. Nobody
had reconciled them.

Measured on a 12-core/24-thread Ryzen 9 3900X, the full 7925-window pass, idle:

| threads | embed | against 5 |
|---|---|---|
| 5 | 1460 s (24.3 min) | -- |
| 8 | 1356 s (22.6 min) | 1.08x |
| 12 | **1188 s (19.8 min)** | **1.23x** |

`diarization.threads: "auto"` now resolves to half the logical cores, clamped
to [5, 12] (`diarize.resolve_threads`). An explicit number still wins.

- **12 is the cap** because the curve flattens: over a fixed slice, 12 to 20
  threads is 67% more threads for 6% less time, and during the concurrent phase
  those threads compete with whisper.
- **5 is the floor** because that is the constant this replaces, not because it
  was measured. `cores // 2` gives 2 on a quad-core, and 2 is the worst number
  in the table. Detection must not make any machine slower than the constant it
  replaces. No small machine has been measured.
- On 12 logical cores this asks for 6 where section 7's budget allows 5.
  Accepted: whisper is GPU-bound and its threads mostly feed the card.

### It is only safe because of 9au

Thread count changes the embeddings by 3.5e-07, and until 9au that was enough to
re-roll the speaker assignment. Verified after the fix -- the full pass at 1, 5,
8 and 12 threads produces **the same partition, same label hash**
(32.6/27.8/27.0/12.6). Before 9au this tuning would have been unshippable: it
would have traded four minutes for a different set of speakers.

### A benchmark that measured the wrong thing

The first attempt timed a 300-window slice, and was then compared against a 953 s
figure lifted from a production job log. That gave "12 threads is 25% slower",
the opposite of the truth. The slice was fine -- it predicted 1.30x against an
actual 1.23x -- but the baseline was a run under different conditions, and a
controlled pair was needed, not a convenient number already written down.

`tools/thread_bench.py` sweeps counts over a slice. Useful for shape; the
decision needs the full pass.
