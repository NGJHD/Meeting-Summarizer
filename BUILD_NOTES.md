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
  Leaving it unset, as section 11.1 advises, is correct: per-request control governs.

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
| GitHub per-asset limit | 2 GB |

So size was never the obstacle for the binaries, and neither was licensing -- everything
in there is MIT, BSD, Apache, PSF or LGPL, all redistributable. **Only the models are
impossible:** `Qwen3.8-27B-UD-Q4_K_M.gguf` is 16.5 GB and `IQ3_XXS` is 10.9 GB, each on
its own over the 2 GB limit, so they can never be release assets whatever the packaging.

A release therefore carries two zips, built by `tools/make_release.py`:

- `Meeting-Summariser-vX.Y.Z-full.zip` (~1.14 GB) -- a first install. Unzip it and only
  the models remain to download.
- `Meeting-Summariser-vX.Y.Z.zip` (~190 KB) -- the update payload.

### The updater must never take the full one

Two reasons, and the second is the real one:

1. A 190 KB update would become 1.14 GB, to ship a few changed `.py` files.
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
"nothing to install", not "install the 1.14 GB one". Exercised against seven asset layouts
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

## 10. Still not measured

- Tokens/second during real map calls on the target hardware.
- Wall time by stage for a full 4-hour and 8-hour run including the LLM stages.
- An 8-hour recording end to end. Everything above is measured at 3h25m; section 0 now
  requires testing at 8 hours.
- Final unzipped folder size is currently **~19 GB** (16.5 GB model, 1.8 GB binaries,
  253 MB runtime). `tools\` adds 180 MB and should be deleted before shipping.
