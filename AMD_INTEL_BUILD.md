# Running on AMD and Intel graphics

An assessment, not an implementation. Question asked: how much work is it to make this
app run on AMD or Intel graphics, auto-detecting the vendor at startup? Plus: do NPUs
help, and does the current build work on integrated graphics?

**Short answers**

| Question | Answer |
|---|---|
| How much code change? | **Small — roughly 150 lines**, concentrated in two files. The bulk of the work is downloading and shipping more binaries. |
| Is it just configuration? | Nearly. One new detection function, one new lookup for which folder to launch from, one build step that is not currently done. |
| Does the current build work on an iGPU? | **No.** See §5. It is CUDA-only, so an Intel or AMD iGPU gets nothing and falls back to CPU, which is unusably slow for a 27B model. |
| Do NPUs help? | **Not usefully, today.** See §4. |
| Confidence it would work | **~85% for AMD/Intel discrete and modern iGPU via Vulkan. ~40% that it is *pleasant* on an iGPU.** See §7. |

---

## 1. What actually depends on NVIDIA

Three components do inference. Only two of them care about the GPU vendor.

| Component | Today | Vendor-locked? |
|---|---|---|
| `bin/llama/llama-server.exe` + `ggml-cuda.dll` | CUDA 12.4 build | **Yes** |
| `bin/whisper/whisper-cli.exe` + `ggml-cuda.dll` | CUDA 12.4 build | **Yes** |
| `sherpa-onnx` diarization | CPU-only by design (CLAUDE.md §8) | **No — already portable** |
| `bin/ffmpeg.exe` | CPU | No |
| `cublas64_12.dll`, `cublasLt64_12.dll`, `cudart64_12.dll` (573 MB) | NVIDIA runtime | Yes, and droppable on other vendors |

Diarization is the pleasant surprise: it was made CPU-only to avoid PyTorch, and that
decision happens to make a third of the pipeline vendor-neutral for free.

---

## 2. What you would ship instead

### llama.cpp — easy, official binaries exist

`ggml-org/llama.cpp` publishes prebuilt Windows binaries every build. As of `b10852`:

```
llama-b10852-bin-win-cuda-12.4-x64.zip     <- what we ship now
llama-b10852-bin-win-vulkan-x64.zip        <- AMD + Intel + NVIDIA, one binary
llama-b10852-bin-win-rocm-10.0-x64.zip     <- AMD discrete, faster than Vulkan
llama-b10852-bin-win-sycl-x64.zip          <- Intel oneAPI
llama-b10852-bin-win-openvino-2026.3.1-x64.zip  <- Intel CPU/GPU/NPU
llama-b10852-bin-win-cpu-x64.zip           <- last-resort fallback
```

**Vulkan is the answer for a shipping app.** One binary covers AMD, Intel and NVIDIA,
integrated and discrete, and it needs no vendor runtime installed — just a current
display driver, which is already the app's only stated external dependency. ROCm and
SYCL are faster on their own hardware but each drags in a large vendor toolchain and
another matrix of driver versions to support.

### whisper.cpp — this is the real work

`ggml-org/whisper.cpp` release assets are **CPU, BLAS and cuBLAS only**. Checked at
v1.9.2:

```
whisper-bin-x64.zip                 whisper-blas-bin-x64.zip
whisper-cublas-11.8.0-bin-x64.zip   whisper-cublas-12.4.0-bin-x64.zip
```

There is **no prebuilt Vulkan, ROCm or SYCL whisper binary.** The source supports all
three (`-DGGML_VULKAN=1`, `-DGGML_HIP=1`, SYCL), so it is a documented one-line CMake
build — but it is a build, on a machine with the Vulkan SDK installed, and then it has to
be tested. That single fact is most of the risk in this whole exercise.

Two ways out:

1. **Build whisper.cpp with Vulkan once**, ship the result. ~1 hour of setup, then it is
   just another folder in `bin\`.
2. **Leave whisper on CPU for non-NVIDIA machines.** Costs roughly 5–10× on the
   transcription stage. On this project's numbers (813 s per audio-hour on GPU) a 4-hour
   meeting goes from ~55 minutes of transcription to perhaps 5 hours. Not acceptable.

So: option 1, and it is the only step that is not "download a zip".

---

## 3. The code changes

Genuinely small. Everything else in the pipeline is already indirection through
`config.py` paths and `hardware.py`.

### 3.1 `server/hardware.py` — detect the vendor (~60 lines, new)

Today VRAM detection is one call to `nvidia-smi` (`detect_vram_mb`). It returns 0 on any
non-NVIDIA machine, which is why the model dropdown would silently pick the small model.

Needs a vendor probe plus a vendor-appropriate VRAM read:

```python
def detect_gpu():
    """-> ("nvidia"|"amd"|"intel"|"none", vram_mb)"""
    # 1. nvidia-smi                     -> nvidia, exact VRAM
    # 2. WMIC/CIM Win32_VideoController -> vendor from AdapterCompatibility,
    #                                      VRAM from AdapterRAM
    # 3. fall back to ("none", 0)
```

The catch worth knowing: `Win32_VideoController.AdapterRAM` is a 32-bit field and
**caps out at 4 GB**, so it reports 4095 MB for a 16 GB card. The reliable read is the
registry value `HKLM\SYSTEM\CurrentControlSet\Control\Class\{4d36e968-...}\NNNN\
HardwareInformation.qwMemorySize`, or parsing `llama-server --list-devices`, which the
Vulkan build prints with correct sizes. The second is better: it is the same code that
will do the allocating.

For an iGPU there is no fixed VRAM at all — it carves out of system RAM — so the
`Q4_MIN_VRAM_MB = 15000` threshold means something different there and would need its own
branch (§5).

### 3.2 `server/config.py` — pick the binary folder (~20 lines)

Currently hard-coded:

```python
WHISPER_CLI  = BIN / "whisper" / "whisper-cli.exe"
LLAMA_SERVER = BIN / "llama"   / "llama-server.exe"
```

Becomes a lookup against the detected vendor, with a documented fallback order — try the
vendor-specific folder, then Vulkan, then CPU:

```
bin\llama-cuda\    bin\llama-vulkan\    bin\llama-cpu\
bin\whisper-cuda\  bin\whisper-vulkan\  bin\whisper-cpu\
```

This is already almost how it works: the two engines are in **separate folders on
purpose** because they ship incompatible builds of `ggml.dll`. Adding a third dimension
to a path that is already computed costs nothing structurally.

### 3.3 `DOWNLOAD_MODELS.bat` — fetch the right binaries (~40 lines)

It already downloads 6 models with resume and size verification. It would gain a vendor
probe and download the matching binary set. Or simpler and more robust for a
non-technical user: **download all of them**. The extra Vulkan binaries are ~200 MB
against the 29 GB of models already being fetched, and it removes an entire class of
"I picked the wrong one" support problem.

### 3.4 `config.json` — one new key

```json
"gpu": { "backend": "auto" }
```

Same shape as `llm.model: "auto"` today. `auto` detects, an explicit `cuda` / `vulkan` /
`cpu` overrides. The operator gets an escape hatch and the code gets a test hook.

### 3.5 What does *not* change

- The pipeline, the prompts, the chunker, the reduce strategy, the UI, the estimator.
- `--override-tensor` and `--n-gpu-layers` are backend-agnostic tensor-placement flags.
  They work identically on Vulkan.
- Cancellation, the job object, temp cleanup — all process-level, unaffected.
- Diarization — already CPU-only.
- `calibration.json` **self-corrects**. It keys timings by model and takes the rate from
  the most recent run, so the first run on new hardware is unestimated and the second is
  right. That was built for a card swap; a vendor swap is the same event.

**Estimated diff: ~150 lines across 4 files, plus one out-of-tree whisper.cpp build.**

---

## 4. NPUs

Short version: **not worth targeting.** Both vendors have a path, both paths are narrow.

### AMD Ryzen AI (300/400 series)

whisper.cpp has genuine support: the **Whisper encoder** can be fully offloaded to the
NPU via VitisAI, and AMD publish pre-converted encoder caches on Hugging Face
(`ggml-<model>-encoder-vitisai.rai`). The README claims significant speedup over CPU.

But: encoder only, the decoder stays on CPU; it needs the Ryzen AI stack and XRT
installed; it is a per-model conversion; and there is nothing for the **LLM**, which is
80–95% of this pipeline's wall time. Speeding up part of the smaller half is not where
the time is.

### Intel NPU (Core Ultra)

llama.cpp's OpenVINO backend does run on Intel NPUs, from the same GGUF files. The
limitations in its own docs are disqualifying for this app:

- **`Q4_0` is the primary supported NPU quantisation.** This app ships `Q4_K_M` and
  `IQ3_XXS` — chosen deliberately, because sub-4-bit quants mangle proper nouns and
  figures (CLAUDE.md §11.1), which is exactly what meeting minutes are made of. A
  requantise-and-revalidate exercise, not a config change.
- Context defaults to the model's training context and **OOMs on NPU** unless `-c` is
  passed explicitly.
- `llama-server -np > 1` unsupported; static graph with a fixed prefill chunk size.
- The backend documents itself as work in progress on accuracy and op coverage.

And the structural problem for both: a 27B model at 10.9–16.5 GB does not fit in an NPU's
working set. NPUs are built for small, fixed, low-power models — good for a 1B assistant,
not for this.

**Verdict: skip NPUs.** Revisit if a future release lands NPU support for K-quants and a
27B-class model, which is not close.

---

## 5. Does this work on integrated graphics?

**Today: no.** The build is CUDA-only. On a machine with only Intel or AMD integrated
graphics, `ggml-cuda.dll` finds no device and both engines run on CPU. Nothing crashes —
it is just slow enough not to count. A 27B model on CPU is on the order of 1 token/second;
the reduce stage alone would run for hours.

Worth being precise, because "iGPU" is two different questions:

**Would a Vulkan build work on an iGPU?** Yes, mechanically. Vulkan runs on Intel Iris
Xe / Arc iGPUs and AMD Radeon 780M-class iGPUs, and llama.cpp explicitly reports them
(`ggml_vulkan: Using Intel(R) Graphics (ADL GT2) | uma: 1`).

**Would it be usable?** That depends entirely on RAM, not on the GPU:

- An iGPU has no dedicated VRAM. It shares system memory, and bandwidth is the binding
  constraint for token generation — roughly 50–100 GB/s on dual-channel DDR5 against
  ~450 GB/s on a mid-range discrete card. Expect **3–6× slower generation** than a
  discrete GPU of nominally similar capability.
- IQ3_XXS is 10.9 GB. On the target 32 GB machine that fits in shared memory with room to
  spare. On a 16 GB laptop it would not, and the `--override-tensor` offload logic would
  have to push most of the model to CPU anyway — at which point the iGPU is buying very
  little.
- The `Q4_MIN_VRAM_MB = 15000` threshold is meaningless on an iGPU and would need a
  separate rule, probably based on total system RAM.

**Honest expectation on a modern 32 GB iGPU laptop:** it would run, IQ3_XXS only, and a
4-hour meeting would take somewhere around 3–5 hours instead of ~70 minutes. Useful for
an overnight job, not for "come back after lunch".

---

## 6. What I would actually do

In order, stopping wherever the value runs out:

1. **Ship the Vulkan llama.cpp build alongside CUDA.** Zero build work, official binary,
   covers every vendor for the stage that dominates the runtime.
2. **Add vendor detection and the folder lookup.** ~80 lines. Now an AMD or Intel machine
   runs the LLM on its GPU and whisper on CPU — already a usable app.
3. **Build whisper.cpp with Vulkan and ship it.** The remaining piece. One CMake build,
   then transcription is accelerated everywhere too.
4. Stop. ROCm and SYCL buy maybe 20–40% over Vulkan on their own hardware in exchange for
   two more toolchains, two more support matrices and two more things to keep in step with
   driver updates. Not worth it for a single-machine internal tool.

Downloads grow by roughly 200 MB (Vulkan binaries) and *shrink* by 573 MB on non-NVIDIA
machines, since the CUDA runtime DLLs are not needed there.

---

## 7. Confidence

| Outcome | Confidence | Why |
|---|---|---|
| llama.cpp runs on AMD/Intel discrete via Vulkan | **95%** | Official prebuilt binary, mature backend, no source build. The one real unknown is whether `--override-tensor` offload tuning behaves the same, and the regex is already a config value precisely so it can be retuned. |
| Auto-detection picks the right backend reliably | **90%** | The logic is simple; the fiddly part is reading VRAM correctly on non-NVIDIA hardware, where `Win32_VideoController` lies above 4 GB. Parsing `llama-server --list-devices` sidesteps it. |
| whisper.cpp Vulkan build works | **80%** | Documented and supported upstream, but it is a from-source build, and `--dtw` token timestamps plus the VAD path are exactly the sort of thing that has already misbehaved once on this project (BUILD_NOTES §3.2: `--dtw` silently returned −1 under flash-attention on CUDA). It would need re-verifying against the merge stage, not just "it produced text". |
| Whole thing works end to end on an AMD/Intel **discrete** GPU | **85%** | Product of the above, with margin for one unforeseen driver quirk. |
| It is *pleasant* on an integrated GPU | **40%** | It will run. Whether anyone is happy with the runtime is a different question, and depends on the machine's RAM bandwidth more than on anything this code does. |
| NPU offload is worth doing | **10%** | Quantisation mismatch, encoder-only on AMD, work-in-progress on Intel, and the LLM — where the time actually goes — is not addressed by either. |

The estimate I would defend: **a day to get AMD/Intel discrete working via Vulkan
including the whisper build**, plus a day of testing on real hardware, which I do not
have. The code change is not the risk. The risk is that nobody has run this pipeline —
VAD, `--dtw` word timestamps, the word-level merge — on a non-CUDA backend, and stage 4 is
already documented as the stage most likely to produce garbage.
