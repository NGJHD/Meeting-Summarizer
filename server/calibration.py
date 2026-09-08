"""Per-machine timing calibration.

The time estimate and the progress bar both need to know how long each stage
takes on *this* machine, and that varies enormously. Measured on the 10GB
development card the LLM stage was 95% of the job; on a card that fits more of
the model it will be far less. A constant cannot serve both, and hardware
detection would only be a guess dressed up as a measurement.

So: measure. Every completed job records seconds-per-audio-hour for each stage
into calibration.json beside the app, and later runs use the median of what has
actually been observed here. The first run on a new machine uses the defaults
below and is labelled as rough; from the second run on, both the estimate and
the progress bar are calibrated to the hardware they are running on.

Section 16 keeps everything inside the app folder, so this file lives next to
config.json and travels with the folder. Deleting it just re-runs cold.
"""

from __future__ import annotations

import json
import math
import threading
from pathlib import Path

from . import config

CALIBRATION_PATH = config.ROOT / "calibration.json"

# Keep the last N runs and take a median, so one pathological recording (an
# hour of silence, a cancelled retry) cannot skew the estimate for good.
MAX_RECORDS = 10

# Seconds of wall time per hour of audio. The LLM stages differ enormously by
# quantisation and by how much of the model fits on the card, so they are held
# per model; everything before them is the same either way.
#
# Q4_K_M measured on a 3h25m recording (RTX 3080 10GB, 24 layers offloaded);
# IQ3_XXS scaled from the 34-minute comparison in BUILD_NOTES §9h, where it ran
# 1.8x faster on map and 2.2x on reduce. Both are cold-start guesses only --
# they are replaced by observation after the first run on a machine.
_SHARED_RATES = {
    "convert": 3.0,
    "transcribe": 104.0,      # transcription and diarization, run concurrently
    "merge": 0.5,
    "group_reduce": 1.0,
}

_LLM_RATES = {
    "q4_k_m":  {"map": 1200.0, "reduce": 990.0},
    "iq3_xxs": {"map": 650.0,  "reduce": 450.0},
}

DEFAULT_MODEL = "q4_k_m"


def _key(model_key: str = "") -> str:
    """The identity a timing record is filed under: model *and* backend.

    Generation speed depends on both. The same model on CUDA, on Vulkan and on
    the processor differs by an order of magnitude, so applying a CUDA run's
    tokens-per-second to a CPU run would make the estimate meaningless. Records
    written before backends existed carry a bare model key and are simply never
    matched, which is the correct outcome -- they describe an unknown machine.
    """
    from . import config

    try:
        backend = config.backend("llama")
    except Exception:  # noqa: BLE001 - never let calibration break a job
        backend = "cuda"
    return "%s@%s" % (model_key or DEFAULT_MODEL, backend)


def default_rates(model_key: str = "") -> dict:
    llm = _LLM_RATES.get(model_key or DEFAULT_MODEL, _LLM_RATES[DEFAULT_MODEL])
    return {**_SHARED_RATES, **llm}


DEFAULT_RATES = default_rates()

# Stages the progress bar must still show even when they cost almost nothing,
# or the bar would appear frozen through them.
MIN_WEIGHT = 1.0

_lock = threading.Lock()


_load_cache: dict = {"mtime": -1.0, "data": None}


def _load() -> dict:
    """Read calibration.json, cached against its modification time.

    The progress bar asks for a call profile every 25 generated tokens, and the
    ETA asks again on every status event. Re-parsing the file each time is
    pointless disk work on the event loop; the mtime check keeps it honest when
    a run writes new records.
    """
    try:
        mtime = CALIBRATION_PATH.stat().st_mtime
    except OSError:
        mtime = -1.0
    if _load_cache["data"] is not None and _load_cache["mtime"] == mtime:
        return _load_cache["data"]
    data = _read()
    _load_cache.update(mtime=mtime, data=data)
    return data


def _read() -> dict:
    try:
        with open(CALIBRATION_PATH, "r", encoding="utf-8") as fh:
            data = json.load(fh)
        if isinstance(data, dict) and isinstance(data.get("runs"), list):
            return data
    except (OSError, json.JSONDecodeError, ValueError):
        pass
    return {"runs": []}


def _median(values: list[float]) -> float:
    values = sorted(values)
    n = len(values)
    if not n:
        return 0.0
    mid = n // 2
    return values[mid] if n % 2 else (values[mid - 1] + values[mid]) / 2.0


def rates(model_key: str = "") -> tuple[dict, bool]:
    """Seconds per audio hour for each stage, and whether it is measured.

    Records are filtered to the model in question: a run on IQ3_XXS says
    nothing useful about how long Q4_K_M will take, and mixing them would make
    both estimates wrong.
    """
    data = _load()
    runs = [r for r in data["runs"] if r.get("audio_hours", 0) > 0.05]
    if model_key:
        wanted = _key(model_key)
        same = [r for r in runs if r.get("model") == wanted]
        # Stages before the LLM do not depend on the model, so fall back to all
        # runs for those rather than discarding them.
        runs = same or []
    if not runs:
        return default_rates(model_key), False

    # The most recent run, not a median of ten. These figures are dominated by
    # hardware -- swap the card and every older run is describing a machine
    # that no longer exists. One run is enough to notice; ten would take five
    # recordings to catch up.
    latest = runs[-1]
    out = default_rates(model_key)
    for stage in out:
        value = (latest.get("rates") or {}).get(stage)
        if value:
            out[stage] = float(value)
    return out, True


def record(stage_seconds: dict, audio_seconds: float, mode: str,
           model_key: str = "") -> None:
    """Store one completed run. Only full runs; a cancelled job teaches nothing."""
    hours = audio_seconds / 3600.0
    if hours <= 0.05:            # under 3 minutes: startup costs dominate
        return
    entry = {
        "audio_hours": round(hours, 4),
        "mode": mode,
        "model": _key(model_key),
        "rates": {k: round(v / hours, 2) for k, v in stage_seconds.items()},
    }
    with _lock:
        data = _load()
        data["runs"].append(entry)
        data["runs"] = data["runs"][-MAX_RECORDS:]
        try:
            CALIBRATION_PATH.write_text(json.dumps(data, indent=2), encoding="utf-8")
            _load_cache["data"] = None      # force a re-read next time
        except OSError:
            pass          # a read-only folder must not fail a finished job


def estimate_seconds(audio_seconds: float, mode: str = "summary",
                     model_key: str = "") -> tuple[float, bool]:
    """Predicted wall time, or (0, False) when this machine is unmeasured.

    Seconds-per-audio-hour is the wrong shape for the LLM stages and it showed:
    a short recording was quoted at 2 minutes and took 10. Writing a document
    costs roughly the same whether the meeting ran twenty minutes or two hours
    -- the model still produces a whole document -- so a purely proportional
    figure collapses on short recordings and is optimistic on long ones.

    Count tokens instead, which is what the work actually is:

        transcript tokens  = audio minutes x a measured baseline (config.json)
        map calls          = transcript tokens / chunk stride
        + group reduces if that exceeds the threshold, + one or two documents
        each call          = its measured token count / this machine's rate

    Transcription, conversion and the merge really are proportional to length,
    so those stay on seconds-per-audio-hour.

    Returns measured=False when nothing has been measured here yet. There is no
    honest number to give in that case, and a guess that is five times out is
    worse than admitting it.
    """
    if audio_seconds <= 0:
        return 0.0, False
    r, measured_stages = rates(model_key)
    if not measured_stages or not has_llm_profile(model_key):
        return 0.0, False

    from . import config

    try:
        cfg = config.load_config()
    except RuntimeError:
        return 0.0, False
    est = cfg.get("estimate", {})
    hours = audio_seconds / 3600.0

    total = (r["convert"] + r["transcribe"] + r["merge"]) * hours
    total += llm_seconds(audio_seconds, mode, model_key, cfg)
    total += float(est.get("fixed_overhead_s", 60))
    return total * (1.0 + float(est.get("buffer_fraction", 0.15))), True


def llm_seconds(audio_seconds: float, mode: str, model_key: str,
                cfg: dict) -> float:
    """Predicted seconds for map + group reduce + final reduce."""
    return sum(llm_stage_seconds(audio_seconds, mode, model_key, cfg).values())


def llm_stage_seconds(audio_seconds: float, mode: str, model_key: str,
                      cfg: dict) -> dict:
    """Predicted seconds per LLM stage, broken out.

    The same arithmetic the live ETA uses once the real chunk count is known;
    here the chunk count is predicted from the audio length instead. It is
    broken out per stage so the live ETA can price the stages it does not yet
    have a call plan for **the same way the up-front estimate did**. Falling
    back to seconds-per-audio-hour there was a real defect: the front page said
    56 minutes and the progress view said 2h21m for the same job.
    """
    est = cfg.get("estimate", {})
    ccfg = cfg.get("chunking", {})
    per_minute = float(est.get("transcript_tokens_per_audio_minute", 206))
    tokens = (audio_seconds / 60.0) * per_minute

    target = float(ccfg.get("target_tokens", 10000))
    overlap = float(ccfg.get("overlap_tokens", 400))
    stride = max(target - overlap, 1000.0)
    chunks = max(1, int(-(-tokens // stride)))

    hours = audio_seconds / 3600.0

    def cost(stage: str, calls: int) -> float:
        if calls <= 0:
            return 0.0
        out_tokens, tok_s = call_profile(stage, model_key, hours)
        return calls * out_tokens / max(tok_s, 0.05)

    threshold = int(ccfg.get("group_reduce_threshold", 8))
    size = max(2, int(ccfg.get("group_size", 5)))
    return {
        "map": cost("map", chunks),
        "group_reduce": cost("group_reduce", -(-chunks // size))
                        if chunks > threshold else 0.0,
        "reduce": cost("reduce", 2 if mode == "both" else 1),
    }


def weights(mode: str = "summary", model_key: str = "") -> dict:
    """Progress-bar weights proportional to what each stage actually costs.

    Section 13 fixes these at transcribe 55 / map 28 / reduce 6. Measured, the
    split on this hardware is closer to transcribe 5 / map 52 / reduce 43, so
    the documented weights make the bar race to 62% and then crawl for an hour.
    Deriving them from observed rates keeps the bar honest on whatever machine
    it is running on.
    """
    r, _ = rates(model_key)
    scaled = dict(r)
    if mode == "both":
        scaled["reduce"] = r["reduce"] * 2
    total = sum(scaled.values()) or 1.0
    raw = {k: max(MIN_WEIGHT, 100.0 * v / total) for k, v in scaled.items()}
    # Renormalise after the floor so the bar still ends at exactly 100.
    scale = 100.0 / sum(raw.values())
    return {k: v * scale for k, v in raw.items()}


def stage_seconds(stage: str, audio_seconds: float, model_key: str = "") -> float:
    """Expected wall time for one whole stage on this machine.

    Used to pace the progress bar *inside* a single LLM call. Token counts
    alone cannot do it: `max_tokens` is a cap, not an expectation, and the real
    output is usually a third of it -- which is why the bar used to crawl to a
    third and then jump. Elapsed time against a measured expectation does not
    have that failure.
    """
    r, _ = rates(model_key)
    return r.get(stage, 0.0) * (max(audio_seconds, 0.0) / 3600.0)


# ---------------------------------------------------------------------------
# per-call LLM profile -- what the ETA is actually built from
# ---------------------------------------------------------------------------

# A stage rate in seconds-per-audio-hour is fine for sizing the whole job, but
# it cannot answer "how much longer is *this* call". For that we need two
# numbers per stage: how many tokens a call of that kind produces, and how fast
# this machine produces them. Both are measured; these are cold-start guesses,
# taken on the 10GB development card and replaced after the first real call.
_CALL_DEFAULTS = {
    "map":          {"tokens": 1400.0, "tok_s": 4.0},
    "group_reduce": {"tokens": 2200.0, "tok_s": 4.0},
    # Thinking is on for the final reduce, and reasoning tokens are counted:
    # 2079 completion tokens in 683s on IQ3_XXS, plus the reasoning before it.
    "reduce":       {"tokens": 4300.0, "tok_s": 3.0},
}

MAX_CALL_RECORDS = 80


def _calls_for(stage: str, model_key: str) -> list:
    wanted = _key(model_key) if model_key else ""
    return [c for c in _load().get("calls", [])
            if c.get("stage") == stage and float(c.get("seconds") or 0) > 1
            and (not wanted or c.get("model") == wanted)]


def has_llm_profile(model_key: str = "", stages=("map", "reduce")) -> bool:
    """True once every named stage has been observed here with this model.

    A whole-run estimate needs `map` as well as `reduce` -- map is the larger
    half of the LLM cost and guessing it would defeat the point. Regenerating a
    single document only needs `reduce`, so that caller asks for less.
    """
    return all(_calls_for(stage, model_key) for stage in stages)


def _expected_tokens(calls: list, audio_hours: float, default: float) -> float:
    """How many tokens a call of this kind produces for a meeting this long.

    Not a single median. Output size grows with the meeting -- measured, a
    document for a 55-second clip is ~810 tokens and one for a 2h05 meeting is
    ~5,500 -- so a pooled median suits neither end. An early attempt used a
    nearest-length window with "use everything" as the fallback, and that
    fallback is what quoted 23 minutes for a clip that took 5: with only long
    meetings on record, a short one inherited their document size.

    Instead: one point per recorded length, then interpolate between them on a
    log scale. Outside the recorded range it extrapolates, clamped to half the
    smallest and twice the largest thing ever actually seen, so a length far
    outside the data cannot produce a wild number.
    """
    tagged = [c for c in calls if float(c.get("audio_h") or 0) > 0]
    if not tagged or audio_hours <= 0:
        # Records from before lengths were stored: nothing better to do.
        return _median([float(c["tokens"]) for c in calls]) or default

    groups: dict = {}
    for c in tagged:
        groups.setdefault(round(float(c["audio_h"]), 3), []).append(float(c["tokens"]))
    points = sorted((h, _median(v)) for h, v in groups.items())
    if len(points) == 1:
        return points[0][1] or default

    xs = [math.log10(max(h, 1e-3)) for h, _ in points]
    ys = [t for _, t in points]
    x = math.log10(max(audio_hours, 1e-3))

    if x <= xs[0]:
        i = 0
    elif x >= xs[-1]:
        i = len(points) - 2
    else:
        i = max(j for j in range(len(xs) - 1) if xs[j] <= x)
    span = xs[i + 1] - xs[i]
    t = (x - xs[i]) / span if span else 0.0
    value = ys[i] + t * (ys[i + 1] - ys[i])
    return min(max(value, min(ys) * 0.5), max(ys) * 2.0) or default


def call_profile(stage: str, model_key: str = "",
                 audio_hours: float = 0.0) -> tuple[float, float]:
    """(expected tokens, tokens per second) for one call of this stage.

    The two halves come from different subsets on purpose:

    - **tokens per second** comes from the most recent run, whatever its
      length. It is the most machine-dependent number in the pipeline -- set by
      how much of the model `cpu_ffn_regex` had to push into system RAM -- so
      after a card is changed every older sample describes a machine that no
      longer exists.
    - **tokens per call** is interpolated across recorded meeting lengths (see
      `_expected_tokens`). How long a document runs is a property of the
      meeting, not the hardware, and it grows with the meeting.
    """
    default = _CALL_DEFAULTS.get(stage, _CALL_DEFAULTS["map"])
    calls = _calls_for(stage, model_key)
    if not calls:
        return default["tokens"], default["tok_s"]

    tokens = _expected_tokens(calls, audio_hours, default["tokens"])

    latest = calls[-1].get("run", "")
    recent = [c for c in calls if c.get("run", "") == latest] or calls
    tok_s = _median([float(c["tokens"]) / float(c["seconds"]) for c in recent])
    return tokens, tok_s or default["tok_s"]


def record_call(stage: str, tokens: float, seconds: float,
                model_key: str = "", run_id: str = "",
                audio_hours: float = 0.0) -> None:
    """Store one completed LLM call. Cheap, and it is what makes the ETA true.

    `run_id` groups the calls of one job so the speed reading can be taken from
    the newest run alone while the token counts still pool across all of them.
    """
    if tokens < 50 or seconds < 1.0:
        return                      # too short to say anything about the rate
    with _lock:
        data = _load()
        calls = data.get("calls")
        if not isinstance(calls, list):
            calls = []
        calls.append({
            "stage": stage,
            "model": _key(model_key),
            "run": run_id,
            "audio_h": round(float(audio_hours), 3),
            "tokens": round(float(tokens), 1),
            "seconds": round(float(seconds), 1),
        })
        data["calls"] = calls[-MAX_CALL_RECORDS:]
        try:
            CALIBRATION_PATH.write_text(json.dumps(data, indent=2), encoding="utf-8")
            _load_cache["data"] = None      # force a re-read next time
        except OSError:
            pass
