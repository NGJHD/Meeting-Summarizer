"""Stage 3 - speaker diarization via sherpa-onnx (CLAUDE.md section 8).

CPU only, ONNX Runtime. Nothing here imports torch, which is the entire reason
sherpa-onnx was chosen over pyannote.audio or whisperX.

The work happens in a child process (see diarize_worker.py): sherpa-onnx holds
the GIL for the whole of process(), which would otherwise stall the thread
reading whisper-cli's output and deadlock the concurrent path.

This stage is allowed to fail. If it does, the caller continues with an empty
speaker list and the merge stage degrades to an unattributed transcript.
"""

from __future__ import annotations

import json
import subprocess
import sys
import threading
import time
from dataclasses import dataclass
from pathlib import Path

from . import config
from .jobs import Cancelled, Job

SEGMENTATION_MODEL = "segmentation-3.0.onnx"
EMBEDDING_MODEL = "speaker-embedding.onnx"

CREATE_NEW_PROCESS_GROUP = getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0)
CREATE_NO_WINDOW = getattr(subprocess, "CREATE_NO_WINDOW", 0)


@dataclass
class Turn:
    start: float
    end: float
    speaker: int


# Diminishing returns set in early. Embedding is closer to memory-bandwidth
# bound than core bound, so threads buy far less than they look like they
# should -- measured on a 12-core/24-thread machine over a fixed slice:
#
#     threads    2      4      6      8     10     12     16     20
#     time    84.8s  71.5s  57.8s  56.1s  50.1s  48.6s  46.8s  45.6s
#
# Ten times the threads is 1.86x the speed. Past twelve, 67% more threads buy
# 6% less time -- and during the concurrent phase those threads are competing
# with whisper for the same machine (section 7's budget). So: half the logical
# cores, capped where the curve flattens.
MAX_AUTO_THREADS = 12
# The floor is the value this replaced, not a measured optimum. `cores // 2`
# falls below it under ten logical cores -- 2 on a quad-core -- and 2 is the
# worst number in the table above, on exactly the machines where diarization
# already hurts most. Detection may not make anything slower than the constant
# it is replacing.
#
# Note this can exceed section 7's budget on a small machine: at 12 logical
# cores it asks for 6 where the budget allows 5. Accepted, because whisper is
# GPU-bound and its threads mostly feed the card rather than saturating them.
# No small machine has been measured; if one ever is, measure before lowering.
MIN_AUTO_THREADS = 5


def resolve_threads(value) -> int:
    """`"auto"` -> half the logical cores, clamped. A number is taken as given.

    Section 4 says every machine-dependent value is detected rather than
    guessed; this one was hard-coded to 5, which oversubscribes a 4-core laptop
    and uses under half of a 24-thread desktop.
    """
    if isinstance(value, str) and value.strip().lower() == "auto":
        import os

        cores = os.cpu_count() or 4
        return max(MIN_AUTO_THREADS, min(MAX_AUTO_THREADS, cores // 2))
    try:
        return max(1, int(value))
    except (TypeError, ValueError):
        return 5


def run(job: Job, cfg: dict, wav: Path) -> list[Turn]:
    """Return speaker turns as (start, end, speaker_id), or [] on any failure."""
    dcfg = cfg["diarization"]
    if not dcfg.get("enabled", True):
        job.log("diarization: disabled in config.json, skipping")
        return []

    seg_path = config.MODELS / SEGMENTATION_MODEL
    # The embedding model is selectable so alternatives can be measured against
    # a real recording without swapping files about. TitaNet-large ships
    # (BUILD_NOTES 3.8); anything sherpa-onnx accepts works. Note that
    # `cluster_threshold`, `merge_centroid_distance` and `reassign_max_distance`
    # are all tuned to TitaNet's cosine scale and do NOT carry over -- with an
    # explicit `num_speakers` the threshold is bypassed, which is the only
    # configuration another model has been tested in.
    emb_path = config.resolve(dcfg.get("embedding_model")
                              or (config.MODELS / EMBEDDING_MODEL))
    for p in (seg_path, emb_path):
        if not p.exists():
            job.log("diarization: %s missing, continuing without speakers" % p.name)
            return []

    params_path = config.TEMP / "diarize_params.json"
    result_path = config.TEMP / "diarize_result.json"
    params = {
        "segmentation_model": str(seg_path),
        "embedding_model": str(emb_path),
        "threads": resolve_threads(dcfg.get("threads", "auto")),
        "num_speakers": int(dcfg.get("num_speakers", 0)),
        "cluster_threshold": float(dcfg.get("cluster_threshold", 0.7)),
        "min_duration_on": float(dcfg.get("min_duration_on", 0.3)),
        "min_duration_off": float(dcfg.get("min_duration_off", 0.5)),
        "max_speakers": int(dcfg.get("max_speakers", 20)),
        "min_cluster_speech_s": float(dcfg.get("min_cluster_speech_s", 30)),
        "min_cluster_speech_fraction": float(dcfg.get("min_cluster_speech_fraction", 0.005)),
        "reassign_max_distance": float(dcfg.get("reassign_max_distance", 0.85)),
        "merge_centroid_distance": float(dcfg.get("merge_centroid_distance", 0.35)),
        "min_embed_duration": float(dcfg.get("min_embed_duration", 1.0)),
        "min_coverage_fraction": float(dcfg.get("min_coverage_fraction", 0.80)),
        "embedding_cache": bool(dcfg.get("embedding_cache", True)),
    }
    # An explicit participant count from the UI overrides the config default.
    if job.num_speakers:
        params["num_speakers"] = int(job.num_speakers)
    params_path.write_text(json.dumps(params), encoding="utf-8")

    cmd = [
        sys.executable,
        "-m",
        "server.diarize_worker",
        str(wav),
        str(params_path),
        str(result_path),
    ]

    try:
        proc = subprocess.Popen(
            cmd,
            cwd=str(config.ROOT),
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
            errors="replace",
            bufsize=1,
            env=config.child_env(),
            creationflags=CREATE_NEW_PROCESS_GROUP | CREATE_NO_WINDOW,
        )
    except OSError as exc:
        job.log("diarization: could not start (%s); continuing without speakers" % exc)
        return []

    job.register_proc(proc)
    job.log("diarization: started on %d threads"
            % resolve_threads(dcfg.get("threads", "auto")))

    # sherpa reports nothing at all during its segmentation phase, which on a
    # multi-hour recording is many minutes of total silence in the log. Without
    # a heartbeat the only visible sign of diarization is the line above, and
    # the operator cannot tell a slow stage from a hung one.
    started = time.time()
    done = threading.Event()

    def heartbeat() -> None:
        while not done.wait(60.0):
            job.log("diarization: still working (%d min elapsed)"
                    % round((time.time() - started) / 60))

    threading.Thread(target=heartbeat, name="diarize-heartbeat", daemon=True).start()

    error = ""
    degraded = ""
    stage = ""
    last_pct = -10
    try:
        assert proc.stdout is not None
        for line in proc.stdout:
            line = line.strip()
            if line.startswith("PROGRESS "):
                try:
                    _, processed, total = line.split()
                    pct = int(int(processed) * 100 / max(1, int(total)))
                except ValueError:
                    continue
                if pct >= last_pct + 10:
                    last_pct = pct
                    job.log("diarization: %s %d%%" % (stage or "working", pct))
            elif line.startswith("STAGE "):
                stage = line[6:]
                last_pct = -10
                job.log("diarization: %s" % {
                    "segment": "finding speech and speaker changes",
                    "embed": "measuring voices (the slow part)",
                    "cluster": "grouping voices",
                    "cache": "reusing cached voice measurements",
                }.get(stage, stage))
            elif line.startswith("DEGRADED "):
                degraded = line[9:]
            elif line.startswith("ERROR "):
                error = line[6:]
    finally:
        done.set()
        try:
            proc.wait(timeout=60)
        except subprocess.TimeoutExpired:
            pass
        job.unregister_proc(proc)

    if job.cancelled:
        raise Cancelled()

    if proc.returncode != 0 or not result_path.exists():
        job.log(
            "diarization: failed (%s); output will be unattributed"
            % (error or "exit %s" % proc.returncode)
        )
        return []

    diag_path = result_path.with_suffix(".diag.json")
    try:
        diag = json.loads(diag_path.read_text(encoding="utf-8"))
        job.log(
            "diarization: %d windows, %d embeddings (%d too short), clusters %d -> %d, "
            "coverage %.0f%%"
            % (
                diag.get("windows", 0),
                diag.get("embeddings", 0),
                diag.get("skipped_short", 0),
                diag.get("clusters_before", 0),
                diag.get("clusters_kept", 0),
                100 * diag.get("coverage", 0.0),
            )
        )
        job.log("diarization: timings %s" % diag.get("timings", {}))
        job.diarization_diag = diag
    except (OSError, json.JSONDecodeError, ValueError):
        pass

    if degraded:
        # Part F: meaningless labels are worse than none. The reader cannot
        # tell clustering debris from a real participant, and will act on it.
        job.log("diarization: UNRELIABLE - %s; dropping attribution" % degraded)
        job.attribution_note = (
            "Speaker identification was unreliable for this recording and has been "
            "omitted."
        )
        return []

    try:
        raw = json.loads(result_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        job.log("diarization: unreadable result (%s); continuing unattributed" % exc)
        return []

    turns = [
        Turn(start=float(t["start"]), end=float(t["end"]), speaker=int(t["speaker"]))
        for t in raw
    ]

    # sherpa returns raw cluster indices, which are sparse: a two-speaker clip
    # can come back as clusters 0, 2 and 6. Renumber by order of first
    # appearance so the labels read SPEAKER_00, SPEAKER_01, ... A reader seeing
    # "SPEAKER_06" in a three-person meeting reasonably assumes the pipeline
    # lost four people.
    mapping: dict[int, int] = {}
    for turn in turns:
        if turn.speaker not in mapping:
            mapping[turn.speaker] = len(mapping)
    for turn in turns:
        turn.speaker = mapping[turn.speaker]

    job.log("diarization: %d turns, %d speakers" % (len(turns), len(mapping)))
    return turns
