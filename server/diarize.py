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


def run(job: Job, cfg: dict, wav: Path) -> list[Turn]:
    """Return speaker turns as (start, end, speaker_id), or [] on any failure."""
    dcfg = cfg["diarization"]
    if not dcfg.get("enabled", True):
        job.log("diarization: disabled in config.json, skipping")
        return []

    seg_path = config.MODELS / SEGMENTATION_MODEL
    emb_path = config.MODELS / EMBEDDING_MODEL
    for p in (seg_path, emb_path):
        if not p.exists():
            job.log("diarization: %s missing, continuing without speakers" % p.name)
            return []

    params_path = config.TEMP / "diarize_params.json"
    result_path = config.TEMP / "diarize_result.json"
    params = {
        "segmentation_model": str(seg_path),
        "embedding_model": str(emb_path),
        "threads": int(dcfg.get("threads", 5)),
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
    job.log("diarization: started on %d threads" % int(dcfg.get("threads", 5)))

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
