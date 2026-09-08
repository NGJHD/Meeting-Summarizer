"""Job state, progress accounting, cancellation and temp cleanup.

One job at a time. There is deliberately no queue (CLAUDE.md section 16).
"""

from __future__ import annotations

import os
import subprocess
import threading
import time
import traceback
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from . import config

# Stage weights. Section 13 fixes these (transcribe 55, map 28, reduce 6), but
# measured on real hardware the split is nothing like that -- the LLM stage was
# 95% of a 3h25m job on the development card. Fixed weights make the bar race to
# 62% and then crawl for an hour, so they are derived from measured per-stage
# rates instead (server/calibration.py) and re-derived per job.
STAGE_WEIGHTS = {
    "upload": 2,
    "convert": 3,
    "transcribe": 55,   # shared with diarization; driven by transcription only
    "merge": 2,
    "map": 28,
    "group_reduce": 4,
    "reduce": 6,
}

STAGE_LABELS = {
    "upload": "Uploading",
    "convert": "Converting audio",
    "transcribe": "Transcribing",
    "merge": "Matching speakers to words",
    "map": "Reading the meeting",
    "group_reduce": "Consolidating sections",
    "reduce": "Writing the document",
}

_ORDER = list(STAGE_WEIGHTS.keys())


def stage_weights(mode: str = "summary", model_key: str = "") -> dict:
    """Measured weights where available, section 13's defaults otherwise."""
    from . import calibration

    try:
        w = calibration.weights(mode, model_key)
    except Exception:  # noqa: BLE001 - a bad calibration file must not stop a job
        return dict(STAGE_WEIGHTS)
    # calibration covers the pipeline stages; upload is client-side.
    out = {"upload": 2.0}
    scale = 98.0 / (sum(w.values()) or 1.0)
    for k in _ORDER:
        if k != "upload":
            out[k] = w.get(k, STAGE_WEIGHTS[k]) * scale
    return out


def stage_base(stage: str, weights: dict | None = None) -> float:
    """Cumulative percentage completed before `stage` starts."""
    weights = weights or STAGE_WEIGHTS
    total = 0.0
    for s in _ORDER:
        if s == stage:
            break
        total += weights[s]
    return total


class Cancelled(Exception):
    """Raised inside the pipeline when the user cancels."""


class JobError(Exception):
    """A failure with a message already fit for a non-technical user."""

    def __init__(self, message: str, detail: str = ""):
        super().__init__(message)
        self.message = message
        self.detail = detail


@dataclass
class Job:
    id: str
    filename: str = ""
    mode: str = "summary"
    # "run" processes a recording; "generate" rewrites one document from cached
    # notes. Cancelling them means different things -- a cancelled run has
    # nothing behind it, a cancelled rewrite still has the finished meeting.
    kind: str = "run"
    upload_path: Optional[Path] = None
    size_bytes: int = 0
    duration_s: float = 0.0

    # Optional participant count from the UI. Not a tuning knob: it is metadata
    # about the user's own recording, which they know and the clusterer does
    # not (DIARIZATION_FIX.md part D4). 0 means "unsure, auto-detect".
    num_speakers: int = 0
    # Model chosen in the UI; blank means use the config/auto default.
    model_key: str = ""
    # Set when attribution was dropped because the labels could not be trusted.
    attribution_note: str = ""
    diarization_diag: Optional[dict] = None
    # Live LlamaServer, so cancel can drop an in-flight request rather
    # than waiting out a whole chunk (CLAUDE.md section 14).
    llm_server: object = None
    # Progress weights derived from this machine's measured stage rates, and
    # the per-stage seconds this run spent, which feed the next calibration.
    weights: dict = field(default_factory=dict)
    samples: list = field(default_factory=list)
    meeting_name: str = ""
    stage_seconds: dict = field(default_factory=dict)
    stage_started_at: float = 0.0
    # What the ETA is built from: how many LLM calls of each kind are still to
    # be dispatched, and the one currently streaming. See remaining_seconds().
    pending_calls: dict = field(default_factory=dict)
    current_call: Optional[dict] = None
    # Set when the transcript succeeded but the document did not.
    document_error: str = ""

    state: str = "new"          # new | uploaded | running | done | error | cancelled
    stage: str = "upload"
    percent: float = 0.0
    message: str = ""
    error: str = ""
    started_at: float = 0.0

    transcript_path: Optional[Path] = None
    # Renamed copy, written only when the user supplies speaker names. The
    # raw SPEAKER_nn transcript above is never overwritten.
    tagged_transcript_path: Optional[Path] = None
    document_path: Optional[Path] = None
    documents: list = field(default_factory=list)

    cancel_event: threading.Event = field(default_factory=threading.Event)
    procs: list = field(default_factory=list)      # live subprocess.Popen handles
    log_lines: list = field(default_factory=list)
    _queues: list = field(default_factory=list)    # asyncio.Queue per SSE listener
    _loop: object = None

    # -- cancellation ----------------------------------------------------

    @property
    def cancelled(self) -> bool:
        return self.cancel_event.is_set()

    def check_cancelled(self) -> None:
        if self.cancel_event.is_set():
            raise Cancelled()

    def register_proc(self, proc: subprocess.Popen) -> None:
        self.procs.append(proc)
        assign_to_job_object(proc.pid)

    def unregister_proc(self, proc: subprocess.Popen) -> None:
        try:
            self.procs.remove(proc)
        except ValueError:
            pass

    def kill_processes(self) -> None:
        """Kill each child *process tree*.

        /T matters: whisper-cli and ffmpeg may have their own children, and a
        surviving one keeps VRAM allocated until reboot (CLAUDE.md section 14).
        """
        for proc in list(self.procs):
            if proc.poll() is not None:
                continue
            try:
                subprocess.run(
                    ["taskkill", "/F", "/T", "/PID", str(proc.pid)],
                    capture_output=True,
                    creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
                    timeout=15,
                )
            except Exception:
                try:
                    proc.kill()
                except Exception:
                    pass
        self.procs.clear()

    def cancel(self) -> None:
        self.cancel_event.set()
        server = self.llm_server
        if server is not None:
            try:
                server.abort()
            except Exception:
                pass
        self.kill_processes()

    # -- progress and logging -------------------------------------------

    def emit(self, event: dict) -> None:
        """Push an event to every SSE listener. Safe from worker threads."""
        loop = self._loop
        for q in list(self._queues):
            if loop is not None and loop.is_running():
                try:
                    loop.call_soon_threadsafe(q.put_nowait, event)
                except RuntimeError:
                    pass
            else:
                try:
                    q.put_nowait(event)
                except Exception:
                    pass

    def log(self, text: str) -> None:
        stamp = time.strftime("%H:%M:%S")
        line = "[%s] %s" % (stamp, text)
        self.log_lines.append(line)
        _append_job_log(line)
        self.emit({"type": "log", "line": line})

    def _w(self) -> dict:
        if not self.weights:
            self.weights = stage_weights(self.mode, self.model_key)
        return self.weights

    def set_stage(self, stage: str, message: str = "") -> None:
        if stage != self.stage:
            self.stage_started_at = time.time()
        self.stage = stage
        self.message = message or STAGE_LABELS.get(stage, stage)
        self.percent = stage_base(stage, self._w())
        self.emit(self._status())
        self.log(self.message)

    def set_progress(self, stage: str, fraction: float, message: str = "") -> None:
        """Report progress within a stage as a 0..1 fraction of that stage."""
        fraction = max(0.0, min(1.0, fraction))
        w = self._w()
        self.stage = stage
        self.percent = stage_base(stage, w) + w[stage] * fraction
        if message:
            self.message = message
        self.emit(self._status())

    def _status(self) -> dict:
        return {
            "type": "status",
            "state": self.state,
            "stage": self.stage,
            "stage_label": STAGE_LABELS.get(self.stage, self.stage),
            "percent": round(self.percent, 2),
            "message": self.message,
            "elapsed": round(time.time() - self.started_at, 1) if self.started_at else 0,
            "remaining": self.remaining_seconds(),
        }

    def remaining_seconds(self):
        """Time left, counted from the work still to do -- not from the bar.

        Extrapolating `elapsed / percent` looks reasonable and is a lie: the
        bar's position inside an LLM call is a deliberately asymptotic curve,
        not a measurement, so dividing by it produced "about 6 seconds left"
        with twenty minutes still to run.

        Count the actual work instead. What remains is a known number of LLM
        calls, each of a known kind, plus whatever pipeline stages have not
        started. Both parts are priced from this machine's own measurements
        (calibration.py): tokens per call and tokens per second for the former,
        seconds per audio-hour for the latter.
        """
        if not self.started_at or self.state != "running":
            return None
        # Stages with a known call plan are priced per call; everything else
        # falls back to seconds-per-audio-hour. The handover happens the moment
        # chunking finishes and the real number of calls is known.
        priced = {s for s, n in self.pending_calls.items() if n > 0}
        if self.current_call:
            priced.add(self.current_call.get("stage", ""))
        total = self._llm_remaining() + self._stage_remaining(priced)
        if total <= 0:
            return None
        return max(0, int(round(total)))

    def _llm_remaining(self) -> float:
        from . import calibration

        seconds = 0.0

        call = self.current_call
        if call:
            stage = call.get("stage", "map")
            tokens, tok_s = calibration.call_profile(
                stage, self.model_key, self.duration_s / 3600.0)
            elapsed = max(time.time() - call.get("started", time.time()), 0.0)
            seen = float(call.get("seen", 0))
            if seen >= 60 and elapsed > 5.0:
                # Live rate beats any stored one: it is this call, on this
                # machine, right now. Only the length is still a prediction.
                tok_s = seen / elapsed
                tokens = max(tokens, seen + 1)
            seconds += max(tokens - seen, 0.0) / max(tok_s, 0.05)
        else:
            # Between calls, or the model is still loading.
            if self.pending_calls:
                seconds += 15.0

        for stage, count in self.pending_calls.items():
            if count <= 0:
                continue
            tokens, tok_s = calibration.call_profile(
                stage, self.model_key, self.duration_s / 3600.0)
            seconds += count * tokens / max(tok_s, 0.05)
        return seconds

    # Stages whose cost really is proportional to how long the recording is.
    # Everything else is priced per LLM call.
    _LINEAR_STAGES = ("convert", "transcribe", "merge")

    def _stage_remaining(self, priced: set) -> float:
        """Everything not yet priced per call.

        This is what carries the ETA through transcription, when the LLM work
        is entirely ahead and its size is not yet known. Without it the ETA
        would report only the minutes left of transcribing and then leap when
        the map loop started.

        The two halves are priced the way section 13 requires and the way the
        up-front estimate does it: convert/transcribe/merge from seconds per
        audio-hour, the LLM stages from predicted tokens. Pricing the LLM
        stages from audio-hours here -- which is what this used to do -- made
        the progress view disagree with the first screen by a factor of two and
        a half on the same job.
        """
        from . import calibration

        if self.kind != "run" or self.duration_s <= 0:
            return 0.0
        hours = self.duration_s / 3600.0
        rates, _ = calibration.rates(self.model_key)

        seconds = 0.0
        for stage in self._LINEAR_STAGES:
            if stage in self.stage_seconds or stage in priced:
                continue
            expected = rates.get(stage, 0.0) * hours
            if stage == self.stage and self.stage_started_at:
                expected -= time.time() - self.stage_started_at
            seconds += max(expected, 0.0)

        try:
            cfg = config.load_config()
        except RuntimeError:
            return seconds
        predicted = calibration.llm_stage_seconds(
            self.duration_s, self.mode, self.model_key, cfg)
        for stage, expected in predicted.items():
            if stage in self.stage_seconds or stage in priced:
                continue
            if stage == self.stage and self.stage_started_at:
                expected -= time.time() - self.stage_started_at
            seconds += max(expected, 0.0)
        return seconds

    def push_status(self) -> None:
        self.emit(self._status())

    def finish(self, documents, transcript: Optional[Path]) -> None:
        if isinstance(documents, Path):
            documents = [documents]
        self.state = "done"
        self.percent = 100.0
        self.documents = list(documents or [])
        self.document_path = self.documents[0] if self.documents else None
        self.transcript_path = transcript
        self.message = "Finished"
        self.emit(
            {
                "type": "done",
                "documents": [p.name for p in self.documents],
                "transcript": transcript.name if transcript else None,
                "tagged": (self.tagged_transcript_path.name
                           if self.tagged_transcript_path else None),
                "elapsed": round(time.time() - self.started_at, 1),
                "document_error": self.document_error,
            }
        )

    def fail(self, message: str, detail: str = "") -> None:
        self.state = "error"
        self.error = message
        self.message = message
        if detail:
            _append_job_log(detail)
        self.emit({"type": "error", "message": message})

    def mark_cancelled(self) -> None:
        self.state = "cancelled"
        self.message = "Cancelled"
        self.emit({"type": "cancelled"})


# ---------------------------------------------------------------------------
# registry
# ---------------------------------------------------------------------------

_jobs: dict[str, Job] = {}
_lock = threading.Lock()


def create(job_id: str) -> Job:
    with _lock:
        job = Job(id=job_id)
        _jobs[job_id] = job
        return job


def get(job_id: str) -> Optional[Job]:
    return _jobs.get(job_id)


def active() -> Optional[Job]:
    for job in _jobs.values():
        if job.state == "running":
            return job
    return None


def all_jobs() -> list[Job]:
    return list(_jobs.values())


# ---------------------------------------------------------------------------
# job log
# ---------------------------------------------------------------------------

_log_lock = threading.Lock()


def job_log_path() -> Path:
    return config.TEMP / "job.log"


def reset_job_log() -> None:
    with _log_lock:
        try:
            config.TEMP.mkdir(parents=True, exist_ok=True)
            job_log_path().write_text("", encoding="utf-8")
        except OSError:
            pass


def _append_job_log(text: str) -> None:
    with _log_lock:
        try:
            config.TEMP.mkdir(parents=True, exist_ok=True)
            with open(job_log_path(), "a", encoding="utf-8", errors="replace") as fh:
                fh.write(text.rstrip() + "\n")
        except OSError:
            pass


def log_exception(exc: BaseException) -> str:
    detail = "".join(traceback.format_exception(type(exc), exc, exc.__traceback__))
    _append_job_log(detail)
    return detail


def read_job_log() -> str:
    """The bundle behind the UI's "Copy diagnostic info" button.

    Includes the tail of llama-server's own log: when the LLM stage fails, the
    reason is almost always in there and almost never in ours.
    """
    parts = []
    try:
        parts.append(job_log_path().read_text(encoding="utf-8", errors="replace"))
    except OSError:
        pass
    llama = config.TEMP / "llama-server.log"
    try:
        tail = llama.read_text(encoding="utf-8", errors="replace").splitlines()[-60:]
        if tail:
            header = "\n--- llama-server (last %d lines) ---\n" % len(tail)
            parts.append(header + "\n".join(tail))
    except OSError:
        pass
    return "\n".join(parts)


# ---------------------------------------------------------------------------
# temp cleanup
# ---------------------------------------------------------------------------

# Everything the pipeline writes into temp\. audio.wav alone is >1GB for an
# 8-hour recording, so a user who cancels a few times would otherwise fill
# their disk with no idea why (CLAUDE.md section 14).
_TEMP_GLOBS = [
    "upload.*",
    "audio.wav",
    "transcript.json",
    "transcript.stderr",
    "diarize_params.json",
    "diarize_result.json",
    "diarize_result.diag.json",
    "llama-server.log",
    # The embedding cache makes re-clustering the same recording free, but each
    # one is ~100MB for a long meeting. Production never re-runs the same audio,
    # so it is swept with everything else rather than filling the disk.
    "embeddings-*.npz",
    "*.tmp",
]


def clean_temp(keep_log: bool = True) -> None:
    if not config.TEMP.exists():
        return
    for pattern in _TEMP_GLOBS:
        for path in config.TEMP.glob(pattern):
            try:
                path.unlink()
            except OSError:
                pass
    if not keep_log:
        try:
            job_log_path().unlink()
        except OSError:
            pass


def shutdown_all() -> None:
    """Kill every child process and clear temp. Safe to call more than once."""
    for job in list(_jobs.values()):
        try:
            job.cancel_event.set()
            job.kill_processes()
        except Exception:
            pass
    clean_temp()


# ---------------------------------------------------------------------------
# kill-on-close job object
# ---------------------------------------------------------------------------

# atexit and signal handlers do not run when a console window is force-quit,
# which would strand whisper-cli.exe or llama-server.exe holding VRAM until
# reboot (CLAUDE.md section 14, acceptance test 7). A Windows job object with
# KILL_ON_JOB_CLOSE is the only mechanism the OS honours unconditionally: when
# this process dies for any reason, its handle closes and every process
# assigned to the job is terminated with it.

_job_handle = None

_JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE = 0x2000
_JOB_OBJECT_EXTENDED_LIMIT_INFORMATION = 9


def install_kill_on_close() -> None:
    global _job_handle
    if _job_handle is not None or os.name != "nt":
        return
    try:
        import ctypes
        from ctypes import wintypes

        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)

        class IO_COUNTERS(ctypes.Structure):
            _fields_ = [
                ("ReadOperationCount", ctypes.c_ulonglong),
                ("WriteOperationCount", ctypes.c_ulonglong),
                ("OtherOperationCount", ctypes.c_ulonglong),
                ("ReadTransferCount", ctypes.c_ulonglong),
                ("WriteTransferCount", ctypes.c_ulonglong),
                ("OtherTransferCount", ctypes.c_ulonglong),
            ]

        class JOBOBJECT_BASIC_LIMIT_INFORMATION(ctypes.Structure):
            _fields_ = [
                ("PerProcessUserTimeLimit", ctypes.c_longlong),
                ("PerJobUserTimeLimit", ctypes.c_longlong),
                ("LimitFlags", wintypes.DWORD),
                ("MinimumWorkingSetSize", ctypes.c_size_t),
                ("MaximumWorkingSetSize", ctypes.c_size_t),
                ("ActiveProcessLimit", wintypes.DWORD),
                ("Affinity", ctypes.POINTER(ctypes.c_ulong)),
                ("PriorityClass", wintypes.DWORD),
                ("SchedulingClass", wintypes.DWORD),
            ]

        class JOBOBJECT_EXTENDED_LIMIT_INFORMATION(ctypes.Structure):
            _fields_ = [
                ("BasicLimitInformation", JOBOBJECT_BASIC_LIMIT_INFORMATION),
                ("IoInfo", IO_COUNTERS),
                ("ProcessMemoryLimit", ctypes.c_size_t),
                ("JobMemoryLimit", ctypes.c_size_t),
                ("PeakProcessMemoryUsed", ctypes.c_size_t),
                ("PeakJobMemoryUsed", ctypes.c_size_t),
            ]

        handle = kernel32.CreateJobObjectW(None, None)
        if not handle:
            return
        info = JOBOBJECT_EXTENDED_LIMIT_INFORMATION()
        info.BasicLimitInformation.LimitFlags = _JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE
        if not kernel32.SetInformationJobObject(
            handle,
            _JOB_OBJECT_EXTENDED_LIMIT_INFORMATION,
            ctypes.byref(info),
            ctypes.sizeof(info),
        ):
            kernel32.CloseHandle(handle)
            return
        _job_handle = handle
    except Exception:  # noqa: BLE001 - best effort; taskkill remains the fallback
        _job_handle = None


def assign_to_job_object(pid: int) -> None:
    """Put a spawned process into the kill-on-close job object."""
    if _job_handle is None or os.name != "nt":
        return
    try:
        import ctypes

        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        PROCESS_SET_QUOTA, PROCESS_TERMINATE = 0x0100, 0x0001
        h = kernel32.OpenProcess(PROCESS_SET_QUOTA | PROCESS_TERMINATE, False, pid)
        if not h:
            return
        try:
            kernel32.AssignProcessToJobObject(_job_handle, h)
        finally:
            kernel32.CloseHandle(h)
    except Exception:  # noqa: BLE001
        pass
