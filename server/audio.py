"""Stage 1 - audio conversion via ffmpeg (CLAUDE.md section 6)."""

from __future__ import annotations

import re
import subprocess
from pathlib import Path

from . import config
from .jobs import Cancelled, Job, JobError

# "Duration: 03:47:12.44, start: ..." from ffmpeg's banner.
_DURATION_RE = re.compile(r"Duration:\s*(\d+):(\d{2}):(\d{2})\.(\d+)")
# "time=03:12:44.10" from the progress lines.
_TIME_RE = re.compile(r"time=\s*(\d+):(\d{2}):(\d{2})\.(\d+)")

BAD_AUDIO = "That file couldn't be read as audio. Try a different recording."

CREATE_NEW_PROCESS_GROUP = getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0)
CREATE_NO_WINDOW = getattr(subprocess, "CREATE_NO_WINDOW", 0)


def _hms(m: re.Match) -> float:
    h, mi, s, frac = m.groups()
    return int(h) * 3600 + int(mi) * 60 + int(s) + float("0." + frac)


def probe_duration(path: Path) -> float:
    """Read the duration from ffmpeg's banner without decoding the file.

    ffmpeg exits non-zero here (no output specified) which is expected; we only
    care about the banner it prints first. Returns 0.0 if it cannot be read.
    """
    try:
        proc = subprocess.run(
            [str(config.FFMPEG), "-hide_banner", "-i", str(path)],
            capture_output=True,
            text=True,
            errors="replace",
            env=config.child_env(),
            creationflags=CREATE_NO_WINDOW,
            timeout=120,
        )
    except (OSError, subprocess.SubprocessError):
        return 0.0
    m = _DURATION_RE.search(proc.stderr or "")
    return _hms(m) if m else 0.0


def convert(job: Job, src: Path, dst: Path) -> float:
    """Convert `src` to 16kHz mono 16-bit WAV at `dst`.

    Returns the duration in seconds. Every downstream time estimate derives
    from this number, never from a constant (CLAUDE.md section 6).
    """
    job.check_cancelled()

    cmd = [
        str(config.FFMPEG),
        "-y",
        "-hide_banner",
        "-i", str(src),
        "-ar", "16000",
        "-ac", "1",
        "-c:a", "pcm_s16le",
        str(dst),
    ]

    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
        text=True,
        errors="replace",
        bufsize=1,
        env=config.child_env(),
        creationflags=CREATE_NEW_PROCESS_GROUP | CREATE_NO_WINDOW,
    )
    job.register_proc(proc)

    duration = job.duration_s or 0.0
    tail: list[str] = []
    try:
        assert proc.stderr is not None
        for line in proc.stderr:
            if job.cancelled:
                break
            tail.append(line)
            if len(tail) > 40:
                tail.pop(0)

            if not duration:
                m = _DURATION_RE.search(line)
                if m:
                    duration = _hms(m)
                    job.duration_s = duration
                    job.emit({"type": "duration", "seconds": duration})

            m = _TIME_RE.search(line)
            if m and duration > 0:
                job.set_progress("convert", _hms(m) / duration)
    finally:
        try:
            proc.wait(timeout=30)
        except subprocess.TimeoutExpired:
            pass
        job.unregister_proc(proc)

    if job.cancelled:
        raise Cancelled()

    if proc.returncode != 0 or not dst.exists() or dst.stat().st_size == 0:
        raise JobError(BAD_AUDIO, "ffmpeg exit=%s\n%s" % (proc.returncode, "".join(tail)))

    if not duration:
        duration = probe_duration(dst)
        job.duration_s = duration

    job.set_progress("convert", 1.0)
    return duration
