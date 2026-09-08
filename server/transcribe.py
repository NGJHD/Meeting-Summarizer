"""Stage 2 - transcription via whisper.cpp (CLAUDE.md section 7).

Two things here are not obvious and are the reason this module is longer than
a subprocess call would suggest. Both are recorded in BUILD_NOTES.md.

1. Token timestamps come back in *VAD-compressed* time while segment
   timestamps are remapped to the original timeline. Diarization works on the
   original audio, so every token time must be mapped back before the merge
   stage can compare them. whisper-cli prints the mapping it used on stderr.

2. --dtw silently produces nothing (t_dtw = -1) unless flash attention is
   disabled with -nfa. Measured cost of -nfa: about +31% wall time.
"""

from __future__ import annotations

import json
import re
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from . import config
from .jobs import Cancelled, Job, JobError

TRANSCRIBE_FAILED = "The recording could not be transcribed."

CREATE_NEW_PROCESS_GROUP = getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0)
CREATE_NO_WINDOW = getattr(subprocess, "CREATE_NO_WINDOW", 0)

_PROGRESS_RE = re.compile(r"progress\s*=\s*(\d+)%")
_VAD_SEG_RE = re.compile(
    r"vad_segment_info:\s*"
    r"orig_start:\s*(-?[\d.]+),\s*orig_end:\s*(-?[\d.]+),\s*"
    r"vad_start:\s*(-?[\d.]+),\s*vad_end:\s*(-?[\d.]+)"
)


@dataclass
class Word:
    text: str
    start: float   # seconds, original audio timeline
    end: float

    @property
    def mid(self) -> float:
        return (self.start + self.end) / 2.0


class VadTimeline:
    """Maps VAD-compressed time back to original-audio time.

    whisper-cli logs one line per retained speech region:

        vad_segment_info: orig_start: 4.48, orig_end: 6.56,
                          vad_start: 1.79, vad_end: 3.87

    Within a region the mapping is linear. Outside any region (a token landing
    in a stripped silence) we clamp to the nearest region edge, which is the
    correct behaviour: the audio there was removed, so no word truly lives in
    that gap.
    """

    def __init__(self, regions: list[tuple[float, float, float, float]]):
        # each entry: (vad_start, vad_end, orig_start, orig_end)
        self.regions = sorted(regions, key=lambda r: r[0])

    @property
    def active(self) -> bool:
        return bool(self.regions)

    def to_original(self, t: float) -> float:
        if not self.regions:
            return t
        for vs, ve, os_, oe in self.regions:
            if vs <= t <= ve:
                span = ve - vs
                if span <= 0:
                    return os_
                return os_ + (t - vs) * (oe - os_) / span
        # before the first region
        if t < self.regions[0][0]:
            return self.regions[0][2]
        # after the last region
        for i in range(len(self.regions) - 1):
            ve = self.regions[i][1]
            next_vs = self.regions[i + 1][0]
            if ve < t < next_vs:
                return self.regions[i][3]
        return self.regions[-1][3] + (t - self.regions[-1][1])


def _tail_lines(path: Path, count: int) -> str:
    try:
        with open(path, "r", encoding="utf-8", errors="replace") as fh:
            return "".join(fh.readlines()[-count:])
    except OSError:
        return ""


def _follow_progress(
    job: Job,
    proc: subprocess.Popen,
    stderr_path: Path,
    regions: list[tuple[float, float, float, float]],
) -> None:
    """Tail whisper's stderr file while it runs, driving progress and the VAD map.

    Polling a file rather than reading a pipe means a slow or starved reader
    costs us only progress resolution, never the transcription itself.
    """
    with open(stderr_path, "r", encoding="utf-8", errors="replace") as reader:
        pending = ""
        while True:
            chunk = reader.read()
            if chunk:
                pending += chunk
                lines = pending.split("\n")
                pending = lines.pop()          # keep any partial last line
                for line in lines:
                    m = _VAD_SEG_RE.search(line)
                    if m:
                        o_s, o_e, v_s, v_e = (float(x) for x in m.groups())
                        regions.append((v_s, v_e, o_s, o_e))
                        continue
                    m = _PROGRESS_RE.search(line)
                    if m:
                        pct = int(m.group(1))
                        job.set_progress(
                            "transcribe", pct / 100.0, "Transcribing (%d%%)" % pct
                        )
                continue

            if proc.poll() is not None:
                # Process gone: drain whatever landed after the last read.
                rest = reader.read()
                for line in (pending + rest).split("\n"):
                    m = _VAD_SEG_RE.search(line)
                    if m:
                        o_s, o_e, v_s, v_e = (float(x) for x in m.groups())
                        regions.append((v_s, v_e, o_s, o_e))
                return
            if job.cancelled:
                return
            time.sleep(0.25)


def _build_command(cfg: dict, wav: Path, out_prefix: Path) -> list[str]:
    w = cfg["whisper"]
    cmd = [
        str(config.WHISPER_CLI),
        "-m", str(config.resolve(w["model"])),
        "-f", str(wav),
        "-l", str(w.get("language", "en")),
        "-ojf",                       # full JSON: per-token offsets (see BUILD_NOTES)
        "-of", str(out_prefix),
        "-pp",                        # progress -> stderr
        "-t", str(int(w.get("threads", 5))),
        # Do not carry decoded text between windows.
        #
        # whisper conditions each window on the previous window's text. On a
        # long, noisy recording it can drift into an unpunctuated lowercase
        # style early, and the carried context then locks that in for the rest
        # of the file. Measured over the first 15 minutes of a 3.5-hour council
        # recording: with carried context, 0 capitals and 0 commas; with
        # -mc 0, 179 capitals and 84 commas over the same audio.
        #
        # That matters beyond looks -- the reduce stages rely on sentence
        # boundaries and capitalised proper nouns to attribute names.
        "-mc", str(int(w.get("max_context", 0))),
    ]
    vad_model = config.resolve(w["vad_model"])
    if vad_model.exists():
        cmd += ["--vad", "--vad-model", str(vad_model)]
    if w.get("dtw", True):
        # -nfa is mandatory for --dtw to populate t_dtw (BUILD_NOTES.md).
        cmd += ["--dtw", "large.v3.turbo", "-nfa"]
    return cmd


def run(job: Job, cfg: dict, wav: Path) -> list[Word]:
    """Transcribe `wav` and return words on the original audio timeline."""
    job.check_cancelled()

    out_prefix = config.TEMP / "transcript"
    json_path = out_prefix.with_suffix(".json")
    if json_path.exists():
        json_path.unlink()

    cmd = _build_command(cfg, wav, out_prefix)
    job.log("whisper: " + " ".join(cmd[1:]))

    # whisper's stderr goes to a FILE, not a pipe.
    #
    # A pipe holds ~64KB. If whatever is reading it stalls, whisper-cli blocks
    # on write and the whole stage silently freezes. That is not hypothetical:
    # a 3.5-hour recording produces far more than 64KB of VAD and progress
    # output, and any GIL-holding native call elsewhere in the process is
    # enough to starve the reader. Writing to a file removes the backpressure
    # entirely -- whisper never waits for us.
    stderr_path = config.TEMP / "transcript.stderr"
    regions: list[tuple[float, float, float, float]] = []

    with open(stderr_path, "w", encoding="utf-8", errors="replace") as sink:
        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.DEVNULL,
            stderr=sink,
            env=config.child_env(),
            creationflags=CREATE_NEW_PROCESS_GROUP | CREATE_NO_WINDOW,
        )
        job.register_proc(proc)

        try:
            _follow_progress(job, proc, stderr_path, regions)
        finally:
            try:
                proc.wait(timeout=60)
            except subprocess.TimeoutExpired:
                pass
            job.unregister_proc(proc)

    tail = _tail_lines(stderr_path, 60)

    if job.cancelled:
        raise Cancelled()

    if proc.returncode != 0:
        raise JobError(
            TRANSCRIBE_FAILED,
            "whisper-cli exit=%s\n%s" % (proc.returncode, tail),
        )
    if not json_path.exists():
        raise JobError(TRANSCRIBE_FAILED, "whisper-cli produced no JSON output")

    timeline = VadTimeline(regions)
    if timeline.active:
        job.log("whisper: %d speech regions kept by VAD" % len(regions))

    words = parse_json(json_path, timeline)
    if not words:
        raise JobError(
            "No speech was found in that recording.",
            "transcript.json parsed to zero words",
        )
    job.log("whisper: %d words" % len(words))
    job.set_progress("transcribe", 1.0)
    return words


def parse_json(path: Path, timeline: Optional[VadTimeline] = None) -> list[Word]:
    """Turn whisper's full JSON into words on the original timeline.

    whisper emits *tokens*, which are sub-word pieces: " Pen" + "cil". A token
    that begins with a space starts a new word; one that does not continues the
    previous one.
    """
    with open(path, "r", encoding="utf-8", errors="replace") as fh:
        data = json.load(fh)

    timeline = timeline or VadTimeline([])
    words: list[Word] = []

    for segment in data.get("transcription", []):
        tokens = segment.get("tokens") or []
        if not tokens:
            # No token detail: fall back to one "word" per segment so the
            # pipeline still produces something usable.
            text = (segment.get("text") or "").strip()
            offs = segment.get("offsets") or {}
            if text and offs:
                words.append(Word(text, offs.get("from", 0) / 1000.0,
                                  offs.get("to", 0) / 1000.0))
            continue

        current: Optional[Word] = None
        for tok in tokens:
            raw = tok.get("text", "")
            if not raw or raw.startswith("[_"):     # [_BEG_], [_TT_123], [_EOT_]
                continue

            offs = tok.get("offsets") or {}
            t_from = offs.get("from", 0) / 1000.0
            t_to = offs.get("to", t_from * 1000) / 1000.0

            dtw = tok.get("t_dtw", -1)
            if dtw is not None and dtw >= 0:
                # t_dtw is in centiseconds and tracks the token's aligned end.
                t_to = max(t_to, dtw / 100.0)

            start = timeline.to_original(t_from)
            end = timeline.to_original(t_to)
            if end < start:
                end = start

            if raw.startswith(" ") or current is None:
                if current is not None:
                    words.append(current)
                current = Word(raw.strip(), start, end)
            else:
                current.text += raw
                current.end = max(current.end, end)

        if current is not None and current.text:
            words.append(current)

    return [w for w in words if w.text]
