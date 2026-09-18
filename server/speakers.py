"""Speaker samples and renaming (CLAUDE.md section 13, items 9 and 10).

Diarization produces anonymous clusters. The reduce prompt infers real names
where the transcript gives it evidence, but there is a hard ceiling: if nobody
says the chair's name aloud, no amount of prompting will recover it. The person
who was in the meeting knows in seconds. Asking beats inferring.

So after a run we extract, for each speaker, the three longest things they said,
as short audio clips. Reading a transcript line tells you what was said; hearing
it tells you who said it, which is the actual question.

Everything one meeting produced lives in its own folder, `output\\<name>\\`, with
the voice clips under `speaker_samples\\`:
    samples.json          speaker rows, clip metadata, and the map notes
    spk00_1.mp3 ...       three clips per speaker

The notes are cached in the same folder so that applying names re-runs only the
final reduce -- one call instead of the whole map loop, which on a 3.5-hour
recording is 68 minutes saved. That cache independently protects a long job
whose reduce stage fails.
"""

from __future__ import annotations

import json
import re
import subprocess
from dataclasses import dataclass
from pathlib import Path

from . import config, merge
from .merge import SpeakerTurn

CREATE_NO_WINDOW = getattr(subprocess, "CREATE_NO_WINDOW", 0)

SAMPLES_PER_SPEAKER = 3
# Long enough to recognise a voice, short enough to stay small and to load
# instantly. Clips are cut on turn boundaries, then capped.
MAX_CLIP_S = 18.0
MIN_CLIP_S = 1.5

# Skip the first seconds of a turn when cutting its clip.
#
# Turn boundaries land a few words early: the start of a turn routinely carries
# the tail of the previous speaker (merge.snap_to_sentences moves the boundary
# but cannot always place it right). That is tolerable in a transcript and fatal
# in a voice sample, because the clip is cut from the turn's first word and so
# opens on the wrong person -- reported on three of twelve clips, each time one
# sentence of somebody else before the right voice starts.
#
# 4 s clears a sentence of lead-in. It is only taken where the turn can spare
# it, so short turns are cut from their start as before.
CLIP_LEAD_IN_S = 4.0
CLIP_KEEP_S = 8.0


@dataclass
class Sample:
    speaker: int
    index: int
    start: float
    duration: float
    text: str
    file: str


def folder_for(meeting_name: str) -> Path:
    return config.meeting_dir(meeting_name) / "speaker_samples"


def _extract(wav: Path, start: float, duration: float, dest: Path) -> bool:
    """Cut one clip. MP3 so the browser can play it with no decoder fuss."""
    cmd = [
        str(config.FFMPEG), "-y", "-v", "error",
        "-ss", "%.3f" % max(0.0, start),
        "-t", "%.3f" % duration,
        "-i", str(wav),
        "-c:a", "libmp3lame", "-b:a", "64k", "-ac", "1",
        str(dest),
    ]
    try:
        proc = subprocess.run(
            cmd, capture_output=True, env=config.child_env(),
            creationflags=CREATE_NO_WINDOW, timeout=120,
        )
    except (OSError, subprocess.SubprocessError):
        return False
    return proc.returncode == 0 and dest.exists() and dest.stat().st_size > 0


def _clip_start(turn: SpeakerTurn) -> float:
    """Where to begin this turn's clip, skipping any borrowed lead-in."""
    first, last = turn.words[0].start, turn.words[-1].end
    if last - first - CLIP_LEAD_IN_S >= CLIP_KEEP_S:
        return first + CLIP_LEAD_IN_S
    return first


def _clip_speech_seconds(turn: SpeakerTurn) -> float:
    """Seconds of actual speech inside the first MAX_CLIP_S of a turn.

    The word intervals are unioned rather than summed: whisper's DTW ends
    routinely overrun the next word's start (32.5% of consecutive pairs on a
    measured 2h12m recording), so summing them reports far more speech than the
    clip can physically hold.
    """
    begin = _clip_start(turn)
    cutoff = begin + MAX_CLIP_S
    spans: list[list[float]] = []
    for w in turn.words:
        if w.start >= cutoff:
            break
        if w.end <= begin:
            continue
        a, b = max(w.start, begin), min(w.end, cutoff)
        if spans and a <= spans[-1][1]:
            spans[-1][1] = max(spans[-1][1], b)
        else:
            spans.append([a, b])
    return sum(b - a for a, b in spans)


def _text_within(turn: SpeakerTurn, begin: float, end: float) -> str:
    """The turn's words inside the clip window -- what the clip contains."""
    words = [w.text for w in turn.words if w.start < end and w.end > begin]
    return " ".join(words).strip() or turn.text[:400]


def build_samples(
    job, wav: Path, turns: list[SpeakerTurn], meeting_name: str
) -> list[Sample]:
    """Extract the longest few utterances per speaker as playable clips."""
    speakers: dict[int, list[SpeakerTurn]] = {}
    for turn in turns:
        if turn.speaker >= 0 and turn.words:
            speakers.setdefault(turn.speaker, []).append(turn)
    if not speakers:
        return []

    out_dir = folder_for(meeting_name)
    out_dir.mkdir(parents=True, exist_ok=True)

    samples: list[Sample] = []
    for speaker in sorted(speakers):
        # Rank by how much SPEECH lands in the clip we would actually cut, not
        # by the turn's wall-clock span. A turn only breaks at TURN_GAP_S, so a
        # span can be mostly silence -- ranking by it preferentially picked the
        # turns with the largest internal gaps, which is precisely backwards.
        # Measured on a 2h12m meeting: the top-ranked clip for one speaker was
        # 18 seconds of audio containing the single word "so".
        ranked = sorted(
            speakers[speaker], key=_clip_speech_seconds, reverse=True
        )
        picked = 0
        for turn in ranked:
            if picked >= SAMPLES_PER_SPEAKER:
                break
            start = _clip_start(turn)
            length = min(turn.words[-1].end - start, MAX_CLIP_S)
            if length < MIN_CLIP_S or _clip_speech_seconds(turn) < MIN_CLIP_S:
                continue
            name = "spk%02d_%d.mp3" % (speaker, picked + 1)
            if not _extract(wav, start, length, out_dir / name):
                continue
            samples.append(
                Sample(
                    speaker=speaker,
                    index=picked + 1,
                    start=start,
                    duration=length,
                    # Only the words the clip actually contains. The turn can
                    # run far past MAX_CLIP_S, and showing its whole text made
                    # the preview disagree with the audio for every long turn.
                    text=_text_within(turn, start, start + length)[:400],
                    file=name,
                )
            )
            picked += 1

    job.log("samples: %d clips for %d speakers" % (len(samples), len(speakers)))
    return samples


def save(meeting_name: str, samples: list[Sample], notes: list[str],
         mode: str, duration_s: float, names: dict | None = None,
         documents: dict | None = None) -> None:
    """Persist everything needed to re-label or re-generate this meeting.

    `documents` holds each generated document **as the model wrote it**, with
    the SPEAKER_nn labels still in place. Naming then rewrites the output files
    from that pristine copy rather than editing them in place, so a second pass
    with corrected names works from the original rather than from the result of
    the first pass -- which would no longer contain any labels to replace.
    """
    out_dir = folder_for(meeting_name)
    out_dir.mkdir(parents=True, exist_ok=True)
    existing = load(meeting_name) or {}
    payload = {
        "meeting_name": meeting_name,
        "mode": mode,
        "duration_s": duration_s,
        "names": names if names is not None else existing.get("names", {}),
        "notes": notes if notes is not None else existing.get("notes", []),
        "samples": ([s.__dict__ for s in samples] if samples is not None
                    else existing.get("samples", [])),
        "documents": {**existing.get("documents", {}), **(documents or {})},
    }
    config.write_atomic(out_dir / "samples.json", json.dumps(payload, indent=2))


def rebuild_documents(meeting_name: str, data: dict, names: dict) -> list:
    """Rewrite each stored document with the current names. No model involved.

    This is a search and replace, and it runs in milliseconds. Re-running the
    final reduce so the model can re-reason about the names is a *separate*,
    explicit action -- naming should not cost minutes.
    """
    stored = dict(data.get("documents") or {})
    adopted: dict[str, str] = {}
    written = []
    for mode in ("summary", "minutes"):
        path = config.meeting_dir(meeting_name) / ("%s_%s.md" % (meeting_name, mode))
        pristine = stored.get(mode)
        if pristine is None:
            # Written before pristine copies were kept. Adopt what is on disk as
            # the original -- it still carries the labels -- so this rename and
            # every later one work from the same text.
            if not path.exists():
                continue
            pristine = path.read_text(encoding="utf-8", errors="replace")
            adopted[mode] = pristine
        config.write_atomic(path, apply_names(pristine, names).rstrip() + "\n")
        written.append(path)
    if adopted:
        save(meeting_name, None, None, data.get("mode", "summary"),
             data.get("duration_s", 0.0), documents=adopted)
    return written


def tagged_transcript(meeting_name: str, names: dict):
    """Write the renamed transcript beside the raw one, never over it."""
    src = config.meeting_dir(meeting_name) / ("%s_transcript.md" % meeting_name)
    if not src.exists():
        return None
    dest = config.meeting_dir(meeting_name) / ("%s_transcript_tagged.md" % meeting_name)
    config.write_atomic(
        dest, apply_names(src.read_text(encoding="utf-8", errors="replace"), names))
    return dest


def names_from_tagged(meeting_name: str) -> dict:
    """Recover the names a previous session applied, for pre-filling the panel.

    `samples.json` is the record; the tagged transcript is a fallback for a
    meeting whose sidecar has been deleted, recovered by reading the raw and
    tagged transcripts side by side and pairing turns on their timestamps. The
    two files are written from the same turn list, so they line up exactly.
    """
    data = load(meeting_name)
    if data and data.get("names"):
        return dict(data["names"])

    raw = config.meeting_dir(meeting_name) / ("%s_transcript.md" % meeting_name)
    tagged = config.meeting_dir(meeting_name) / ("%s_transcript_tagged.md" % meeting_name)
    if not (raw.exists() and tagged.exists()):
        return {}

    pattern = re.compile(r"^\[(.+?)\]\s*\((\d{2}:\d{2}:\d{2})\)")

    def turns(path: Path) -> list[tuple[str, str]]:
        out = []
        with path.open(encoding="utf-8", errors="replace") as fh:
            for line in fh:
                m = pattern.match(line)
                if m:
                    out.append((m.group(1), m.group(2)))
        return out

    names: dict[str, str] = {}
    for (label, t_raw), (name, t_tag) in zip(turns(raw), turns(tagged)):
        if t_raw != t_tag:
            # The files have drifted; anything after this point is guesswork.
            break
        m = re.fullmatch(r"SPEAKER_(\d+)", label)
        if m and name != label:
            names.setdefault(str(int(m.group(1))), name)
    return names


def load(meeting_name: str) -> dict | None:
    path = folder_for(meeting_name) / "samples.json"
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


def speaker_rows(data: dict) -> list[dict]:
    """Group clips by speaker for the UI, longest first."""
    rows: dict[int, dict] = {}
    for s in data.get("samples", []):
        row = rows.setdefault(
            int(s["speaker"]),
            {"speaker": int(s["speaker"]),
             "label": merge.speaker_label(int(s["speaker"])),
             "name": data.get("names", {}).get(str(s["speaker"]), ""),
             "total_s": 0.0,
             "clips": []},
        )
        row["clips"].append({
            "file": s["file"],
            "start": s["start"],
            "start_hms": merge.hms(s["start"]),
            "duration": round(s["duration"], 1),
            "text": s["text"],
        })
    for row in rows.values():
        row["total_s"] = round(sum(c["duration"] for c in row["clips"]), 1)
    return [rows[k] for k in sorted(rows)]


def apply_names(text: str, names: dict) -> str:
    """Replace SPEAKER_nn with the supplied names throughout a document.

    Longest labels first so SPEAKER_1 cannot partially match SPEAKER_10.
    """
    for key in sorted(names, key=lambda k: -len(str(k))):
        value = str(names[key]).strip()
        if not value:
            continue
        label = merge.speaker_label(int(key))
        text = text.replace("[%s]" % label, "[%s]" % value)
        text = text.replace(label, value)
    return text
