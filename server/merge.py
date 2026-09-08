"""Stage 4 - merge transcript with speaker turns (CLAUDE.md section 9).

Whisper segments and diarization segments are two independent timelines that
do not line up; a single whisper segment routinely spans a speaker change.
Merging at segment level is the classic source of wrong attribution, so this
works at word level throughout.
"""

from __future__ import annotations

import bisect
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from . import config
from .diarize import Turn
from .transcribe import Word

NO_SPEAKER = -1

# Also break a turn when the same speaker pauses for longer than this.
#
# Grouping purely by speaker means an unattributed transcript (diarization off,
# or failed) becomes ONE turn containing every word -- 27,000 of them on a
# 3.5-hour recording. Section 10 requires the chunker to split only on turn
# boundaries, so a single turn that large cannot be chunked at all. Splitting on
# a natural pause keeps turns to a usable size without inventing a speaker
# change, and it reads better in the transcript either way.
TURN_GAP_S = 2.0


@dataclass
class SpeakerTurn:
    speaker: int
    words: list[Word] = field(default_factory=list)

    @property
    def start(self) -> float:
        return self.words[0].start if self.words else 0.0

    @property
    def text(self) -> str:
        return " ".join(w.text for w in self.words).strip()


def hms(seconds: float) -> str:
    seconds = max(0, int(round(seconds)))
    return "%02d:%02d:%02d" % (seconds // 3600, (seconds % 3600) // 60, seconds % 60)


def speaker_label(speaker: int) -> str:
    return "SPEAKER_%02d" % speaker


def assign_speakers(words: list[Word], turns: list[Turn]) -> list[SpeakerTurn]:
    """Attach a speaker to every word, then group into turns."""
    if not words:
        return []

    if turns:
        ordered = sorted(turns, key=lambda t: t.start)
        starts = [t.start for t in ordered]
        # Diarization segments can overlap (pyannote emits concurrent speech),
        # so the newest-starting segment is not necessarily the one containing
        # a given instant. Scanning back past the longest segment seen so far
        # is what makes the lookup correct rather than merely usually right.
        longest = max((t.end - t.start) for t in ordered)

        previous = NO_SPEAKER
        assigned: list[int] = []
        for word in words:
            mid = word.mid
            idx = bisect.bisect_right(starts, mid) - 1
            speaker = NO_SPEAKER
            j = idx
            while j >= 0 and starts[j] >= mid - longest:
                turn = ordered[j]
                if turn.start <= mid <= turn.end:
                    speaker = turn.speaker     # latest-starting match wins
                    break
                j -= 1
            if speaker == NO_SPEAKER:
                # The word fell in a gap between diarization segments: inherit
                # the previous word's speaker (CLAUDE.md section 9, step 3).
                speaker = previous
            previous = speaker
            assigned.append(speaker)

        # A leading run before the first diarization segment has nothing to
        # inherit from; give it the first speaker we did resolve.
        first_real = next((s for s in assigned if s != NO_SPEAKER), NO_SPEAKER)
        assigned = [first_real if s == NO_SPEAKER else s for s in assigned]

        # Diarization boundaries run late against the words; pull each change
        # onto the nearest sentence boundary. Only meaningful when there are
        # speakers to move between.
        assigned = snap_to_sentences(words, assigned)
    else:
        assigned = [NO_SPEAKER] * len(words)

    grouped: list[SpeakerTurn] = []
    for word, speaker in zip(words, assigned):
        current = grouped[-1] if grouped else None
        if (
            current is not None
            and current.speaker == speaker
            and word.start - current.words[-1].end <= TURN_GAP_S
        ):
            current.words.append(word)
        else:
            grouped.append(SpeakerTurn(speaker=speaker, words=[word]))

    return _drop_clustering_noise(grouped)


# A word that closes a sentence, allowing a trailing quote or bracket.
_SENTENCE_END = re.compile(r'[.!?]["\')\]]*$')

# How far a speaker change may be moved to reach a sentence boundary.
#
# Measured on a 34-minute council recording, as the share of speaker changes
# that begin a new sentence rather than cutting one in half:
#
#     window   changes   clean    words moved
#     0 (off)       37   18.9%              0
#     4            34   50.0%             30
#     6            34   70.6%             68  <- chosen
#     12           34   82.4%            104
#
# 6 is the knee. Beyond it the gain is small and the claim is large: moving a
# boundary a dozen words is no longer correcting a lag, it is guessing. An
# asymmetric window biased backwards was also tried, on the theory that the
# error is systematically late; it scored worse (64.7% at 6-back/3-forward),
# so the lag is not purely one-directional and the window stays symmetric.
SNAP_WINDOW = 6


def snap_to_sentences(words: list[Word], assigned: list[int]) -> list[int]:
    """Move each speaker change onto a nearby sentence boundary.

    Diarization boundaries land consistently *late* against the words: the
    first word or two of the new speaker's sentence keep the previous
    speaker's label, which reads as a speaker flapping mid-sentence:

        [SPEAKER_00] ... any inquiries going up. My name
        [SPEAKER_03] is Candice Starr. I'm at 67085 Pine Ridge Road.

    "My name" belongs to Candice. The rule is simply that a sentence should not
    span a speaker change, so a change that falls mid-sentence is pulled to the
    nearest sentence boundary within SNAP_WINDOW words.

    It only fires where punctuation exists, which is another reason `-mc 0`
    matters (BUILD_NOTES 3.4) -- on an unpunctuated transcript there is nothing
    to snap to.
    """
    n = len(words)
    if n < 2:
        return assigned

    ends_sentence = [bool(_SENTENCE_END.search(w.text)) for w in words]
    changes = [i for i in range(1, n) if assigned[i] != assigned[i - 1]]
    if not changes:
        return assigned

    out = list(assigned)
    for idx, i in enumerate(changes):
        if ends_sentence[i - 1]:
            continue                       # already on a boundary

        # Do not let a snap cross a neighbouring change, or it would swallow
        # a whole short turn rather than nudging a boundary.
        low = changes[idx - 1] + 1 if idx > 0 else 1
        high = changes[idx + 1] - 1 if idx + 1 < len(changes) else n - 1
        lo = max(low, i - SNAP_WINDOW)
        hi = min(high, i + SNAP_WINDOW)

        best = None
        for j in range(lo, hi + 1):
            if j >= 1 and ends_sentence[j - 1]:
                if best is None or abs(j - i) < abs(best - i):
                    best = j
        if best is None or best == i:
            continue

        before, after = out[i - 1], out[i]
        if best < i:
            for k in range(best, i):       # trailing words belong to the next speaker
                out[k] = after
        else:
            for k in range(i, best):       # leading words belong to the previous speaker
                out[k] = before

    return out


def _drop_clustering_noise(turns: list[SpeakerTurn]) -> list[SpeakerTurn]:
    """Discard turns under 3 words sitting between two turns of one other
    speaker. These are almost always clustering noise, not real interjections
    (CLAUDE.md section 9, step 5).
    """
    if len(turns) < 3:
        return turns

    keep: list[SpeakerTurn] = [turns[0]]
    i = 1
    while i < len(turns):
        current = turns[i]
        nxt = turns[i + 1] if i + 1 < len(turns) else None
        prev = keep[-1]

        if (
            nxt is not None
            and len(current.words) < 3
            and prev.speaker == nxt.speaker
            and prev.speaker != current.speaker
        ):
            # absorb the noise turn and the following turn into prev
            prev.words.extend(current.words)
            prev.words.extend(nxt.words)
            i += 2
            continue

        keep.append(current)
        i += 1

    return keep


def render(turns: list[SpeakerTurn], attributed: bool) -> str:
    """Render the transcript in the form the LLM stages consume.

        [SPEAKER_01] (00:14:22) So the vendor contract -- we agreed to defer.

    Timestamps are carried through to the final document: on a multi-hour
    recording they are what lets a reader jump back to the source.
    """
    lines: list[str] = []
    for turn in turns:
        text = turn.text
        if not text:
            continue
        stamp = "(%s)" % hms(turn.start)
        if attributed and turn.speaker != NO_SPEAKER:
            lines.append("[%s] %s %s" % (speaker_label(turn.speaker), stamp, text))
        else:
            lines.append("%s %s" % (stamp, text))
    return "\n".join(lines)


def write_transcript(
    path: Path,
    turns: list[SpeakerTurn],
    attributed: bool,
    meeting_name: str,
    duration_s: float,
    attribution_note: str = "",
) -> str:
    """Write output\\<name>_transcript.md and return its body.

    Kept after the run finishes: it is a useful deliverable in its own right
    and it is what you debug against when a summary is wrong (section 16).
    """
    body = render(turns, attributed)
    speakers = sorted({t.speaker for t in turns if t.speaker != NO_SPEAKER})
    header = [
        "# Transcript - %s" % meeting_name,
        "",
        "*Meeting Duration: %s*" % hms(duration_s),
        "*Speakers detected: %s*"
        % (", ".join(speaker_label(s) for s in speakers) if speakers else "none (unattributed)"),
    ]
    if attribution_note:
        header += ["", "> %s" % attribution_note]
    header += ["", "---", ""]
    config.write_atomic(path, "\n".join(header) + body + "\n")
    return body
