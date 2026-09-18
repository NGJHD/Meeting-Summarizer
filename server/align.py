r"""Forced alignment of known words to audio, under onnxruntime. No torch.

Whisper's DTW word timestamps are unreliable: on a measured 2h12m recording
32.5% of consecutive word pairs overlapped and the worst single word claimed 43
seconds (BUILD_NOTES 9ad). That is tolerable in a transcript and expensive
downstream -- the map stage reads speaker-attributed *turns*, and turn
boundaries are computed from these timings, so bad timings cost summary quality.
Measured blind at 4.67 against 2.33 out of 5 over six summaries (9aj).

This re-times words that are already known, which is a far easier problem than
recognising them: the CTC emissions of a wav2vec2 model are scored against the
one transcript we already have, and the best monotonic path through them says
when each character was spoken. The algorithm is torchaudio's, reimplemented in
numpy so nothing here needs torch:

    emissions -> trellis -> Viterbi backtrace -> character spans -> word spans

`models/wav2vec2-align.onnx` is torchaudio's WAV2VEC2_ASR_BASE_960H exported
once, offline (tools/export_align_onnx.py). English only -- the label set is
26 letters, an apostrophe, a word separator and CTC blank.
"""

from __future__ import annotations

import json
import wave
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np

SAMPLE_RATE = 16000
BLANK_ID = 0
SEPARATOR = "|"

# Segments handed to the aligner. Short enough that one bad segment cannot
# poison much, long enough to give the model context.
SEG_MIN_S = 3.0
SEG_MAX_S = 20.0
# Below this the conv stack has nothing to work with (torchaudio pads to 400).
MIN_SAMPLES = 400


@dataclass
class AlignedWord:
    text: str
    start: float
    end: float
    score: float


def _log_softmax(x: np.ndarray) -> np.ndarray:
    m = x.max(axis=-1, keepdims=True)
    e = np.exp(x - m)
    return x - m - np.log(e.sum(axis=-1, keepdims=True))


def _trellis(emission: np.ndarray, tokens: np.ndarray) -> np.ndarray:
    """Best score for having emitted the first j tokens by frame t."""
    n_frame, n_tok = emission.shape[0], len(tokens)
    tr = np.empty((n_frame + 1, n_tok + 1), dtype=np.float32)
    tr[0, 0] = 0.0
    tr[1:, 0] = np.cumsum(emission[:, BLANK_ID])
    tr[0, 1:] = -np.inf
    for t in range(n_frame):
        stay = tr[t, 1:] + emission[t, BLANK_ID]
        change = tr[t, :-1] + emission[t, tokens]
        tr[t + 1, 1:] = np.maximum(stay, change)
    return tr


def _backtrack(tr: np.ndarray, emission: np.ndarray, tokens: np.ndarray):
    """Walk the trellis back, returning (token_index, frame_index, prob)."""
    j = tr.shape[1] - 1
    t = int(np.argmax(tr[:, j]))
    path = []
    while t > 0:
        stayed = tr[t - 1, j] + emission[t - 1, BLANK_ID]
        changed = tr[t - 1, j - 1] + emission[t - 1, tokens[j - 1]]
        took_token = changed > stayed
        prob = float(np.exp(emission[t - 1, tokens[j - 1] if took_token else BLANK_ID]))
        path.append((j - 1, t - 1, prob))
        if took_token:
            j -= 1
            if j == 0:
                break
        t -= 1
    else:
        return None                      # ran out of frames before tokens
    return path[::-1]


def _merge_repeats(path, transcript: str):
    """Collapse the frame-wise path into one span per character."""
    out, i1 = [], 0
    while i1 < len(path):
        i2 = i1
        while i2 < len(path) and path[i2][0] == path[i1][0]:
            i2 += 1
        score = sum(p[2] for p in path[i1:i2]) / (i2 - i1)
        out.append((transcript[path[i1][0]], path[i1][1], path[i2 - 1][1] + 1, score))
        i1 = i2
    return out


def _merge_words(chars):
    """Group character spans into words on the separator."""
    words, i1 = [], 0
    while i1 < len(chars):
        i2 = i1
        while i2 < len(chars) and chars[i2][0] != SEPARATOR:
            i2 += 1
        if i2 > i1:
            segs = chars[i1:i2]
            span = sum(s[2] - s[1] for s in segs) or 1
            words.append((
                "".join(s[0] for s in segs), segs[0][1], segs[-1][2],
                sum(s[3] * (s[2] - s[1]) for s in segs) / span,
            ))
        i1 = i2 + 1
    return words


class Aligner:
    """One ONNX session, reused for every segment of a recording."""

    def __init__(self, model_path: Path, threads: int = 4):
        import onnxruntime as ort

        meta_path = model_path.with_suffix(".json")
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        self.labels = [str(c) for c in meta["labels"]]
        # torchaudio's labels are upper case; match on lower case as whisperx
        # does, so "Don't" and "DON'T" tokenise identically.
        self.index = {c.lower(): i for i, c in enumerate(self.labels)}

        opts = ort.SessionOptions()
        opts.intra_op_num_threads = max(1, int(threads))
        opts.inter_op_num_threads = 1
        self.session = ort.InferenceSession(
            str(model_path), opts, providers=["CPUExecutionProvider"]
        )
        self.input_name = self.session.get_inputs()[0].name

    def clean(self, text: str) -> tuple[str, list[int]]:
        """Reduce text to the model's alphabet, keeping a map back to it.

        Returns the cleaned string and, for each cleaned character, the index
        of the source character it came from -- which is how a character span
        is attributed back to a word.
        """
        out, src = [], []
        for i, ch in enumerate(text):
            low = ch.lower()
            if low == " ":
                if out and out[-1] != SEPARATOR:
                    out.append(SEPARATOR)
                    src.append(i)
            elif low in self.index:
                out.append(low)
                src.append(i)
        while out and out[-1] == SEPARATOR:
            out.pop(); src.pop()
        return "".join(out), src

    def emissions(self, audio: np.ndarray) -> np.ndarray:
        if audio.shape[0] < MIN_SAMPLES:
            audio = np.pad(audio, (0, MIN_SAMPLES - audio.shape[0]))
        logits = self.session.run(
            None, {self.input_name: audio[None, :].astype(np.float32)}
        )[0][0]
        return _log_softmax(logits.astype(np.float32))

    def tokenise(self, texts: list[str]):
        """Clean each word separately, keeping the map back to its index.

        A word can vanish entirely -- "20" and "..." have no character in an
        alphabet of 26 letters and an apostrophe. Those keep their original
        timing; dropping them would shift every word after them, and rejecting
        the whole segment (the first attempt here) lost 24% of the recording.
        """
        parts, owners = [], []
        for i, t in enumerate(texts):
            chars = [c.lower() for c in t if c.lower() in self.index and c != " "]
            if chars:
                parts.append("".join(chars))
                owners.append(i)
        return SEPARATOR.join(parts), owners

    def align(self, audio: np.ndarray, texts: list[str], t0: float, duration: float):
        """Align `texts` to `audio`; returns {word index: AlignedWord}."""
        clean, owners = self.tokenise(texts)
        if not clean:
            return {}
        emission = self.emissions(audio)

        unknown = [c for c in clean if c not in self.index and c != SEPARATOR]
        if unknown:
            mask = np.ones(emission.shape[1], dtype=bool)
            mask[BLANK_ID] = False
            wild = emission[:, mask].max(axis=1, keepdims=True)
            emission = np.concatenate([emission, wild], axis=1)
            wild_id = emission.shape[1] - 1
            tokens = np.array([self.index.get(c, wild_id) for c in clean], dtype=np.int64)
        else:
            tokens = np.array([self.index[c] for c in clean], dtype=np.int64)

        if emission.shape[0] <= len(tokens):
            return {}                    # more characters than frames
        tr = _trellis(emission, tokens)
        path = _backtrack(tr, emission, tokens)
        if not path:
            return {}

        words = _merge_words(_merge_repeats(path, clean))
        if len(words) != len(owners):
            return {}
        ratio = duration / max(tr.shape[0] - 1, 1)
        return {owners[k]: AlignedWord(w[0], w[1] * ratio + t0,
                                       w[2] * ratio + t0, w[3])
                for k, w in enumerate(words)}


def _read_block(wav_path: Path, start_s: float, end_s: float) -> np.ndarray:
    """Read [start_s, end_s) from a 16kHz mono PCM WAV as float32 in [-1, 1].

    Random access via `setpos`, so a 250MB file is never loaded whole
    (CLAUDE.md section 16).
    """
    with wave.open(str(wav_path), "rb") as fh:
        rate = fh.getframerate()
        total = fh.getnframes()
        a = max(0, min(total, int(start_s * rate)))
        b = max(a, min(total, int(end_s * rate)))
        if b <= a:
            return np.zeros(0, dtype=np.float32)
        fh.setpos(a)
        raw = fh.readframes(b - a)
    return np.frombuffer(raw, dtype="<i2").astype(np.float32) / 32768.0


def plan_segments(words, sentence_end) -> list[tuple[int, int]]:
    """Group word indices into utterances the aligner can chew on.

    Break on a sentence boundary once the span is worth aligning, and hard-cap
    the length so one runaway span cannot swallow minutes of audio.
    """
    spans, start = [], 0
    for i, w in enumerate(words):
        span = w.end - words[start].start
        if (sentence_end(w.text) and span >= SEG_MIN_S) or span >= SEG_MAX_S:
            spans.append((start, i + 1))
            start = i + 1
    if start < len(words):
        spans.append((start, len(words)))
    return spans


def align_words(words, wav_path: Path, model_path: Path, sentence_end,
                threads: int = 4, should_cancel=None, on_progress=None):
    """Re-time `words` against the audio. Returns (start, end) per word.

    Words the aligner cannot place keep their original timing: the caller
    indexes positionally and a dropped word would shift every speaker
    assignment after it. A segment that fails is left entirely alone.
    """
    if not words:
        return [], 0
    aligner = Aligner(model_path, threads)
    out = [(w.start, w.end) for w in words]
    spans = plan_segments(words, sentence_end)
    replaced = 0

    for n, (lo, hi) in enumerate(spans):
        if should_cancel is not None and should_cancel():
            break
        t0, t1 = words[lo].start, words[hi - 1].end
        if t1 <= t0:
            continue
        audio = _read_block(wav_path, t0, t1)
        if audio.shape[0] < MIN_SAMPLES:
            continue
        try:
            got = aligner.align(audio, [w.text for w in words[lo:hi]], t0, t1 - t0)
        except Exception:                # noqa: BLE001 - never fail the job
            got = {}
        for k, aw in got.items():
            out[lo + k] = (aw.start, max(aw.end, aw.start))
        replaced += len(got)
        if on_progress is not None and (n % 25) == 0:
            on_progress(n + 1, len(spans))

    if on_progress is not None:
        on_progress(len(spans), len(spans))
    return out, replaced
