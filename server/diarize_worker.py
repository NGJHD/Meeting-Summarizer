"""Diarization worker - runs as its own process. Do not import this in-process.

sherpa-onnx holds the CPython GIL for the whole of its native calls. Measured on
this build: a 166 s call let a competing Python thread run twice, against the
~3300 ticks it should have had. Running this on a thread freezes every other
Python thread in the server, including the one draining whisper-cli's output,
and the two stages deadlock. The child process is load-bearing, not a
convenience -- SpeakerEmbeddingExtractor below is the same pybind11 surface and
behaves the same way.

Structure (DIARIZATION_FIX.md part D2) -- three stages, not one call:

    1. SEGMENT  pyannote segmentation-3.0 via onnxruntime  -> per-frame labels
    2. EMBED    sherpa SpeakerEmbeddingExtractor           -> one vector per
                                                              (window, local speaker)
    3. CLUSTER  sherpa FastClustering, then prune          -> speaker ids

Stages 1 and 2 are deterministic and expensive; stage 3 is seconds. Splitting
them lets stages 1-2 be cached, so re-clustering the same recording -- a
threshold sweep, a different cap -- costs seconds instead of ~40 minutes.
sherpa's OfflineSpeakerDiarization does all three in one call, which is why it
had to be dropped.

Protocol (stdout, one per line):
    STAGE <name>
    PROGRESS <processed> <total>
    ERROR <message>
    DONE
Results go to the output path as JSON; diagnostics alongside it.
"""

from __future__ import annotations

import hashlib
import json
import sys
import time
import wave
from pathlib import Path

import numpy as np


def _emit(line: str) -> None:
    sys.stdout.write(line + "\n")
    sys.stdout.flush()


# ---------------------------------------------------------------------------
# audio
# ---------------------------------------------------------------------------

def read_wav(path: Path):
    """Read 16kHz mono 16-bit WAV into float32 [-1, 1], in blocks."""
    with wave.open(str(path), "rb") as wf:
        if wf.getnchannels() != 1 or wf.getsampwidth() != 2:
            raise RuntimeError(
                "expected 16-bit mono, got %d ch / %d bytes"
                % (wf.getnchannels(), wf.getsampwidth())
            )
        rate = wf.getframerate()
        total = wf.getnframes()
        out = np.empty(total, dtype=np.float32)
        pos = 0
        while pos < total:
            raw = wf.readframes(min(1 << 20, total - pos))
            if not raw:
                break
            chunk = np.frombuffer(raw, dtype="<i2")
            out[pos:pos + len(chunk)] = chunk.astype(np.float32) / 32768.0
            pos += len(chunk)
    return out[:pos], rate


def cache_key(wav: Path, params: dict) -> str:
    """Identify the audio plus everything that changes the embeddings.

    Hashing 400MB would cost seconds on every run for no benefit, so this uses
    size plus the head and tail of the file, which cannot collide in practice
    for successive recordings.
    """
    h = hashlib.sha1()
    size = wav.stat().st_size
    h.update(str(size).encode())
    with open(wav, "rb") as fh:
        h.update(fh.read(1 << 20))
        if size > (2 << 20):
            fh.seek(-(1 << 20), 2)
            h.update(fh.read(1 << 20))
    for k in ("segmentation_model", "embedding_model", "min_embed_duration"):
        h.update(str(params.get(k)).encode())
    return h.hexdigest()[:16]


# ---------------------------------------------------------------------------
# stage 1: segmentation
# ---------------------------------------------------------------------------

class SegmentationModel:
    """pyannote segmentation-3.0 under onnxruntime.

    Follows the reference implementation shipped in the sherpa-onnx
    segmentation model tarball (speaker-diarization-onnx.py).
    """

    def __init__(self, filename: str, threads: int):
        import onnxruntime as ort

        opts = ort.SessionOptions()
        opts.log_severity_level = 3
        opts.inter_op_num_threads = 1
        opts.intra_op_num_threads = max(1, threads)
        self.session = ort.InferenceSession(
            filename, sess_options=opts, providers=["CPUExecutionProvider"]
        )
        meta = self.session.get_modelmeta().custom_metadata_map
        self.window_size = int(meta["window_size"])
        self.sample_rate = int(meta["sample_rate"])
        self.window_shift = int(0.1 * self.window_size)
        self.receptive_field_size = int(meta["receptive_field_size"])
        self.receptive_field_shift = int(meta["receptive_field_shift"])
        self.num_speakers = int(meta["num_speakers"])
        self.powerset_max_classes = int(meta["powerset_max_classes"])
        self.num_classes = int(meta["num_classes"])
        self._in = self.session.get_inputs()[0].name
        self._out = self.session.get_outputs()[0].name

    def __call__(self, x: np.ndarray) -> np.ndarray:
        (y,) = self.session.run([self._out], {self._in: np.expand_dims(x, axis=1)})
        return y


def powerset_mapping(num_classes: int, num_speakers: int, max_classes: int) -> np.ndarray:
    """Map each powerset class index to a multi-hot speaker vector."""
    mapping = np.zeros((num_classes, num_speakers), dtype=np.int8)
    k = 1
    for size in range(1, max_classes + 1):
        if size == 1:
            for j in range(num_speakers):
                mapping[k, j] = 1
                k += 1
        elif size == 2:
            for j in range(num_speakers):
                for m in range(j + 1, num_speakers):
                    mapping[k, j] = 1
                    mapping[k, m] = 1
                    k += 1
        else:
            raise RuntimeError("unsupported powerset size %d" % size)
    return mapping


def segment(seg_m: SegmentationModel, audio: np.ndarray):
    """Return (labels, has_last_chunk).

    labels: (num_windows, num_frames, num_local_speakers) multi-hot int8.
    """
    from numpy.lib.stride_tricks import as_strided

    n = (audio.shape[0] - seg_m.window_size) // seg_m.window_shift + 1
    n = max(n, 0)
    windows = as_strided(
        audio,
        shape=(n, seg_m.window_size),
        strides=(seg_m.window_shift * audio.strides[0], audio.strides[0]),
    )
    has_last = (
        audio.shape[0] < seg_m.window_size
        or (audio.shape[0] - seg_m.window_size) % seg_m.window_shift > 0
    )

    batch = 32
    out = []
    total = n + (1 if has_last else 0)
    for i in range(0, n, batch):
        out.append(seg_m(windows[i:i + batch]))
        _emit("PROGRESS %d %d" % (min(i + batch, n), total))
    if has_last:
        tail = audio[n * seg_m.window_shift:]
        tail = np.pad(tail, (0, seg_m.window_size - tail.shape[0]))
        out.append(seg_m(np.expand_dims(tail, axis=0)))
        _emit("PROGRESS %d %d" % (total, total))

    y = np.vstack(out) if out else np.zeros((0, 0, seg_m.num_classes), dtype=np.float32)
    mapping = powerset_mapping(
        seg_m.num_classes, seg_m.num_speakers, seg_m.powerset_max_classes
    )
    idx = np.argmax(y, axis=-1)
    labels = mapping[idx.reshape(-1)].reshape(idx.shape[0], idx.shape[1], -1)
    return labels.astype(np.int8), has_last


def frame_speaker_count(labels: np.ndarray, seg_m: SegmentationModel) -> np.ndarray:
    """How many people are talking in each global frame."""
    per = labels.sum(axis=-1)
    num_frames = (
        int(
            (seg_m.window_size + (per.shape[0] - 1) * seg_m.window_shift)
            / seg_m.receptive_field_shift
        )
        + 1
    )
    acc = np.zeros(num_frames)
    cnt = np.zeros(num_frames)
    for i in range(per.shape[0]):
        start = int(i * seg_m.window_shift / seg_m.receptive_field_shift + 0.5)
        end = start + per.shape[1]
        acc[start:end] += per[i]
        cnt[start:end] += 1
    acc /= np.maximum(cnt, 1e-12)
    return (acc + 0.5).astype(np.int8)


# ---------------------------------------------------------------------------
# stage 2: embedding
# ---------------------------------------------------------------------------

def _append(buffer: np.ndarray, idx: int, seg: np.ndarray) -> int:
    """Copy `seg` into `buffer` at `idx`, clamped to the buffer's capacity."""
    n = min(seg.shape[0], buffer.shape[0] - idx)
    if n > 0:
        buffer[idx:idx + n] = seg[:n]
    return idx + max(n, 0)


def embed(
    audio: np.ndarray,
    labels: np.ndarray,
    seg_m: SegmentationModel,
    embedding_model: str,
    threads: int,
    min_embed_duration: float,
):
    """One embedding per (window, local speaker) that speaks enough in it.

    `min_embed_duration` gates out slivers whose embeddings are dominated by
    noise. The reference implementation hard-codes roughly 0.2 s here; this is
    configurable because those slivers are a major source of spurious clusters
    (DIARIZATION_FIX.md part E).
    """
    import sherpa_onnx

    cfg = sherpa_onnx.SpeakerEmbeddingExtractorConfig(
        model=embedding_model, num_threads=max(1, threads)
    )
    if not cfg.validate():
        raise RuntimeError("invalid speaker embedding configuration")
    extractor = sherpa_onnx.SpeakerEmbeddingExtractor(cfg)

    num_windows, num_frames, num_local = labels.shape
    min_samples = int(min_embed_duration * seg_m.sample_rate)
    buffer = np.empty(seg_m.window_size, dtype=np.float32)

    pairs: list[tuple[int, int]] = []
    vectors: list[np.ndarray] = []
    active_samples: list[int] = []
    skipped = 0

    # Frame index -> sample offset within the window, precomputed once. Walking
    # the frames in Python instead costs num_windows x num_local x num_frames
    # iterations -- about 22 million on a 3.5-hour recording, which dominated
    # the stage until it was vectorised.
    frame_to_sample = (
        np.arange(num_frames + 1) / num_frames * seg_m.window_size
    ).astype(np.int64)

    for i in range(num_windows):
        offset = i * seg_m.window_shift
        lt = labels[i].T                       # (num_local, num_frames)
        for j in range(num_local):
            frames = lt[j]
            if not frames.any():
                continue

            # run boundaries of this speaker's active frames, via diff
            edge = np.diff(np.concatenate(([0], frames, [0])).astype(np.int8))
            run_starts = np.flatnonzero(edge == 1)
            run_ends = np.flatnonzero(edge == -1)

            idx = 0
            for rs, re_ in zip(run_starts, run_ends):
                a = int(frame_to_sample[rs]) + offset
                b = int(frame_to_sample[re_]) + offset
                idx = _append(buffer, idx, audio[a:min(b, audio.shape[0])])

            if idx < min_samples:
                skipped += 1
                continue

            stream = extractor.create_stream()
            stream.accept_waveform(sample_rate=seg_m.sample_rate, waveform=buffer[:idx])
            stream.input_finished()
            if not extractor.is_ready(stream):
                continue
            vectors.append(np.array(extractor.compute(stream), dtype=np.float32))
            pairs.append((i, j))
            active_samples.append(idx)

        if (i % 200) == 0:
            _emit("PROGRESS %d %d" % (i, num_windows))
    _emit("PROGRESS %d %d" % (num_windows, num_windows))

    emb = np.array(vectors, dtype=np.float32) if vectors else np.zeros((0, 1), np.float32)
    return np.array(pairs, dtype=np.int32), emb, np.array(active_samples, np.int64), skipped


# ---------------------------------------------------------------------------
# stage 3: clustering and pruning
# ---------------------------------------------------------------------------

def cluster(embeddings: np.ndarray, num_speakers: int, threshold: float) -> np.ndarray:
    import sherpa_onnx

    cfg = sherpa_onnx.FastClusteringConfig(
        num_clusters=int(num_speakers) or -1, threshold=float(threshold)
    )
    return np.asarray(sherpa_onnx.FastClustering(cfg)(embeddings), dtype=np.int32)


def _unit(v: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(v, axis=-1, keepdims=True)
    return v / np.maximum(n, 1e-12)


def merge_close_clusters(
    assignment: np.ndarray,
    embeddings: np.ndarray,
    weights: np.ndarray,
    threshold: float,
):
    """Merge surviving clusters whose centroids are nearly identical.

    Pruning removes debris but leaves genuine fragmentation: one person can
    hold two or three large clusters, which then alternate mid-sentence in the
    transcript. Deciding that at the level of individual embeddings is what
    fails -- a single 5-second vector is noisy. Centroids are averages over
    hundreds of them, so the same decision is far better conditioned.

    Measured on the 3h25m meeting: the 190 centroid pairs split into six at
    cosine distance 0.03-0.15 (all the same voice) and nothing else below 0.57.
    A threshold anywhere in that gap is safe, which is why this works where
    threshold tuning on raw embeddings did not.
    """
    if threshold <= 0:
        return assignment, 0

    out = assignment.copy()
    merges = 0
    while True:
        ids = sorted(set(int(c) for c in out if c >= 0))
        if len(ids) < 2:
            break
        unit = _unit(embeddings)
        cent = np.stack([_unit(unit[out == c].mean(axis=0, keepdims=True))[0] for c in ids])
        dist = 1.0 - cent @ cent.T
        np.fill_diagonal(dist, np.inf)

        i, j = np.unravel_index(np.argmin(dist), dist.shape)
        if dist[i, j] > threshold:
            break
        # fold the smaller cluster (by speech) into the larger
        a, b = ids[int(i)], ids[int(j)]
        wa = weights[out == a].sum()
        wb = weights[out == b].sum()
        loser, winner = (a, b) if wa < wb else (b, a)
        out[out == loser] = winner
        merges += 1

    return out, merges


def prune(
    cluster_labels: np.ndarray,
    embeddings: np.ndarray,
    weights: np.ndarray,
    params: dict,
    total_speech_s: float,
):
    """Drop tiny clusters, cap the rest, reassign what we can.

    Duration-independent by construction, which is exactly what threshold
    tuning can never be (DIARIZATION_FIX.md part D3).
    """
    sample_rate = 16000
    ids = np.unique(cluster_labels)
    seconds = {int(c): float(weights[cluster_labels == c].sum()) / sample_rate for c in ids}

    floor = max(
        float(params.get("min_cluster_speech_s", 30)),
        float(params.get("min_cluster_speech_fraction", 0.005)) * total_speech_s,
    )
    keep = [c for c in ids if seconds[int(c)] >= floor]
    keep.sort(key=lambda c: -seconds[int(c)])
    max_speakers = int(params.get("max_speakers", 20))
    dropped_by_cap = max(0, len(keep) - max_speakers)
    keep = keep[:max_speakers]
    keep_set = set(int(c) for c in keep)

    stats = {
        "clusters_before": int(len(ids)),
        "clusters_kept": int(len(keep)),
        "dropped_by_cap": int(dropped_by_cap),
        "min_cluster_speech_s_used": round(floor, 2),
        "speech_seconds_by_cluster_before": {
            str(int(c)): round(seconds[int(c)], 2) for c in ids
        },
    }

    if not keep_set:
        return cluster_labels, stats, 0.0

    # centroids of survivors, on unit vectors so this is cosine distance
    unit = _unit(embeddings)
    centroids = {}
    for c in keep_set:
        m = cluster_labels == c
        centroids[c] = _unit(unit[m].mean(axis=0, keepdims=True))[0]

    order = sorted(keep_set)
    cmat = np.stack([centroids[c] for c in order])
    max_dist = float(params.get("reassign_max_distance", 0.85))

    out = cluster_labels.copy()
    orphan = np.array([int(c) not in keep_set for c in cluster_labels])
    reassigned = 0
    unknown = 0
    if orphan.any():
        d = 1.0 - unit[orphan] @ cmat.T          # cosine distance
        best = d.argmin(axis=1)
        bestd = d[np.arange(d.shape[0]), best]
        new = np.where(bestd <= max_dist, np.array(order)[best], -1)
        out[orphan] = new
        reassigned = int((new >= 0).sum())
        unknown = int((new < 0).sum())

    stats["reassigned"] = reassigned
    stats["left_unknown"] = unknown

    kept_seconds = float(weights[out >= 0].sum()) / sample_rate
    coverage = kept_seconds / total_speech_s if total_speech_s > 0 else 0.0
    stats["coverage"] = round(coverage, 4)
    return out, stats, coverage


# ---------------------------------------------------------------------------
# reconstruct time segments
# ---------------------------------------------------------------------------

def build_segments(
    labels: np.ndarray,
    pairs: np.ndarray,
    assignment: np.ndarray,
    speakers_per_frame: np.ndarray,
    seg_m: SegmentationModel,
    audio_len: int,
    has_last: bool,
    params: dict,
):
    """Turn per-(window, speaker) cluster ids back into (start, end, speaker)."""
    valid = assignment >= 0
    if not valid.any():
        return []
    speaker_ids = sorted(set(int(x) for x in assignment[valid]))
    remap = {c: i for i, c in enumerate(speaker_ids)}
    k_total = len(speaker_ids)

    num_windows, num_frames, _ = labels.shape
    num_global = (
        int(
            (seg_m.window_size + (num_windows - 1) * seg_m.window_shift)
            / seg_m.receptive_field_shift
        )
        + 1
    )
    count = np.zeros((num_global, k_total), dtype=np.float32)

    for (i, j), c in zip(pairs, assignment):
        if c < 0:
            continue
        t = remap[int(c)]
        start = int(i * seg_m.window_shift / seg_m.receptive_field_shift + 0.5)
        count[start:start + num_frames, t] += labels[i, :, j]

    if has_last:
        stop = int(audio_len / seg_m.receptive_field_shift)
        count = count[:stop]
        speakers_per_frame = speakers_per_frame[:stop]

    n = min(count.shape[0], speakers_per_frame.shape[0])
    count = count[:n]
    spf = speakers_per_frame[:n]

    order = np.argsort(-count, axis=-1)
    final = np.zeros_like(count)
    for i in range(n):
        for k in range(int(spf[i])):
            if k < k_total:
                final[i, order[i, k]] = 1

    scale = seg_m.receptive_field_shift / seg_m.sample_rate
    scale_offset = seg_m.receptive_field_size / seg_m.sample_rate * 0.5
    min_on = float(params.get("min_duration_on", 0.3))
    min_off = float(params.get("min_duration_off", 0.5))

    segments: list[dict] = []
    for t in range(k_total):
        frames = final[:, t]
        runs: list[list[float]] = []
        active = frames[0] > 0.5
        start = 0 if active else None
        for i in range(1, n):
            if active and frames[i] < 0.5:
                runs.append([start * scale + scale_offset, i * scale + scale_offset])
                active = False
            elif not active and frames[i] > 0.5:
                start = i
                active = True
        if active and start is not None:
            runs.append([start * scale + scale_offset, (n - 1) * scale + scale_offset])

        # bridge short gaps, then drop what is still too short
        merged: list[list[float]] = []
        for r in runs:
            if merged and r[0] - merged[-1][1] <= min_off:
                merged[-1][1] = r[1]
            else:
                merged.append(r)
        for r in merged:
            if r[1] - r[0] >= min_on:
                segments.append({"start": r[0], "end": r[1], "speaker": t})

    segments.sort(key=lambda s: s["start"])
    return segments


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main(argv: list[str]) -> int:
    if len(argv) != 4:
        _emit("ERROR usage: diarize_worker <wav> <params.json> <out.json>")
        return 2

    wav_path, params_path, out_path = (Path(a) for a in argv[1:])
    timings: dict[str, float] = {}

    try:
        params = json.loads(params_path.read_text(encoding="utf-8"))

        t0 = time.time()
        audio, rate = read_wav(wav_path)
        timings["load_audio"] = time.time() - t0

        seg_m = SegmentationModel(params["segmentation_model"], int(params["threads"]))
        if rate != seg_m.sample_rate:
            _emit("ERROR model expects %dHz but audio is %dHz" % (seg_m.sample_rate, rate))
            return 1

        key = cache_key(wav_path, params)
        use_cache = bool(params.get("embedding_cache", True))
        # Segmentation and embedding are cached separately: embedding is by far
        # the longer stage, and losing 3 minutes of segmentation to a crash or a
        # cancel partway through it is pure waste.
        seg_cache = wav_path.parent / ("embeddings-%s-seg.npz" % key)
        emb_cache = wav_path.parent / ("embeddings-%s-emb.npz" % key)

        if use_cache and seg_cache.exists():
            _emit("STAGE cache")
            z = np.load(seg_cache)
            labels, has_last = z["labels"], bool(z["has_last"][0])
            timings["segment"] = float(z["t_segment"][0])
        else:
            _emit("STAGE segment")
            t0 = time.time()
            labels, has_last = segment(seg_m, audio)
            timings["segment"] = time.time() - t0
            if use_cache:
                np.savez_compressed(
                    seg_cache, labels=labels, has_last=np.array([has_last]),
                    t_segment=np.array([timings["segment"]]),
                )

        if use_cache and emb_cache.exists():
            _emit("STAGE cache")
            z = np.load(emb_cache)
            pairs, embeddings, weights = z["pairs"], z["embeddings"], z["weights"]
            skipped = int(z["skipped"][0])
            timings["embed"] = float(z["t_embed"][0])
        else:
            _emit("STAGE embed")
            t0 = time.time()
            pairs, embeddings, weights, skipped = embed(
                audio, labels, seg_m,
                params["embedding_model"],
                int(params["threads"]),
                float(params.get("min_embed_duration", 1.0)),
            )
            timings["embed"] = time.time() - t0
            if use_cache and embeddings.shape[0]:
                np.savez_compressed(
                    emb_cache, pairs=pairs, embeddings=embeddings, weights=weights,
                    skipped=np.array([skipped]),
                    t_embed=np.array([timings["embed"]]),
                )

        if embeddings.shape[0] == 0:
            _emit("ERROR no speech segments found")
            return 1

        _emit("STAGE cluster")
        t0 = time.time()
        raw = cluster(
            embeddings,
            int(params.get("num_speakers", 0)),
            float(params.get("cluster_threshold", 0.7)),
        )
        timings["cluster"] = time.time() - t0

        total_speech_s = float(weights.sum()) / seg_m.sample_rate
        assignment, stats, coverage = prune(raw, embeddings, weights, params, total_speech_s)
        assignment, merges = merge_close_clusters(
            assignment, embeddings, weights,
            float(params.get("merge_centroid_distance", 0.35)),
        )
        stats["centroid_merges"] = merges
        stats["clusters_kept"] = len(set(int(c) for c in assignment if c >= 0))

        spf = frame_speaker_count(labels, seg_m)
        t0 = time.time()
        segments = build_segments(
            labels, pairs, assignment, spf, seg_m, audio.shape[0], has_last, params
        )
        timings["reconstruct"] = time.time() - t0

        # Automatic degradation (part F): meaningless labels are worse than
        # none, because a reader cannot tell clustering debris from a person.
        min_cov = float(params.get("min_coverage_fraction", 0.80))
        max_speakers = int(params.get("max_speakers", 20))
        degraded = ""
        if stats["clusters_kept"] > max_speakers:
            degraded = "surviving cluster count %d exceeds max_speakers %d" % (
                stats["clusters_kept"], max_speakers)
        elif coverage < min_cov:
            degraded = "surviving clusters cover only %.1f%% of speech (need %.0f%%)" % (
                coverage * 100, min_cov * 100)
        if degraded:
            _emit("DEGRADED %s" % degraded)
            segments = []

        diag = {
            "timings": {k: round(v, 2) for k, v in timings.items()},
            "audio_seconds": round(audio.shape[0] / seg_m.sample_rate, 1),
            "total_speech_seconds": round(total_speech_s, 1),
            "windows": int(labels.shape[0]),
            "embeddings": int(embeddings.shape[0]),
            "skipped_short": int(skipped),
            "segments": len(segments),
            "degraded": degraded,
            "cache": str(emb_cache.name),
            **stats,
        }
        out_path.write_text(json.dumps(segments), encoding="utf-8")
        out_path.with_suffix(".diag.json").write_text(json.dumps(diag, indent=2), encoding="utf-8")
        _emit("DONE")
        return 0

    except Exception as exc:  # noqa: BLE001
        import traceback

        sys.stderr.write(traceback.format_exc())
        _emit("ERROR %s" % str(exc).replace("\n", " ")[:400])
        return 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
