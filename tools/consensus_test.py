r"""Does consensus clustering stabilise the partition? Development only.

The partition is decided by a coin flip: a 3e-07 perturbation of the embeddings
-- the magnitude a different thread count produces -- gives the balanced answer
about half the time and a merged-speaker one the rest (BUILD_NOTES 9as).

A single clustering therefore cannot be trusted. The idea tested here is to run
it K times under that same perturbation and keep the partition that agrees most
with the others: if one answer is genuinely better supported by the data, it
should recur, and the knife-edge ones should not.

Agreement between two labellings is measured by pair counting, which needs no
correspondence between their labels: over a sample of index pairs, how often do
the two agree about whether the pair belongs together?

    runtime\python.exe -m tools.consensus_test <emb.npz> [K] [trials]
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from server.diarize_worker import cluster, prune  # noqa: E402

SAMPLE_RATE = 16000
NOISE = 3e-7
PARAMS = {"max_speakers": 20, "min_cluster_speech_s": 30,
          "min_cluster_speech_fraction": 0.005, "reassign_max_distance": 0.85}


def agreement(a: np.ndarray, b: np.ndarray, idx: np.ndarray) -> float:
    """Pair-counting agreement, label-permutation invariant."""
    i, j = idx[:, 0], idx[:, 1]
    return float(((a[i] == a[j]) == (b[i] == b[j])).mean())


def one_partition(emb, w, total, n, seed):
    rng = np.random.default_rng(seed)
    e = (emb + rng.normal(0, NOISE, emb.shape)).astype(np.float32)
    labels = cluster(e, n, 0.70)
    kept, _stats, _cov = prune(labels, e, w, PARAMS, total)
    return kept


def shares(kept, w):
    secs = sorted((float(w[kept == c].sum()) / SAMPLE_RATE
                   for c in set(int(x) for x in kept if x >= 0)), reverse=True)
    tot = sum(secs) or 1.0
    return "  ".join("%.1f%%" % (100 * s / tot) for s in secs)


def main(argv):
    path = Path(argv[1])
    K = int(argv[2]) if len(argv) > 2 else 11
    trials = int(argv[3]) if len(argv) > 3 else 4
    n = int(argv[4]) if len(argv) > 4 else 4

    z = np.load(path)
    emb, w = z["embeddings"], z["weights"]
    total = float(w.sum()) / SAMPLE_RATE
    rng = np.random.default_rng(0)
    idx = rng.integers(0, emb.shape[0], size=(200_000, 2))

    # Warm the library once so every measured call is in the same regime.
    cluster(emb[:64], 2, 0.70)

    print("K=%d runs per trial, %d trials, noise %.0e, num_speakers=%d\n"
          % (K, trials, NOISE, n))
    for t in range(trials):
        parts = [one_partition(emb, w, total, n, 1000 * t + k) for k in range(K)]
        # medoid: the partition agreeing most with all the others
        scores = [sum(agreement(p, q, idx) for q in parts if q is not p)
                  for p in parts]
        best = int(np.argmax(scores))
        mean_agree = np.mean([s / (K - 1) for s in scores])
        print("trial %d" % (t + 1))
        counts = {}
        for p in parts:
            counts[shares(p, w)] = counts.get(shares(p, w), 0) + 1
        for share, c in sorted(counts.items(), key=lambda kv: -kv[1]):
            print("   %2d/%d  %s" % (c, K, share))
        print("   consensus pick : %s" % shares(parts[best], w))
        print("   mean agreement : %.4f" % mean_agree)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
