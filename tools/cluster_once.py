r"""Cluster one embedding cache, once, in a fresh process. Development only.

Exists because `sherpa_onnx.FastClustering` is not stateless: the first call in
a process returns a different partition from every call after it, on identical
input (BUILD_NOTES 9as). `diarize_worker` clusters exactly once per job, so only
a first call describes what the app does -- and any tool that sweeps parameters
in a loop is measuring a regime the app never enters.

    runtime\python.exe -m tools.cluster_once <emb.npz> <num_speakers> <threshold>

Prints one JSON object: cluster count, kept count, coverage and speech seconds.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from server.diarize_worker import cluster, merge_close_clusters, prune  # noqa: E402

SAMPLE_RATE = 16000
DEFAULTS = {
    "max_speakers": 20,
    "min_cluster_speech_s": 30,
    "min_cluster_speech_fraction": 0.005,
    "reassign_max_distance": 0.85,
}


def main(argv: list[str]) -> int:
    path = Path(argv[1])
    num_speakers = int(argv[2]) if len(argv) > 2 else 0
    threshold = float(argv[3]) if len(argv) > 3 else 0.70
    merge_distance = float(argv[4]) if len(argv) > 4 else 0.0

    z = np.load(path)
    emb, w = z["embeddings"], z["weights"]
    total = float(w.sum()) / SAMPLE_RATE

    labels = cluster(emb, num_speakers, threshold)
    raw_clusters = len(set(int(c) for c in labels))
    kept, stats, coverage = prune(labels, emb, w, DEFAULTS, total)
    merges = 0
    # The app skips this when a count was supplied (section 8.1).
    if merge_distance > 0 and num_speakers <= 0:
        kept, merges = merge_close_clusters(kept, emb, w, merge_distance)

    seconds = sorted(
        (float(w[kept == c].sum()) / SAMPLE_RATE
         for c in set(int(x) for x in kept if x >= 0)),
        reverse=True,
    )
    print(json.dumps({
        "raw": raw_clusters,
        "pruned": stats["clusters_kept"],
        "merges": merges,
        "final": len(seconds),
        "coverage": round(coverage, 4),
        "seconds": [round(s, 1) for s in seconds],
    }))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
