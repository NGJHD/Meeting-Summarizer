r"""Build the two release assets. Development only; not shipped.

    runtime/python.exe tools/make_release.py

A release carries two zips and they are not variants of the same thing:

  Meeting-Summariser-vX.Y.Z.zip        the source tree, ~190 KB
  Meeting-Summariser-vX.Y.Z-full.zip   the same plus runtime\, bin\ and the small models, ~1.24 GB

The small one is the **update payload** -- what the in-app updater downloads and
robocopies over an install. It must stay small and must never contain
`runtime\python.exe`: the updater would be overwriting the interpreter the
running app is executing from, and a half-copied interpreter cannot start, so
it cannot self-repair. `version.pick_asset` skips anything with `-full` in the
name for exactly that reason.

The full one is for a **first install**: unzip it and only the three large
models are left to download. Everything in it is the set that was actually
tested, rather than whatever the pinned URLs happen to serve later.

The three large models are never bundled -- see BUNDLED_FILES below for which
ones do travel and why.
"""

from __future__ import annotations

import subprocess
import sys
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from server import version  # noqa: E402

PREFIX = "Meeting-Summariser"
BUNDLED_DIRS = ("runtime", "bin")

# The models small enough to travel, named individually -- `models\` as a whole
# is 27 GB. Together these are 103 MB and they carry their weight twice over:
# they are also the only two downloads that were never pinned to immutable
# bytes (sherpa-onnx publishes them on floating release tags), and the
# segmentation one arrives as a tar.bz2 that has to be unpacked and renamed.
# Shipping them retires the most fragile step in DOWNLOAD_MODELS.bat.
#
# Whisper large-v3-turbo is deliberately not here. It compresses to 1.42 GB,
# which would take the bundle to 2.56 GB against GitHub's 2 GB per-asset limit.
BUNDLED_FILES = (
    "models/ggml-silero-v5.1.2.bin",
    "models/segmentation-3.0.onnx",
    "models/speaker-embedding.onnx",
)
SKIP_PARTS = {"__pycache__"}
# This script is tracked so that a release is reproducible from the repository,
# but it is development tooling and has no business in an install.
EXCLUDE_PREFIXES = ("tools/",)


def source_entries() -> list:
    """Every tracked file, so the zip matches the commit rather than the disk."""
    out = subprocess.run(["git", "ls-files"], cwd=ROOT, capture_output=True,
                         text=True, check=True)
    return [line for line in out.stdout.splitlines()
            if line.strip() and not line.startswith(EXCLUDE_PREFIXES)]


def write(target: Path, extras: bool) -> None:
    files = source_entries()
    with zipfile.ZipFile(target, "w", zipfile.ZIP_DEFLATED, compresslevel=6) as z:
        for rel in files:
            src = ROOT / rel
            if src.is_file():
                z.write(src, "%s/%s" % (PREFIX, rel))
        if not extras:
            return
        for folder in BUNDLED_DIRS:
            base = ROOT / folder
            if not base.is_dir():
                raise SystemExit("missing %s -- run DOWNLOAD_MODELS.bat first" % folder)
            for src in base.rglob("*"):
                if not src.is_file() or SKIP_PARTS & set(src.parts):
                    continue
                z.write(src, "%s/%s" % (PREFIX, src.relative_to(ROOT).as_posix()))
        for rel in BUNDLED_FILES:
            src = ROOT / rel
            if not src.is_file():
                raise SystemExit("missing %s -- run DOWNLOAD_MODELS.bat first" % rel)
            z.write(src, "%s/%s" % (PREFIX, rel))


def main() -> None:
    v = version.APP_VERSION
    out = ROOT.parent
    plain = out / ("%s-v%s.zip" % (PREFIX, v))
    full = out / ("%s-v%s-full.zip" % (PREFIX, v))

    for target, extras in ((plain, False), (full, True)):
        print("building %s ..." % target.name)
        if target.exists():
            target.unlink()
        write(target, extras)
        size = target.stat().st_size
        print("  %d bytes (%.2f GB)" % (size, size / 1e9))
        if size > 2_000_000_000:
            raise SystemExit("%s exceeds GitHub's 2 GB per-asset limit" % target.name)

    print("\nassets for v%s:" % v)
    print("  %s   <- update payload, what pick_asset chooses" % plain.name)
    print("  %s   <- first install" % full.name)


if __name__ == "__main__":
    main()
