r"""Build the two release assets. Development only; not shipped.

    runtime/python.exe tools/make_release.py

A release carries two zips and they are not variants of the same thing:

  Meeting-Summariser-vX.Y.Z.zip        the source tree, ~190 KB
  Meeting-Summariser-vX.Y.Z-full.zip   the same plus runtime\ and bin\, ~1.14 GB

The small one is the **update payload** -- what the in-app updater downloads and
robocopies over an install. It must stay small and must never contain
`runtime\python.exe`: the updater would be overwriting the interpreter the
running app is executing from, and a half-copied interpreter cannot start, so
it cannot self-repair. `version.pick_asset` skips anything with `-full` in the
name for exactly that reason.

The full one is for a **first install**: unzip it and only the models are left
to download. Everything in it is the binary set that was actually tested,
rather than whatever the pinned URLs happen to serve later.

Models are never bundled. Two of them are individually larger than GitHub's
2 GB per-asset limit, so `DOWNLOAD_MODELS.bat` remains the only way to get them.
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
