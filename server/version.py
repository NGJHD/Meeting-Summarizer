"""App identity and version arithmetic.

The only file that changes when this code is reused for a different app
(BUILD_NOTES.md section 9q), and the only place the version number lives.

Everything here is pure -- no network, no filesystem -- so the comparison logic
can be exercised without downloading anything.
"""

from __future__ import annotations

import re

APP_NAME = "Meeting Summariser"
APP_AUTHOR = "Darren Ng"
APP_VERSION = "1.0.2"

GITHUB_REPO = "NGJHD/Meeting-Summarizer"
REPO_URL = "https://github.com/%s" % GITHUB_REPO
RELEASES_API = "https://api.github.com/repos/%s/releases/latest" % GITHUB_REPO

# A release carries two zips: the small source tree, which is what the updater
# downloads, and a full bundle that also contains `runtime\` and `bin\` for a
# first install. This marker tells them apart.
#
# The updater must never take the full one. It would turn a 190 KB update into
# 1.14 GB, and -- far worse -- it would robocopy `runtime\python.exe` over the
# interpreter the running app is executing from. A half-copied interpreter
# cannot start, so it cannot self-repair.
FULL_ASSET_MARKER = "-full"

# Files the update zip must contain before it is believed to be this app.
EXPECTED_FILES = ("run.bat", "server/main.py", "server/version.py", "web/index.html")

# Never overwritten by an update: the operator tunes it by hand, and
# config.load_config() merges any new keys in from _DEFAULTS anyway, so a
# stale file loses nothing.
PRESERVE = ("config.json",)


def parse_version(text: str):
    """"v1.2.0" -> (1, 2, 0). None if it cannot be read as one.

    An unparseable tag must answer "not newer" rather than raise: offering an
    update you cannot reason about is worse than offering none.
    """
    if not text:
        return None
    m = re.match(r"^\s*v?(\d+)\.(\d+)(?:\.(\d+))?", str(text))
    if not m:
        return None
    return (int(m.group(1)), int(m.group(2)), int(m.group(3) or 0))


def is_newer(candidate: str, current: str = APP_VERSION) -> bool:
    """Numeric comparison. "1.10.0" > "1.9.0" is false as strings."""
    a, b = parse_version(candidate), parse_version(current)
    if a is None or b is None:
        return False
    return a > b


def pick_asset(assets: list) -> dict | None:
    """The source zip attached to the release -- the update payload.

    Anything carrying FULL_ASSET_MARKER is a first-install bundle and is
    skipped. A release from before that split had a single zip and still
    resolves. Any other ambiguity is refused rather than guessed at: offering
    the wrong asset would overwrite an install with something unintended.
    """
    zips = [a for a in (assets or [])
            if str(a.get("name", "")).lower().endswith(".zip")
            and a.get("browser_download_url")]
    updates = [a for a in zips
               if FULL_ASSET_MARKER not in str(a.get("name", "")).lower()]
    # Exactly one candidate or nothing. A release carrying only a full bundle
    # yields no update rather than the dangerous one -- "nothing to install" is
    # the correct answer there, not "install the 1.14 GB one".
    return updates[0] if len(updates) == 1 else None
