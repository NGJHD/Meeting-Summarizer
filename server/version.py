"""App identity and version arithmetic.

The only file that changes when this code is reused for a different app
(UPDATE_BUTTON.md section 3), and the only place the version number lives.

Everything here is pure -- no network, no filesystem -- so the comparison logic
can be exercised without downloading anything.
"""

from __future__ import annotations

import re

APP_NAME = "Meeting Summariser"
APP_AUTHOR = "Darren Ng"
APP_VERSION = "1.0.0"

GITHUB_REPO = "NGJHD/Meeting-Summarizer"
REPO_URL = "https://github.com/%s" % GITHUB_REPO
RELEASES_API = "https://api.github.com/repos/%s/releases/latest" % GITHUB_REPO

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
    """The one .zip attached to the release. Nothing else is looked at."""
    zips = [a for a in (assets or [])
            if str(a.get("name", "")).lower().endswith(".zip")
            and a.get("browser_download_url")]
    if len(zips) != 1:
        # Zero is a release with nothing to install; more than one is an
        # ambiguity this deliberately refuses to guess at.
        return zips[0] if len(zips) == 1 else None
    return zips[0]
