"""Check GitHub Releases for a newer build, download it, and swap it in.

This is the **one** place the app touches the network, and only when somebody
presses a button. CLAUDE.md section 0 forbids internet access at runtime; that
rule is about the pipeline, which must work with the adapter disabled, and it
still does. An explicit "check for updates" is a deliberate, documented
exception -- nothing here runs on startup, on a timer, or in the background.

The workflow is UPDATE_BUTTON.md section 1, with two simplifications that fall
out of this being a Python app in a plain folder rather than a packaged exe:

- Download, unpack and verify all happen **in Python, before anything is
  replaced**. The batch script only waits, copies and restarts, so there is far
  less to get wrong in cmd.
- Nothing is overwritten until the staged copy has been verified. Every failure
  path leaves the installed app exactly as it was.

The traps in UPDATE_BUTTON.md section 4 are respected where they apply:
cmd.exe is the executable and the script is a separate argv entry (4.1); the
wait is on a marker file and needs no pipe, so there is no `find.exe` to hang
(4.2); robocopy retries locked files and skips unchanged ones (4.4); and the
staged copy is verified against the tag before it is trusted (4.5).
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import threading
import time
import urllib.error
import urllib.request
import zipfile
from pathlib import Path

from . import config, version

CREATE_NO_WINDOW = getattr(subprocess, "CREATE_NO_WINDOW", 0)
DETACHED_PROCESS = getattr(subprocess, "DETACHED_PROCESS", 0x00000008)

STAGING_PREFIX = "meetsum-update-"
# Marker the app holds for its whole life. The update script waits for it to
# disappear rather than polling the process list.
RUNNING_MARKER = config.TEMP / "running.lock"

USER_AGENT = "%s/%s" % (version.APP_NAME.replace(" ", "-"), version.APP_VERSION)
NETWORK_TIMEOUT = 30

# Plain sentences; a non-technical user never sees a traceback (section 13).
NO_NETWORK = ("Couldn't reach GitHub. Check the network connection and try "
              "again -- everything else in this app works offline.")
NO_RELEASE = "No update has been published yet."
NO_ASSET = ("The latest release has no download attached to it. Try again "
            "later.")
NOT_WRITABLE = ("This folder can't be written to, so the update can't be "
                "installed. Move the app somewhere like your Desktop, or ask "
                "whoever installed it.")


_state_lock = threading.Lock()
_state: dict = {"phase": "idle", "message": "", "downloaded": 0, "total": 0}
_cancel = threading.Event()


# ---------------------------------------------------------------------------
# progress state, polled by the UI
# ---------------------------------------------------------------------------

def state() -> dict:
    with _state_lock:
        return dict(_state)


def _set(**fields) -> None:
    with _state_lock:
        _state.update(fields)


def cancel() -> None:
    _cancel.set()


# ---------------------------------------------------------------------------
# marker file
# ---------------------------------------------------------------------------

def mark_running() -> None:
    try:
        config.TEMP.mkdir(parents=True, exist_ok=True)
        RUNNING_MARKER.write_text(str(os.getpid()), encoding="utf-8")
    except OSError:
        pass


def clear_running() -> None:
    try:
        RUNNING_MARKER.unlink()
    except OSError:
        pass


# ---------------------------------------------------------------------------
# check
# ---------------------------------------------------------------------------

def _get_json(url: str) -> dict:
    req = urllib.request.Request(url, headers={
        "User-Agent": USER_AGENT,
        "Accept": "application/vnd.github+json",
    })
    with urllib.request.urlopen(req, timeout=NETWORK_TIMEOUT) as resp:
        return json.loads(resp.read().decode("utf-8", errors="replace"))


def check() -> dict:
    """Ask GitHub what the latest release is. Never raises."""
    sweep_staging()
    try:
        data = _get_json(version.RELEASES_API)
    except urllib.error.HTTPError as exc:
        if exc.code == 404:
            return {"status": "none", "message": NO_RELEASE,
                    "current": version.APP_VERSION}
        return {"status": "error", "message": NO_NETWORK,
                "current": version.APP_VERSION}
    except Exception:  # noqa: BLE001 - offline, DNS, TLS, timeout, bad JSON
        return {"status": "error", "message": NO_NETWORK,
                "current": version.APP_VERSION}

    tag = str(data.get("tag_name") or "")
    if not version.is_newer(tag):
        return {
            "status": "current",
            "current": version.APP_VERSION,
            "latest": tag or version.APP_VERSION,
            "message": "Version %s is the latest." % version.APP_VERSION,
        }

    asset = version.pick_asset(data.get("assets"))
    if not asset:
        return {"status": "error", "message": NO_ASSET,
                "current": version.APP_VERSION, "latest": tag}

    return {
        "status": "available",
        "current": version.APP_VERSION,
        "latest": version.parse_version(tag) and tag,
        "asset": asset["name"],
        "url": asset["browser_download_url"],
        "size_bytes": int(asset.get("size") or 0),
        "notes": (data.get("body") or "").strip()[:2000],
        "writable": is_writable(),
    }


def is_writable() -> bool:
    """Check before the download, not after (UPDATE_BUTTON.md section 5)."""
    probe = config.ROOT / (".write-probe-%d" % os.getpid())
    try:
        probe.write_text("x", encoding="utf-8")
        probe.unlink()
        return True
    except OSError:
        return False


# ---------------------------------------------------------------------------
# staging
# ---------------------------------------------------------------------------

def _staging_root() -> Path:
    import tempfile

    return Path(tempfile.gettempdir())


def sweep_staging(max_age_s: float = 86400) -> None:
    """Delete abandoned staging folders. A power cut mid-update leaves one.

    Only ever touches names carrying this app's own prefix -- pointed anywhere
    else this would take a real folder with it.
    """
    now = time.time()
    try:
        entries = list(_staging_root().iterdir())
    except OSError:
        return
    for path in entries:
        if not path.name.startswith(STAGING_PREFIX):
            continue
        try:
            if now - path.stat().st_mtime < max_age_s:
                continue
            if path.is_dir():
                shutil.rmtree(path, ignore_errors=True)
            else:
                path.unlink()
        except OSError:
            pass


def _download(url: str, dest: Path, total_hint: int) -> None:
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(req, timeout=NETWORK_TIMEOUT) as resp:
        total = int(resp.headers.get("Content-Length") or total_hint or 0)
        _set(total=total, downloaded=0)
        done = 0
        with open(dest, "wb") as fh:
            while True:
                if _cancel.is_set():
                    raise RuntimeError("cancelled")
                block = resp.read(262144)
                if not block:
                    break
                fh.write(block)
                done += len(block)
                _set(downloaded=done)


def _verify(folder: Path, expected_tag: str) -> Path:
    """Return the folder that actually holds the app, or raise.

    A release zip may wrap everything in a single top-level directory; look one
    level down before giving up.
    """
    candidates = [folder]
    try:
        children = [c for c in folder.iterdir() if c.is_dir()]
    except OSError:
        children = []
    if len(children) == 1:
        candidates.append(children[0])

    for root in candidates:
        if all((root / rel).exists() for rel in version.EXPECTED_FILES):
            found = _read_version(root / "server" / "version.py")
            want = version.parse_version(expected_tag)
            if found and want and version.parse_version(found) != want:
                # Read it and it disagrees: a hard stop (section 4.5).
                raise RuntimeError(
                    "the download says version %s but the release is tagged %s"
                    % (found, expected_tag))
            return root
    raise RuntimeError("the download does not look like this application")


def _read_version(path: Path) -> str:
    """Pull APP_VERSION out without importing the downloaded code."""
    import re

    try:
        text = path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return ""
    m = re.search(r'^APP_VERSION\s*=\s*["\']([^"\']+)["\']', text, re.M)
    return m.group(1) if m else ""


# ---------------------------------------------------------------------------
# install
# ---------------------------------------------------------------------------

def install(url: str, size_bytes: int, tag: str, port: int) -> None:
    """Download, unpack, verify, then hand over to a script. Worker thread."""
    _cancel.clear()
    stage = None
    try:
        if not is_writable():
            _set(phase="error", message=NOT_WRITABLE)
            return

        import tempfile

        stage = Path(tempfile.mkdtemp(prefix=STAGING_PREFIX))
        _set(phase="downloading", message="Downloading version %s" % tag,
             downloaded=0, total=size_bytes)
        zip_path = stage / "update.zip"
        _download(url, zip_path, size_bytes)

        _set(phase="unpacking", message="Unpacking")
        unpacked = stage / "unpacked"
        unpacked.mkdir()
        with zipfile.ZipFile(zip_path) as zf:
            _safe_extract(zf, unpacked)

        _set(phase="verifying", message="Checking the download")
        ready = _verify(unpacked, tag)

        _set(phase="applying",
             message="Restarting to finish the update")
        script = _write_script(stage, ready, port)
        # cmd.exe is the executable and the script is its own argument: never
        # the script as the executable, never shell=True with an interpolated
        # path (UPDATE_BUTTON.md section 4.1).
        subprocess.Popen(
            [os.environ.get("ComSpec") or "cmd.exe", "/c", str(script)],
            creationflags=DETACHED_PROCESS | subprocess.CREATE_NEW_PROCESS_GROUP,
            close_fds=True,
        )
        _set(phase="restarting",
             message="The application will close and reopen by itself.")
        threading.Timer(1.5, _quit).start()

    except RuntimeError as exc:
        if _cancel.is_set():
            _set(phase="idle", message="", downloaded=0, total=0)
        else:
            _set(phase="error", message="The update couldn't be installed: %s." % exc)
        _cleanup(stage)
    except Exception:  # noqa: BLE001 - never show a traceback
        import sys

        from . import jobs

        jobs.log_exception(sys.exc_info()[1])
        _set(phase="error",
             message="The update couldn't be downloaded. Nothing was changed.")
        _cleanup(stage)


def _safe_extract(zf: zipfile.ZipFile, dest: Path) -> None:
    """Extract, refusing any member that would escape the destination."""
    dest = dest.resolve()
    for member in zf.infolist():
        target = (dest / member.filename).resolve()
        if not str(target).startswith(str(dest)):
            raise RuntimeError("the download contains an unexpected path")
    zf.extractall(dest)


def _cleanup(stage) -> None:
    if stage:
        shutil.rmtree(stage, ignore_errors=True)


def _quit() -> None:
    """Stop the server so the script can replace the files behind us."""
    from . import jobs

    jobs.shutdown_all()
    clear_running()
    os._exit(0)


def _write_script(stage: Path, ready: Path, port: int) -> Path:
    """The .cmd that waits, copies and restarts.

    Every path is baked in rather than passed as an argument: the install
    folder can contain spaces, and a `set "X=..."` line has no quoting left to
    get wrong (UPDATE_BUTTON.md section 5).
    """
    excludes = " ".join('"%s"' % name for name in version.PRESERVE)
    script = stage / "apply-update.cmd"
    body = f"""@echo off
title Updating {version.APP_NAME}
setlocal
set "READY={ready}"
set "TARGET={config.ROOT}"
set "MARK={RUNNING_MARKER}"
set "LOG={config.ROOT / 'update.log'}"
set "STAGE={stage}"

echo Updating to a new version... >"%LOG%"

rem Wait for the app to let go. `if exist` needs no pipe, so there is no
rem find.exe to hang the way UPDATE_BUTTON.md section 4.2 describes.
set /a TRIES=0
:waitloop
if not exist "%MARK%" goto copy
set /a TRIES+=1
if %TRIES% GEQ 60 goto copy
ping -n 2 127.0.0.1 >nul
goto waitloop

:copy
robocopy "%READY%" "%TARGET%" /E /R:3 /W:2 /XF {excludes} /NFL /NDL /NJH /NJS /NP >>"%LOG%" 2>&1
rem Robocopy exit codes 0-7 are all success; only 8 and above are failures.
if errorlevel 8 (
  echo update failed >>"%LOG%"
) else (
  echo update applied >>"%LOG%"
)

start "" "%TARGET%\\run.bat"

rem The script cannot delete itself; the sweep in updater.py collects it.
cd /d "%TEMP%"
rd /s /q "%STAGE%\\unpacked" 2>nul
del /q "%STAGE%\\update.zip" 2>nul
"""
    script.write_text(body, encoding="utf-8")
    return script
