"""Check GitHub Releases for a newer build, download it, and swap it in.

This is the **one** place the app touches the network, and only when somebody
presses a button. CLAUDE.md section 0 forbids internet access at runtime; that
rule is about the pipeline, which must work with the adapter disabled, and it
still does. An explicit "check for updates" is a deliberate, documented
exception -- nothing here runs on startup, on a timer, or in the background.

The workflow is BUILD_NOTES.md section 9q, with two simplifications that fall
out of this being a Python app in a plain folder rather than a packaged exe:

- Download, unpack and verify all happen **in Python, before anything is
  replaced**. The batch script only waits, copies and restarts, so there is far
  less to get wrong in cmd.
- Nothing is overwritten until the staged copy has been verified. Every failure
  path leaves the installed app exactly as it was.

The five traps in BUILD_NOTES.md section 9q are respected where they apply:
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
    """Check before the download, not after (BUILD_NOTES.md section 9q)."""
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


# Models a newer version needs that an older install cannot have.
#
# The update payload is the source zip and must stay small -- it is 230 KB, and
# the whole point of not shipping the full bundle is that an update should not
# be a gigabyte (BUILD_NOTES section 9q). But a release that adds a model then
# leaves an updated install silently missing it: the app degrades quietly to
# whisper's own word timings and says so only in temp\job.log, where nobody is
# looking. Telling the user to go and run DOWNLOAD_MODELS.bat is not an answer
# either -- they updated from a button and have no reason to suspect a second
# step exists.
#
# So the updater fetches them itself, after the payload and before the restart.
# Each is optional by construction: the app runs without it, so a failure here
# warns and carries on rather than abandoning an update that is otherwise fine.
EXTRA_MODELS = (
    {
        "path": "models/wav2vec2-align.onnx",
        # What the user loses without it, named once however many files it
        # takes. Two files are one capability, and the notice should say so.
        "component": "Word alignment",
        "reason": "speaker attribution and summary quality are noticeably worse",
        "min_size": 350_000_000,
        "size_hint": 377_811_056,
        "label": "word alignment model",
        "url": "https://github.com/%s/releases/download/"
               "align-wav2vec2-base-960h/wav2vec2-align.onnx" % version.GITHUB_REPO,
    },
    {
        "path": "models/wav2vec2-align.json",
        "component": "Word alignment",
        "reason": "speaker attribution and summary quality are noticeably worse",
        "min_size": 200,
        "size_hint": 277,
        "label": "word alignment labels",
        "url": "https://github.com/%s/releases/download/"
               "align-wav2vec2-base-960h/wav2vec2-align.json" % version.GITHUB_REPO,
    },
)


def offered_language_models() -> list:
    r"""Every shipped language model this install does not have.

    **Both**, not just the one this machine would pick. The folder is portable
    by design (CLAUDE.md section 16: copy it to another drive and it works), so
    the machine that downloads is often not the machine that runs -- somebody
    fetches it on a laptop and copies it to an on-prem box with a far better
    card. Offering only what the *downloading* machine needs quietly strips the
    folder of the model the destination wanted, and that is why
    DOWNLOAD_MODELS.bat fetches both rather than choosing.

    Never in EXTRA_MODELS. Those are fetched automatically during an update,
    which is right for 360 MB and completely wrong for 25 GB: an update must
    never silently become a download that size. These are only ever offered,
    with the size stated, behind a button.

    The candidates are the shipped models -- NOT hardware.choose_key(), which
    answers "what will this run?" from what is on disk, so an install carrying
    the legacy Q4_K_M is told nothing is missing and never hears about the
    model that is 3.4x faster (BUILD_NOTES 9z).
    """
    from . import hardware

    out = []
    try:
        for model in hardware.MODELS:
            if not model.get("url"):
                continue
            path = hardware.model_path(model["key"])
            try:
                if path.exists() and path.stat().st_size >= model.get("min_size", 0):
                    continue
            except OSError:
                pass
            out.append({
                "path": str(path.relative_to(config.ROOT)).replace("\\", "/"),
                "component": model["label"].split(":")[0].strip() + " language model",
                "label": model["label"],
                "reason": "the app falls back to whatever model it already has, "
                          "which may be slower or lower quality",
                "url": model["url"],
                "min_size": model.get("min_size", 0),
                "size_hint": int(model.get("size_gb", 0) * 1_000_000_000),
            })
    except Exception:  # noqa: BLE001 - never break the page over this
        return []
    return out


def missing_models() -> list:
    """Which EXTRA_MODELS this install does not already have.

    Size is checked as well as existence: a half-finished download from a
    previous attempt is worse than nothing, because the app would load it.
    """
    out = []
    for m in EXTRA_MODELS:
        target = config.ROOT / m["path"]
        try:
            if target.exists() and target.stat().st_size >= m["min_size"]:
                continue
        except OSError:
            pass
        out.append(m)
    return out


def fetch_models(items: list) -> list:
    r"""Download each into models\, via .partial so a failure leaves nothing.

    Returns the ones that failed. Never raises for a download problem: the
    update itself has already succeeded by this point.
    """
    failed = []
    for n, m in enumerate(items, 1):
        target = config.ROOT / m["path"]
        partial = target.with_suffix(target.suffix + ".partial")
        # Count components, not files. Word alignment is two files and one
        # thing the user is waiting for; "1 of 3" against a notice that named
        # two components invites the question of what the third one is.
        groups = []
        for x in items:
            name = x.get("component") or x["label"]
            if name not in groups:
                groups.append(name)
        this = (m.get("component") or m["label"])
        _set(phase="downloading",
             message=("Downloading %s (%d of %d)"
                      % (this, groups.index(this) + 1, len(groups))
                      if len(groups) > 1 else "Downloading %s" % this),
             downloaded=0, total=m.get("size_hint") or 0)
        try:
            target.parent.mkdir(parents=True, exist_ok=True)
            _download(m["url"], partial, m.get("size_hint") or 0)
            if partial.stat().st_size < m["min_size"]:
                raise RuntimeError("file is smaller than expected")
            partial.replace(target)
        except Exception as exc:  # noqa: BLE001 - an optional model
            if _cancel.is_set():
                raise
            failed.append(m)
            try:
                partial.unlink()
            except OSError:
                pass
            from . import jobs
            jobs.log_exception(exc)
    return failed


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

        # Anything this version needs that the running one never had. Done here
        # rather than after the restart: the app is still up, the user is
        # watching a progress bar, and a model that arrives now is in place the
        # first time they process a recording.
        missing = missing_models()
        if missing:
            failed = fetch_models(missing)
            if failed:
                _set(extra_failed=", ".join(m["label"] for m in failed))

        _set(phase="applying",
             message="Restarting to finish the update")
        script = _write_script(stage, ready, port)
        # cmd.exe is the executable and the script is its own argument: never
        # the script as the executable, never shell=True with an interpolated
        # path (BUILD_NOTES.md section 9q, trap 1).
        # CREATE_NO_WINDOW, not DETACHED_PROCESS. Both keep the script alive
        # after this process exits, but a detached cmd.exe allocates a console
        # of its own -- so the update showed a second black window running the
        # wait loop, beside the new app's window. CREATE_NO_WINDOW gives it a
        # console with no window at all. The two flags are mutually exclusive;
        # fall back if the constant is somehow unavailable, because a script
        # that dies with its parent cannot copy the files.
        flags = (CREATE_NO_WINDOW or DETACHED_PROCESS) | subprocess.CREATE_NEW_PROCESS_GROUP
        subprocess.Popen(
            [os.environ.get("ComSpec") or "cmd.exe", "/c", str(script)],
            creationflags=flags,
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


UPDATING_MARKER = config.TEMP / "updating.flag"


def _quit() -> None:
    """Stop the server so the script can replace the files behind us.

    Leaves a marker first. run.bat runs uvicorn in the foreground and pauses
    when it returns, which is right when the app has crashed and wrong here:
    the user watched a new window appear while the old one sat behind it
    saying "Meeting Summariser has stopped", which reads as a failure. The
    marker tells run.bat this exit was intentional so it closes quietly.
    """
    from . import jobs

    try:
        UPDATING_MARKER.parent.mkdir(parents=True, exist_ok=True)
        UPDATING_MARKER.write_text("1", encoding="utf-8")
    except OSError:
        pass
    jobs.shutdown_all()
    clear_running()
    os._exit(0)


def _write_script(stage: Path, ready: Path, port: int) -> Path:
    """The .cmd that waits, copies and restarts.

    Every path is baked in rather than passed as an argument: the install
    folder can contain spaces, and a `set "X=..."` line has no quoting left to
    get wrong (BUILD_NOTES.md section 9q).
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
rem find.exe to hang the way BUILD_NOTES.md section 9q, trap 2 describes.
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
