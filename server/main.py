"""FastAPI app: static frontend, streaming upload, SSE progress, cancellation.

Bound to 127.0.0.1 only, never 0.0.0.0 (CLAUDE.md section 16).
"""

from __future__ import annotations

import asyncio
import atexit
import json
import re
import shutil
import signal
import threading
import uuid
from contextlib import asynccontextmanager
from pathlib import Path
from urllib.parse import unquote

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import (
    FileResponse,
    HTMLResponse,
    JSONResponse,
    PlainTextResponse,
    StreamingResponse,
)

from . import config, hardware, jobs, merge, pipeline, speakers


@asynccontextmanager
async def lifespan(_app: FastAPI):
    config.TEMP.mkdir(parents=True, exist_ok=True)
    config.OUTPUT.mkdir(parents=True, exist_ok=True)
    # Sweep temp in case a previous run died badly (CLAUDE.md section 14).
    jobs.clean_temp(keep_log=False)
    # Outputs produced before per-meeting folders existed would otherwise
    # disappear from the history list, which reads as data loss.
    try:
        moved = config.migrate_flat_output()
        if moved:
            jobs._append_job_log(
                "output: moved %d file(s) into per-meeting folders" % moved)
    except OSError as exc:
        jobs.log_exception(exc)
    jobs.install_kill_on_close()
    try:
        yield
    finally:
        jobs.shutdown_all()


app = FastAPI(
    title="Meeting Summariser", docs_url=None, redoc_url=None, lifespan=lifespan
)

ACCEPTED_EXTENSIONS = {".mp3", ".wav", ".m4a", ".mp4", ".mkv"}
DISK_HEADROOM_MULTIPLIER = 3        # upload + WAV + headroom (section 5)


# ---------------------------------------------------------------------------
# static frontend
# ---------------------------------------------------------------------------

@app.get("/", response_class=HTMLResponse)
async def index() -> HTMLResponse:
    return HTMLResponse((config.WEB / "index.html").read_text(encoding="utf-8"))


@app.get("/app.js")
async def app_js() -> FileResponse:
    return FileResponse(
        config.WEB / "app.js",
        # charset matters: app.js contains literal play/stop glyphs, and
        # without it some browsers decode the file as latin-1.
        media_type="application/javascript; charset=utf-8",
    )


@app.get("/style.css")
async def style_css() -> FileResponse:
    return FileResponse(config.WEB / "style.css", media_type="text/css; charset=utf-8")


@app.get("/api/models")
async def model_choices() -> dict:
    """The model dropdown: what is installed, and what this card should use."""
    return hardware.describe(hardware.detect_vram_mb())


@app.get("/api/current")
async def current_job() -> dict:
    """The job in flight, if any.

    Closing the tab must not orphan a two-hour run. The page asks this on load
    and reattaches to the live progress stream, which is also the only way to
    reach the Cancel button again after a reload.
    """
    job = jobs.active()
    if job is None:
        return {"job": None}
    return {
        "job": {
            "id": job.id,
            "filename": job.filename,
            "mode": job.mode,
            "kind": job.kind,
            "meeting": job.meeting_name,
            "percent": round(job.percent, 2),
            "stage_label": jobs.STAGE_LABELS.get(job.stage, job.stage),
        }
    }


@app.get("/api/history")
async def history() -> dict:
    """Everything already processed, newest first.

    Derived from `output\\` rather than from a database: the files are the
    record, they survive a restart, and a meeting deleted from the folder
    should disappear from the list without further ceremony.
    """
    items = []
    for path in config.OUTPUT.glob("*/*_transcript.md"):
        name = path.name[: -len("_transcript.md")]
        data = speakers.load(name) or {}
        items.append({
            "meeting": name,
            "modified": path.stat().st_mtime,
            "duration_s": data.get("duration_s", 0),
            "documents": [
                mode for mode in ("summary", "minutes")
                if (config.meeting_dir(name) / ("%s_%s.md" % (name, mode))).exists()
            ],
            "tagged": (config.meeting_dir(name) / ("%s_transcript_tagged.md" % name)).exists(),
            "named": bool(data.get("names"))
                     or (config.meeting_dir(name) / ("%s_transcript_tagged.md" % name)).exists(),
            "can_rebuild": bool(data.get("notes")),
        })
    items.sort(key=lambda i: -i["modified"])
    return {"meetings": items}


@app.post("/api/history/open")
async def history_open(request: Request):
    """Reopen a past meeting so it can be renamed or have documents added."""
    body = await request.json()
    name = Path(str(body.get("meeting") or "")).name
    transcript = config.meeting_dir(name) / ("%s_transcript.md" % name)
    if not name or not transcript.exists():
        return JSONResponse({"error": "That meeting is no longer in the output folder."},
                            status_code=404)
    if jobs.active() is not None:
        return JSONResponse({"error": "Another recording is still being processed."},
                            status_code=409)

    data = speakers.load(name) or {}
    # Reuse the handle if this meeting is already open, rather than growing the
    # registry by one every time somebody clicks down the list.
    job = next((j for j in jobs.all_jobs()
                if j.meeting_name == name and j.state == "done"), None)
    if job is None:
        job = jobs.create(uuid.uuid4().hex[:12])
    job.state = "done"
    job.filename = name
    job.meeting_name = name
    job.mode = data.get("mode", "summary")
    job.duration_s = float(data.get("duration_s") or 0)
    job.transcript_path = transcript
    tagged = config.meeting_dir(name) / ("%s_transcript_tagged.md" % name)
    job.tagged_transcript_path = tagged if tagged.exists() else None
    job.documents = [
        config.meeting_dir(name) / ("%s_%s.md" % (name, mode))
        for mode in ("summary", "minutes")
        if (config.meeting_dir(name) / ("%s_%s.md" % (name, mode))).exists()
    ]
    return {"job_id": job.id, "meeting": name}


@app.get("/api/health")
async def health() -> dict:
    missing = [p.name for p in config.missing_files()]
    return {"ok": not missing, "missing": missing}


# ---------------------------------------------------------------------------
# upload
# ---------------------------------------------------------------------------

@app.post("/api/upload")
async def upload(request: Request):
    """Stream the request body straight to disk as it arrives.

    Never read the body into memory: an 8-hour MP4 can be several hundred
    megabytes and awaiting it whole would spike RAM or fail outright. Section 5
    asks for 1MB blocks; `request.stream()` hands us whatever size the transport
    delivers (typically 64KB) and each is written immediately, which meets the
    intent -- peak memory is one chunk, not one file. Multipart is deliberately
    avoided: it would buffer the upload a second time.
    """
    filename = unquote(request.query_params.get("filename", "recording"))
    suffix = Path(filename).suffix.lower()
    if suffix not in ACCEPTED_EXTENSIONS:
        return JSONResponse(
            {"error": "That file type isn't supported. Use MP3, WAV, M4A, MP4 or MKV."},
            status_code=400,
        )

    declared = int(request.headers.get("content-length") or 0)
    if declared:
        need = declared * DISK_HEADROOM_MULTIPLIER
        free = shutil.disk_usage(config.ROOT).free
        if free < need:
            return JSONResponse(
                {
                    "error": "Not enough free disk space. This recording needs about "
                    "%d GB free and there is %d GB available."
                    % (round(need / 1e9) or 1, round(free / 1e9)),
                },
                status_code=400,
            )

    if jobs.active() is not None:
        return JSONResponse(
            {"error": "Another recording is still being processed."}, status_code=409
        )

    jobs.clean_temp()
    jobs.reset_job_log()

    job_id = uuid.uuid4().hex[:12]
    job = jobs.create(job_id)
    job.filename = Path(filename).name

    config.TEMP.mkdir(parents=True, exist_ok=True)
    dest = config.TEMP / ("upload" + suffix)
    written = 0
    try:
        with open(dest, "wb") as fh:
            async for chunk in request.stream():
                if not chunk:
                    continue
                fh.write(chunk)
                written += len(chunk)
    except Exception as exc:  # noqa: BLE001
        jobs.log_exception(exc)
        try:
            dest.unlink()
        except OSError:
            pass
        return JSONResponse({"error": "The upload didn't complete."}, status_code=500)

    if written == 0:
        return JSONResponse({"error": "That file is empty."}, status_code=400)

    job.upload_path = dest
    job.size_bytes = written
    job.state = "uploaded"

    # Read the duration now so the UI can set expectations before the user
    # commits to a run (CLAUDE.md section 13).
    from . import audio

    duration = await asyncio.to_thread(audio.probe_duration, dest)
    if duration <= 0:
        try:
            dest.unlink()
        except OSError:
            pass
        return JSONResponse({"error": audio.BAD_AUDIO}, status_code=400)
    job.duration_s = duration

    return {
        "job_id": job_id,
        "filename": job.filename,
        "size_bytes": written,
        "duration_s": duration,
        "duration_hms": merge.hms(duration),
        "estimates": {
            key: {
                mode: estimate_minutes(duration, mode, key)
                for mode in ("summary", "minutes", "both")
            }
            for key in hardware.BY_KEY
        },
    }


def estimate_minutes(duration_s: float, mode: str = "summary",
                     model_key: str = ""):
    """Predicted minutes, measured on this machine wherever possible.

    Section 13 proposes a fixed 9 minutes per audio hour. That is wrong by
    roughly four times here, and more importantly it is wrong *differently* on
    different hardware: the figure is dominated by generation speed, which is
    set by how much of the model `cpu_ffn_regex` pushes onto the CPU. A
    constant cannot be right for both this 10GB card and a 17.9GB one.

    So: measure. Every completed run records what each stage cost and what each
    LLM call produced, and the estimate is built from those. Before the first
    run there is nothing to build from, and this returns None -- the UI says
    "first run" rather than printing a number that will be several times out.
    """
    from . import calibration

    seconds, measured = calibration.estimate_seconds(duration_s, mode, model_key)
    if not measured:
        # Nothing measured on this machine yet, and the LLM stage varies by
        # several times across cards. None tells the UI to say so rather than
        # print a number that will be wrong.
        return None
    return max(1, int(round(seconds / 60.0)))


# ---------------------------------------------------------------------------
# job control
# ---------------------------------------------------------------------------

@app.post("/api/jobs/{job_id}/start")
async def start(job_id: str, request: Request):
    job = jobs.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="No such job")
    if job.state == "running":
        return {"ok": True}
    if job.state != "uploaded" or not job.upload_path or not job.upload_path.exists():
        return JSONResponse({"error": "Upload that recording again."}, status_code=400)

    body = {}
    try:
        body = await request.json()
    except Exception:  # noqa: BLE001
        pass
    mode = str(body.get("mode", "summary"))
    job.mode = mode if mode in {"summary", "minutes", "both"} else "summary"

    # Optional participant count. Blank/0 means auto-detect and prune; a number
    # is passed straight to the clusterer, which is far more reliable than any
    # distance threshold on a long recording (DIARIZATION_FIX.md part D4).
    try:
        n = int(body.get("num_speakers") or 0)
    except (TypeError, ValueError):
        n = 0
    job.num_speakers = n if 1 <= n <= 50 else 0

    # Model choice. Detection picks the default; an explicit choice from the
    # dropdown overrides it, because the user may know something we do not.
    requested = str(body.get("model") or "")
    job.model_key = requested if requested in hardware.BY_KEY else ""

    try:
        cfg = config.load_config()
    except RuntimeError as exc:
        return JSONResponse({"error": str(exc)}, status_code=500)

    job._loop = asyncio.get_running_loop()
    job.state = "running"
    threading.Thread(
        target=pipeline.run, args=(job, cfg), name="pipeline-%s" % job_id, daemon=True
    ).start()
    return {"ok": True, "mode": job.mode}


@app.post("/api/jobs/{job_id}/cancel")
async def cancel(job_id: str):
    job = jobs.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="No such job")
    job.cancel()
    return {"ok": True}


@app.get("/api/events/{job_id}")
async def events(job_id: str):
    job = jobs.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="No such job")

    queue: asyncio.Queue = asyncio.Queue()
    job._queues.append(queue)
    job._loop = asyncio.get_running_loop()

    async def stream():
        # Replay current state so a reconnecting client is never blank.
        yield _sse({"type": "status", **{k: v for k, v in job._status().items() if k != "type"}})
        for line in job.log_lines[-200:]:
            yield _sse({"type": "log", "line": line})
        try:
            while True:
                try:
                    event = await asyncio.wait_for(queue.get(), timeout=15.0)
                except asyncio.TimeoutError:
                    yield ": keepalive\n\n"
                    continue
                yield _sse(event)
                if event.get("type") in {"done", "error", "cancelled"}:
                    break
        finally:
            try:
                job._queues.remove(queue)
            except ValueError:
                pass

    return StreamingResponse(
        stream(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


def _sse(event: dict) -> str:
    return "data: %s\n\n" % json.dumps(event)


# ---------------------------------------------------------------------------
# results
# ---------------------------------------------------------------------------

@app.get("/api/jobs/{job_id}/result")
async def result(job_id: str):
    job = jobs.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="No such job")
    out = {}
    for path in list(job.documents or []):
        if path and path.exists():
            # key by mode so the UI can label the tabs
            key = "minutes" if path.stem.endswith("_minutes") else "summary"
            out[key] = {
                "name": path.name,
                "markdown": path.read_text(encoding="utf-8", errors="replace"),
            }
    data = speakers.load(job.meeting_name) or {}
    out["_can_rebuild"] = bool(data.get("notes"))
    # What regenerating a document actually costs here, so the finished page
    # can say so instead of promising "a few minutes" on a five-hour meeting.
    from . import calibration

    key = job.model_key or hardware.choose_key(hardware.detect_vram_mb())
    if calibration.has_llm_profile(key, stages=("reduce",)):
        seconds_of_audio = job.duration_s or float(data.get("duration_s") or 0)
        tokens, tok_s = calibration.call_profile(
            "reduce", key, seconds_of_audio / 3600.0)
        out["_rebuild_minutes"] = max(1, int(round(tokens / max(tok_s, 0.05) / 60.0)))
    else:
        out["_rebuild_minutes"] = 0
    for key, path in (("transcript", job.transcript_path),
                      ("tagged", job.tagged_transcript_path)):
        if path and path.exists():
            out[key] = {
                "name": path.name,
                "markdown": path.read_text(encoding="utf-8", errors="replace"),
            }
    return out


@app.get("/api/jobs/{job_id}/speakers")
async def speaker_list(job_id: str):
    """Speakers with their three longest utterances, for the naming panel."""
    job = jobs.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="No such job")
    data = speakers.load(job.meeting_name)
    if not data:
        return {"meeting": job.meeting_name, "speakers": []}
    # Pre-fill from whatever was applied last time, so reopening a meeting to
    # correct one name does not mean retyping the other nine.
    if not data.get("names"):
        data["names"] = speakers.names_from_tagged(job.meeting_name)
    return {"meeting": job.meeting_name, "speakers": speakers.speaker_rows(data)}


@app.get("/api/samples/{meeting}/{filename}")
async def speaker_sample(meeting: str, filename: str):
    """Serve one voice clip.

    Both path parts are resolved and checked against the samples folder before
    anything is opened: they arrive from the URL, and a name like `..\\..\\`
    would otherwise read anywhere on the disk.
    """
    if not re.fullmatch(r"spk\d{2}_\d\.mp3", filename):
        raise HTTPException(status_code=404, detail="No such sample")
    base = speakers.folder_for(Path(unquote(meeting)).name).resolve()
    path = (base / filename).resolve()
    if base != path.parent or not path.exists():
        raise HTTPException(status_code=404, detail="No such sample")
    return FileResponse(path, media_type="audio/mpeg")


@app.post("/api/jobs/{job_id}/names")
async def apply_speaker_names(job_id: str, request: Request):
    """Apply real names, then rebuild the documents from the cached notes.

    Rewriting the transcript is trivial; the documents are regenerated so the
    names appear in the prose and, in minutes mode, against the action items --
    which is the whole point. Regeneration re-runs only the final reduce.
    """
    job = jobs.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="No such job")
    if jobs.active() is not None:
        return JSONResponse(
            {"error": "Another recording is still being processed."}, status_code=409
        )

    body = await request.json()
    names = {str(k): str(v).strip() for k, v in (body.get("names") or {}).items()}
    names = {k: v for k, v in names.items() if v}
    if not names:
        return JSONResponse({"error": "No names were entered."}, status_code=400)

    data = speakers.load(job.meeting_name)
    if not data:
        return JSONResponse(
            {"error": "The speaker information for this meeting is no longer available."},
            status_code=400,
        )

    try:
        cfg = config.load_config()
    except RuntimeError as exc:
        return JSONResponse({"error": str(exc)}, status_code=500)

    # Pure search and replace: milliseconds, no model. Re-running the reduce so
    # the model can re-reason about the names is a separate, explicit action
    # (/generate) -- naming a speaker should never cost minutes.
    try:
        tagged = speakers.tagged_transcript(job.meeting_name, names)
        if tagged:
            job.tagged_transcript_path = tagged
        docs = speakers.rebuild_documents(job.meeting_name, data, names)
        if docs:
            job.documents = docs
        speakers.save(job.meeting_name, None, None, job.mode,
                      job.duration_s, names=names)
    except OSError as exc:
        jobs.log_exception(exc)
        return JSONResponse(
            {"error": "The renamed files could not be written."}, status_code=500
        )
    return {"ok": True, "instant": True, "documents": [p.name for p in job.documents]}


@app.post("/api/jobs/{job_id}/generate")
async def generate_document(job_id: str, request: Request):
    """Produce a document the run did not make, or redo one with the model.

    Runs the final reduce over the cached notes -- one call, seconds to minutes
    rather than the whole pipeline. This is how you get minutes after a
    summary-only run without transcribing the recording again.
    """
    job = jobs.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="No such job")
    if jobs.active() is not None:
        return JSONResponse(
            {"error": "Another recording is still being processed."}, status_code=409
        )

    body = await request.json()
    mode = str(body.get("mode") or "")
    if mode not in ("summary", "minutes"):
        return JSONResponse({"error": "Unknown document type."}, status_code=400)

    data = speakers.load(job.meeting_name)
    if not data or not data.get("notes"):
        return JSONResponse(
            {"error": "The notes for this meeting are no longer available, so it "
                      "cannot be rebuilt. Process the recording again."},
            status_code=400,
        )

    try:
        cfg = config.load_config()
    except RuntimeError as exc:
        return JSONResponse({"error": str(exc)}, status_code=500)

    job._loop = asyncio.get_running_loop()
    job.state = "running"
    job.cancel_event.clear()
    threading.Thread(
        target=pipeline.generate_one,
        args=(job, cfg, mode, data),
        name="generate-%s" % job_id,
        daemon=True,
    ).start()
    return {"ok": True}


@app.get("/api/diagnostics")
async def diagnostics() -> PlainTextResponse:
    return PlainTextResponse(jobs.read_job_log())


@app.post("/api/open-output")
async def open_output(request: Request):
    """Open a folder in Explorer -- this meeting's, or the output root."""
    import subprocess

    body = {}
    try:
        body = await request.json()
    except Exception:  # noqa: BLE001 - no body is a request for the root
        pass
    target = config.OUTPUT
    name = Path(str(body.get("meeting") or "")).name
    if name and config.meeting_dir(name).is_dir():
        target = config.meeting_dir(name)
    target.mkdir(parents=True, exist_ok=True)
    try:
        subprocess.Popen(["explorer", str(target)])
    except OSError:
        return JSONResponse({"error": "Couldn't open the folder."}, status_code=500)
    return {"ok": True}


# ---------------------------------------------------------------------------
# lifecycle
# ---------------------------------------------------------------------------

def _emergency_cleanup(*_args) -> None:
    jobs.shutdown_all()


atexit.register(_emergency_cleanup)

# SIGINT is deliberately NOT handled here: uvicorn installs its own handler and
# uses it to run a clean shutdown (which calls jobs.shutdown_all via lifespan).
# Overriding it would leave Ctrl+C unable to stop the server.
for _sig in ("SIGTERM", "SIGBREAK"):
    if hasattr(signal, _sig):
        try:
            signal.signal(getattr(signal, _sig), _emergency_cleanup)
        except (ValueError, OSError):
            pass
