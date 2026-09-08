"""Stage orchestration.

convert -> transcribe + diarize -> merge -> chunk -> map -> group reduce ->
final reduce. Stage weights come from section 13 and sum to 100.
"""

from __future__ import annotations

import concurrent.futures
import re
import time
from pathlib import Path

from . import (audio, calibration, chunker, config, diarize, jobs, llm, merge,
               reduce, speakers, transcribe)
from .jobs import Cancelled, Job, JobError


def _safe_name(filename: str) -> str:
    """Meeting name from the uploaded filename (no UI field for it)."""
    stem = Path(filename).stem.strip() or "meeting"
    stem = re.sub(r'[<>:"/\\|?*\x00-\x1f]', "_", stem)
    return stem[:100] or "meeting"


def run(job: Job, cfg: dict) -> None:
    """Run the whole pipeline. Called on a worker thread."""
    job.started_at = time.time()
    job.state = "running"
    job.kind = "run"
    job.push_status()

    meeting_name = _safe_name(job.filename)
    wav = config.TEMP / "audio.wav"
    timings: dict[str, float] = {}

    try:
        # -- Stage 1: convert -------------------------------------------
        job.set_stage("convert")
        t0 = time.time()
        duration = audio.convert(job, job.upload_path, wav)
        timings["convert"] = job.stage_seconds["convert"] = time.time() - t0
        job.duration_s = duration
        job.log(
            "audio: %s, %.1f MB WAV"
            % (merge.hms(duration), wav.stat().st_size / 1e6)
        )

        # The upload is no longer needed once the WAV exists; for an 8-hour
        # MP4 that reclaims hundreds of megabytes mid-job.
        try:
            if job.upload_path and job.upload_path.exists():
                job.upload_path.unlink()
        except OSError:
            pass

        # -- Stages 2 and 3: transcribe (GPU) + diarize (CPU) -----------
        job.set_stage("transcribe")
        t0 = time.time()
        words, turns = _transcribe_and_diarize(job, cfg, wav)
        timings["transcribe+diarize"] = job.stage_seconds["transcribe"] = time.time() - t0

        # -- Stage 4: merge ---------------------------------------------
        job.set_stage("merge")
        t0 = time.time()
        job.check_cancelled()
        speaker_turns = merge.assign_speakers(words, turns)
        attributed = bool(turns)
        transcript_path = config.meeting_dir(meeting_name) / ("%s_transcript.md" % meeting_name)
        merge.write_transcript(
            transcript_path,
            speaker_turns,
            attributed,
            meeting_name,
            job.duration_s,
            job.attribution_note,
        )
        # Extract voice samples now: the WAV is deleted when the job ends, and
        # hearing a speaker is the fastest way to work out who they are.
        job.samples = speakers.build_samples(job, wav, speaker_turns, meeting_name)
        job.meeting_name = meeting_name
        timings["merge"] = job.stage_seconds["merge"] = time.time() - t0
        job.set_progress("merge", 1.0)
        job.log(
            "merge: %d turns written to %s"
            % (len(speaker_turns), transcript_path.name)
        )

        # -- Stages 5 and 6: chunk, map, reduce -------------------------
        t0 = time.time()
        documents = _produce_document(job, cfg, speaker_turns, attributed, meeting_name)
        timings["chunk+llm"] = time.time() - t0

        for stage, seconds in timings.items():
            job.log("timing: %s took %.1fs" % (stage, seconds))
        if job.duration_s > 0:
            total = sum(timings.values())
            job.log(
                "timing: %.1fs total = %.2fx realtime"
                % (total, job.duration_s / total if total else 0)
            )

        # Feed this run back into the machine's calibration so the next
        # estimate and progress bar are measured rather than assumed.
        try:
            calibration.record(job.stage_seconds, job.duration_s, job.mode,
                               job.model_key)
        except Exception as exc:  # noqa: BLE001 - never fail a finished job
            jobs.log_exception(exc)

        job.finish(documents, transcript_path)

    except Cancelled:
        job.mark_cancelled()
        job.log("cancelled by user")
    except JobError as exc:
        jobs.log_exception(exc)
        job.fail(exc.message, exc.detail)
        job.log("failed: %s" % exc.message)
    except Exception as exc:  # noqa: BLE001
        jobs.log_exception(exc)
        job.fail(
            "Something went wrong while processing that recording.",
            str(exc),
        )
        job.log("failed: %s" % exc)
    finally:
        job.kill_processes()
        jobs.clean_temp()


def generate_one(job: Job, cfg: dict, mode: str, data: dict) -> None:
    """Produce one document from the cached notes. Worker thread.

    Used for "give me minutes too" after a summary-only run, and for
    regenerating a document once speakers have real names so the model can
    reason about people rather than labels. Either way this is one final-reduce
    call over notes that already exist -- not the 68-minute map loop, and not
    the recording again.
    """
    job.started_at = time.time()
    job.kind = "generate"
    job.stage_seconds = {}
    job.document_error = ""
    # This job is one reduce call and nothing else, so give that stage the
    # whole bar. With the run weights it would open at 49% -- the width of
    # everything it is skipping -- and look stuck there.
    job.weights = {stage: 0.0 for stage in jobs.STAGE_WEIGHTS}
    job.weights["reduce"] = 100.0
    job.push_status()

    meeting_name = job.meeting_name
    names = data.get("names") or {}
    server = None
    try:
        job.set_stage("reduce", "Loading language model (up to 2 minutes on first run)")
        # One call, declared now so the ETA is populated during the model load
        # rather than appearing out of nowhere when generation starts.
        job.pending_calls = {"reduce": 1}
        notes = [speakers.apply_names(n, names) for n in (data.get("notes") or [])]

        server = llm.LlamaServer(job, cfg)
        server.start()
        job.llm_server = server
        document = reduce.run_final_reduce(
            job, server, notes, cfg, mode, meeting_name, job.duration_s
        )

        # Store the model's own text before names are applied, so a later
        # rename rewrites from it rather than from an already-renamed file.
        speakers.save(meeting_name, None, None, job.mode, job.duration_s,
                      documents={mode: document})
        path = config.meeting_dir(meeting_name) / ("%s_%s.md" % (meeting_name, mode))
        config.write_atomic(
            path, speakers.apply_names(document, names).rstrip() + "\n")
        job.log("generate: wrote %s" % path.name)

        docs = [p for p in (job.documents or []) if p.name != path.name]
        docs.append(path)
        job.finish(sorted(docs, key=lambda p: p.name), job.transcript_path)

    except Cancelled:
        job.mark_cancelled()
        job.log("cancelled by user")
    except JobError as exc:
        jobs.log_exception(exc)
        job.fail(exc.message, exc.detail)
    except Exception as exc:  # noqa: BLE001
        jobs.log_exception(exc)
        job.fail("That document could not be produced.", str(exc))
    finally:
        job.llm_server = None
        if server is not None:
            server.stop()
        job.kill_processes()


def _produce_document(job: Job, cfg: dict, turns, attributed: bool, meeting_name: str):
    """Chunk, map, reduce. Returns the written documents (possibly empty).

    Whisper has exited by now, so the two never hold VRAM at the same time
    (section 11.1), and the server is shut down as soon as the document is
    written rather than sitting on 14GB.
    """
    if not turns:
        return []
    job.set_stage("map", "Loading language model (up to 2 minutes on first run)")
    server = None
    try:
        server = llm.LlamaServer(job, cfg)
        server.start()
        job.llm_server = server
        chunks = chunker.build_chunks(job, server, turns, attributed, cfg)
        if not chunks:
            return []
        return reduce.produce_document(job, server, chunks, cfg, meeting_name)
    except Cancelled:
        raise
    except JobError as exc:
        # The transcript is already written and is a deliverable in its own
        # right, so a failure here must not throw away the whole job.
        jobs.log_exception(exc)
        job.log("llm: %s - transcript kept, document not produced" % exc.message)
        job.document_error = exc.message
        return []
    finally:
        job.llm_server = None
        if server is not None:
            server.stop()


def _transcribe_and_diarize(job: Job, cfg: dict, wav: Path):
    """Run stages 2 and 3, concurrently by default (CLAUDE.md section 7).

    Transcription is GPU-bound, diarization is CPU-only, and stage 3 does not
    depend on stage 2 -- only the merge needs both. Failures are independent:
    diarization dying degrades the output to unattributed, transcription dying
    fails the job because there is nothing to merge.
    """
    concurrent_ok = bool(cfg["pipeline"].get("concurrent_diarization", True))
    diar_enabled = bool(cfg["diarization"].get("enabled", True))

    if not diar_enabled:
        words = transcribe.run(job, cfg, wav)
        return words, []

    if not concurrent_ok:
        job.log("pipeline: sequential transcribe then diarize")
        words = transcribe.run(job, cfg, wav)
        job.check_cancelled()
        turns = diarize.run(job, cfg, wav)
        return words, turns

    job.log("pipeline: transcribe (GPU) and diarize (CPU) concurrently")
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
        diar_future = pool.submit(diarize.run, job, cfg, wav)
        try:
            words = transcribe.run(job, cfg, wav)
        except BaseException:
            # Transcription failed or was cancelled: stop diarization too
            # rather than leaving it burning CPU for output nothing will read.
            # Killing the child is what actually stops it -- a flag alone would
            # leave the worker running until it finished on its own.
            job.kill_processes()
            concurrent.futures.wait([diar_future], timeout=60)
            raise

        try:
            turns = diar_future.result(timeout=None)
        except Cancelled:
            raise
        except Exception as exc:  # noqa: BLE001
            jobs.log_exception(exc)
            job.log("diarization: failed (%s); output will be unattributed" % exc)
            turns = []

    return words, turns
