"""Stages 5-6 - map, group reduce, final reduce (CLAUDE.md section 10).

Two tiers when the chunk count is at or below `group_reduce_threshold`, three
(or more) when it is above. The extra tier is not optional: with 11 chunks a
straight reduce feeds ~16.5k tokens of notes plus 8k of output plus thinking
into a 32k context, and it does not fail cleanly -- it silently drops material
from the middle of the meeting, which is the failure least likely to be noticed.
"""

from __future__ import annotations

import time

from . import calibration, chunker, config, jobs, llm, merge, speakers
from .chunker import Chunk
from .jobs import Cancelled, Job, JobError

PLACEHOLDER = "_(This section of the recording could not be summarised.)_"


def _guard(job: Job, server: llm.LlamaServer, prompt: str, max_out: int, cfg: dict) -> int:
    """Return the prompt's token count, refusing to send one that will not fit."""
    ctx = int(cfg["llm"].get("ctx_size", 32768))
    n = server.token_count(prompt)
    if not chunker.fits(n, max_out, ctx):
        raise JobError(
            "That recording produced more material than the model can work with "
            "in one pass. The full transcript was still saved.",
            "prompt %d + output %d + 2000 exceeds ctx_size %d" % (n, max_out, ctx),
        )
    return n


def run_map(job: Job, server: llm.LlamaServer, chunks: list[Chunk], cfg: dict) -> list[str]:
    """One call per chunk. Thinking off: this is mechanical extraction."""
    template = llm.load_prompt("map")
    max_out = int(cfg["chunking"].get("max_map_output_tokens", 1500))
    think = bool(cfg["thinking"].get("map", False))
    # How long one map call is expected to take here, so the bar can move
    # inside it rather than only between chunks.
    # Bar and ETA are priced from the same measurement, so they agree.
    exp_tokens, tok_s = calibration.call_profile("map", job.model_key,
                                            job.duration_s / 3600.0)
    per_call = exp_tokens / max(tok_s, 0.05)
    notes: list[str] = []

    # The ETA is priced off these counts. Group reduce is included up front,
    # unknown though its group count is at this point, because otherwise the
    # estimate would step upward the moment the map loop ended.
    ccfg = cfg["chunking"]
    threshold = int(ccfg.get("group_reduce_threshold", 8))
    size = max(2, int(ccfg.get("group_size", 5)))
    groups = -(-len(chunks) // size) if len(chunks) > threshold else 0
    job.pending_calls = {
        "map": len(chunks),
        "group_reduce": groups,
        "reduce": 2 if job.mode == "both" else 1,
    }

    for chunk in chunks:
        job.check_cancelled()
        job.pending_calls["map"] = len(chunks) - chunk.index + 1
        job.set_progress(
            "map",
            (chunk.index - 1) / max(len(chunks), 1),
            "Reading the meeting (part %d of %d)" % (chunk.index, len(chunks)),
        )
        prompt = llm.fill(
            template,
            transcript=chunk.text,
            chunk_index=chunk.index,
            chunk_total=chunk.total,
            time_start=chunk.time_start,
            time_end=chunk.time_end,
        )
        _guard(job, server, prompt, max_out, cfg)
        done = chunk.index - 1
        total = max(len(chunks), 1)

        def within(frac: float, _d=done, _t=total) -> None:
            job.set_progress("map", (_d + frac) / _t)

        try:
            job.pending_calls["map"] = len(chunks) - chunk.index
            text = server.chat(prompt, max_out, thinking=think, progress=within,
                               expect_seconds=per_call, expect_tokens=exp_tokens,
                               stage="map")
        except Cancelled:
            raise
        except JobError as exc:
            # One bad chunk out of eleven must not destroy a job that has
            # already consumed 40 minutes (section 11.3).
            jobs.log_exception(exc)
            job.log("map: chunk %d failed, inserting a placeholder" % chunk.index)
            text = "## Themes\n%s\n" % PLACEHOLDER
        notes.append(
            "### Section %d of %d (%s - %s)\n\n%s"
            % (chunk.index, chunk.total, chunk.time_start, chunk.time_end, text)
        )
        job.set_progress("map", chunk.index / max(len(chunks), 1))

    job.set_progress("map", 1.0)
    return notes


def run_group_reduce(
    job: Job,
    server: llm.LlamaServer,
    notes: list[str],
    spans: list[tuple[str, str]],
    cfg: dict,
) -> tuple[list[str], list[tuple[str, str]]]:
    """Consolidate consecutive groups of notes into section summaries.

    Applied repeatedly until the count fits under the threshold. Section 10.3
    requires this to generalise rather than assume exactly three tiers -- it
    should not trigger below 40 hours of audio, but the code must not care.
    """
    ccfg = cfg["chunking"]
    threshold = int(ccfg.get("group_reduce_threshold", 8))
    size = max(2, int(ccfg.get("group_size", 5)))
    max_out = int(ccfg.get("max_group_output_tokens", 2500))
    think = bool(cfg["thinking"].get("group_reduce", False))
    template = llm.load_prompt("group_reduce")

    tier = 0
    while len(notes) > threshold:
        tier += 1
        groups = [notes[i:i + size] for i in range(0, len(notes), size)]
        group_spans = [spans[i:i + size] for i in range(0, len(spans), size)]
        job.log(
            "group reduce: tier %d, %d notes -> %d groups of up to %d"
            % (tier, len(notes), len(groups), size)
        )

        out_notes: list[str] = []
        out_spans: list[tuple[str, str]] = []
        exp_tokens, tok_s = calibration.call_profile(
            "group_reduce", job.model_key, job.duration_s / 3600.0)
        per_call = exp_tokens / max(tok_s, 0.05)
        for i, (group, gspan) in enumerate(zip(groups, group_spans), 1):
            job.check_cancelled()
            job.set_progress(
                "group_reduce",
                (i - 1) / max(len(groups), 1),
                "Consolidating section %d of %d" % (i, len(groups)),
            )
            start, end = gspan[0][0], gspan[-1][1]
            prompt = llm.fill(
                template, notes="\n\n".join(group), time_start=start, time_end=end
            )
            _guard(job, server, prompt, max_out, cfg)

            def within(frac: float, _d=i - 1, _t=max(len(groups), 1)) -> None:
                job.set_progress("group_reduce", (_d + frac) / _t)

            job.pending_calls["group_reduce"] = len(groups) - i
            try:
                text = server.chat(prompt, max_out, thinking=think, progress=within,
                                   expect_seconds=per_call, expect_tokens=exp_tokens,
                                   stage="group_reduce")
            except Cancelled:
                raise
            except JobError as exc:
                jobs.log_exception(exc)
                job.log("group reduce: group %d failed, passing its notes through" % i)
                text = "\n\n".join(group)
            out_notes.append("### Section %s - %s\n\n%s" % (start, end, text))
            out_spans.append((start, end))
        notes, spans = out_notes, out_spans
        job.set_progress("group_reduce", 1.0)

    return notes, spans


def run_final_reduce(
    job: Job,
    server: llm.LlamaServer,
    notes: list[str],
    cfg: dict,
    mode: str,
    meeting_name: str,
    duration_s: float,
    index: int = 0,
    total: int = 1,
) -> str:
    """The one call where thinking earns its cost (section 11.2)."""
    name = "reduce_minutes" if mode == "minutes" else "reduce_summary"
    template = llm.load_prompt(name)
    think = bool(cfg["thinking"].get("reduce", True))
    effort = str(cfg["thinking"].get("reduce_effort", "medium"))
    max_out = 8000

    label = "minutes" if mode == "minutes" else "summary"
    if total > 1:
        label += " (%d of %d)" % (index + 1, total)
    job.set_stage("reduce", "Writing the %s" % label)
    # In "both" mode the reduce stage covers two calls, so the bar must not
    # reach 100% after the first one.
    job.set_progress("reduce", index / total)
    prompt = llm.fill(
        template,
        notes="\n\n".join(notes),
        meeting_name=meeting_name,
        duration=merge.hms(duration_s),
    )
    n = _guard(job, server, prompt, max_out, cfg)
    job.log("reduce: %d prompt tokens, thinking=%s effort=%s" % (n, think, effort))
    def within(frac: float) -> None:
        job.set_progress("reduce", (index + frac) / total)

    job.pending_calls = {"reduce": total - index - 1}
    exp_tokens, tok_s = calibration.call_profile("reduce", job.model_key,
                                               duration_s / 3600.0)
    text = server.chat(
        prompt, max_out, thinking=think, effort=effort, progress=within,
        expect_seconds=exp_tokens / max(tok_s, 0.05), expect_tokens=exp_tokens,
        stage="reduce",
    )
    job.set_progress("reduce", (index + 1) / total)
    return text


def _fit_for_final(
    job: Job,
    server: llm.LlamaServer,
    notes: list[str],
    spans: list[tuple[str, str]],
    cfg: dict,
    meeting_name: str,
) -> tuple[list[str], list[tuple[str, str]]]:
    """Force extra group-reduce tiers until the final prompt actually fits.

    The chunk-count threshold is a proxy for size, and a good one, but notes
    can run long individually. This measures the real assembled prompt.
    """
    ctx = int(cfg["llm"].get("ctx_size", 32768))
    template = llm.load_prompt("reduce_minutes" if job.mode == "minutes" else "reduce_summary")
    max_out = 8000

    for attempt in range(4):
        prompt = llm.fill(
            template,
            notes="\n\n".join(notes),
            meeting_name=meeting_name,
            duration=merge.hms(job.duration_s),
        )
        n = server.token_count(prompt)
        if chunker.fits(n, max_out, ctx):
            return notes, spans
        if len(notes) < 2:
            break
        job.log(
            "reduce: final prompt is %d tokens against a %d context; "
            "consolidating one more tier" % (n, ctx)
        )
        forced = dict(cfg)
        forced["chunking"] = dict(cfg["chunking"])
        # Force one more pass by pretending the threshold is just under the
        # current count, rather than duplicating run_group_reduce's body.
        forced["chunking"]["group_reduce_threshold"] = max(1, len(notes) - 1)
        notes, spans = run_group_reduce(job, server, notes, spans, forced)

    return notes, spans


def produce_document(
    job: Job,
    server: llm.LlamaServer,
    chunks: list[Chunk],
    cfg: dict,
    meeting_name: str,
) -> list:
    """Full map -> (group reduce) -> reduce. Returns the written documents."""
    started = time.time()

    t_map = time.time()
    notes = run_map(job, server, chunks, cfg)
    job.stage_seconds["map"] = time.time() - t_map
    spans = [(c.time_start, c.time_end) for c in chunks]

    threshold = int(cfg["chunking"].get("group_reduce_threshold", 8))
    t_gr = time.time()
    if len(notes) > threshold:
        notes, spans = run_group_reduce(job, server, notes, spans, cfg)
        job.stage_seconds["group_reduce"] = time.time() - t_gr
    else:
        job.stage_seconds["group_reduce"] = 0.1
        job.log("group reduce: skipped, %d notes is within the threshold of %d"
                % (len(notes), threshold))
        job.set_progress("group_reduce", 1.0)

    # Section 10.3: the count check above is necessary but not sufficient --
    # notes can individually run long. If the assembled final prompt still will
    # not fit, log the overflow and split further rather than sending it.
    notes, spans = _fit_for_final(job, server, notes, spans, cfg, meeting_name)

    # "both" runs the final reduce twice over the SAME notes. Everything before
    # this point -- transcription, diarization, chunking, map, group reduce --
    # is identical for the two modes and is by far the expensive part, so the
    # second document costs one extra call rather than a second pipeline.
    modes = ["summary", "minutes"] if job.mode == "both" else [job.mode]
    t_red = time.time()
    paths = []
    pristine: dict[str, str] = {}
    for i, mode in enumerate(modes):
        if len(modes) > 1:
            job.log("reduce: document %d of %d (%s)" % (i + 1, len(modes), mode))
        document = run_final_reduce(
            job, server, notes, cfg, mode, meeting_name, job.duration_s,
            index=i, total=len(modes),
        )
        if job.attribution_note:
            document = document.rstrip() + "\n\n---\n\n*%s*\n" % job.attribution_note
        # Keep the model's own text, labels and all, so renaming later can
        # rewrite from it instead of editing the file in place.
        pristine[mode] = document
        path = config.meeting_dir(meeting_name) / ("%s_%s.md" % (meeting_name, mode))
        config.write_atomic(path, document.rstrip() + "\n")
        job.log("reduce: wrote %s" % path.name)
        paths.append(path)

    # Cache the consolidated notes with the samples. Applying speaker names
    # later then costs one final-reduce call instead of the whole map loop.
    try:
        speakers.save(meeting_name, job.samples, notes, job.mode, job.duration_s,
                      documents=pristine)
    except Exception as exc:  # noqa: BLE001 - a cache failure must not fail the job
        jobs.log_exception(exc)
    job.stage_seconds["reduce"] = (time.time() - t_red) / max(len(modes), 1)
    job.log("reduce: %d document(s) in %.0fs" % (len(paths), time.time() - started))
    return paths
