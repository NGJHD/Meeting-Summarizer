"""Stage 5 - tokenizer-aware chunking (CLAUDE.md section 10.1).

Never splits mid-turn, carries an overlap so a point discussed across a
boundary is not lost, and counts tokens with llama-server's /tokenize rather
than a characters-per-token estimate.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from .merge import SpeakerTurn, hms, speaker_label
from .jobs import Job


@dataclass
class Chunk:
    index: int = 0
    total: int = 0
    lines: list[str] = field(default_factory=list)
    tokens: int = 0
    start_s: float = 0.0
    end_s: float = 0.0

    @property
    def text(self) -> str:
        return "\n".join(self.lines)

    @property
    def time_start(self) -> str:
        return hms(self.start_s)

    @property
    def time_end(self) -> str:
        return hms(self.end_s)


def render_turn(turn: SpeakerTurn, attributed: bool) -> str:
    stamp = "(%s)" % hms(turn.start)
    if attributed and turn.speaker >= 0:
        return "[%s] %s %s" % (speaker_label(turn.speaker), stamp, turn.text)
    return "%s %s" % (stamp, turn.text)


def _batched_counts(job: Job, server, texts: list[str], batch: int = 400) -> list[int]:
    """Exact token count per turn.

    One /tokenize call per turn. That is a few thousand round trips on a long
    meeting, but they are localhost calls costing a couple of milliseconds each
    -- seconds in total against a stage measured in minutes. Estimating instead
    would reintroduce exactly the drift section 10.1 forbids, and packing
    chunks to a token budget needs a per-turn figure, which a single whole-
    transcript call cannot give.
    """
    if not texts:
        return []
    # The newline that joins turns costs tokens too; measure it once.
    sep_tokens = max(server.token_count("\n\n") - server.token_count("\n"), 0)

    counts: list[int] = []
    for i, line in enumerate(texts):
        if i % batch == 0:
            job.check_cancelled()
            if i:
                job.log("chunker: counted %d/%d turns" % (i, len(texts)))
        counts.append(server.token_count(line) + sep_tokens)
    return counts


def build_chunks(
    job: Job,
    server,
    turns: list[SpeakerTurn],
    attributed: bool,
    cfg: dict,
) -> list[Chunk]:
    """Split the merged transcript into ~target_tokens chunks on turn boundaries."""
    ccfg = cfg["chunking"]
    target = int(ccfg.get("target_tokens", 10000))
    overlap = int(ccfg.get("overlap_tokens", 400))

    lines = [render_turn(t, attributed) for t in turns]
    counts = _batched_counts(job, server, lines)
    starts = [t.start for t in turns]
    ends = [t.words[-1].end if t.words else t.start for t in turns]

    chunks: list[Chunk] = []
    i = 0
    n = len(lines)
    while i < n:
        job.check_cancelled()
        chunk = Chunk(lines=[], tokens=0, start_s=starts[i])
        j = i
        while j < n and (chunk.tokens + counts[j] <= target or not chunk.lines):
            chunk.lines.append(lines[j])
            chunk.tokens += counts[j]
            chunk.end_s = ends[j]
            j += 1
        chunks.append(chunk)

        if j >= n:
            break

        # Step back far enough to carry `overlap` tokens into the next chunk,
        # but never so far that we fail to advance.
        back = 0
        k = j - 1
        while k > i and back + counts[k] <= overlap:
            back += counts[k]
            k -= 1
        i = max(k + 1, i + 1)

    for idx, c in enumerate(chunks, 1):
        c.index = idx
        c.total = len(chunks)

    job.log(
        "chunker: %d turns -> %d chunks (target %d tokens, overlap %d)"
        % (n, len(chunks), target, overlap)
    )
    for c in chunks:
        job.log("chunker:   chunk %d/%d  %s-%s  %d tokens"
                % (c.index, c.total, c.time_start, c.time_end, c.tokens))
    return chunks


def fits(prompt_tokens: int, max_output_tokens: int, ctx_size: int) -> bool:
    """Section 10.3: assert the assembled prompt fits before sending it."""
    return prompt_tokens + max_output_tokens + 2000 <= ctx_size
