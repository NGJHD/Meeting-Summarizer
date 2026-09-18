"""Stage 6 - llama-server lifecycle and chat calls (CLAUDE.md section 11).

Uses http.client rather than a third-party client for two reasons: it is in the
standard library, so nothing extra is vendored, and it exposes the connection
object, which is what lets a cancel abort a request that is already in flight
instead of waiting out a whole chunk.
"""

from __future__ import annotations

import json
import math
import re
import socket
import subprocess
import threading
import time
from http.client import HTTPConnection
from pathlib import Path
from typing import Optional

from . import config
from .jobs import Cancelled, Job, JobError

CREATE_NEW_PROCESS_GROUP = getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0)
CREATE_NO_WINDOW = getattr(subprocess, "CREATE_NO_WINDOW", 0)

LLM_FAILED = "The language model could not be started."
LLM_CALL_FAILED = "The language model stopped responding."
EXTERNAL_FAILED = (
    "Nothing answered on the port you chose. Start the language model server "
    "first, or pick High Quality or Low Quality instead."
)

# The Model dropdown's third entry: not a quantisation but "somebody else's
# server". Kept out of hardware.MODELS deliberately -- it is not a file we
# ship, size, or place on a GPU, and every one of those code paths has to
# skip it rather than special-case it.
EXTERNAL = "external"

_THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL)
_OPEN_THINK_RE = re.compile(r"^.*?</think>", re.DOTALL)

# The chat template raises a Jinja exception - surfacing as HTTP 500 - for any
# effort outside this set, and silently rewrites "high" to "xhigh"
# (BUILD_NOTES.md section 5). Validate before dispatch rather than after.
# Qwen3.8 accepts xhigh / medium / low / none (Unsloth's published set).
# Anything else falls back to medium, so an unrecognised value is never sent.
VALID_EFFORT = {"xhigh", "medium", "low", "none"}

# How long to wait for a server we did not start. See LlamaServer.attach.
ATTACH_TIMEOUT_S = 10.0

# Passed to --reasoning-budget-message. It is injected in place of the rest of
# the thinking when the budget runs out, so the model stops reasoning and
# starts writing rather than simply being cut off mid-thought. Phrased as an
# instruction because that is what it becomes: it lands inside the reasoning
# block, immediately before the closing tag.
BUDGET_MESSAGE = (
    "\n\nThat is enough planning. Write the finished document now, in full, "
    "using the structure asked for.\n"
)


def strip_thinking(text: str) -> str:
    """Remove <think> blocks. Applied even when thinking is disabled.

    Belt and braces (section 11.2): a stray block that reaches the document is
    far more damaging than an unnecessary regex pass.
    """
    text = _THINK_RE.sub("", text)
    if "</think>" in text:            # unbalanced open tag, truncated stream
        text = _OPEN_THINK_RE.sub("", text)
    return text.strip()


def load_prompt(name: str) -> str:
    """Load a prompt from prompts\\ and inline the shared rules.

    Substitution is by str.replace, never str.format: the prompts contain
    literal braces and Markdown that format() would choke on (section 12).
    """
    path = config.PROMPTS / ("%s.txt" % name)
    try:
        text = path.read_text(encoding="utf-8")
    except OSError as exc:
        raise JobError(
            "A prompt file is missing from the application folder.",
            "cannot read %s: %s" % (path, exc),
        ) from exc
    if "{{SHARED_RULES}}" in text:
        shared = (config.PROMPTS / "_shared.txt").read_text(encoding="utf-8").strip()
        text = text.replace("{{SHARED_RULES}}", shared)
    return text


def _ctx_from_props(props: dict) -> int:
    """The context size a running llama-server reports. 0 if it will not say.

    llama.cpp has moved this around between builds, so try the places it has
    lived rather than trusting one. Returning 0 is safe: the caller then falls
    back to our configured value, which is what the code did before.
    """
    gen = props.get("default_generation_settings")
    for candidate in (gen, (gen or {}).get("params"), props):
        if isinstance(candidate, dict):
            for key in ("n_ctx", "ctx_size"):
                try:
                    value = int(candidate.get(key) or 0)
                except (TypeError, ValueError):
                    continue
                if value > 0:
                    return value
    return 0


def loading_message(cfg: dict, job: Job) -> str:
    """What the progress panel says while the LLM stage is getting ready.

    "up to 2 minutes on first run" is about loading 16 GB of weights off a cold
    disk, and it is a promise we cannot keep about a server we did not start --
    that one either answers at once or is not there. Saying the wrong one is
    worse than saying nothing: the user waits out two minutes that were never
    going to happen.
    """
    saved, port = config.external_llm(cfg)
    external = (job.model_key == EXTERNAL) or (not job.model_key and saved)
    if external:
        return "Connecting to the language model on port %d" % port
    return "Loading language model (up to 2 minutes on first run)"


def fill(template: str, **values: str) -> str:
    for key, value in values.items():
        template = template.replace("{{%s}}" % key.upper(), str(value))
    return template


class LlamaServer:
    """Owns the llama-server process for the life of one job."""

    def __init__(self, job: Job, cfg: dict):
        self.job = job
        self.cfg = cfg
        self.llm = cfg["llm"]
        # "Port" in the Model dropdown means a llama-server somebody else is
        # already running. The job's choice wins over config.json, so a user
        # who switches back to High or Low gets our own server for that run
        # even before the saved preference has been re-read.
        saved, saved_port = config.external_llm(cfg)
        if self.job.model_key == EXTERNAL:
            self.external = True
        elif self.job.model_key:
            self.external = False
        else:
            self.external = saved            # rebuilds, and tools with no UI
        self.port = saved_port if self.external else int(self.llm.get("port", 8080))
        self.proc: Optional[subprocess.Popen] = None
        self._remote_ctx = 0
        self._conn: Optional[HTTPConnection] = None
        self._conn_lock = threading.Lock()
        self._resolved = None

    # -- lifecycle -------------------------------------------------------

    def resolve_model(self) -> tuple[Path, str, str]:
        """Return (weights path, gpu_layers, offload regex).

        "auto" means: pick the quantisation this machine can actually hold, and
        then get out of llama.cpp's way. It fits the model to free device
        memory itself and does that better than we can -- the exception being
        a unified-memory GPU, where the figure it fits against is a fiction
        (hardware.gpu_layers_for).
        """
        from . import hardware

        # Cached: this runs nvidia-smi and logs, and is called from both
        # command() and start().
        if getattr(self, "_resolved", None) is not None:
            return self._resolved

        if self.external:
            # There are no weights of ours to find and nothing to place on a
            # GPU: the server on that port was started by somebody else, with
            # whatever model and offload they chose.
            self.job.model_key = EXTERNAL
            self._resolved = (Path(), "", "")
            return self._resolved

        layers = str(self.llm.get("gpu_layers", "auto"))
        regex = str(self.llm.get("cpu_ffn_regex") or "")

        requested = self.job.model_key or ""
        if requested and requested not in hardware.BY_KEY:
            # job.model_key is the dropdown's vocabulary, not a filename. A
            # value that is not one of its keys is a label -- tools/ sets one
            # so a comparison run's timings are filed apart from real jobs --
            # and must not be resolved as a path to a model that will not be
            # found. Fall back to what config.json asked for.
            requested = ""
        requested = requested or self.llm.get("model") or "auto"
        if requested not in hardware.BY_KEY and requested not in ("auto", ""):
            # An explicit path in config.json still wins.
            self._resolved = (config.resolve(requested),
                              "" if layers == "auto" else layers,
                              "" if regex == "auto" else regex)
            return self._resolved

        gpu = hardware.detect_gpu()
        key = hardware.resolve_key(requested, gpu["vram_mb"])
        path = hardware.model_path(key)
        auto_layers, auto_regex = hardware.placement(key, gpu)
        if layers == "auto":
            layers = auto_layers
        if regex == "auto":
            regex = auto_regex

        placement = ("llama.cpp decides" if layers == ""
                     else "on the processor" if layers == "0"
                     else "%d FFN blocks to system RAM" % (regex.count("|") + 1)
                     if regex else "fully on the GPU")
        self.job.log(
            "llm: %s | %s via %s (%d MiB%s) | %s"
            % (path.name, gpu["device"], config.backend("llama"), gpu["vram_mb"],
               ", unified memory" if gpu.get("uma") else "", placement)
        )
        # Record what was actually chosen. Calibration is keyed by model, and a
        # job that never went through the dropdown -- one reopened from the
        # history list, say -- would otherwise be timed against the wrong one.
        self.job.model_key = key
        self._resolved = (path, layers, regex)
        return self._resolved

    def command(self) -> list[str]:
        from . import hardware

        model, layers, regex = self.resolve_model()
        cmd = [
            str(config.LLAMA_SERVER),
            "-m", str(model),
            "--ctx-size", str(int(self.llm.get("ctx_size", 32768))),
            "--flash-attn", "on",
            "--cache-type-k", str(self.llm.get("cache_type_k", "q8_0")),
            "--cache-type-v", str(self.llm.get("cache_type_v", "q8_0")),
            "--jinja",
            "--batch-size", "512", "--ubatch-size", "512",
            "--parallel", "1",
            "--host", "127.0.0.1", "--port", str(self.port),
            "--no-webui",
        ]
        # No --n-gpu-layers unless we have a reason. Section 11.1 specified
        # `-ngl 99` plus a hand-written --override-tensor; both are withdrawn.
        # llama.cpp now fits the model to free device memory on its own, and
        # passing -ngl at all makes it give up and do as it is told:
        #
        #   common_fit_params: failed to fit params to free device memory:
        #   n_gpu_layers already set by user to 99, abort
        threads = str(self.llm.get("threads", "auto"))
        if threads and threads != "auto":
            cmd += ["--threads", threads]
        if layers:
            cmd += ["--n-gpu-layers", layers]
        # Only when there is a choice to get wrong. Detection sized the model
        # against one specific adapter; llama.cpp must use that one and not
        # whichever it would have picked by itself.
        device_id = hardware.detect_gpu().get("device_id")
        if device_id and layers != "0":
            cmd += ["--device", device_id]
        if regex:
            # Keep attention on the GPU and push the upper blocks' FFN tensors
            # into system RAM. Measured 2.4x faster than letting llama.cpp fit
            # whole layers, at identical VRAM -- see hardware.placement.
            cmd += ["--override-tensor", regex]
        spec = self.spec_type(regex)
        if spec:
            cmd += ["--spec-type", spec]
        budget = self._reasoning_budget()
        if budget > 0:
            cmd += ["--reasoning-budget", str(budget)]
            cmd += ["--reasoning-budget-message", BUDGET_MESSAGE]
        return cmd

    def spec_type(self, regex: str) -> str:
        """Speculative decoding for this model on this card. "" = off.

        "draft-mtp" uses the multi-token-prediction layer carried inside the
        GGUF itself (blk.64.nextn.*), so there is no second model to ship. It
        is lossless -- draft-and-verify, so the accepted tokens are the ones
        the target model would have produced -- and measured worth roughly 50%
        on generation (BUILD_NOTES 9v, 9y).

        **It is not free in VRAM.** The draft context costs about 700 MiB: its
        own KV cache and compute buffers, plus a larger recurrent-state
        allocation. So `"auto"` enables it only when the model already fits
        without an FFN offload. Spending 700 MiB on a faster draft while
        pushing more layers onto the processor to pay for it would be a poor
        trade -- the offload is the single largest cost there is on this
        pipeline (9w).

        `llm.spec_type` overrides: "" forces it off, "draft-mtp" forces it on
        regardless of fit. Files without the tensors get nothing either way --
        IQ2_XXS has 64 blocks and no nextn at all.
        """
        from . import hardware

        requested = str(self.llm.get("spec_type", "auto"))
        if requested != "auto":
            return requested
        if regex:
            return ""                     # already offloading; do not add to it
        key = self.job.model_key
        return str(hardware.BY_KEY.get(key, {}).get("spec") or "")

    def _reasoning_budget(self) -> int:
        """Ceiling on thinking tokens for the final reduce. 0 = unrestricted.

        The failure this prevents is specific and expensive: thinking and the
        answer come out of one allowance (reduce.REDUCE_MAX_TOKENS), and a
        model that reasons for the whole of it returns `finish_reason: length`
        with an *empty* `content`. llm.chat() sees an empty completion, retries
        twice, and fails the job -- after the recording has already been
        transcribed, diarized and mapped.

        Measured on a real 3h51m recording: the reduce wanted ~3440 thinking
        tokens and then wrote a 4158-token document, so 4000 leaves the
        document the larger half of the allowance (BUILD_NOTES 9x).
        """
        think = self.cfg["thinking"]
        if not any(bool(think.get(k)) for k in ("map", "group_reduce", "reduce")):
            return 0                      # nothing thinks; nothing to cap
        try:
            return max(0, int(think.get("reduce_budget_tokens") or 0))
        except (TypeError, ValueError):
            return 0

    def start(self) -> None:
        if self.external:
            self.attach()
            return
        model, _layers, _regex = self.resolve_model()
        if not model.exists():
            raise JobError(LLM_FAILED, "model file missing: %s" % model)

        self.job.log("llm: starting llama-server on port %d" % self.port)
        log_path = config.TEMP / "llama-server.log"
        # stderr to a file, never a pipe: llama-server is verbose and a full
        # pipe would block it exactly as it did whisper (BUILD_NOTES 3.7).
        self._log_handle = open(log_path, "w", encoding="utf-8", errors="replace")
        try:
            self.proc = subprocess.Popen(
                self.command(),
                stdout=self._log_handle,
                stderr=subprocess.STDOUT,
                env=config.child_env(),
                creationflags=CREATE_NEW_PROCESS_GROUP | CREATE_NO_WINDOW,
            )
        except OSError as exc:
            raise JobError(LLM_FAILED, "could not spawn llama-server: %s" % exc) from exc
        self.job.register_proc(self.proc)

        timeout = float(self.llm.get("startup_timeout_s", 180))
        self.job.log("llm: loading model (up to 2 minutes on first run)")
        deadline = time.time() + timeout
        while time.time() < deadline:
            self.job.check_cancelled()
            if self.proc.poll() is not None:
                raise JobError(LLM_FAILED, self._log_tail())
            if self._healthy():
                self.job.log("llm: ready after %.0fs" % (timeout - (deadline - time.time())))
                return
            time.sleep(1.0)
        raise JobError(
            LLM_FAILED,
            "llama-server did not become ready within %.0fs\n%s" % (timeout, self._log_tail()),
        )

    def attach(self) -> None:
        """Use a llama-server already running on this port instead of our own.

        The wait is short on purpose. Our own server has a model to load off a
        cold disk, which is why `startup_timeout_s` is 180; a server somebody
        started by hand is either up or it is not, and making the user watch a
        three-minute countdown for a port with nothing behind it is the wrong
        way to tell them they mistyped it. If it happens to be loading, the
        health check fails closed and they can start the job again.
        """
        self.resolve_model()
        deadline = time.time() + ATTACH_TIMEOUT_S
        while time.time() < deadline:
            self.job.check_cancelled()
            if self._healthy():
                props = self._props()
                self._remote_ctx = _ctx_from_props(props)
                name = str(props.get("model_path")
                           or props.get("model_alias") or "")
                self.job.log("llm: using the server on port %d (%s, %s context)"
                             % (self.port, Path(name).name or "model unknown",
                                "%d" % self._remote_ctx if self._remote_ctx
                                else "unknown"))
                if self._remote_ctx and self._remote_ctx < int(
                        self.llm.get("ctx_size", 32768)):
                    # Not a failure -- the reduce simply splits into more
                    # tiers -- but it is why a job through this port may take
                    # more calls than the same recording through our own.
                    self.job.log(
                        "llm: that is smaller than our own %d, so long "
                        "meetings will be consolidated in more steps"
                        % int(self.llm.get("ctx_size", 32768)))
                # Our own runs keep whisper and the LLM out of VRAM at the same
                # time (section 11.1). We cannot do that for a server we did
                # not start, so say so rather than let a mysterious slowdown or
                # an out-of-memory during transcription go unexplained.
                self.job.log("llm: that server holds its own memory for the "
                             "whole job, including while transcribing")
                return
            time.sleep(0.5)
        raise JobError(
            EXTERNAL_FAILED,
            "nothing healthy on 127.0.0.1:%d after %.0fs" % (self.port, ATTACH_TIMEOUT_S),
        )

    @property
    def ctx_size(self) -> int:
        """The context window actually in force, which may not be ours.

        For our own server this is `llm.ctx_size`, because we passed it on the
        command line. For a server on a port it is whatever that one was
        started with, and assuming ours would be a silent correctness bug: the
        overflow guard in reduce.py exists to force another group-reduce tier
        rather than send a prompt that will not fit, and a guard measuring
        against the wrong number does not guard. The operator's own launcher
        uses `-c 16384`, half of our default, so this is not hypothetical.
        """
        if self.external and self._remote_ctx:
            return self._remote_ctx
        return int(self.llm.get("ctx_size", 32768))

    def _props(self) -> dict:
        try:
            conn = HTTPConnection("127.0.0.1", self.port, timeout=5)
            conn.request("GET", "/props")
            resp = conn.getresponse()
            data = json.loads(resp.read())
            conn.close()
            return data if isinstance(data, dict) else {}
        except Exception:  # noqa: BLE001 - a server that answers /health but
            return {}                      # not /props is still usable

    def _remote_model(self) -> str:
        """What the external server is actually serving, for the log.

        Worth a line in the log because nothing else in the job records it: the
        user chose a port, not a model, and six months later the only way to
        know what wrote a document is if we wrote it down.
        """
        data = self._props()
        name = str(data.get("model_path") or data.get("model_alias") or "")
        return Path(name).name or "model unknown"

    def _log_tail(self, lines: int = 40) -> str:
        try:
            with open(config.TEMP / "llama-server.log", "r", encoding="utf-8", errors="replace") as fh:
                return "".join(fh.readlines()[-lines:])
        except OSError:
            return ""

    def _healthy(self) -> bool:
        try:
            conn = HTTPConnection("127.0.0.1", self.port, timeout=3)
            conn.request("GET", "/health")
            resp = conn.getresponse()
            resp.read()
            conn.close()
            return resp.status == 200
        except Exception:  # noqa: BLE001 - not up yet is the normal case
            return False

    def stop(self) -> None:
        """Shut the server down. 14GB of VRAM must not stay allocated."""
        self.abort()
        if self.external:
            # Somebody else's process. Dropping the in-flight request is the
            # whole of our responsibility here; killing it would take down a
            # server that was running before this job started and will be
            # wanted after it.
            return
        proc, self.proc = self.proc, None
        if proc is None:
            return
        self.job.unregister_proc(proc)
        try:
            if proc.poll() is None:
                subprocess.run(
                    ["taskkill", "/F", "/T", "/PID", str(proc.pid)],
                    capture_output=True, creationflags=CREATE_NO_WINDOW, timeout=20,
                )
        except Exception:  # noqa: BLE001
            try:
                proc.kill()
            except Exception:
                pass
        try:
            self._log_handle.close()
        except Exception:  # noqa: BLE001
            pass
        self.job.log("llm: server stopped")

    def abort(self) -> None:
        """Drop any in-flight request so a cancel does not wait out a chunk."""
        with self._conn_lock:
            conn, self._conn = self._conn, None
        if conn is not None:
            try:
                conn.close()
            except Exception:  # noqa: BLE001
                pass

    def __enter__(self) -> "LlamaServer":
        self.start()
        return self

    def __exit__(self, *exc) -> None:
        self.stop()

    # -- requests --------------------------------------------------------

    def _post(self, path: str, payload: dict, timeout: float) -> dict:
        body = json.dumps(payload).encode("utf-8")
        conn = HTTPConnection("127.0.0.1", self.port, timeout=timeout)
        with self._conn_lock:
            self._conn = conn
        try:
            conn.request(
                "POST", path, body=body,
                headers={"Content-Type": "application/json", "Content-Length": str(len(body))},
            )
            resp = conn.getresponse()
            raw = resp.read()
            if resp.status != 200:
                raise RuntimeError(
                    "HTTP %d from %s: %s" % (resp.status, path, raw[:400].decode("utf-8", "replace"))
                )
            return json.loads(raw)
        finally:
            with self._conn_lock:
                if self._conn is conn:
                    self._conn = None
            try:
                conn.close()
            except Exception:  # noqa: BLE001
                pass

    def token_count(self, text: str) -> int:
        """Exact token count via /tokenize.

        Section 10.1 forbids a characters-per-token heuristic: it drifts badly
        on transcript text full of names, disfluencies and timestamps, and the
        whole chunking scheme depends on this number being right.
        """
        self.job.check_cancelled()
        data = self._post("/tokenize", {"content": text}, timeout=300)
        return len(data.get("tokens", []))

    def _stream(self, payload: dict, idle_timeout: float, on_token) -> tuple[str, str, int, str]:
        """POST a streaming completion, returning (content, reasoning, tokens, finish).

        Streaming is not for show. A non-streaming call needs a single total
        timeout covering the whole generation, and there is no good value for
        it: the final reduce can legitimately emit many thousands of tokens,
        which at the 1.8 tok/s slow hardware manages is over an hour, while a
        genuinely hung server should be caught in minutes. Streaming replaces that guess with
        an *idle* timeout -- the socket read only blocks between tokens -- so a
        slow call runs as long as it needs and a dead one is caught quickly.
        """
        body = json.dumps({**payload, "stream": True}).encode("utf-8")
        conn = HTTPConnection("127.0.0.1", self.port, timeout=idle_timeout)
        with self._conn_lock:
            self._conn = conn

        content: list[str] = []
        reasoning_chars = 0
        reasoning_tokens = 0
        tokens = 0
        finish = ""
        try:
            conn.request(
                "POST", "/v1/chat/completions", body=body,
                headers={"Content-Type": "application/json", "Content-Length": str(len(body))},
            )
            resp = conn.getresponse()
            if resp.status != 200:
                raw = resp.read()
                raise RuntimeError(
                    "HTTP %d: %s" % (resp.status, raw[:400].decode("utf-8", "replace"))
                )
            while True:
                line = resp.readline()
                if not line:
                    break
                line = line.strip()
                if not line.startswith(b"data:"):
                    continue
                data = line[5:].strip()
                if data == b"[DONE]":
                    break
                try:
                    event = json.loads(data)
                except ValueError:
                    continue
                choices = event.get("choices") or [{}]
                # Why the completion ended, not just that it did. "length"
                # with nothing in `content` is the specific, expensive failure
                # this whole budget mechanism exists to catch: the model spent
                # its entire allowance thinking. It has to be distinguishable
                # from a server that simply died, because the answer to one is
                # to ask again differently and the answer to the other is not
                # to ask again at all.
                finish = choices[0].get("finish_reason") or finish
                delta = choices[0].get("delta") or {}
                piece = delta.get("content") or ""
                if piece:
                    content.append(piece)
                    tokens += 1
                    on_token(tokens, False)
                # Thinking arrives as reasoning_content and can run for many
                # minutes before the first word of the answer. Counting it for
                # liveness too keeps the log moving through that phase, which
                # is otherwise the longest silence in the whole job.
                thought = delta.get("reasoning_content") or ""
                if thought:
                    reasoning_chars += len(thought)
                    reasoning_tokens += 1
                    on_token(reasoning_tokens, True)
        finally:
            with self._conn_lock:
                if self._conn is conn:
                    self._conn = None
            try:
                conn.close()
            except Exception:  # noqa: BLE001
                pass

        return "".join(content), str(reasoning_chars), tokens, finish

    def chat(
        self,
        prompt: str,
        max_tokens: int,
        thinking: bool,
        effort: str = "medium",
        timeout: float = 0.0,
        progress=None,
        expect_seconds: float = 0.0,
        expect_tokens: float = 0.0,
        stage: str = "",
    ) -> str:
        """One chat completion, retried twice with backoff (section 11.3)."""
        if thinking:
            sampling = {
                "temperature": 1.0, "top_p": 0.95, "top_k": 20, "min_p": 0.0,
                "presence_penalty": 0.0, "repeat_penalty": 1.0,
            }
            kwargs = {"enable_thinking": True}
            effort = effort if effort in VALID_EFFORT else "medium"
            kwargs["reasoning_effort"] = effort
        else:
            sampling = {
                "temperature": 0.7, "top_p": 0.8, "top_k": 20, "min_p": 0.0,
                "presence_penalty": 1.5, "repeat_penalty": 1.0,
            }
            kwargs = {"enable_thinking": False}

        payload = {
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": int(max_tokens),
            "chat_template_kwargs": kwargs,
            **sampling,
        }
        # No per-request budget field. `reasoning_control` exists in this
        # build's request schema and is silently ignored: measured against a
        # server launched with --reasoning-budget 60, requests asking for
        # 2000 came back capped at 60 all the same (BUILD_NOTES section 9s).
        # The launch flag in command() is the whole mechanism, which means an
        # external server is capped only if whoever started it said so -- and
        # that is what the empty-completion retry below is for.
        idle = float(timeout or self.llm.get("idle_timeout_s", 300))

        last = ""
        for attempt in range(3):
            self.job.check_cancelled()
            started = time.time()
            state = {"logged": 0, "thinking_logged": 0, "seen": 0}
            # Publish this call so the ETA can price what is left of it from
            # the live token rate rather than from the progress bar.
            if stage:
                self.job.current_call = {"stage": stage, "started": started, "seen": 0}

            # A single reduce call can run for an hour. Without progress inside
            # the call the bar sits still for that whole time and then jumps,
            # which is exactly what makes a long job look hung.
            #
            # Two signals, whichever is further along:
            #
            #   tokens   -- against how many tokens a call of this kind has
            #               actually produced on this machine before. Against
            #               `max_tokens` instead this is useless: the cap is
            #               12000 and a document is 2300-4200, so the bar
            #               crawls to a third and leaps. The measured figure
            #               tracks the real completion and reaches the end
            #               with it.
            #   elapsed  -- against how long such a call actually takes here.
            #               Approached asymptotically, 1 - e^-t: 63% at the
            #               expected time, 86% at twice it, never quite 100%.
            #               It cannot stall however wrong the expectation is,
            #               which is what makes it a safe floor for the above.
            budget = expect_tokens or float(max_tokens) * (1.5 if thinking else 1.05)

            def fraction() -> float:
                frac = state["seen"] / budget
                if expect_seconds > 0:
                    t = (time.time() - started) / expect_seconds
                    frac = max(frac, 1.0 - math.exp(-t))
                return min(0.99, frac)

            def on_token(n: int, is_thinking: bool) -> None:
                state["seen"] += 1
                if state["seen"] % 25 == 0:
                    if self.job.current_call is not None:
                        self.job.current_call["seen"] = state["seen"]
                    if progress is not None:
                        progress(fraction())
                # Cheap liveness in the UI: a single map call is ~14 minutes on
                # slow hardware, and silence that long reads as a hang.
                key = "thinking_logged" if is_thinking else "logged"
                if n - state[key] >= 250:
                    state[key] = n
                    self.job.log(
                        "llm: %d %s tokens so far (%.1f/s)"
                        % (n, "thinking" if is_thinking else "output",
                           n / max(time.time() - started, 1e-6))
                    )

            try:
                text, reasoning_chars, tokens, finish = self._stream(payload, idle, on_token)
                elapsed = time.time() - started
                self.job.log(
                    "llm: %d completion tokens in %.0fs (%.1f tok/s), %s reasoning chars"
                    % (tokens, elapsed, tokens / max(elapsed, 1e-6), reasoning_chars)
                )
                if stage:
                    from . import calibration

                    calibration.record_call(
                        stage, state["seen"], elapsed, self.job.model_key,
                        self.job.id, self.job.duration_s / 3600.0)
                    self.job.current_call = None
                text = strip_thinking(text)
                if text:
                    return text
                last = "empty completion (finish_reason=%s)" % (finish or "unknown")
                if finish == "length" and payload["chat_template_kwargs"].get(
                        "enable_thinking"):
                    # The model reasoned for the whole of max_tokens and never
                    # started the document. Asking again identically just buys
                    # the same wait and the same nothing -- measured at 400
                    # tokens: 1700 characters of reasoning, zero of content,
                    # three times over. So change the question instead: drop
                    # thinking for the retry and take a document written
                    # without it over no document at all after an hour of
                    # transcription, diarization and mapping.
                    self.job.log("llm: the model used its whole allowance "
                                 "thinking; retrying without thinking")
                    payload["chat_template_kwargs"] = {"enable_thinking": False}
                    payload.update({"temperature": 0.7, "top_p": 0.8,
                                    "presence_penalty": 1.5, "min_p": 0.0})
            except Cancelled:
                self.job.current_call = None
                raise
            except Exception as exc:  # noqa: BLE001
                self.job.current_call = None
                last = str(exc)
                if self.job.cancelled:
                    raise Cancelled() from exc
                if isinstance(exc, (TimeoutError, socket.timeout)):
                    # An idle timeout means the server produced nothing for
                    # `idle` seconds. Retrying re-runs the whole generation at
                    # the same speed, so it costs the same wait again and
                    # almost never succeeds. Fail now and keep the transcript.
                    raise JobError(
                        LLM_CALL_FAILED,
                        "no output for %.0fs; giving up rather than retrying" % idle,
                    ) from exc
            if attempt < 2:
                delay = 2.0 * (attempt + 1)
                self.job.log("llm: call failed (%s); retrying in %.0fs" % (last, delay))
                time.sleep(delay)

        raise JobError(LLM_CALL_FAILED, "three attempts failed; last error: %s" % last)
