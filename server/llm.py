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

_THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL)
_OPEN_THINK_RE = re.compile(r"^.*?</think>", re.DOTALL)

# The chat template raises a Jinja exception - surfacing as HTTP 500 - for any
# effort outside this set, and silently rewrites "high" to "xhigh"
# (BUILD_NOTES.md section 5). Validate before dispatch rather than after.
VALID_EFFORT = {"xhigh", "medium", "low"}


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
        self.port = int(self.llm.get("port", 8080))
        self.proc: Optional[subprocess.Popen] = None
        self._conn: Optional[HTTPConnection] = None
        self._conn_lock = threading.Lock()
        self._resolved = None

    # -- lifecycle -------------------------------------------------------

    def resolve_model(self) -> tuple[Path, str]:
        """Return (weights path, offload regex), honouring "auto" for both.

        "auto" means: pick the quantisation this card can actually hold, and
        offload only the excess. A fixed regex written for one card is wrong on
        every other one (BUILD_NOTES §9j), so both are computed unless the
        operator has pinned them in config.json.
        """
        from . import hardware

        # Cached: this runs nvidia-smi and logs, and is called from both
        # command() and start().
        if getattr(self, "_resolved", None) is not None:
            return self._resolved

        requested = self.job.model_key or self.llm.get("model") or "auto"
        if requested not in hardware.BY_KEY and requested not in ("auto", ""):
            # An explicit path in config.json still wins.
            path = config.resolve(requested)
            regex = self.llm.get("cpu_ffn_regex") or ""
            self._resolved = (path, "" if regex == "auto" else regex)
            return self._resolved

        vram = hardware.detect_vram_mb()
        key = hardware.resolve_key(requested, vram)
        path = hardware.model_path(key)
        regex = self.llm.get("cpu_ffn_regex")
        if regex == "auto" or regex is None:
            regex = hardware.offload_regex(key, vram)
        self.job.log(
            "llm: %s | %d MiB VRAM | %s"
            % (path.name, vram,
               ("%d layers to CPU" % (regex.count("|") + 1)) if regex else "fully on GPU")
        )
        # Record what was actually chosen. Calibration is keyed by model, and a
        # job that never went through the dropdown -- one reopened from the
        # history list, say -- would otherwise be timed against the wrong one.
        self.job.model_key = key
        self._resolved = (path, regex)
        return self._resolved

    def command(self) -> list[str]:
        model, _regex = self.resolve_model()
        cmd = [
            str(config.LLAMA_SERVER),
            "-m", str(model),
            "--ctx-size", str(int(self.llm.get("ctx_size", 32768))),
            "--n-gpu-layers", str(int(self.llm.get("gpu_layers", 99))),
            "--flash-attn", "on",
            "--cache-type-k", str(self.llm.get("cache_type_k", "q8_0")),
            "--cache-type-v", str(self.llm.get("cache_type_v", "q8_0")),
            "--jinja",
            "--threads", str(int(self.llm.get("threads", 8))),
            "--batch-size", "512", "--ubatch-size", "512",
            "--parallel", "1",
            "--host", "127.0.0.1", "--port", str(self.port),
            "--no-webui",
        ]
        if _regex:
            # A 27B model does not fit alongside a 32k KV cache on a small
            # card. This pushes the upper FFN tensors into system RAM and keeps
            # attention on the GPU; how many layers is computed for this card.
            cmd += ["--override-tensor", _regex]
        return cmd

    def start(self) -> None:
        model, _ = self.resolve_model()
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

    def _stream(self, payload: dict, idle_timeout: float, on_token) -> tuple[str, str, int]:
        """POST a streaming completion, returning (content, reasoning, tokens).

        Streaming is not for show. A non-streaming call needs a single total
        timeout covering the whole generation, and there is no good value for
        it: the final reduce can legitimately emit 8000 tokens, which at the
        1.8 tok/s this hardware manages is over an hour, while a genuinely hung
        server should be caught in minutes. Streaming replaces that guess with
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

        return "".join(content), str(reasoning_chars), tokens

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
                "temperature": 0.7, "top_p": 0.8, "top_k": 20,
                "presence_penalty": 1.5, "repeat_penalty": 1.0,
            }
            kwargs = {"enable_thinking": False}

        payload = {
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": int(max_tokens),
            "chat_template_kwargs": kwargs,
            **sampling,
        }
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
            #               8000 and a document is 2300, so the bar crawls to a
            #               third and leaps. The measured figure tracks the real
            #               completion closely and reaches the end with it.
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
                text, reasoning_chars, tokens = self._stream(payload, idle, on_token)
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
                last = "empty completion"
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
