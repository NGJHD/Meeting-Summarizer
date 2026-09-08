/* Meeting Summariser front end. No framework, no bundler, no external requests. */
(function () {
  "use strict";

  var $ = function (id) { return document.getElementById(id); };

  // `phase` says what the progress bar is currently running, which decides
  // where Cancel goes back to: a cancelled run has nothing to show, a
  // cancelled rewrite still has the finished meeting behind it.
  var state = { jobId: null, file: null, events: null, timer: null, started: 0,
                docs: {}, phase: "run" };

  /* ---------------------------------------------------------------- boot */

  // run.bat now waits for the server before opening the browser, but the page
  // can still be reloaded early or the server can be slow, so this keeps
  // retrying. It reports elapsed time and escalates the wording rather than
  // sitting on a bare spinner: an unexplained "Starting up..." is
  // indistinguishable from a hang, and users reasonably conclude the worse one.
  var bootStarted = Date.now();

  function bootStage(seconds) {
    if (seconds < 10) {
      return ["Contacting the application…", null];
    }
    if (seconds < 30) {
      return ["Still starting. The first launch takes a little longer.", null];
    }
    if (seconds < 90) {
      return ["Still starting.",
              "This is longer than usual. Check the black console window " +
              "titled “Meeting Summariser” — if it has closed or shows an " +
              "error, close this tab and run run.bat again."];
    }
    return ["The application isn’t responding.",
            "Close this tab, close the black console window, and run run.bat " +
            "again. If it keeps happening, restart the computer — a previous " +
            "run may still be holding the graphics card."];
  }

  function waitForServer(attempt) {
    var seconds = Math.floor((Date.now() - bootStarted) / 1000);
    $("boot-elapsed").textContent = seconds + "s";
    // Ramps toward 90% over ~20s so it moves without ever claiming to be done.
    $("boot-bar").style.width = Math.min(90, 8 + seconds * 4) + "%";
    var stage = bootStage(seconds);
    $("boot-msg").textContent = stage[0];
    if (stage[1]) { $("boot-detail").textContent = stage[1]; $("boot-detail").hidden = false; }
    if (seconds >= 90) { $("boot-spinner").hidden = true; $("boot-title").textContent = "Not responding"; }

    fetch("/api/health")
      .then(function (r) { return r.json(); })
      .then(function (h) {
        if (!h.ok) {
          show("connecting", false);
          show("failed", true);
          $("fail-msg").textContent =
            "These files are missing from the application folder:\n\n  " +
            h.missing.join("\n  ") +
            "\n\nThe folder may not have copied completely, or the models " +
            "have not been downloaded yet. Run DOWNLOAD_MODELS.bat, or copy " +
            "the folder again.";
          return;
        }
        $("boot-bar").style.width = "100%";
        show("connecting", false);
        loadModels();
        reattach();
      })
      .catch(function () {
        setTimeout(function () { waitForServer(attempt + 1); }, 400);
      });
  }

  function show(id, on) { $(id).hidden = !on; }

  /* ------------------------------------------------------------- reattach */

  // Closing the tab must not orphan a two-hour run, and reopening it must not
  // present a fresh upload form while the graphics card is busy -- that reads
  // as "my job is gone" and there would be no way back to Cancel.
  function reattach() {
    fetch("/api/current")
      .then(function (r) { return r.json(); })
      .then(function (d) {
        if (!d.job) { show("setup", true); loadHistory(); return; }
        state.jobId = d.job.id;
        // A rewrite that is cancelled after a reload must still go back to the
        // meeting it was rewriting, not to an empty upload form.
        state.phase = d.job.kind === "generate" ? "generate" : "run";
        state.meeting = d.job.meeting || null;
        show("progress", true);
        $("log").textContent = "";
        // The server reports elapsed with every status event, so the local
        // clock is corrected from that rather than counting from this reload.
        state.started = Date.now();
        startClock();
        $("stage-label").textContent = d.job.stage_label || "Working";
        $("pct").textContent = Math.round(d.job.percent) + "%";
        $("bar-fill").style.width = d.job.percent + "%";
        listen();
      })
      .catch(function () { show("setup", true); });
  }

  /* -------------------------------------------------------------- history */

  function loadHistory() {
    fetch("/api/history")
      .then(function (r) { return r.json(); })
      .then(function (d) {
        var list = d.meetings || [];
        var host = $("hist-list");
        host.textContent = "";
        if (!list.length) { show("history", false); return; }
        list.forEach(function (m) { host.appendChild(historyRow(m)); });
        show("history", true);
      })
      .catch(function () { show("history", false); });
  }

  $("hist-refresh").addEventListener("click", loadHistory);

  function historyRow(m) {
    var row = document.createElement("button");
    row.className = "hist-row";
    row.type = "button";

    var name = document.createElement("span");
    name.className = "hist-name";
    name.textContent = m.meeting;
    row.appendChild(name);

    var meta = document.createElement("span");
    meta.className = "hist-meta";
    var bits = [];
    if (m.duration_s) bits.push(humanClock(m.duration_s));
    if (m.documents.length) bits.push(m.documents.join(" + "));
    else bits.push("transcript only");
    if (m.named) bits.push("named");
    meta.textContent = bits.join(" · ");
    row.appendChild(meta);

    row.addEventListener("click", function () { openHistory(m.meeting, row); });
    return row;
  }

  function openHistory(meeting, row) {
    row.disabled = true;
    fetch("/api/history/open", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ meeting: meeting })
    })
      .then(function (r) { return r.json().then(function (b) { return { ok: r.ok, body: b }; }); })
      .then(function (res) {
        row.disabled = false;
        if (!res.ok) { fail(res.body.error || "That meeting couldn't be opened."); return; }
        state.jobId = res.body.job_id;
        state.meeting = res.body.meeting;
        show("setup", false); show("history", false);
        openResult(res.body.meeting);
      })
      .catch(function () { row.disabled = false; fail("That meeting couldn't be opened."); });
  }

  /* --------------------------------------------------------------- models */

  // The dropdown defaults to whatever this card can actually hold, but the
  // choice is the user's: they may know something the detection does not.
  function loadModels() {
    fetch("/api/models")
      .then(function (r) { return r.json(); })
      .then(function (d) {
        var sel = $("model");
        sel.textContent = "";
        (d.models || []).forEach(function (m) {
          var o = document.createElement("option");
          o.value = m.key;
          o.textContent = m.label + (m.available ? "" : "  (not downloaded)");
          o.disabled = !m.available;
          if (m.key === d.recommended) o.selected = true;
          sel.appendChild(o);
        });
        state.recommended = d.recommended;
        var gb = d.vram_mb ? (d.vram_mb / 1024).toFixed(1) + " GB" : "unknown";
        $("model-note").textContent =
          "Detected " + gb + " of video memory, so " +
          (d.recommended === "q4_k_m" ? "High Quality" : "Low Quality") +
          " is selected. You can change it.";
        sel.addEventListener("change", showEstimate);
      })
      .catch(function () { /* the dropdown just stays empty; config still applies */ });
  }

  /* -------------------------------------------------------------- picking */

  var drop = $("drop"), fileInput = $("file");

  drop.addEventListener("click", function () { fileInput.click(); });
  drop.addEventListener("keydown", function (e) {
    if (e.key === "Enter" || e.key === " ") { e.preventDefault(); fileInput.click(); }
  });
  ["dragenter", "dragover"].forEach(function (ev) {
    drop.addEventListener(ev, function (e) { e.preventDefault(); drop.classList.add("over"); });
  });
  ["dragleave", "drop"].forEach(function (ev) {
    drop.addEventListener(ev, function (e) { e.preventDefault(); drop.classList.remove("over"); });
  });
  drop.addEventListener("drop", function (e) {
    if (e.dataTransfer.files && e.dataTransfer.files.length) { choose(e.dataTransfer.files[0]); }
  });
  fileInput.addEventListener("change", function () {
    if (fileInput.files.length) { choose(fileInput.files[0]); }
  });

  function humanSize(n) {
    if (n >= 1e9) return (n / 1e9).toFixed(2) + " GB";
    if (n >= 1e6) return (n / 1e6).toFixed(0) + " MB";
    return (n / 1e3).toFixed(0) + " KB";
  }

  function humanClock(sec) {
    sec = Math.round(sec);
    var h = Math.floor(sec / 3600), m = Math.floor((sec % 3600) / 60);
    return h > 0 ? h + "h " + m + "m" : m + "m " + (sec % 60) + "s";
  }

  function choose(file) {
    state.file = file;
    $("setup-error").hidden = true;
    $("fi-name").textContent = file.name;
    $("fi-size").textContent = humanSize(file.size);
    $("fi-duration").textContent = "reading…";
    $("fi-estimate").textContent = "—";
    $("fi-note").textContent = "";
    show("fileinfo", true);
    $("process").disabled = true;
    upload(file);
  }

  /* ------------------------------------------------------------- uploading */

  function upload(file) {
    var xhr = new XMLHttpRequest();
    xhr.open("POST", "/api/upload?filename=" + encodeURIComponent(file.name));
    xhr.setRequestHeader("Content-Type", "application/octet-stream");

    xhr.upload.onprogress = function (e) {
      if (!e.lengthComputable) return;
      var pct = Math.round((e.loaded / e.total) * 100);
      $("fi-duration").textContent = "uploading " + pct + "%";
    };

    xhr.onload = function () {
      var body = {};
      try { body = JSON.parse(xhr.responseText); } catch (err) { /* fall through */ }
      if (xhr.status !== 200) {
        $("fi-duration").textContent = "—";
        fail(body.error || "That file couldn't be read.");
        return;
      }
      state.jobId = body.job_id;
      state.estimates = body.estimates || {};
      state.durationS = body.duration_s;
      $("fi-duration").textContent = body.duration_hms;
      showEstimate();
      $("process").disabled = false;
    };

    xhr.onerror = function () { fail("The upload didn't complete."); };
    xhr.send(file);
  }

  // "Both" adds one extra final-reduce call over the same notes, not a second
  // pipeline, so the extra cost is modest but worth mentioning.
  $("mode").addEventListener("change", showEstimate);

  function showEstimate() {
    if (!state.durationS) return;
    var mode = $("mode").value, both = mode === "both";
    var perModel = state.estimates[$("model").value] || {};
    var mins = perModel[mode];

    // Null means nothing has been measured on this machine with this model.
    // Say so. The old behaviour was a hardcoded rate that quoted 2 minutes for
    // a job that took 10, which is worse than no number at all.
    if (!mins) {
      $("fi-estimate").textContent = "not known yet";
      $("fi-note").textContent =
        "This recording is " + humanClock(state.durationS) + ". This is the " +
        "first run on this computer with this model, so there is no reliable " +
        "time estimate yet — it will be timed as it goes, and every " +
        "recording after this one will be estimated up front. Expect it to " +
        "take a while; you can leave this window open and come back.";
      return;
    }
    $("fi-estimate").textContent = "about " + mins + " minutes";
    $("fi-note").textContent =
      "This recording is " + humanClock(state.durationS) + ". Processing will take " +
      "roughly " + mins + " minutes" +
      (both ? ", producing both a summary and minutes" : "") +
      ". You can leave this window open and come back.";
  }

  function fail(msg) {
    $("setup-error").textContent = msg;
    $("setup-error").hidden = false;
  }

  /* ------------------------------------------------------------ processing */

  $("process").addEventListener("click", function () {
    if (!state.jobId) return;
    $("process").disabled = true;
    fetch("/api/jobs/" + state.jobId + "/start", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ mode: $("mode").value,
                             num_speakers: $("speakers").value,
                             model: $("model").value })
    })
      .then(function (r) { return r.json().then(function (b) { return { ok: r.ok, body: b }; }); })
      .then(function (res) {
        if (!res.ok) { $("process").disabled = false; fail(res.body.error || "Couldn't start."); return; }
        state.phase = "run";
        state.tab = null;
        show("setup", false);
        show("history", false);
        show("another", false);
        show("progress", true);
        $("log").textContent = "";
        state.started = Date.now();
        startClock();
        listen();
      })
      .catch(function () { $("process").disabled = false; fail("Couldn't start."); });
  });

  function startClock() {
    stopClock();
    state.timer = setInterval(function () {
      var s = Math.floor((Date.now() - state.started) / 1000);
      $("elapsed").textContent = Math.floor(s / 60) + ":" + ("0" + (s % 60)).slice(-2);
    }, 1000);
  }
  function stopClock() { if (state.timer) { clearInterval(state.timer); state.timer = null; } }

  function listen() {
    if (state.events) state.events.close();
    var es = new EventSource("/api/events/" + state.jobId);
    state.events = es;

    es.onmessage = function (e) {
      var ev;
      try { ev = JSON.parse(e.data); } catch (err) { return; }

      if (ev.type === "status") {
        // Trust the server's elapsed over the local clock: after a reload the
        // page has no idea when the job actually started.
        if (ev.elapsed) state.started = Date.now() - ev.elapsed * 1000;
        $("stage-label").textContent = ev.stage_label || "Working";
        $("stage-msg").textContent = ev.message || "";
        $("pct").textContent = Math.round(ev.percent) + "%";
        $("bar-fill").style.width = ev.percent + "%";
        // Live ETA from this run's own progress, which already accounts for the
        // machine, the recording and the settings. Far better than any constant.
        $("remaining").textContent =
          (ev.remaining === null || ev.remaining === undefined)
            ? "" : "about " + humanClock(ev.remaining) + " left";
      } else if (ev.type === "log") {
        var box = $("log");
        var atBottom = box.scrollHeight - box.scrollTop - box.clientHeight < 40;
        box.textContent += ev.line + "\n";
        if (atBottom) box.scrollTop = box.scrollHeight;
      } else if (ev.type === "done") {
        es.close(); stopClock(); finish(ev);
      } else if (ev.type === "error") {
        es.close(); stopClock();
        show("progress", false); show("failed", true);
        $("fail-msg").textContent = ev.message;
      } else if (ev.type === "cancelled") {
        es.close(); stopClock();
        show("progress", false);
        if (state.phase === "generate") {
          // Nothing was lost: the meeting behind this is still finished, so go
          // back to it rather than to an empty upload form.
          state.phase = "run";
          openResult(state.meeting || "Finished");
        } else {
          show("setup", true);
          $("process").disabled = true;
          show("fileinfo", false);
          state.jobId = null;
          loadHistory();
        }
      }
    };
  }

  $("cancel").addEventListener("click", function () {
    $("cancel").disabled = true;
    fetch("/api/jobs/" + state.jobId + "/cancel", { method: "POST" })
      .finally(function () { $("cancel").disabled = false; });
  });

  /* --------------------------------------------------------------- result */

  function finish(ev) {
    openResult("Finished in " + humanClock(ev.elapsed), ev.document_error);
  }

  // Shared by a completed run, a reopened meeting from the history list, and
  // the end of a rewrite -- all three land on exactly the same page.
  function openResult(title, documentError) {
    fetch("/api/jobs/" + state.jobId + "/result")
      .then(function (r) { return r.json(); })
      .then(function (docs) {
        state.docs = docs;
        show("progress", false);
        show("failed", false);
        show("result", true);
        $("result-title").textContent = title || "Finished";
        $("build-error").hidden = true;

        var note = $("result-note");
        if (documentError) {
          note.textContent = documentError +
            " The full transcript below was produced successfully.";
          note.hidden = false;
        } else {
          note.hidden = true;
        }

        // One transcript tab. When names have been applied the named version
        // is the one worth reading, so it takes the slot -- the raw labelled
        // file is still on disk, which is what it is for.
        if (docs.tagged) { docs.transcript = docs.tagged; }
        var keys = [];
        ["summary", "minutes", "transcript"].forEach(function (k) {
          if (docs[k]) keys.push(k);
        });

        var tabs = $("tabs");
        tabs.hidden = keys.length < 2;
        Array.prototype.forEach.call(tabs.querySelectorAll(".tab"), function (t) {
          t.hidden = keys.indexOf(t.dataset.target) === -1;
        });
        if (keys.length) select(keys.indexOf(state.tab) === -1 ? keys[0] : state.tab);
        buildButtons(docs);
        show("another", true);
        loadSpeakers();
      });
  }

  /* -------------------------------------------------- generate / regenerate */

  // A run produces whichever document was asked for. Wanting the other one
  // afterwards, or wanting one rewritten now that the speakers have names,
  // should cost a single final-reduce call -- not the recording again.
  function buildButtons(docs) {
    var can = !!docs._can_rebuild;
    ["summary", "minutes"].forEach(function (mode) {
      var btn = $("gen-" + mode);
      var have = !!docs[mode];
      var noun = mode === "minutes" ? "minutes" : "summary";
      btn.textContent = have ? "Regenerate the " + noun : "Generate " + noun;
      btn.disabled = !can;
    });
    if (!can) {
      $("build-note").textContent =
        "The working notes for this meeting are gone, so it can't be rebuilt.";
      return;
    }
    // Writing a document is the heaviest single step in the pipeline and on a
    // long recording it is not "a few minutes". Quote the same measured figure
    // the estimate on the first screen is built from.
    var mins = docs._rebuild_minutes;
    $("build-note").textContent = mins
      ? "About " + mins + " minute" + (mins === 1 ? "" : "s") +
        " on this machine — the model writes the whole document again. " +
        "The recording itself is not processed again."
      : "The model writes the whole document again, which on a long recording " +
        "can take a while. The recording itself is not processed again.";
  }

  Array.prototype.forEach.call(document.querySelectorAll("#build .btn"), function (btn) {
    btn.addEventListener("click", function () {
      if (!state.jobId) return;
      $("build-error").hidden = true;
      fetch("/api/jobs/" + state.jobId + "/generate", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ mode: btn.dataset.mode })
      })
        .then(function (r) { return r.json().then(function (b) { return { ok: r.ok, body: b }; }); })
        .then(function (res) {
          if (!res.ok) {
            $("build-error").textContent = res.body.error || "That couldn't be started.";
            $("build-error").hidden = false;
            return;
          }
          state.phase = "generate";
          state.tab = btn.dataset.mode;
          show("result", false); show("naming", false); show("another", false);
          show("progress", true);
          $("log").textContent = "";
          state.started = Date.now();
          startClock();
          listen();
        })
        .catch(function () {
          $("build-error").textContent = "That couldn't be started.";
          $("build-error").hidden = false;
        });
    });
  });

  /* ------------------------------------------------------- speaker naming */

  // One <audio> reused for every clip: several players on a page invites two
  // voices at once, which defeats the point of listening to tell them apart.
  var player = new Audio();
  var playingBtn = null;

  function playClip(url, btn) {
    if (playingBtn) { playingBtn.textContent = "▶"; }
    if (playingBtn === btn && !player.paused) {
      player.pause(); playingBtn = null; return;
    }
    player.src = url;
    player.play().then(function () {
      btn.textContent = "■";
      playingBtn = btn;
    }).catch(function () { btn.textContent = "▶"; });
  }
  player.addEventListener("ended", function () {
    if (playingBtn) { playingBtn.textContent = "▶"; playingBtn = null; }
  });

  function loadSpeakers() {
    fetch("/api/jobs/" + state.jobId + "/speakers")
      .then(function (r) { return r.json(); })
      .then(function (d) {
        var rows = d.speakers || [];
        if (!rows.length) { show("naming", false); return; }
        state.meeting = d.meeting;
        var host = $("speaker-rows");
        host.textContent = "";
        rows.forEach(function (row) { host.appendChild(speakerRow(d.meeting, row)); });
        show("naming", true);
      })
      .catch(function () { show("naming", false); });
  }

  function speakerRow(meeting, row) {
    var wrap = document.createElement("div");
    wrap.className = "spk-row";
    wrap.dataset.speaker = row.speaker;

    var head = document.createElement("div");
    head.className = "spk-head";

    var tag = document.createElement("span");
    tag.className = "spk-tag";
    tag.textContent = row.label;
    head.appendChild(tag);

    var input = document.createElement("input");
    input.type = "text";
    input.className = "spk-name";
    input.placeholder = "name (optional)";
    input.value = row.name || "";
    input.dataset.speaker = row.speaker;
    head.appendChild(input);

    var secs = document.createElement("span");
    secs.className = "spk-secs";
    secs.textContent = row.clips.length + " samples";
    head.appendChild(secs);

    wrap.appendChild(head);

    row.clips.forEach(function (clip) {
      var line = document.createElement("div");
      line.className = "clip";

      var btn = document.createElement("button");
      btn.className = "clip-play";
      btn.type = "button";
      btn.textContent = "▶";
      btn.title = "Play " + clip.duration + " seconds from " + clip.start_hms;
      var url = "/api/samples/" + encodeURIComponent(meeting) + "/" + clip.file;
      btn.addEventListener("click", function () { playClip(url, btn); });
      line.appendChild(btn);

      var ts = document.createElement("span");
      ts.className = "clip-ts";
      ts.textContent = clip.start_hms;
      line.appendChild(ts);

      var txt = document.createElement("span");
      txt.className = "clip-text";
      txt.textContent = clip.text;
      line.appendChild(txt);

      wrap.appendChild(line);
    });

    return wrap;
  }

  $("apply-names").addEventListener("click", function () {
    var names = {};
    Array.prototype.forEach.call(document.querySelectorAll(".spk-name"), function (i) {
      if (i.value.trim()) names[i.dataset.speaker] = i.value.trim();
    });
    if (!Object.keys(names).length) {
      $("naming-error").textContent = "Type at least one name first.";
      $("naming-error").hidden = false;
      return;
    }
    $("naming-error").hidden = true;
    $("apply-names").disabled = true;
    fetch("/api/jobs/" + state.jobId + "/names", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ names: names })
    })
      .then(function (r) { return r.json().then(function (b) { return { ok: r.ok, body: b }; }); })
      .then(function (res) {
        $("apply-names").disabled = false;
        if (!res.ok) {
          $("naming-error").textContent = res.body.error || "Couldn't apply those names.";
          $("naming-error").hidden = false;
          return;
        }
        // Substitution, not generation: the files are already rewritten, so
        // just reload them. The tab in view is kept.
        var done = $("naming-done");
        done.hidden = false;
        setTimeout(function () { done.hidden = true; }, 2500);
        state.tab = state.tab || currentTab();
        openResult($("result-title").textContent);
      })
      .catch(function () {
        $("apply-names").disabled = false;
        $("naming-error").textContent = "Couldn't apply those names.";
        $("naming-error").hidden = false;
      });
  });

  Array.prototype.forEach.call(document.querySelectorAll(".tab"), function (tab) {
    tab.addEventListener("click", function () { select(tab.dataset.target); });
  });

  function currentTab() {
    var active = document.querySelector(".tab.active");
    return active ? active.dataset.target : null;
  }

  function select(key) {
    state.tab = key;
    Array.prototype.forEach.call(document.querySelectorAll(".tab"), function (t) {
      t.classList.toggle("active", t.dataset.target === key);
    });
    var d = state.docs[key];
    $("doc").innerHTML = d ? renderMarkdown(d.markdown) : "";
  }

  // A speaker label in the transcript is the natural place to think "who is
  // that?", so clicking one jumps to that speaker's row in the naming panel
  // and puts the cursor in the box, with its clips a keypress away.
  $("doc").addEventListener("click", function (e) {
    var tag = e.target.closest ? e.target.closest(".spk") : null;
    if (!tag) return;
    var label = tag.textContent.trim();
    var rows = document.querySelectorAll("#speaker-rows .spk-row");
    for (var i = 0; i < rows.length; i++) {
      var t = rows[i].querySelector(".spk-tag");
      var n = rows[i].querySelector(".spk-name");
      if (!t) continue;
      // Match the label, or the name already typed against it -- once the
      // transcript is renamed, the label no longer appears in it.
      if (t.textContent.trim() === label || (n && n.value.trim() === label)) {
        show("naming", true);
        rows[i].scrollIntoView({ behavior: "smooth", block: "center" });
        rows[i].classList.add("flash");
        setTimeout(function (r) { return function () { r.classList.remove("flash"); }; }(rows[i]), 1200);
        if (n) n.focus();
        return;
      }
    }
  });

  $("open-folder").addEventListener("click", function () {
    fetch("/api/open-output", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ meeting: state.meeting || "" })
    });
  });

  $("again").addEventListener("click", reset);
  $("retry").addEventListener("click", reset);

  function reset() {
    if (state.events) state.events.close();
    stopClock();
    state.jobId = null; state.file = null; state.docs = {};
    state.phase = "run"; state.tab = null;
    fileInput.value = "";
    show("result", false); show("failed", false); show("progress", false);
    show("naming", false); show("another", false);
    show("fileinfo", false); show("setup", true);
    loadHistory();
    $("process").disabled = true;
    $("setup-error").hidden = true;
    $("log").textContent = "";
    $("bar-fill").style.width = "0";
    $("pct").textContent = "0%";
  }

  $("copy-diag").addEventListener("click", function () {
    fetch("/api/diagnostics")
      .then(function (r) { return r.text(); })
      .then(function (text) {
        var btn = $("copy-diag");
        var done = function () { btn.textContent = "Copied"; setTimeout(function () { btn.textContent = "Copy diagnostic info"; }, 1800); };
        if (navigator.clipboard && navigator.clipboard.writeText) {
          navigator.clipboard.writeText(text).then(done, function () { legacyCopy(text); done(); });
        } else { legacyCopy(text); done(); }
      });
  });

  function legacyCopy(text) {
    var ta = document.createElement("textarea");
    ta.value = text; document.body.appendChild(ta); ta.select();
    try { document.execCommand("copy"); } catch (e) { /* nothing to do */ }
    document.body.removeChild(ta);
  }

  /* ------------------------------------------------------------- markdown */

  function esc(s) {
    return s.replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;");
  }

  function inline(s) {
    return esc(s)
      .replace(/`([^`]+)`/g, "<code>$1</code>")
      .replace(/\*\*([^*]+)\*\*/g, "<strong>$1</strong>")
      .replace(/(^|[^*])\*([^*]+)\*/g, "$1<em>$2</em>");
  }

  function isRow(line) {
    return typeof line === "string" && /^\s*\|.*\|\s*$/.test(line);
  }

  // The ---|---|--- line under the header. Colons for alignment are accepted
  // and ignored: nothing the pipeline emits depends on them.
  function isDivider(line) {
    return typeof line === "string" && /^\s*\|[\s:|-]+\|\s*$/.test(line) &&
           line.indexOf("-") !== -1;
  }

  function cells(line) {
    var t = line.trim().replace(/^\|/, "").replace(/\|$/, "");
    return t.split("|").map(function (c) { return c.trim(); });
  }

  function table(head, body) {
    var html = '<div class="tablewrap"><table><thead><tr>';
    head.forEach(function (c) { html += "<th>" + inline(c) + "</th>"; });
    html += "</tr></thead><tbody>";
    body.forEach(function (row) {
      html += "<tr>";
      for (var c = 0; c < head.length; c++) {
        html += "<td>" + inline(row[c] === undefined ? "" : row[c]) + "</td>";
      }
      html += "</tr>";
    });
    return html + "</tbody></table></div>";
  }

  // Just enough Markdown for what the pipeline emits: headings, bullets,
  // checkboxes, rules, tables, and the [SPEAKER_nn] (hh:mm:ss) transcript form.
  function renderMarkdown(md) {
    var out = [], list = null;
    var lines = md.split(/\r?\n/);

    function closeList() { if (list) { out.push("</" + list + ">"); list = null; } }

    for (var i = 0; i < lines.length; i++) {
      var line = lines[i];

      // Accepts a real name as well as SPEAKER_nn: once the transcript has
      // been renamed the labels are gone, and it should still render as speech.
      var turn = line.match(/^\[([^\]]{1,60})\]\s*\((\d{2}:\d{2}:\d{2})\)\s*(.*)$/);
      if (turn) {
        closeList();
        out.push('<p class="turn"><span class="spk" title="Name this speaker">' +
                 esc(turn[1]) + '</span> ' +
                 '<span class="ts">(' + turn[2] + ')</span> ' + inline(turn[3]) + "</p>");
        continue;
      }
      var plain = line.match(/^\((\d{2}:\d{2}:\d{2})\)\s*(.*)$/);
      if (plain) {
        closeList();
        out.push('<p class="turn"><span class="ts">(' + plain[1] + ')</span> ' + inline(plain[2]) + "</p>");
        continue;
      }

      // Markdown tables. The reduce prompts ask for a Participants section
      // with speaker, inferred name and confidence, and the model quite
      // reasonably answers with a table -- which rendered as a stack of
      // pipe-laden paragraphs until this existed.
      if (isRow(line) && isDivider(lines[i + 1])) {
        closeList();
        var head = cells(line);
        var body = [];
        i += 2;
        while (i < lines.length && isRow(lines[i])) { body.push(cells(lines[i])); i++; }
        i--;
        out.push(table(head, body));
        continue;
      }

      var h = line.match(/^(#{1,4})\s+(.*)$/);
      if (h) { closeList(); out.push("<h" + h[1].length + ">" + inline(h[2]) + "</h" + h[1].length + ">"); continue; }

      if (/^---+\s*$/.test(line)) { closeList(); out.push("<hr>"); continue; }

      var task = line.match(/^\s*[-*]\s+\[( |x|X)\]\s+(.*)$/);
      if (task) {
        if (list !== "ul") { closeList(); out.push("<ul>"); list = "ul"; }
        out.push("<li><input type=\"checkbox\" disabled" +
                 (task[1].toLowerCase() === "x" ? " checked" : "") + "> " + inline(task[2]) + "</li>");
        continue;
      }

      var bullet = line.match(/^\s*[-*]\s+(.*)$/);
      if (bullet) {
        if (list !== "ul") { closeList(); out.push("<ul>"); list = "ul"; }
        out.push("<li>" + inline(bullet[1]) + "</li>");
        continue;
      }

      var num = line.match(/^\s*\d+\.\s+(.*)$/);
      if (num) {
        if (list !== "ol") { closeList(); out.push("<ol>"); list = "ol"; }
        out.push("<li>" + inline(num[1]) + "</li>");
        continue;
      }

      if (line.trim() === "") { closeList(); continue; }

      closeList();
      out.push("<p>" + inline(line) + "</p>");
    }
    closeList();
    return out.join("\n");
  }

  /* ----------------------------------------------------------- about */

  // The only outbound request the app ever makes lives behind this button.
  // Everything about the check goes through the server: the page never talks
  // to an external host, so section 16's rule about the frontend holds.

  var about = { url: "", timer: null };

  function openAbout() {
    show("about", true);
    resetUpdate();
    fetch("/api/about")
      .then(function (r) { return r.json(); })
      .then(function (d) {
        $("about-author").textContent = d.author || "";
        $("about-version").textContent = d.version || "";
        about.url = d.repo_url || "";
        var link = $("about-repo");
        link.textContent = (d.repo_url || "").replace(/^https:\/\//, "");
        link.href = d.repo_url || "#";
      })
      .catch(function () { $("about-version").textContent = "unknown"; });
  }

  function closeAbout() {
    show("about", false);
    if (about.timer) { clearInterval(about.timer); about.timer = null; }
  }

  function resetUpdate() {
    $("update-install").hidden = true;
    $("update-cancel").hidden = true;
    $("update-bar").hidden = true;
    $("update-msg").hidden = true;
    $("update-fill").style.width = "0";
    $("update-check").disabled = false;
    $("update-check").textContent = "Check for updates";
  }

  function updateMsg(text) {
    var el = $("update-msg");
    el.textContent = text;
    el.hidden = !text;
  }

  $("about-open").addEventListener("click", openAbout);
  $("about-close").addEventListener("click", closeAbout);
  $("about").addEventListener("click", function (e) {
    if (e.target === $("about")) closeAbout();      // click the backdrop
  });
  document.addEventListener("keydown", function (e) {
    if (e.key === "Escape" && !$("about").hidden) closeAbout();
  });

  $("update-check").addEventListener("click", function () {
    $("update-check").disabled = true;
    $("update-check").textContent = "Checking…";
    $("update-install").hidden = true;
    updateMsg("");
    fetch("/api/update/check", { method: "POST" })
      .then(function (r) { return r.json(); })
      .then(function (d) {
        $("update-check").disabled = false;
        $("update-check").textContent = "Check for updates";
        if (d.status !== "available") { updateMsg(d.message || "Nothing to do."); return; }
        about.pending = d;
        updateMsg("Version " + d.latest + " is available (" + humanSize(d.size_bytes) + ").");
        if (d.writable === false) {
          updateMsg("Version " + d.latest + " is available, but this folder " +
                    "can't be written to, so it can't be installed here.");
          return;
        }
        $("update-install").hidden = false;
      })
      .catch(function () {
        $("update-check").disabled = false;
        $("update-check").textContent = "Check for updates";
        updateMsg("Couldn't reach GitHub. Check the network connection.");
      });
  });

  $("update-install").addEventListener("click", function () {
    var d = about.pending;
    if (!d) return;
    $("update-install").hidden = true;
    $("update-check").disabled = true;
    $("update-cancel").hidden = false;
    $("update-bar").hidden = false;
    fetch("/api/update/install", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ url: d.url, size_bytes: d.size_bytes, tag: d.latest })
    })
      .then(function (r) { return r.json().then(function (b) { return { ok: r.ok, body: b }; }); })
      .then(function (res) {
        if (!res.ok) { resetUpdate(); updateMsg(res.body.error || "Couldn't start."); return; }
        pollUpdate();
      })
      .catch(function () { resetUpdate(); updateMsg("Couldn't start the update."); });
  });

  $("update-cancel").addEventListener("click", function () {
    fetch("/api/update/cancel", { method: "POST" });
  });

  function pollUpdate() {
    if (about.timer) clearInterval(about.timer);
    about.timer = setInterval(function () {
      fetch("/api/update/progress")
        .then(function (r) { return r.json(); })
        .then(function (p) {
          // Bytes, not just a percentage: a bare percentage does not tell you
          // whether it is stuck.
          if (p.total) {
            $("update-fill").style.width = (p.downloaded / p.total * 100) + "%";
            updateMsg(p.message + " — " + humanSize(p.downloaded) +
                      " of " + humanSize(p.total));
          } else if (p.message) {
            updateMsg(p.message);
          }
          // Nothing left to abort once the download is done.
          if (p.phase !== "downloading") $("update-cancel").hidden = true;
          if (p.phase === "restarting") {
            clearInterval(about.timer); about.timer = null;
            $("update-fill").style.width = "100%";
            updateMsg("Update ready. The application will close and reopen by " +
                      "itself — this page will reconnect.");
          } else if (p.phase === "error") {
            clearInterval(about.timer); about.timer = null;
            resetUpdate();
            updateMsg(p.message);
          } else if (p.phase === "idle") {
            clearInterval(about.timer); about.timer = null;
            resetUpdate();
            updateMsg("Update cancelled. Nothing was changed.");
          }
        })
        .catch(function () { /* the server may already be restarting */ });
    }, 500);
  }

  waitForServer(0);
})();
