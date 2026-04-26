(function(){
  const state = { date: todayStr(), view: "journal" };

  function todayStr() {
    const d = new Date();
    const pad = n => String(n).padStart(2, "0");
    return `${d.getFullYear()}-${pad(d.getMonth()+1)}-${pad(d.getDate())}`;
  }

  function fmtTime(iso) {
    try {
      const d = new Date(iso);
      return d.toLocaleTimeString([], { hour: "numeric", minute: "2-digit" });
    } catch(e) { return iso || ""; }
  }

  function fmtRelDate(iso) {
    try {
      const d = new Date(iso);
      const today = new Date();
      today.setHours(0,0,0,0);
      const dd = new Date(d); dd.setHours(0,0,0,0);
      const diffDays = Math.round((today - dd) / 86400000);
      if (diffDays === 0) return "today";
      if (diffDays === 1) return "yesterday";
      if (diffDays < 7) return d.toLocaleDateString([], { weekday: "long" }).toLowerCase();
      return d.toLocaleDateString([], { month: "short", day: "numeric" });
    } catch(e) { return iso || ""; }
  }

  function el(tag, attrs, children) {
    const node = document.createElement(tag);
    if (attrs) for (const k in attrs) {
      if (k === "class") node.className = attrs[k];
      else if (k === "html") node.innerHTML = attrs[k];
      else if (k.startsWith("on")) node.addEventListener(k.slice(2), attrs[k]);
      else node.setAttribute(k, attrs[k]);
    }
    if (children) for (const c of [].concat(children)) {
      if (c == null) continue;
      node.appendChild(typeof c === "string" ? document.createTextNode(c) : c);
    }
    return node;
  }

  // Tiny markdown renderer that builds DOM nodes instead of assigning HTML.
  // Important because this renders LLM-produced text that could contain
  // prompt-injected markup; building via textContent kills the XSS path.
  // Supported: **bold**, `- ` bullets, blank-line paragraphs.
  function renderMarkdown(src) {
    const root = document.createElement("div");
    const lines = String(src || "").split("\n");
    let currentList = null;
    let paraBuf = [];

    function flushPara() {
      if (!paraBuf.length) return;
      const p = document.createElement("div");
      for (const frag of parseInline(paraBuf.join(" "))) {
        p.appendChild(frag);
      }
      root.appendChild(p);
      paraBuf = [];
    }

    for (const raw of lines) {
      const ln = raw;
      if (/^\s*-\s+/.test(ln)) {
        flushPara();
        if (!currentList) {
          currentList = document.createElement("ul");
          root.appendChild(currentList);
        }
        const li = document.createElement("li");
        for (const frag of parseInline(ln.replace(/^\s*-\s+/, ""))) {
          li.appendChild(frag);
        }
        currentList.appendChild(li);
      } else if (ln.trim() === "") {
        flushPara();
        currentList = null;
      } else {
        if (currentList) currentList = null;
        paraBuf.push(ln);
      }
    }
    flushPara();
    return root;
  }

  // Inline parser — handles **bold**, emits an array of Node children.
  // All non-marker text goes through createTextNode, so no HTML interpretation.
  function parseInline(s) {
    const nodes = [];
    const re = /\*\*(.+?)\*\*/g;
    let i = 0, m;
    while ((m = re.exec(s)) !== null) {
      if (m.index > i) {
        nodes.push(document.createTextNode(s.slice(i, m.index)));
      }
      const strong = document.createElement("strong");
      strong.textContent = m[1];
      nodes.push(strong);
      i = m.index + m[0].length;
    }
    if (i < s.length) {
      nodes.push(document.createTextNode(s.slice(i)));
    }
    return nodes;
  }

  // ---------- Tab switching ----------
  document.querySelectorAll(".tab").forEach(tab => {
    tab.addEventListener("click", () => switchTab(tab.dataset.view));
  });

  function switchTab(view) {
    state.view = view;
    document.querySelectorAll(".tab").forEach(t => {
      t.classList.toggle("active", t.dataset.view === view);
    });
    document.getElementById("journal_view").style.display = view === "journal" ? "" : "none";
    document.getElementById("queue_view").style.display = view === "queue" ? "" : "none";
    document.getElementById("patterns_view").style.display = view === "patterns" ? "" : "none";
    document.getElementById("journal_nav").style.visibility = view === "journal" ? "" : "hidden";
    // Clear stale search on tab-switch
    if (view !== "journal") {
      const si = document.getElementById("search_input");
      const sc = document.getElementById("search_clear");
      if (si) si.value = "";
      if (sc) sc.style.display = "none";
    }
    if (view === "queue") loadQueue();
    else if (view === "patterns") loadPatterns();
    else load(state.date);
  }

  // ---------- Journal (existing) ----------
  async function load(date) {
    state.date = date;
    document.getElementById("datepicker").value = date;
    const r = await fetch(`/api/day?date=${date}`);
    const data = await r.json();
    render(data);
  }

  function render(data) {
    document.getElementById("daylabel").textContent = data.day_label;
    const s = data.stats;
    document.getElementById("stats").textContent =
      `${s.transcriptions} transcription${s.transcriptions===1?"":"s"} · ${s.reflections} reflection${s.reflections===1?"":"s"}`;

    document.getElementById("prev").disabled = !data.prev_day;
    document.getElementById("prev").onclick = () => data.prev_day && load(data.prev_day);
    document.getElementById("next").disabled = !data.next_day;
    document.getElementById("next").onclick = () => data.next_day && load(data.next_day);
    document.getElementById("today").onclick = () => load(todayStr());
    document.getElementById("datepicker").onchange = (e) => {
      if (e.target.value) load(e.target.value);
    };

    const summarySlot = document.getElementById("summary_slot");
    summarySlot.innerHTML = "";
    if (data.entries.length > 0) {
      if (data.summary_enabled) {
        if (data.summary && data.summary.summary) {
          const meta = el("div", {class:"meta"},[
            `Summary · ${data.summary.provider} · ${shortModel(data.summary.model)}`,
            el("button", {class:"refresh", onclick: () => regen(data.date)}, "Refresh"),
          ]);
          const body = el("div", {class:"body"}, data.summary.summary);
          summarySlot.appendChild(el("div", {class:"summary"}, [meta, body]));
        } else {
          const label = data.summary_model
            ? `Generate a summary? · ${data.summary_provider} · ${shortModel(data.summary_model)}`
            : "Generate a summary for this day?";
          const msg = el("span", null, label);
          const btn = el("button", {onclick: () => regen(data.date)}, "Generate");
          summarySlot.appendChild(el("div", {class:"generate"}, [msg, btn]));
        }
      } else {
        const hint = el("div", {class:"generate"}, [
          el("span", null, [
            "💡 Summaries are off. Enable them by adding ",
            el("code", {class:"inline"}, '"summaries": {"provider": "local"}'),
            " to ~/.voiceclip/config.json — then restart ",
            el("code", {class:"inline"}, "voiceclip view"),
            ".",
          ]),
        ]);
        summarySlot.appendChild(hint);
      }
    }

    const appsSlot = document.getElementById("apps_slot");
    appsSlot.innerHTML = "";
    if (s.apps && s.apps.length) {
      const max = Math.max(...s.apps.map(a => a.count));
      const list = el("div", {class:"apps"}, [
        el("h3", null, "Where your voice went"),
        ...s.apps.map(a => {
          const pct = Math.round((a.count / max) * 100);
          return el("div", {class:"appbar"}, [
            el("div", {class:"name"}, a.name),
            el("div", {class:"bar"}, el("span", {style:`width:${pct}%`})),
            el("div", {class:"count"}, String(a.count)),
          ]);
        }),
      ]);
      appsSlot.appendChild(list);
    }

    const entriesSlot = document.getElementById("entries_slot");
    entriesSlot.innerHTML = "";
    if (!data.entries.length) {
      entriesSlot.appendChild(el("div", {class:"empty"}, "Nothing captured on this day."));
      return;
    }
    const reflections = data.entries.filter(e => e.kind === "reflection");
    const transcriptions = data.entries.filter(e => e.kind === "transcription");

    if (reflections.length) {
      entriesSlot.appendChild(el("h3", null, "Reflections"));
      reflections.forEach(e => entriesSlot.appendChild(renderEntry(e)));
    }
    if (transcriptions.length) {
      const details = el("details", {class:"transcriptions"});
      details.appendChild(el("summary", null,
        `Show ${transcriptions.length} transcription${transcriptions.length===1?"":"s"}`));
      transcriptions.forEach(e => details.appendChild(renderEntry(e)));
      entriesSlot.appendChild(details);
    }
  }

  function renderEntry(e) {
    const copyBtn = el("button", {onclick: ev => {
      const node = ev.target.closest(".entry").querySelector(".text");
      navigator.clipboard.writeText(node.textContent).then(() => flash(ev.target, "Copied"));
    }}, "Copy");
    const actions = [copyBtn];
    const wrap = el("div", {class:`entry ${e.kind}`});
    wrap.dataset.entryId = String(e.id);
    if (e.kind === "transcription") {
      const promoteBtn = el("button", {onclick: async ev => {
        const r = await fetch("/api/promote", {
          method: "POST",
          headers: {"Content-Type":"application/json"},
          body: JSON.stringify({id: e.id}),
        });
        if (r.ok) {
          flash(ev.target, "Promoted");
          setTimeout(() => fadeOutAndReconcile(wrap), 500);
        } else flash(ev.target, "Failed");
      }}, "💭 Keep");
      actions.push(promoteBtn);
    }
    const deleteBtn = el("button", {class: "danger", onclick: ev => handleDelete(ev.target, wrap, e)}, "Delete");
    actions.push(deleteBtn);

    const text = el("div", {class:"text", contenteditable: "true", spellcheck: "true"}, e.text);
    wireInlineEdit(text, e);

    const metaParts = [
      `${e.kind === "reflection" ? "💭" : "📝"} ${fmtTime(e.timestamp)}`,
      e.duration ? `· ${e.duration}s` : null,
      e.app_name ? `· ${e.app_name}` : null,
    ];
    const meta = el("div", {class:"meta"}, [
      ...metaParts,
      e.edited_at ? el("span", {class:"edited"}, "· edited") : null,
      el("div", {class:"actions"}, actions),
    ]);
    wrap.appendChild(text);
    wrap.appendChild(meta);
    return wrap;
  }

  // Inline editing — auto-save on blur, Esc to cancel, Cmd+Enter to save.
  function wireInlineEdit(node, entry) {
    node.dataset.original = entry.text;
    // Prevent pasted HTML — always insert as plain text.
    node.addEventListener("paste", (ev) => {
      ev.preventDefault();
      const text = (ev.clipboardData || window.clipboardData).getData("text/plain");
      document.execCommand("insertText", false, text);
    });
    node.addEventListener("keydown", (ev) => {
      if (ev.key === "Escape") {
        node.textContent = node.dataset.original;
        node.blur();
        ev.preventDefault();
      } else if ((ev.metaKey || ev.ctrlKey) && ev.key === "Enter") {
        node.blur();
        ev.preventDefault();
      }
    });
    node.addEventListener("blur", async () => {
      const newText = node.textContent.trim();
      const oldText = node.dataset.original;
      if (!newText || newText === oldText) {
        if (!newText) node.textContent = oldText;  // don't allow empty
        return;
      }
      try {
        const r = await fetch("/api/update", {
          method: "POST",
          headers: {"Content-Type":"application/json"},
          body: JSON.stringify({id: entry.id, text: newText}),
        });
        if (!r.ok) {
          node.textContent = oldText;
          return;
        }
        node.dataset.original = newText;
        // Show the "edited" marker if the meta line doesn't have it yet.
        const metaLine = node.parentElement.querySelector(".meta");
        if (metaLine && !metaLine.querySelector(".edited")) {
          const actions = metaLine.querySelector(".actions");
          metaLine.insertBefore(
            el("span", {class:"edited"}, "· edited"),
            actions,
          );
        }
        // Little saved pill flash inside the text node.
        const pill = el("span", {class:"saved-pill"}, "saved");
        node.appendChild(pill);
        setTimeout(() => pill.remove(), 1200);
      } catch(e) {
        node.textContent = oldText;
      }
    });
  }

  function fadeOutAndReconcile(node) {
    node.classList.add("removing");
    setTimeout(() => {
      const parent = node.parentNode;
      node.remove();
      if (parent && parent.children && parent.children.length === 0 &&
          (parent.tagName === "SECTION" || parent.tagName === "DETAILS")) {
        parent.remove();
      }
      refreshCounts();
    }, 360);
  }

  async function refreshCounts() {
    try {
      const r = await fetch(`/api/day?date=${state.date}`);
      const data = await r.json();
      const s = data.stats;
      document.getElementById("stats").textContent =
        `${s.transcriptions} transcription${s.transcriptions===1?"":"s"} · ${s.reflections} reflection${s.reflections===1?"":"s"}`;
      const details = document.querySelector("details.transcriptions summary");
      if (details) {
        const remaining = document.querySelectorAll(".entry.transcription").length;
        details.textContent = `Show ${remaining} transcription${remaining===1?"":"s"}`;
      }
      if (data.entries.length === 0) load(state.date);
    } catch(e) { /* best effort */ }
  }

  function handleDelete(btn, node, entry) {
    if (btn.dataset.armed === "1") {
      btn.dataset.armed = "";
      fetch("/api/delete", {
        method: "POST",
        headers: {"Content-Type":"application/json"},
        body: JSON.stringify({id: entry.id}),
      }).then(r => {
        if (r.ok) fadeOutAndReconcile(node);
        else flash(btn, "Failed");
      });
      return;
    }
    btn.dataset.armed = "1";
    const prev = btn.textContent;
    btn.textContent = "Really delete?";
    btn.classList.add("armed");
    setTimeout(() => {
      if (btn.dataset.armed === "1") {
        btn.dataset.armed = "";
        btn.textContent = prev;
        btn.classList.remove("armed");
      }
    }, 3000);
  }

  function flash(btn, label) {
    const prev = btn.textContent;
    btn.textContent = label;
    btn.classList.add("flash");
    setTimeout(() => { btn.textContent = prev; btn.classList.remove("flash"); }, 900);
  }

  async function regen(date) {
    const slot = document.getElementById("summary_slot");
    slot.innerHTML = `<div class="generate"><span><span class="spinner"></span>Thinking — local models take 15-60 seconds on first run…</span></div>`;
    try {
      const r = await fetch("/api/summarize", {
        method: "POST",
        headers: {"Content-Type":"application/json"},
        body: JSON.stringify({date, force: true}),
      });
      const data = await r.json();
      if (!r.ok) {
        slot.innerHTML = `<div class="generate"><span style="color:#c44; white-space:pre-line">${escapeHtml(data.error || "failed")}</span></div>`;
        return;
      }
      load(date);
    } catch(e) {
      slot.innerHTML = `<div class="generate"><span style="color:#c44">${escapeHtml(e.message)}</span></div>`;
    }
  }

  function escapeHtml(s) {
    return String(s).replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;");
  }

  function shortModel(id) {
    if (!id) return "";
    const slash = id.lastIndexOf("/");
    return slash >= 0 ? id.slice(slash + 1) : id;
  }

  // ---------- Queue (new) ----------

  async function loadQueue() {
    const r = await fetch("/api/queue");
    const data = await r.json();
    renderQueue(data);
  }

  function renderQueue(data) {
    const topics = data.topics || [];
    const hint = document.getElementById("queue_hint");
    if (!data.research_enabled) {
      hint.innerHTML = `💡 Research is off. Enable it by adding <code class="inline">"research": {"provider": "openai"}</code> to ~/.voiceclip/config.json and exporting <code class="inline">OPENAI_API_KEY</code>.`;
    } else {
      hint.innerHTML = `Using ${data.research_provider} · ${shortModel(data.research_model)}. The model decides whether to search the web per topic.`;
    }
    document.getElementById("queue_stats").textContent =
      `${topics.length} topic${topics.length===1?"":"s"}`;

    const slot = document.getElementById("topics_slot");
    slot.innerHTML = "";
    if (!topics.length) {
      slot.appendChild(el("div", {class:"empty"}, "No research topics yet. Add one above."));
      return;
    }
    topics.forEach(t => slot.appendChild(renderTopic(t, data.research_enabled)));
  }

  function renderTopic(t, researchEnabled) {
    const wrap = el("div", {class:`topic ${t.status}`});
    wrap.dataset.topicId = String(t.id);

    const actions = [];
    if (t.status === "pending" || t.status === "failed") {
      const btn = el("button", {class:"primary",
        disabled: researchEnabled ? null : "disabled",
        onclick: ev => runResearch(wrap, t.id, ev.target)}, "Research");
      actions.push(btn);
    } else if (t.status === "running") {
      actions.push(el("button", {disabled: "disabled"}, "Running…"));
    } else if (t.status === "ready") {
      const btn = el("button", {
        onclick: ev => runResearch(wrap, t.id, ev.target, true)
      }, "Re-research");
      actions.push(btn);
    }
    const deleteBtn = el("button", {class:"danger", onclick: ev => handleTopicDelete(ev.target, wrap, t)}, "Delete");
    actions.push(deleteBtn);

    const head = el("div", {class:"topic-head"}, [
      el("div", {class:"title"}, t.text),
      el("div", {class:"actions"}, actions),
    ]);
    wrap.appendChild(head);

    const metaBits = [
      fmtRelDate(t.timestamp),
      `· ${t.status}`,
    ];
    wrap.appendChild(el("div", {class:"topic-meta"}, metaBits.join(" ")));

    if (t.brief) {
      wrap.appendChild(renderBrief(t.brief));
    }
    return wrap;
  }

  function renderBrief(brief) {
    const body = el("div", {class:"brief"});
    body.appendChild(renderMarkdown(brief.text));
    if (brief.used_web_search) {
      const badge = el("span", {class:"web-badge"}, "web");
      body.insertBefore(badge, body.firstChild);
    }
    const wrap = el("div", null, [body]);
    if (brief.sources && brief.sources.length) {
      const src = el("div", {class:"sources"});
      src.appendChild(el("div", null, `Sources (${brief.sources.length}):`));
      brief.sources.forEach(s => {
        src.appendChild(el("a", {href: s.url, target: "_blank", rel: "noopener"}, s.title));
      });
      wrap.appendChild(src);
    }
    return wrap;
  }

  async function runResearch(wrap, id, btn, isRerun) {
    const prev = btn.textContent;
    btn.disabled = true;
    btn.textContent = "Researching…";
    // Optimistically mark the topic running
    wrap.classList.remove("pending", "failed", "ready");
    wrap.classList.add("running");
    try {
      const r = await fetch("/api/research/run", {
        method: "POST",
        headers: {"Content-Type":"application/json"},
        body: JSON.stringify({id}),
      });
      const data = await r.json();
      if (!r.ok) {
        wrap.classList.remove("running");
        wrap.classList.add("failed");
        btn.disabled = false;
        btn.textContent = prev;
        const err = el("div", {class:"brief",
          style:"color:#c44; white-space:pre-line"}, data.error || "failed");
        // Replace any existing brief with the error
        const existing = wrap.querySelector(".brief");
        if (existing) existing.parentNode.replaceChild(err, existing);
        else wrap.appendChild(err);
        return;
      }
      // Reload the whole queue so counts + status line refresh consistently
      loadQueue();
    } catch(e) {
      wrap.classList.remove("running");
      wrap.classList.add("failed");
      btn.disabled = false;
      btn.textContent = prev;
    }
  }

  function handleTopicDelete(btn, node, topic) {
    if (btn.dataset.armed === "1") {
      btn.dataset.armed = "";
      fetch("/api/delete", {
        method: "POST",
        headers: {"Content-Type":"application/json"},
        body: JSON.stringify({id: topic.id}),
      }).then(r => {
        if (r.ok) {
          node.classList.add("removing");
          setTimeout(() => { node.remove(); loadQueue(); }, 360);
        } else flash(btn, "Failed");
      });
      return;
    }
    btn.dataset.armed = "1";
    const prev = btn.textContent;
    btn.textContent = "Really delete?";
    btn.classList.add("armed");
    setTimeout(() => {
      if (btn.dataset.armed === "1") {
        btn.dataset.armed = "";
        btn.textContent = prev;
        btn.classList.remove("armed");
      }
    }, 3000);
  }

  // Topic input
  document.getElementById("topic_add").addEventListener("click", addTopic);
  document.getElementById("topic_input").addEventListener("keydown", (ev) => {
    if (ev.key === "Enter") { ev.preventDefault(); addTopic(); }
  });

  async function addTopic() {
    const input = document.getElementById("topic_input");
    const text = input.value.trim();
    if (!text) return;
    try {
      const r = await fetch("/api/research/create", {
        method: "POST",
        headers: {"Content-Type":"application/json"},
        body: JSON.stringify({text}),
      });
      if (r.ok) {
        input.value = "";
        loadQueue();
      }
    } catch(e) {}
  }

  // Auto-refresh today's page every 20s while viewing today's journal
  setInterval(() => {
    if (state.view === "journal" && state.date === todayStr()) load(state.date);
  }, 20000);

  // ---------- Patterns (new) ----------

  async function loadPatterns() {
    const hint = document.getElementById("patterns_hint");
    const slot = document.getElementById("patterns_slot");
    const stats = document.getElementById("patterns_stats");
    const r = await fetch("/api/patterns/config");
    const cfg = await r.json();
    stats.textContent = `past ${cfg.window_days} days`;
    if (!cfg.enabled) {
      hint.innerHTML = `💡 Patterns are off. Enable them by adding <code class="inline">"patterns": {"provider": "local"}</code> to ~/.voiceclip/config.json — then restart <code class="inline">voiceclip view</code>.`;
      slot.innerHTML = "";
      return;
    }
    hint.textContent = `Using ${cfg.provider} · ${shortModel(cfg.model)}. This reads your last ${cfg.window_days} days — local keeps it private, cloud sends it to the provider.`;
    slot.innerHTML = "";
    const card = el("div", {class: "patterns-generate"}, [
      el("div", null, "Generate a patterns view to see what you've been occupied with, recurring themes, and suggested things to learn."),
      el("button", {onclick: () => runPatterns()}, "Generate"),
    ]);
    slot.appendChild(card);
  }

  async function runPatterns() {
    const slot = document.getElementById("patterns_slot");
    slot.innerHTML = `<div class="patterns-generate"><span class="spinner"></span> Reading your recent entries — this can take 30-60 seconds locally…</div>`;
    try {
      const r = await fetch("/api/patterns/run", {
        method: "POST",
        headers: {"Content-Type":"application/json"},
        body: JSON.stringify({force: true}),
      });
      const data = await r.json();
      if (!r.ok) {
        slot.innerHTML = `<div class="patterns-generate" style="color:#c44; white-space:pre-line">${escapeHtml(data.error || "failed")}</div>`;
        return;
      }
      renderPatterns(data.patterns);
    } catch(e) {
      slot.innerHTML = `<div class="patterns-generate" style="color:#c44">${escapeHtml(e.message)}</div>`;
    }
  }

  function renderPatterns(p) {
    const slot = document.getElementById("patterns_slot");
    slot.innerHTML = "";

    if (p.empty) {
      slot.appendChild(el("div", {class:"empty"},
        "Nothing captured in this window. Dictate a few things and come back."));
      return;
    }

    const stats = p.stats || {};
    // Occupied with
    if (p.occupied_with) {
      const sec = el("div", {class:"pattern-section"}, [
        el("h3", null, "Occupied with"),
        el("div", {class:"pattern-body"}, p.occupied_with),
      ]);
      slot.appendChild(sec);
    }

    // Recurring themes
    if (p.themes && p.themes.length) {
      const themes = el("div", {class:"pattern-section"});
      themes.appendChild(el("h3", null, "Recurring themes"));
      p.themes.forEach(t => {
        const card = el("div", {class:"theme"}, [
          el("div", null, [
            el("span", {class:"title"}, t.title || ""),
            t.reflection_count
              ? el("span", {class:"count"}, `· ${t.reflection_count} reflection${t.reflection_count === 1 ? "" : "s"}`)
              : null,
          ]),
          t.quote ? el("div", {class:"quote"}, `"${t.quote}"`) : null,
        ]);
        themes.appendChild(card);
      });
      slot.appendChild(themes);
    }

    // Suggested to learn
    if (p.suggestions && p.suggestions.length) {
      const sug = el("div", {class:"pattern-section"});
      sug.appendChild(el("h3", null, "Suggested to learn"));
      p.suggestions.forEach(s => {
        const btn = el("button", {class:"queue-btn",
          onclick: (ev) => queueSuggestion(ev.target, s)}, "Queue it");
        const card = el("div", {class:"suggestion"}, [
          el("div", {class:"content"}, [
            el("div", {class:"topic-line"}, s.topic || ""),
            s.reason ? el("div", {class:"reason"}, s.reason) : null,
            s.grounding_quote
              ? el("div", {class:"quote"}, `"${s.grounding_quote}"`)
              : null,
          ]),
          btn,
        ]);
        sug.appendChild(card);
      });
      slot.appendChild(sug);
    }

    // Footer with window stats
    if (stats.transcription_count != null || stats.reflection_count != null) {
      slot.appendChild(el("div", {class:"queue-hint"},
        `Window: ${stats.start_date} to ${stats.end_date} · ` +
        `${stats.transcription_count || 0} transcriptions · ` +
        `${stats.reflection_count || 0} reflections · ` +
        `${p.provider || ""} ${shortModel(p.model || "")}`));
    }
    // Regen button
    slot.appendChild(el("button", {class:"queue-btn",
      onclick: () => runPatterns(),
      style: "margin-top:16px"}, "↻ Regenerate"));
  }

  async function queueSuggestion(btn, suggestion) {
    btn.disabled = true;
    btn.textContent = "Queuing…";
    try {
      const r = await fetch("/api/patterns/queue", {
        method: "POST",
        headers: {"Content-Type":"application/json"},
        body: JSON.stringify({
          topic: suggestion.topic,
          reason: suggestion.reason,
          grounding_quote: suggestion.grounding_quote,
        }),
      });
      const data = await r.json();
      if (!r.ok) {
        btn.textContent = "Failed";
        setTimeout(() => { btn.disabled = false; btn.textContent = "Queue it"; }, 1500);
        return;
      }
      btn.textContent = data.duplicate ? "Already queued ✓" : "Queued ✓";
    } catch(e) {
      btn.textContent = "Failed";
      setTimeout(() => { btn.disabled = false; btn.textContent = "Queue it"; }, 1500);
    }
  }

  // ---------- Search (journal tab, spans all days) ----------

  let _searchSeq = 0;
  const searchInput = document.getElementById("search_input");
  const searchClear = document.getElementById("search_clear");

  searchInput.addEventListener("input", debounceSearch);
  searchClear.addEventListener("click", () => {
    searchInput.value = "";
    searchClear.style.display = "none";
    load(state.date);
  });

  // Cmd/Ctrl+K focuses search
  document.addEventListener("keydown", (ev) => {
    if ((ev.metaKey || ev.ctrlKey) && ev.key.toLowerCase() === "k") {
      ev.preventDefault();
      if (state.view !== "journal") switchTab("journal");
      searchInput.focus();
      searchInput.select();
    }
  });

  function debounceSearch() {
    const term = searchInput.value.trim();
    searchClear.style.display = term ? "" : "none";
    const mySeq = ++_searchSeq;
    clearTimeout(debounceSearch._t);
    debounceSearch._t = setTimeout(() => {
      if (mySeq !== _searchSeq) return;   // superseded
      if (!term) { load(state.date); return; }
      runSearch(term, mySeq);
    }, 180);
  }

  async function runSearch(term, seq) {
    try {
      const r = await fetch(`/api/search?q=${encodeURIComponent(term)}`);
      const data = await r.json();
      if (seq !== _searchSeq) return;  // a newer search started; discard
      renderSearchResults(term, data.entries || []);
    } catch(e) { /* best effort */ }
  }

  function renderSearchResults(term, entries) {
    // Reuse the existing slots: blank the day-specific sections and replace
    // the entries list. The search header replaces the day label's stats line.
    document.getElementById("daylabel").textContent = `Results for "${term}"`;
    document.getElementById("stats").textContent =
      `${entries.length} match${entries.length === 1 ? "" : "es"}`;
    document.getElementById("summary_slot").innerHTML = "";
    document.getElementById("apps_slot").innerHTML = "";
    // Nav buttons — disable during search
    document.getElementById("prev").disabled = true;
    document.getElementById("next").disabled = true;
    document.getElementById("today").onclick = () => {
      searchInput.value = ""; searchClear.style.display = "none";
      load(todayStr());
    };

    const slot = document.getElementById("entries_slot");
    slot.innerHTML = "";
    if (!entries.length) {
      slot.appendChild(el("div", {class:"empty"}, "No matches."));
      return;
    }
    // Keep original order (fts rank, then id desc — server-side)
    entries.forEach(e => slot.appendChild(renderEntry(e)));
  }

  load(state.date);
})();
