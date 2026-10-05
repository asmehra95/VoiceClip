// @ts-check
/**
 * VoiceClip viewer frontend. All UI logic lives here — data shape comes
 * from voiceclip/viewer.py's JSON endpoints.
 *
 * Type-checked via `tsc --noEmit --project jsconfig.json`. Editors with
 * TypeScript support (VS Code, etc.) surface problems inline. The checker
 * is lenient (strict: false, noImplicitAny: false) — we catch "you called
 * .value on a plain HTMLElement" class mistakes without forcing full
 * annotation coverage.
 *
 * Common idioms:
 *   - $id(...) is a typed getElementById that throws on missing ids, so
 *     downstream code can treat the result as HTMLInputElement etc.
 *   - renderMarkdown and parseInline build Text/Element nodes only,
 *     never assigning innerHTML to LLM output (XSS defense).
 *   - _t properties stashed on DOM nodes hold setTimeout IDs for
 *     debouncing; typed via the TimedEl typedef so the checker accepts.
 */

/** @typedef {HTMLElement & { _t?: number }} TimedEl */

(function(){
  /**
   * Typed getElementById. Throws on missing ids — every call site
   * depends on a DOM node the static HTML shell guarantees, so a
   * missing id means someone edited the HTML without updating JS.
   * @param {string} id
   * @returns {HTMLElement}
   */
  function $id(id) {
    const n = document.getElementById(id);
    if (!n) throw new Error(`expected #${id} in DOM`);
    return n;
  }

  /** @param {string} id @returns {HTMLInputElement} */
  function $input(id) { return /** @type {HTMLInputElement} */ ($id(id)); }

  /** @param {string} id @returns {HTMLButtonElement} */
  function $button(id) { return /** @type {HTMLButtonElement} */ ($id(id)); }

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
      // .getTime() on both sides so TS sees a numeric subtraction; the
      // Date coercion in `(today - dd)` works at runtime but the checker
      // rightly flags it.
      const diffDays = Math.round((today.getTime() - dd.getTime()) / 86400000);
      if (diffDays === 0) return "today";
      if (diffDays === 1) return "yesterday";
      if (diffDays < 7) return d.toLocaleDateString([], { weekday: "long" }).toLowerCase();
      return d.toLocaleDateString([], { month: "short", day: "numeric" });
    } catch(e) { return iso || ""; }
  }

  function el(tag, attrs, children) {
    const node = document.createElement(tag);
    if (attrs) for (const k in attrs) {
      const v = attrs[k];
      // Null/undefined means "skip this attribute" — matches React-ish
      // ergonomics and fixes the `disabled: condition ? null : "disabled"`
      // idiom, which used to silently set disabled="null" (truthy to the
      // browser) and block clicks on otherwise-live buttons.
      if (v == null) continue;
      if (k === "class") node.className = v;
      else if (k === "html") node.innerHTML = v;
      else if (k.startsWith("on")) node.addEventListener(k.slice(2), v);
      else node.setAttribute(k, v);
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
    const t = /** @type {HTMLElement} */ (tab);
    tab.addEventListener("click", () => switchTab(t.dataset.view));
  });

  function switchTab(view) {
    state.view = view;
    document.querySelectorAll(".tab").forEach(t => {
      t.classList.toggle("active", /** @type {HTMLElement} */ (t).dataset.view === view);
    });
    $id("journal_view").style.display = view === "journal" ? "" : "none";
    $id("queue_view").style.display = view === "queue" ? "" : "none";
    $id("patterns_view").style.display = view === "patterns" ? "" : "none";
    $id("settings_view").style.display = view === "settings" ? "" : "none";
    $id("agent_view").style.display = view === "agent" ? "" : "none";
    $id("vocab_view").style.display = view === "vocab" ? "" : "none";
    $id("stats_view").style.display = view === "stats" ? "" : "none";
    $id("journal_nav").style.visibility = view === "journal" ? "" : "hidden";
    // Clear stale search on tab-switch
    if (view !== "journal") {
      const si = $input("search_input");
      const sc = $id("search_clear");
      si.value = "";
      sc.style.display = "none";
    }
    if (view !== "agent") agentDeactivate();
    if (view === "queue") loadQueue();
    else if (view === "patterns") loadPatterns();
    else if (view === "settings") loadSettings();
    else if (view === "agent") agentActivate();
    else if (view === "vocab") loadVocab();
    else if (view === "stats") loadStats();
    else load(state.date);
  }

  // ---------- Journal (existing) ----------
  async function load(date) {
    state.date = date;
    $input("datepicker").value = date;
    const r = await fetch(`/api/day?date=${date}`);
    const data = await r.json();
    render(data);
  }

  function render(data) {
    $id("daylabel").textContent = data.day_label;
    const s = data.stats;
    $id("stats").textContent =
      `${s.transcriptions} transcription${s.transcriptions===1?"":"s"} · ${s.reflections} reflection${s.reflections===1?"":"s"}`;

    $button("prev").disabled = !data.prev_day;
    $button("prev").onclick = () => data.prev_day && load(data.prev_day);
    $button("next").disabled = !data.next_day;
    $button("next").onclick = () => data.next_day && load(data.next_day);
    $button("today").onclick = () => load(todayStr());
    $input("datepicker").onchange = (e) => {
      const t = /** @type {HTMLInputElement} */ (e.target);
      if (t.value) load(t.value);
    };

    const summarySlot = $id("summary_slot");
    summarySlot.innerHTML = "";
    if (data.entries.length > 0) {
      if (data.summary_enabled) {
        if (data.summary && data.summary.summary) {
          // Render as a collapsed <details> so the summary text doesn't
          // take up space by default. A one-line preview lives in the
          // <summary> line so the user knows what's inside without
          // expanding. Matches the research-brief + archived-topics
          // collapsible pattern already in the UI.
          const previewText = previewOf(data.summary.summary);
          const summaryEl = el("details", {class: "summary-details"});
          summaryEl.appendChild(el("summary", null, [
            el("span", {class: "summary-label"}, "Summary"),
            el("span", {class: "summary-meta"},
              `${data.summary.provider} · ${shortModel(data.summary.model)}`),
            el("span", {class: "summary-preview"}, previewText),
            el("button", {
              class: "refresh",
              // Stop the click from toggling the <details> — refreshing
              // shouldn't expand or collapse the panel.
              onclick: (ev) => { ev.preventDefault(); ev.stopPropagation(); regen(data.date); },
            }, "Refresh"),
          ]));
          summaryEl.appendChild(el("div", {class: "body"}, data.summary.summary));
          summarySlot.appendChild(summaryEl);
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

      // Timeline block — same collapsed-by-default pattern as the
      // summary, rendered right below it. Shares the summaries.*
      // provider/model config, so we only render when summaries are
      // enabled (mirrors the summary gating above).
      if (data.summary_enabled) {
        if (data.timeline && data.timeline.timeline) {
          const tlPreview = previewOf(data.timeline.timeline);
          const tlEl = el("details", {class: "summary-details timeline-details"});
          tlEl.appendChild(el("summary", null, [
            el("span", {class: "summary-label"}, "Timeline"),
            el("span", {class: "summary-meta"},
              `${data.timeline.provider} · ${shortModel(data.timeline.model)}`),
            el("span", {class: "summary-preview"}, tlPreview),
            el("button", {
              class: "refresh",
              onclick: (ev) => {
                ev.preventDefault();
                ev.stopPropagation();
                regenTimeline(data.date);
              },
            }, "Refresh"),
          ]));
          tlEl.appendChild(el("div", {class: "body"}, data.timeline.timeline));
          summarySlot.appendChild(tlEl);
        } else {
          // No timeline cached — offer to generate one. Different copy
          // from the summary's generate affordance so users understand
          // this is a separate, optional thing.
          const label = data.summary_model
            ? `Generate a chronological timeline? · ${data.summary_provider} · ${shortModel(data.summary_model)}`
            : "Generate a chronological timeline for this day?";
          const msg = el("span", null, label);
          const btn = el("button", {onclick: () => regenTimeline(data.date)}, "Generate");
          summarySlot.appendChild(el("div", {class:"generate timeline-generate"}, [msg, btn]));
        }
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

  // Wire the standard contenteditable behaviors onto a node:
  //   - paste inserts plain text (no HTML)
  //   - Esc restores the previous value and blurs
  //   - Cmd/Ctrl+Enter blurs (callers wire save-on-blur)
  //
  // `getOriginal()` is a callback so callers can pull the latest value
  // (e.g. from `node.dataset.original`, which they set after each save).
  // Intentionally does NOT own the save logic — each caller's commit
  // path has slightly different UI reactions (edited-pill, topic.text
  // sync) that aren't worth parameterizing.
  function wireContentEditableKeybinds(node, getOriginal) {
    node.addEventListener("paste", (ev) => {
      ev.preventDefault();
      /** @type {any} */
      const win = window;  // window.clipboardData is a legacy IE-only fallback
      const text = (ev.clipboardData || win.clipboardData).getData("text/plain");
      document.execCommand("insertText", false, text);
    });
    node.addEventListener("keydown", (ev) => {
      if (ev.key === "Escape") {
        node.textContent = getOriginal();
        node.blur();
        ev.preventDefault();
      } else if ((ev.metaKey || ev.ctrlKey) && ev.key === "Enter") {
        ev.preventDefault();
        node.blur();
      }
    });
  }

  // Inline editing — auto-save on blur, Esc to cancel, Cmd+Enter to save.
  function wireInlineEdit(node, entry) {
    node.dataset.original = entry.text;
    wireContentEditableKeybinds(node, () => node.dataset.original);
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

  // Two-click confirm pattern used by entry/topic/model delete buttons.
  // First click arms the button with a warning label; second click within
  // timeoutMs calls onConfirm. If no second click comes, the button reverts.
  //
  // The `confirmLabel` override lets callers like the model-delete button
  // produce a richer warning ("Delete 4.3 GB? (summaries will redownload)").
  function twoClickConfirm(btn, onConfirm, {timeoutMs = 3000, confirmLabel = "Really delete?"} = {}) {
    if (btn.dataset.armed === "1") {
      btn.dataset.armed = "";
      onConfirm();
      return;
    }
    btn.dataset.armed = "1";
    const prev = btn.textContent;
    btn.textContent = confirmLabel;
    btn.classList.add("armed");
    setTimeout(() => {
      if (btn.dataset.armed === "1") {
        btn.dataset.armed = "";
        btn.textContent = prev;
        btn.classList.remove("armed");
      }
    }, timeoutMs);
  }

  function handleDelete(btn, node, entry) {
    twoClickConfirm(btn, () => {
      fetch("/api/delete", {
        method: "POST",
        headers: {"Content-Type":"application/json"},
        body: JSON.stringify({id: entry.id}),
      }).then(r => {
        if (r.ok) fadeOutAndReconcile(node);
        else flash(btn, "Failed");
      });
    });
  }

  function flash(btn, label) {
    const prev = btn.textContent;
    btn.textContent = label;
    btn.classList.add("flash");
    setTimeout(() => { btn.textContent = prev; btn.classList.remove("flash"); }, 900);
  }

  async function regen(date) {
    // Surgical replace: swap out ONLY the Summary card with a loading
    // placeholder, leave the Timeline card (if present) untouched. Avoids
    // the user losing both views for 10-60s while one regenerates.
    const slot = document.getElementById("summary_slot");
    const existingSummary = slot.querySelector("details.summary-details:not(.timeline-details), .generate:not(.timeline-generate)");
    const placeholder = el("div", {class: "generate"}, [
      el("span", null, [el("span", {class: "spinner"}), "Thinking — local models take 15-60 seconds on first run…"]),
    ]);
    if (existingSummary) {
      slot.replaceChild(placeholder, existingSummary);
    } else {
      slot.insertBefore(placeholder, slot.firstChild);
    }
    try {
      const r = await fetch("/api/summarize", {
        method: "POST",
        headers: {"Content-Type":"application/json"},
        body: JSON.stringify({date, force: true}),
      });
      const data = await r.json();
      if (!r.ok) {
        placeholder.innerHTML = `<span style="color:#c44; white-space:pre-line">${escapeHtml(data.error || "failed")}</span>`;
        return;
      }
      load(date);
    } catch(e) {
      placeholder.innerHTML = `<span style="color:#c44">${escapeHtml(e.message)}</span>`;
    }
  }

  async function regenTimeline(date) {
    // Same surgical pattern as regen — swap only the Timeline card.
    const slot = document.getElementById("summary_slot");
    const existingTimeline = slot.querySelector("details.timeline-details, .timeline-generate");
    const placeholder = el("div", {class: "generate timeline-generate"}, [
      el("span", null, [
        el("span", {class: "spinner"}),
        "Building a chronological timeline — local models take 20-60 seconds…",
      ]),
    ]);
    if (existingTimeline) {
      slot.replaceChild(placeholder, existingTimeline);
    } else {
      // Append — timeline always sits below summary
      slot.appendChild(placeholder);
    }
    try {
      const r = await fetch("/api/timeline", {
        method: "POST",
        headers: {"Content-Type": "application/json"},
        body: JSON.stringify({date, force: true}),
      });
      const data = await r.json();
      if (!r.ok) {
        placeholder.innerHTML = `<span style="color:#c44; white-space:pre-line">${escapeHtml(data.error || "failed")}</span>`;
        return;
      }
      load(date);
    } catch(e) {
      placeholder.innerHTML = `<span style="color:#c44">${escapeHtml(e.message)}</span>`;
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
    const archived = data.archived || [];
    const hint = document.getElementById("queue_hint");
    if (!data.research_enabled) {
      hint.innerHTML = `💡 Research is off. Enable it by adding <code class="inline">"research": {"provider": "local"}</code> (or <code class="inline">"openai"</code>) to ~/.voiceclip/config.json.`;
    } else if (data.research_provider === "local") {
      hint.innerHTML = `Using local · ${shortModel(data.research_model)}. Answers from model knowledge only — no web search, no sources. Fine for conceptual topics, weak for time-sensitive ones.`;
    } else {
      hint.innerHTML = `Using ${data.research_provider} · ${shortModel(data.research_model)}. The model decides whether to search the web per topic.`;
    }
    document.getElementById("queue_stats").textContent =
      `${topics.length} topic${topics.length===1?"":"s"}`;

    const slot = document.getElementById("topics_slot");
    slot.innerHTML = "";
    if (!topics.length) {
      slot.appendChild(el("div", {class:"empty"}, "No research topics yet. Add one above."));
    } else {
      topics.forEach(t => slot.appendChild(renderTopic(t, data.research_enabled)));
    }

    // Archived section — collapsed by default so it doesn't add noise
    // until the user actually wants to revisit something.
    if (archived.length) {
      const det = el("details", {class: "archived-topics"});
      det.appendChild(el("summary", null,
        `Archived (${archived.length})`));
      archived.forEach(a => det.appendChild(renderArchivedTopic(a)));
      slot.appendChild(det);
    }
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

    // "Done" — archive a topic that's served its purpose. Hides it from
    // the active queue but keeps the row in the DB so FTS search still
    // finds it. Only offered once the topic has a brief to "be done with."
    if (t.status === "ready") {
      const doneBtn = el("button", {
        title: "Archive this topic — keeps it searchable but removes it from the queue",
        onclick: ev => archiveTopic(wrap, t.id, ev.target),
      }, "Done");
      actions.push(doneBtn);
    }

    // "Copy prompt" — produces a self-contained prompt you can paste into
    // ChatGPT / Claude / Perplexity to get the same brief format as the
    // built-in research. Works regardless of status; useful when the
    // user doesn't have a cloud provider configured or just prefers
    // their own chatbot for a given topic.
    const copyPromptBtn = el("button", {
      title: "Copy a ready-to-paste prompt for an external AI",
      onclick: ev => copyResearchPrompt(ev.target, t.text),
    }, "Copy prompt");
    actions.push(copyPromptBtn);

    const deleteBtn = el("button", {class:"danger", onclick: ev => handleTopicDelete(ev.target, wrap, t)}, "Delete");
    actions.push(deleteBtn);

    // Topic title is click-to-edit. Uses the same /api/update endpoint
    // the Journal's inline edits use, since research topics are stored
    // as transcriptions with is_research_topic=1.
    const title = el("div", {
      class: "title",
      contenteditable: "true",
      spellcheck: "true",
      title: "Click to edit",
    }, t.text);
    wireTopicTitleEdit(title, t);

    const head = el("div", {class:"topic-head"}, [
      title,
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

  function buildExternalPrompt(topicText) {
    const topic = String(topicText || "").trim();
    return [
      "Research this topic. Use web search if it's time-sensitive or product-specific; otherwise answer from knowledge.",
      "",
      "Topic:",
      "",
      topic,
    ].join("\n");
  }

  async function copyResearchPrompt(btn, topicText) {
    const prompt = buildExternalPrompt(topicText);
    try {
      await navigator.clipboard.writeText(prompt);
      flash(btn, "Copied ✓");
    } catch(e) {
      // Some browsers block clipboard from non-focused contexts; fall back
      // to the legacy execCommand path.
      const ta = document.createElement("textarea");
      ta.value = prompt;
      ta.style.position = "fixed";
      ta.style.opacity = "0";
      document.body.appendChild(ta);
      ta.focus(); ta.select();
      try { document.execCommand("copy"); flash(btn, "Copied ✓"); }
      catch(e2) { flash(btn, "Copy failed"); }
      document.body.removeChild(ta);
    }
  }

  // Click-to-edit on the topic title. Mirrors the Journal's wireInlineEdit:
  // plain-text paste, Esc cancels, Cmd+Enter or blur saves. Uses the
  // existing /api/update endpoint (research topics live in transcriptions
  // with is_research_topic=1, so the same update path works).
  function wireTopicTitleEdit(node, topic) {
    node.dataset.original = topic.text;
    wireContentEditableKeybinds(node, () => node.dataset.original);
    node.addEventListener("blur", async () => {
      const newText = node.textContent.trim();
      const oldText = node.dataset.original;
      if (!newText || newText === oldText) {
        if (!newText) node.textContent = oldText;
        return;
      }
      try {
        const r = await fetch("/api/update", {
          method: "POST",
          headers: {"Content-Type": "application/json"},
          body: JSON.stringify({id: topic.id, text: newText}),
        });
        if (!r.ok) {
          node.textContent = oldText;
          return;
        }
        node.dataset.original = newText;
        topic.text = newText;  // keep in sync so Research / Copy-prompt use the new text
        const pill = el("span", {class:"saved-pill"}, "saved");
        node.appendChild(pill);
        setTimeout(() => pill.remove(), 1200);
      } catch(e) {
        node.textContent = oldText;
      }
    });
  }

  function renderBrief(brief) {
    // The brief is shown as a <details> so long briefs don't take over the
    // queue. Two modes inside:
    //   - view mode (default): rendered markdown, not editable
    //   - edit mode: plain textarea with the raw text, save on blur
    //
    // Clicking the rendered body flips to edit mode. Blur commits to the
    // server and re-renders. Escape cancels without saving.
    const wrap = el("details", {class: "brief-details", open: "open"});

    const summary = el("summary", null, [
      el("span", {class: "brief-summary-label"}, "Brief"),
      brief.used_web_search
        ? el("span", {class: "web-badge"}, "web")
        : null,
      el("span", {class: "brief-summary-preview"}, previewOf(brief.text)),
    ]);
    wrap.appendChild(summary);

    const body = el("div", {class: "brief"});
    renderBriefBody(body, brief);
    wrap.appendChild(body);

    if (brief.sources && brief.sources.length) {
      const src = el("div", {class: "sources"});
      src.appendChild(el("div", null, `Sources (${brief.sources.length}):`));
      brief.sources.forEach(s => {
        src.appendChild(el("a", {href: s.url, target: "_blank", rel: "noopener"}, s.title));
      });
      wrap.appendChild(src);
    }
    return wrap;
  }

  // One-line preview derived from the brief text. Strips markdown to keep
  // the <summary> line clean.
  function previewOf(text) {
    const s = String(text || "")
      .replace(/\*\*/g, "")
      .replace(/\s+/g, " ")
      .trim();
    return s.length > 140 ? s.slice(0, 140).trimEnd() + "…" : s;
  }

  function renderBriefBody(container, brief) {
    container.innerHTML = "";
    const rendered = el("div", {class: "brief-rendered"});
    rendered.appendChild(renderMarkdown(brief.text));
    // Click-to-edit affordance
    rendered.title = "Click to edit";
    rendered.addEventListener("click", () => {
      enterEditMode(container, brief);
    });
    container.appendChild(rendered);
  }

  function enterEditMode(container, brief) {
    container.innerHTML = "";
    const ta = document.createElement("textarea");
    ta.className = "brief-editor";
    ta.value = brief.text;
    ta.rows = Math.min(30, Math.max(6, brief.text.split("\n").length + 2));
    container.appendChild(ta);

    const hint = el("div", {class: "brief-edit-hint"},
      "Cmd/Ctrl+Enter to save · Esc to cancel · click elsewhere to save");
    container.appendChild(hint);

    ta.focus();
    // Move caret to end
    ta.selectionStart = ta.selectionEnd = ta.value.length;

    let committed = false;
    const save = async () => {
      if (committed) return;
      committed = true;
      const newText = ta.value.trim();
      if (!newText || newText === brief.text) {
        renderBriefBody(container, brief);
        return;
      }
      try {
        const r = await fetch("/api/research/update_brief", {
          method: "POST",
          headers: {"Content-Type": "application/json"},
          body: JSON.stringify({brief_id: brief.id, text: newText}),
        });
        if (r.ok) {
          brief.text = newText;
          // Update the <summary> preview to match
          const details = container.closest("details.brief-details");
          if (details) {
            const preview = details.querySelector(".brief-summary-preview");
            if (preview) preview.textContent = previewOf(newText);
          }
          renderBriefBody(container, brief);
          flashNote(container, "Saved");
        } else {
          renderBriefBody(container, brief);
          flashNote(container, "Save failed");
        }
      } catch (e) {
        renderBriefBody(container, brief);
        flashNote(container, "Save failed");
      }
    };

    const cancel = () => {
      if (committed) return;
      committed = true;
      renderBriefBody(container, brief);
    };

    ta.addEventListener("blur", save);
    ta.addEventListener("keydown", (ev) => {
      if (ev.key === "Escape") {
        ev.preventDefault();
        cancel();
      } else if ((ev.metaKey || ev.ctrlKey) && ev.key === "Enter") {
        ev.preventDefault();
        ta.blur();  // triggers save
      }
    });
  }

  // Small inline toast appended to the brief container. Self-removes after
  // a beat. Distinct from the button flash() so the source of truth stays clear.
  function flashNote(container, text) {
    const note = el("div", {class: "brief-save-note"}, text);
    container.appendChild(note);
    setTimeout(() => note.remove(), 1200);
  }

  async function runResearch(wrap, id, btn, isRerun) {
    const prev = btn.textContent;
    btn.disabled = true;
    btn.textContent = "Researching…";
    // Optimistically mark the topic running
    wrap.classList.remove("pending", "failed", "ready");
    wrap.classList.add("running");

    // Live progress panel: spinner + elapsed seconds + model hint.
    // Replaces any existing brief/error block while the call is in flight,
    // so the user always sees *something* moving. Research can take 10-60s.
    const existing = wrap.querySelector(".brief");
    if (existing) existing.remove();
    const elapsed = el("span", {class:"progress-elapsed"}, "0s");
    const progress = el("div", {class:"research-progress"}, [
      el("span", {class:"spinner"}),
      el("span", null, "Researching"),
      elapsed,
      el("span", {class:"progress-hint"},
        "— the model may search the web; this usually takes 10-30 seconds"),
    ]);
    wrap.appendChild(progress);

    const t0 = Date.now();
    const tick = setInterval(() => {
      const s = Math.round((Date.now() - t0) / 1000);
      elapsed.textContent = `${s}s`;
      // After 90s something is clearly wrong; the user can cancel by reloading
      if (s >= 90) {
        elapsed.textContent = `${s}s — this is longer than usual`;
        elapsed.style.color = "#c44";
      }
    }, 500);

    try {
      const r = await fetch("/api/research/run", {
        method: "POST",
        headers: {"Content-Type":"application/json"},
        body: JSON.stringify({id}),
      });
      const data = await r.json();
      clearInterval(tick);
      progress.remove();
      if (!r.ok) {
        wrap.classList.remove("running");
        wrap.classList.add("failed");
        btn.disabled = false;
        btn.textContent = prev;
        const msg = data.error || "Research failed (no error message)";
        const err = el("div", {class:"research-error"}, [
          el("strong", null, "Research failed."),
          el("div", {class:"error-body", style:"white-space:pre-line"}, msg),
          el("div", {class:"error-hint"},
            "Check ~/.voiceclip/config.json (research.openai_model) or your " +
            "OPENAI_API_KEY env var. Run `voiceclip doctor` to verify setup."),
        ]);
        wrap.appendChild(err);
        return;
      }
      // Reload the whole queue so counts + status line refresh consistently
      loadQueue();
    } catch(e) {
      clearInterval(tick);
      progress.remove();
      wrap.classList.remove("running");
      wrap.classList.add("failed");
      btn.disabled = false;
      btn.textContent = prev;
      const err = el("div", {class:"research-error"}, [
        el("strong", null, "Research failed."),
        el("div", {class:"error-body"}, e.message || "Network error"),
      ]);
      wrap.appendChild(err);
    }
  }

  // Archive a research topic. Single-click commit (delete gets the "really?"
  // guard; archive is reversible so it doesn't need one). Fades the card
  // out, then reloads the queue so the archived section picks it up.
  async function archiveTopic(node, id, btn) {
    const prev = btn.textContent;
    btn.disabled = true;
    btn.textContent = "Archiving…";
    try {
      const r = await fetch("/api/research/archive", {
        method: "POST",
        headers: {"Content-Type": "application/json"},
        body: JSON.stringify({id}),
      });
      if (!r.ok) {
        btn.disabled = false;
        btn.textContent = prev;
        flash(btn, "Failed");
        return;
      }
      node.classList.add("removing");
      setTimeout(() => { node.remove(); loadQueue(); }, 360);
    } catch (e) {
      btn.disabled = false;
      btn.textContent = prev;
      flash(btn, "Failed");
    }
  }

  async function unarchiveTopic(id, btn) {
    const prev = btn.textContent;
    btn.disabled = true;
    btn.textContent = "Restoring…";
    try {
      const r = await fetch("/api/research/unarchive", {
        method: "POST",
        headers: {"Content-Type": "application/json"},
        body: JSON.stringify({id}),
      });
      if (!r.ok) {
        btn.disabled = false;
        btn.textContent = prev;
        flash(btn, "Failed");
        return;
      }
      // Full reload so the topic reappears under the active list
      loadQueue();
    } catch (e) {
      btn.disabled = false;
      btn.textContent = prev;
      flash(btn, "Failed");
    }
  }

  // Render an archived topic as a compact card inside the collapsible.
  // Fewer affordances than an active topic — Unarchive + Copy prompt is
  // enough. The brief (if present) stays collapsed inside its own <details>.
  function renderArchivedTopic(t) {
    const wrap = el("div", {class: "topic archived"});
    wrap.dataset.topicId = String(t.id);

    const unarchiveBtn = el("button", {
      title: "Move this topic back into the active queue",
      onclick: ev => unarchiveTopic(t.id, ev.target),
    }, "Unarchive");
    const copyPromptBtn = el("button", {
      title: "Copy a ready-to-paste prompt for an external AI",
      onclick: ev => copyResearchPrompt(ev.target, t.text),
    }, "Copy prompt");

    const title = el("div", {class: "title"}, t.text);

    const head = el("div", {class: "topic-head"}, [
      title,
      el("div", {class: "actions"}, [unarchiveBtn, copyPromptBtn]),
    ]);
    wrap.appendChild(head);

    const archivedWhen = t.archived_at ? fmtRelDate(t.archived_at) : "";
    wrap.appendChild(el("div", {class: "topic-meta"},
      archivedWhen ? `archived ${archivedWhen}` : "archived"));

    if (t.brief) {
      wrap.appendChild(renderBrief(t.brief));
    }
    return wrap;
  }

  function handleTopicDelete(btn, node, topic) {
    twoClickConfirm(btn, () => {
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
    });
  }

  // Topic input
  document.getElementById("topic_add").addEventListener("click", addTopic);
  document.getElementById("topic_input").addEventListener("keydown", (ev) => {
    if (ev.key === "Enter") { ev.preventDefault(); addTopic(); }
  });

  async function addTopic() {
    const input = $input("topic_input");
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

  // Auto-refresh today's page every 20s while viewing today's journal.
  // Skip the poll entirely when the tab is hidden — saves a /api/day
  // round-trip (and a re-render that would wipe any in-flight inline edit)
  // on every laptop lid-close or tab-switch. The interval itself keeps
  // running; only the fetch is gated.
  setInterval(() => {
    if (document.visibilityState !== "visible") return;
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

    // Footer with window stats.
    // cached_summaries / cached_timelines show how much pre-digested
    // input the model actually had. "0/7" here is a useful signal that
    // the patterns output is running on raw reflections + app counts
    // alone, and the user might want to generate daily summaries or
    // timelines for better longitudinal synthesis next run.
    if (stats.transcription_count != null || stats.reflection_count != null) {
      const wd = stats.window_days;
      const parts = [
        `Window: ${stats.start_date} to ${stats.end_date}`,
        `${stats.transcription_count || 0} transcriptions`,
        `${stats.reflection_count || 0} reflections`,
      ];
      if (wd) {
        parts.push(`${stats.cached_summaries || 0}/${wd} daily summaries`);
        parts.push(`${stats.cached_timelines || 0}/${wd} daily timelines`);
      }
      parts.push(`${p.provider || ""} ${shortModel(p.model || "")}`);
      slot.appendChild(el("div", {class:"queue-hint"}, parts.join(" · ")));
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

  // Debounce + stale-response handling via AbortController — the browser
  // primitive for "cancel this fetch because a newer one just started."
  // Holding a reference to the most-recent controller lets every new
  // keystroke abort in-flight requests. The AbortError arrives in the
  // fetch's rejected-promise path and we silently swallow it, so only
  // the latest search's response ever hits renderSearchResults.

  /** @type {AbortController | null} */
  let _searchAbort = null;
  /** @type {number | undefined} */
  let _searchDebounceTimer;
  const searchInput = $input("search_input");
  const searchClear = $id("search_clear");

  searchInput.addEventListener("input", debounceSearch);
  searchClear.addEventListener("click", () => {
    searchInput.value = "";
    searchClear.style.display = "none";
    _searchAbort?.abort();
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
    clearTimeout(_searchDebounceTimer);
    _searchDebounceTimer = setTimeout(() => {
      if (!term) {
        _searchAbort?.abort();
        load(state.date);
        return;
      }
      runSearch(term);
    }, 180);
  }

  async function runSearch(term) {
    // Any in-flight search is now stale — cancel it so its response
    // never renders over this newer one.
    _searchAbort?.abort();
    _searchAbort = new AbortController();
    const signal = _searchAbort.signal;
    try {
      const r = await fetch(
        `/api/search?q=${encodeURIComponent(term)}`,
        { signal },
      );
      const data = await r.json();
      if (signal.aborted) return;  // superseded while we awaited
      renderSearchResults(term, data.entries || []);
    } catch (e) {
      // AbortError is expected when a newer search supersedes this one.
      // Other errors are silent best-effort — search is non-critical, and
      // an intermittent failure doesn't block anything the user can't retry.
    }
  }

  function renderSearchResults(term, entries) {
    // Reuse the existing slots: blank the day-specific sections and replace
    // the entries list. The search header replaces the day label's stats line.
    $id("daylabel").textContent = `Results for "${term}"`;
    $id("stats").textContent =
      `${entries.length} match${entries.length === 1 ? "" : "es"}`;
    $id("summary_slot").innerHTML = "";
    $id("apps_slot").innerHTML = "";
    // Nav buttons — disable during search
    $button("prev").disabled = true;
    $button("next").disabled = true;
    $button("today").onclick = () => {
      searchInput.value = ""; searchClear.style.display = "none";
      load(todayStr());
    };

    const slot = $id("entries_slot");
    slot.innerHTML = "";
    if (!entries.length) {
      slot.appendChild(el("div", {class:"empty"}, "No matches."));
      return;
    }
    // Keep original order (fts rank, then id desc — server-side)
    entries.forEach(e => slot.appendChild(renderEntry(e)));
  }

  // ---------- Ask (question over a date range) ----------

  (function initAsk() {
    const askInput = $input("ask_input");
    const askStart = $input("ask_start");
    const askEnd = $input("ask_end");
    const askBtn = $button("ask_btn");
    const askResult = $id("ask_result");

    // Default date range: last 7 days
    const today = new Date();
    const weekAgo = new Date(today);
    weekAgo.setDate(weekAgo.getDate() - 7);
    const pad = n => String(n).padStart(2, "0");
    askEnd.value = `${today.getFullYear()}-${pad(today.getMonth()+1)}-${pad(today.getDate())}`;
    askStart.value = `${weekAgo.getFullYear()}-${pad(weekAgo.getMonth()+1)}-${pad(weekAgo.getDate())}`;

    askBtn.addEventListener("click", runAsk);
    askInput.addEventListener("keydown", (ev) => {
      if (ev.key === "Enter") { ev.preventDefault(); runAsk(); }
    });

    async function runAsk() {
      const question = askInput.value.trim();
      if (!question) return;

      askResult.innerHTML = "";
      askResult.appendChild(el("div", {class: "ask-loading"}, [
        el("span", {class: "spinner"}),
        " Thinking…",
      ]));
      askBtn.disabled = true;

      try {
        const r = await fetch("/api/ask", {
          method: "POST",
          headers: {"Content-Type": "application/json"},
          body: JSON.stringify({
            question,
            start_date: askStart.value || undefined,
            end_date: askEnd.value || undefined,
          }),
        });
        const data = await r.json();
        askResult.innerHTML = "";

        if (!r.ok) {
          askResult.appendChild(el("div", {class: "ask-error"},
            data.error || "Something went wrong."));
          return;
        }

        const meta = `${data.entry_count} entries · ${data.start_date} to ${data.end_date} · ${data.provider} · ${shortModel(data.model)}`;
        askResult.appendChild(el("div", {class: "ask-answer"}, [
          renderMarkdown(data.answer),
          el("div", {class: "ask-meta"}, meta),
        ]));
      } catch (e) {
        askResult.innerHTML = "";
        askResult.appendChild(el("div", {class: "ask-error"},
          e.message || "Network error"));
      } finally {
        askBtn.disabled = false;
      }
    }
  })();

  // ---------- Settings ----------
  //
  // Loads the schema + current values from /api/settings, renders one input
  // per setting grouped by category, and commits changes on change (with
  // debounce for text inputs). Cloud-provider flips gate through a confirm
  // modal that mirrors the consent banner's language.

  // Plain-language labels and descriptions. Keyed by dotted config key.
  const SETTING_COPY = {
    "engine":                  ["Engine", "Where speech turns into text"],
    "model":                   ["Whisper model", "Runs on this Mac — also the cloud fallback"],
    "parakeet_model":          ["Parakeet model", "Runs on this Mac"],
    "english_only":            ["English only", "Faster, and no wrong-language mix-ups"],
    "custom_vocabulary":       ["Custom vocabulary", "Names and jargon to spell right — one per line"],
    "hotkey":                  ["Dictate", "Transcribe and paste"],
    "reflection_hotkey":       ["Reflect", "Save a thought to the journal — no paste"],
    "assistant_hotkey":        ["Ask assistant", "It talks back, grounded in your journal"],
    "polish_hotkey":           ["Polish & paste", "AI cleans it up before pasting"],
    "cloud.streaming":         ["Stream while talking", "Text is ready the moment you let go"],
    "cloud.fallback_engine":   ["If the cloud is down", "Keep dictating on this Mac with this model"],
    "cloud.base_url":          ["Server URL", null],
    "cloud.model":             ["Server model", null],
    "cloud.instance_id":       ["EC2 instance", null],
    "cloud.region":            ["AWS region", null],
    "cloud.auto_tunnel":       ["Auto-connect", "Open the secure tunnel when VoiceClip starts"],
    "history":                 ["Keep a journal", "Powers Journal, Queue and Patterns"],
    "window_title_capture":    ["Remember the app", "Note which window you dictated into"],
    "polish_prompt":           ["Polish instructions", null],
    "ai.model":                ["AI model", "One model for summaries, research, patterns, polish and Ask"],
    "autocorrect":             ["Fix mishearings", "AI corrects names and jargon in every dictation (~1s, uses your vocabulary)"],
    "summaries.provider":      ["Provider", "Who writes your daily recap"],
    "summaries.style":         ["Style", null],
    "research.provider":       ["Provider", "Only OpenAI and Anthropic can search the web"],
    "patterns.provider":       ["Provider", "Reads your last week of entries"],
    "patterns.window_days":    ["Look back (days)", null],
  };
  // Every *.local_model / *.openai_model / ... reads simply as "Model"
  // inside its feature section.
  for (const f of ["summaries", "research", "patterns"]) {
    for (const prov of ["local", "openai", "anthropic"]) SETTING_COPY[`${f}.${prov}_model`] = ["Model", null];
    SETTING_COPY[`${f}.cloud_model`] = ["Model", "Route name on your gateway"];
  }
  const KEY_LABELS = {
    alt_r: "Right ⌥ Option", alt_l: "Left ⌥ Option",
    ctrl_r: "Right ⌃ Control", ctrl_l: "Left ⌃ Control",
    shift_r: "Right ⇧ Shift", shift_l: "Left ⇧ Shift",
    cmd_r: "Right ⌘ Command", cmd_l: "Left ⌘ Command",
    caps_lock: "⇪ Caps Lock", space: "Space", esc: "Esc",
  };
  const CHOICE_LABELS = {
    "engine": {auto: "Automatic", whisper: "Whisper · this Mac", whisper_cpp: "whisper.cpp · this Mac",
               parakeet: "Parakeet · this Mac", cloud: "Cloud · your server"},
    "cloud.fallback_engine": {none: "Off — show an error", whisper: "Whisper on this Mac",
                              whisper_cpp: "whisper.cpp on this Mac", parakeet: "Parakeet on this Mac"},
    "model": {tiny: "Tiny — fastest", base: "Base", small: "Small", medium: "Medium",
              "large-v3-turbo": "Large v3 Turbo — recommended", "large-v3": "Large v3 — most accurate"},
    "summaries.style": {descriptive: "What I did", reflective: "What I was thinking"},
  };
  const PROVIDER_LABELS = {none: "Off", local: "On this Mac", openai: "OpenAI",
                           anthropic: "Anthropic", cloud: "Your cloud server"};
  function choiceLabel(key, choice) {
    if (/hotkey$/.test(key)) return KEY_LABELS[choice] || String(choice).toUpperCase();
    if (/hotkey_mode$/.test(key)) return {hold: "Hold to talk", toggle: "Tap on / off"}[choice] || choice;
    if (key.endsWith(".provider")) return PROVIDER_LABELS[choice] || choice;
    if (key === "parakeet_model") return String(choice).replace("mlx-community/", "");
    return (CHOICE_LABELS[key] || {})[choice] || choice;
  }

  // Cloud-provider confirm copy, keyed by the dotted setting key.
  // Used by confirmCloudSwitch().
  const CLOUD_DISCLOSURES = {
    "ai.model": "Every AI feature (daily summaries, research topics, patterns, polish, Ask) will send your entries to this provider.",
    "summaries.provider": "Your entries for each day (every transcription and reflection) will be sent to this provider when a summary is generated.",
    "research.provider":  "The research topic you dictate or type will be sent to this provider. If the model uses its web search tool, that topic also goes to the search backend.",
    "patterns.provider":  "Up to a week of your reflections and daily summaries will be sent in a single prompt.",
  };

  let _settingsCache = null;

  // ---------- Stats ----------
  function fmtNum(n) { return Math.round(n).toLocaleString(); }
  function fmtDuration(min) {
    if (min < 1) return "< 1 min";
    if (min < 60) return `${Math.round(min)} min`;
    const h = Math.floor(min / 60), m = Math.round(min % 60);
    return m ? `${h} h ${m} min` : `${h} h`;
  }
  function countUp(node, target, format) {
    const reduce = window.matchMedia("(prefers-reduced-motion: reduce)").matches;
    if (reduce || target <= 0) { node.textContent = format(target); return; }
    const t0 = performance.now(), dur = 900;
    const step = (t) => {
      const k = Math.min(1, (t - t0) / dur);
      node.textContent = format(target * (1 - Math.pow(1 - k, 3)));
      if (k < 1) requestAnimationFrame(step);
    };
    requestAnimationFrame(step);
  }
  async function loadStats() {
    const slot = $id("stats_slot");
    let d;
    try { d = await (await fetch("/api/stats")).json(); }
    catch (e) { slot.textContent = "Couldn't load stats."; return; }
    if (d.error) { slot.textContent = d.error; return; }
    slot.innerHTML = "";
    $id("stats_streak").textContent = d.current_streak
      ? `🔥 ${d.current_streak}-day streak` : "Dictate today to start a streak";

    const hero = (emoji, value, label, sub, fmt) => {
      const v = el("div", {class: "stat-value"}, "0");
      countUp(v, value, fmt);
      return el("div", {class: "stat-hero"}, [
        el("div", {class: "stat-emoji", "aria-hidden": "true"}, emoji), v,
        el("div", {class: "stat-label"}, label),
        sub ? el("div", {class: "stat-sub"}, sub) : null,
      ]);
    };
    slot.appendChild(el("div", {class: "stat-heroes"}, [
      hero("✍️", d.words, "words dictated", `${fmtNum(d.words_today)} today · ${fmtNum(d.words_week)} this week`, fmtNum),
      hero("⏱️", d.minutes_saved, "of typing saved", `${fmtDuration(d.minutes_saved_week)} this week`, fmtDuration),
      hero("🪄", d.ai_fixes, "words fixed by AI",
           !d.autocorrect_on && !d.ai_checked ? "Turn on “Fix mishearings” in Settings → AI"
             : [`${fmtNum(d.ai_checked)} dictation${d.ai_checked === 1 ? "" : "s"} checked`,
                d.ai_polished ? `${fmtNum(d.ai_polished)} polished` : null].filter(Boolean).join(" · "),
           fmtNum),
    ]));

    // Last 14 days
    const max = Math.max(1, ...d.daily.map(x => x.words));
    const bars = el("div", {class: "stat-bars", role: "img",
                            "aria-label": `Words per day for the last 14 days, up to ${fmtNum(max)}`});
    d.daily.forEach((x, i) => {
      const dt = new Date(x.date + "T12:00:00");
      bars.appendChild(el("div", {class: "stat-bar-col", title: `${dt.toLocaleDateString(undefined, {weekday: "short", month: "short", day: "numeric"})}: ${fmtNum(x.words)} words`}, [
        el("div", {class: "stat-bar" + (x.words ? "" : " zero"),
                   style: `--h:${(x.words / max) * 100}%; animation-delay:${i * 30}ms`}),
        el("div", {class: "stat-bar-day"}, dt.toLocaleDateString(undefined, {weekday: "narrow"})),
      ]));
    });
    const facts = [
      ["Dictations", fmtNum(d.dictations)],
      ["Days active", fmtNum(d.days_active)],
      ["Best streak", `${d.best_streak} day${d.best_streak === 1 ? "" : "s"}`],
      ["Best day", d.best_day ? `${fmtNum(d.best_day.words)} words` : "—"],
      ["Vocabulary", `${d.vocab} words`],
      ["Questions asked", fmtNum(d.questions)],
    ];
    slot.appendChild(el("div", {class: "stat-row"}, [
      el("div", {class: "stat-card stat-chart"}, [el("h3", null, "Last 14 days"), bars]),
      el("div", {class: "stat-card"}, [el("h3", null, "At a glance"),
        el("dl", {class: "stat-facts"}, facts.flatMap(([k, v]) => [el("dt", null, k), el("dd", null, v)]))]),
    ]));

    const lower = [];
    if (d.top_apps.length) {
      const top = d.top_apps[0].words || 1;
      lower.push(el("div", {class: "stat-card"}, [el("h3", null, "Where your words went"),
        ...d.top_apps.map(a => el("div", {class: "stat-app"}, [
          el("span", {class: "stat-app-name"}, a.name),
          el("span", {class: "stat-app-bar"}, el("span", {style: `width:${(a.words / top) * 100}%`})),
          el("span", {class: "stat-app-n"}, fmtNum(a.words)),
        ]))]));
    }
    if (d.ai_recent.length) {
      lower.push(el("div", {class: "stat-card"}, [el("h3", null, "Recent AI fixes"),
        ...d.ai_recent.map(([a, b]) => el("div", {class: "stat-fix"}, [
          el("s", null, a), el("span", {"aria-hidden": "true"}, " → "), el("strong", null, b)]))]));
    }
    if (lower.length) slot.appendChild(el("div", {class: "stat-row"}, lower));

    const earned = d.badges.filter(b => b.earned).length;
    slot.appendChild(el("div", {class: "stat-card"}, [
      el("h3", null, ["Achievements", el("span", {class: "vocab-group-count"}, `${earned} / ${d.badges.length}`)]),
      el("div", {class: "badge-grid"}, d.badges.map(b => el("div", {
        class: "badge" + (b.earned ? " earned" : ""),
        title: b.earned ? `${b.title} — ${b.description}` : `${b.description} (${Math.round(b.progress * 100)}%)`,
      }, [
        el("div", {class: "badge-emoji", "aria-hidden": "true"}, b.emoji),
        el("div", {class: "badge-title"}, b.title),
        el("div", {class: "badge-desc"}, b.description),
        b.earned ? null : el("div", {class: "badge-progress"}, el("span", {style: `width:${b.progress * 100}%`})),
      ]))),
    ]));
    slot.appendChild(el("p", {class: "stat-note"},
      `Time saved compares typing at ${d.assumptions.typing_wpm} words per minute with speaking at ${d.assumptions.speaking_wpm}.`));
  }

  // ---------- Vocabulary ----------
  const VOCAB_KIND = {acronym: "acronym", name: "name", phrase: "phrase"};
  let _vocab = {terms: [], suggestions: []};
  async function saveVocab(terms) {
    const r = await fetch("/api/settings/update", {method: "POST", headers: {"Content-Type": "application/json"},
                                                   body: JSON.stringify({custom_vocabulary: terms})});
    if (!r.ok) {
      const d = await r.json().catch(() => ({}));
      $id("vocab_stats").textContent = d.error || "Couldn't save";
      return false;
    }
    return true;
  }
  async function loadVocab() {
    try {
      _vocab = await (await fetch("/api/vocab")).json();
    } catch (e) {
      $id("vocab_suggestions").textContent = "Couldn't load vocabulary.";
      return;
    }
    renderVocab();
  }
  function renderVocab() {
    const terms = _vocab.terms || [];
    $id("vocab_stats").textContent = `${terms.length} word${terms.length === 1 ? "" : "s"}`;
    const box = $id("vocab_chips");
    box.innerHTML = "";
    $id("vocab_list_head").hidden = terms.length < 12;
    if (!terms.length) box.appendChild(el("div", {class: "vocab-none"}, "Nothing yet — add a word above or pick from the suggestions."));
    const q = ($input("vocab_filter").value || "").trim().toLowerCase();
    const kinds = _vocab.kinds || {};
    const GROUPS = [["name", "👤 People & names"], ["acronym", "🔠 Acronyms"],
                    ["phrase", "💬 Terms & phrases"], ["word", "📝 Words"]];
    for (const [kind, title] of GROUPS) {
      const words = terms.filter(t => (kinds[t] || "word") === kind && (!q || t.toLowerCase().includes(q)))
                         .sort((a, b) => a.localeCompare(b, undefined, {sensitivity: "base"}));
      if (!words.length) continue;
      const list = el("ul", {class: "vocab-words"});
      for (const t of words) {
        list.appendChild(el("li", null, [
          el("span", {class: "vocab-word"}, t),
          el("button", {type: "button", class: "vocab-x", "aria-label": `Remove ${t}`, title: "Remove",
                        onclick: async () => {
            if (await saveVocab(terms.filter(x => x !== t))) loadVocab();
          }}, "×"),
        ]));
      }
      box.appendChild(el("section", {class: "vocab-group"}, [
        el("h4", null, [title, el("span", {class: "vocab-group-count"}, String(words.length))]),
        list,
      ]));
    }
    if (q && !box.children.length) box.appendChild(el("div", {class: "vocab-none"}, `No words match “${q}”.`));
    const sug = _vocab.suggestions || [];
    $id("vocab_scanned").textContent = _vocab.scanned ? `from your last ${_vocab.scanned} entries` : "";
    const list = $id("vocab_suggestions");
    list.innerHTML = "";
    if (!sug.length) {
      list.appendChild(el("div", {class: "vocab-none"}, "No new suggestions — keep dictating and check back."));
      return;
    }
    for (const s of sug) {
      list.appendChild(el("div", {class: "vocab-sug"}, [
        el("div", {class: "vocab-sug-main"}, [
          el("span", {class: "vocab-sug-term"}, s.term),
          el("span", {class: `vocab-kind ${s.kind}`}, VOCAB_KIND[s.kind] || s.kind),
          el("span", {class: "vocab-count"}, `×${s.count}`),
          el("div", {class: "vocab-example"}, s.example || ""),
        ]),
        el("div", {class: "vocab-sug-actions"}, [
          el("button", {type: "button", class: "primary", onclick: async () => {
            if (await saveVocab([...terms, s.term])) loadVocab();
          }}, "Add"),
          el("button", {type: "button", onclick: async () => {
            await fetch("/api/vocab/ignore", {method: "POST", headers: {"Content-Type": "application/json"},
                                              body: JSON.stringify({term: s.term})});
            loadVocab();
          }}, "Ignore"),
        ]),
      ]));
    }
  }
  async function addVocabFromInput() {
    const inp = $input("vocab_input");
    const t = inp.value.trim();
    if (!t) return;
    const terms = _vocab.terms || [];
    if (terms.some(x => x.toLowerCase() === t.toLowerCase())) { inp.value = ""; return; }
    if (await saveVocab([...terms, t])) { inp.value = ""; loadVocab(); }
  }
  $id("vocab_add").addEventListener("click", addVocabFromInput);
  $input("vocab_filter").addEventListener("input", renderVocab);
  $input("vocab_input").addEventListener("keydown", ev => { if (ev.key === "Enter") addVocabFromInput(); });

  // ---------- Voice agent ----------
  const AGENT_STATE_COPY = {
    idle: "Tap to start talking",
    starting: "Waking up…",
    ready: "Listening — just talk",
    listening: "Hearing you…",
    thinking: "Thinking…",
    speaking: "Speaking — interrupt any time",
    stopped: "Tap to start talking",
    error: "Something went wrong",
  };
  const AGENT_TOOL_COPY = {
    web_search: "🔎 Searching the web",
    journal_search: "📓 Looking through your journal",
    save_note: "📝 Saving a note",
    daily_summary: "🗓️ Summarizing the day",
    cloud_status: "☁️ Checking the server",
    park_server: "🅿️ Parking the server",
  };
  const agent = {timer: null, active: false, running: false, after: 0, session: null, state: "idle",
                 busy: false, quietSince: null, startedAt: 0};
  function agentSetState(state, message) {
    agent.state = state;
    const orb = $id("agent_orb");
    orb.className = `agent-orb ${state}`;
    const live = agent.running && state !== "stopped" && state !== "error";
    orb.setAttribute("aria-label", live ? "Stop the voice agent" : "Start the voice agent");
    orb.setAttribute("aria-pressed", live ? "true" : "false");
    $id("agent_state").textContent = message || AGENT_STATE_COPY[state] || state;
  }
  function agentAppend(ev) {
    const log = $id("agent_log");
    let node = null;
    if (ev.type === "user") node = el("div", {class: "bubble user"}, ev.text);
    else if (ev.type === "assistant") node = el("div", {class: "bubble bot"}, ev.text);
    else if (ev.type === "tool") node = el("div", {class: "tool-chip"}, AGENT_TOOL_COPY[ev.name] || `🛠️ ${ev.name}`);
    if (!node) return;
    log.appendChild(node);
    log.scrollTop = log.scrollHeight;
    node.scrollIntoView({block: "nearest", behavior: "smooth"});
  }
  async function agentPoll() {
    try {
      const r = await fetch(`/api/agent/status?after=${agent.after}`);
      const s = await r.json();
      if (s.session !== agent.session) {          // new session → fresh transcript
        agent.session = s.session;
        agent.after = 0;
        $id("agent_log").innerHTML = "";
        return agentPoll();
      }
      agent.running = !!s.running;
      let lastState = null, lastMsg = null;
      for (const ev of s.events || []) {
        agent.after = Math.max(agent.after, ev.seq || 0);
        if (ev.type === "state") { lastState = ev.state; lastMsg = ev.message || null; }
        else agentAppend(ev);
      }
      if (lastState) agentSetState(lastState, lastMsg);
      else if (!agent.running && !["error", "starting"].includes(agent.state)) agentSetState("idle");
      // Process died without saying goodbye (crash, missing dependency...)
      if (!agent.running && agent.state === "starting" && Date.now() - agent.startedAt > 8000) {
        agentSetState("error", "The agent stopped unexpectedly — details in ~/.voiceclip/agent/agent.log");
      }
      // Live mic meter; nudge when it stays near-silent while listening
      const lvl = typeof s.level === "number" ? s.level : null;
      $id("agent_meter").style.setProperty("--level", lvl == null ? 0 : lvl);
      $id("agent_meter").hidden = !(agent.running && lvl != null);
      if (agent.running && agent.state === "ready" && lvl != null && lvl < 0.15) {
        agent.quietSince = agent.quietSince || Date.now();
        if (Date.now() - agent.quietSince > 6000) {
          $id("agent_state").textContent = "I can't hear you — check your mic input (System Settings → Sound)";
        }
      } else {
        agent.quietSince = null;
      }
      if (!$id("agent_log").children.length) {
        $id("agent_log").appendChild(el("div", {class: "agent-empty"},
          "Try: “What did I work on this week?” · “Search the web for flights to Lisbon” · “Note that I owe Sam a reply.”"));
      }
    } catch (e) { /* viewer restarting — keep polling */ }
    if (agent.active) agent.timer = setTimeout(agentPoll, agent.running ? 600 : 2500);
  }
  function agentActivate() {
    agent.active = true;
    clearTimeout(agent.timer);
    agentPoll();
  }
  function agentDeactivate() {
    agent.active = false;
    clearTimeout(agent.timer);
  }
  $id("agent_orb").addEventListener("click", async () => {
    if (agent.busy) return;
    agent.busy = true;
    const live = agent.running && !["stopped", "error", "idle"].includes(agent.state);
    try {
      const path = live ? "/api/agent/stop" : "/api/agent/start";
      if (live) {
        agentSetState("stopped", "Stopping…");
      } else {
        $id("agent_log").innerHTML = "";
        agentSetState("starting");
        agent.running = true;
        agent.startedAt = Date.now();
      }
      const r = await fetch(path, {method: "POST", headers: {"Content-Type": "application/json"}, body: "{}"});
      if (!r.ok) {
        agent.running = false;
        agentSetState("error", r.status === 404
          ? "This viewer is out of date — quit `voiceclip view` and start it again"
          : `Couldn't ${live ? "stop" : "start"} the agent (HTTP ${r.status})`);
      }
    } catch (e) {
      agent.running = false;
      agentSetState("error", "Lost contact with the viewer — is `voiceclip view` still running?");
    } finally {
      agent.busy = false;
      clearTimeout(agent.timer);
      agent.timer = setTimeout(agentPoll, 400);
    }
  });

  async function loadSettings() {
    try {
      const r = await fetch("/api/settings");
      const data = await r.json();
      _settingsCache = data;
      renderSettings(data);
    } catch(e) {
      document.getElementById("settings_slot").innerHTML = "";
      document.getElementById("settings_slot").appendChild(
        el("div", {class:"empty"}, "Could not load settings.")
      );
    }
    // Models list is best-effort and slower (it scans the filesystem) —
    // fetch independently so a slow scan doesn't delay the rest of the
    // settings UI.
    loadModels();
  }

  const SETTINGS_GROUPS = [
    ["Dictation", "🎙️", "voice → text"],
    ["Hotkeys", "⌨️", "one key per job"],
    ["AI", "✨", "one model for everything"],
    ["Remote infra", "☁️", "your server"],
  ];
  const _advancedOpen = new Set();
  // visible_when: {key: value | [values] | "!value" | "!null"} (AND across
  // keys), or a list of such objects (OR).
  function _matchCond(cond, values) {
    return Object.keys(cond).every(k => {
      const want = cond[k], actual = values[k];
      if (Array.isArray(want)) return want.includes(actual);
      if (typeof want === "string" && want.startsWith("!")) {
        const neg = want.slice(1);
        if (neg === "null") return actual !== null && actual !== undefined && actual !== "";
        return actual !== neg;
      }
      return actual === want;
    });
  }
  function isSettingVisible(schema, values) {
    const vw = schema.visible_when;
    if (!vw) return true;
    return Array.isArray(vw) ? vw.some(c => _matchCond(c, values)) : _matchCond(vw, values);
  }
  function visibilityKeys(vw) {
    if (!vw) return [];
    return (Array.isArray(vw) ? vw : [vw]).flatMap(c => Object.keys(c));
  }
  function renderSettings(data) {
    const slot = document.getElementById("settings_slot");
    slot.innerHTML = "";
    document.getElementById("settings_status").textContent =
      "Saves as you go · ↻ takes effect after a restart";
    const byGroup = {};
    for (const key in data.schema) {
      const g = data.schema[key].group;
      (byGroup[g] = byGroup[g] || []).push(key);
    }
    // "<x>_mode" settings render inline next to "<x>" (the hotkey).
    const paired = new Set(Object.keys(data.schema).filter(k => data.schema[k + "_mode"]).map(k => k + "_mode"));
    const known = new Set(SETTINGS_GROUPS.map(g => g[0]));
    const groups = SETTINGS_GROUPS.concat(
      Object.keys(byGroup).filter(g => !known.has(g)).map(g => [g, "•", ""]));
    for (const [gname, icon, blurb] of groups) {
      const keys = (byGroup[gname] || []).filter(
        k => !paired.has(k) && !data.schema[k].hidden && isSettingVisible(data.schema[k], data.values));
      if (!keys.length) continue;
      const groupEl = el("div", {class: "setting-group", "data-group": gname}, [
        el("h3", null, [
          el("span", {class: "group-icon", "aria-hidden": "true"}, icon), gname,
          blurb ? el("span", {class: "group-blurb"}, blurb) : null,
        ]),
      ]);
      const toggles = [];
      let sectionEl = null, sectionName = null;
      const advanced = [];
      for (const key of keys) {
        const s = data.schema[key];
        if (s.advanced) { advanced.push(key); continue; }
        if (s.type === "bool") { toggles.push(key); continue; }
        let target = groupEl;
        if (s.section) {
          if (s.section !== sectionName) {
            sectionName = s.section;
            sectionEl = el("div", {class: "setting-section"}, [el("h4", null, s.section)]);
            groupEl.appendChild(sectionEl);
          }
          target = sectionEl;
        }
        target.appendChild(renderSettingRow(key, s, data.values[key], data));
      }
      if (toggles.length) {
        const grid = el("div", {class: "toggle-grid"});
        toggles.forEach(k => grid.appendChild(renderToggleChip(k, data.schema[k], data.values[k])));
        groupEl.appendChild(grid);
      }
      if (advanced.length) {
        const det = el("details", {class: "setting-advanced"}, [
          el("summary", null, "Connection details"),
        ]);
        if (_advancedOpen.has(gname)) det.open = true;
        det.addEventListener("toggle", () => {
          if (det.open) _advancedOpen.add(gname); else _advancedOpen.delete(gname);
        });
        advanced.forEach(k => det.appendChild(renderSettingRow(k, data.schema[k], data.values[k], data)));
        groupEl.appendChild(det);
      }
      slot.appendChild(groupEl);
    }
    renderCloudPanel(slot, data);
    renderSystemInfo(data.system || {});
  }

  // "Cloud server" panel — instance state + stop/start controls. Shown only
  // when the cloud engine is selected and an instance is configured.
  function renderCloudPanel(slot, data) {
    if (data.values["engine"] !== "cloud") return;
    if (!data.values["cloud.instance_id"] || !data.values["cloud.region"]) return;

    const dot = el("span", {class:"cloud-dot"});
    const stateEl = el("span", {class:"cloud-state"}, "checking\u2026");
    const typeEl = el("span", {class:"desc"}, "");
    const msgEl = el("div", {class:"cloud-msg"}, "");
    const tunnelEl = el("span", {class:"cloud-tunnel"}, "");
    const stopBtn = el("button", null, "Park");
    const startBtn = el("button", null, "Wake");
    stopBtn.disabled = startBtn.disabled = true;

    const panel = el("div", {class:"setting-group"}, [
      el("h3", null, "Cloud server"),
      el("div", {class:"cloud-row"}, [
        el("div", {class:"setting-label"}, [
          el("span", null, "Instance"),
          el("span", {class:"desc"},
             `${data.values["cloud.instance_id"]} \u00b7 ${data.values["cloud.region"]}`),
        ]),
        el("div", {class:"cloud-status"}, [dot, stateEl, typeEl, tunnelEl]),
        el("div", {class:"cloud-actions"}, [stopBtn, startBtn]),
      ]),
      msgEl,
    ]);
    // Which model the server runs (vLLM "assistant" route)
    const llmSel = document.createElement("select");
    llmSel.className = "cloud-llm-select";
    const llmCustom = el("input", {type: "text", class: "cloud-llm-custom", placeholder: "Org/Model-Name on Hugging Face", hidden: "hidden"});
    const llmApply = el("button", {type: "button"}, "Apply");
    const llmMsg = el("div", {class: "cloud-msg"}, "");
    llmApply.disabled = true;
    const llmRow = el("div", {class: "cloud-llm-row"}, [
      el("div", {class: "setting-label"}, [
        el("span", null, "Cloud AI model"),
        el("span", {class: "desc"}, "What your server runs for AI features, the assistant and the agent"),
      ]),
      el("div", {class: "cloud-llm-inputs"}, [llmSel, llmCustom, llmApply]),
    ]);
    panel.appendChild(llmRow);
    panel.appendChild(llmMsg);
    let llmCurrent = null;
    fetch("/api/cloud/llm").then(r => r.json()).then(d => {
      llmCurrent = d.current;
      llmSel.innerHTML = "";
      const ids = new Set();
      for (const c of d.choices || []) {
        ids.add(c.id);
        const o = document.createElement("option"); o.value = c.id; o.textContent = c.label; llmSel.appendChild(o);
      }
      if (!ids.has(d.current)) {
        const o = document.createElement("option"); o.value = d.current; o.textContent = d.current; llmSel.appendChild(o);
      }
      const custom = document.createElement("option"); custom.value = "__custom__"; custom.textContent = "Custom…";
      llmSel.appendChild(custom);
      llmSel.value = d.current;
    });
    const llmTarget = () => llmSel.value === "__custom__" ? llmCustom.value.trim() : llmSel.value;
    const llmSync = () => {
      llmCustom.hidden = llmSel.value !== "__custom__";
      llmApply.disabled = !llmTarget() || llmTarget() === llmCurrent;
    };
    llmSel.addEventListener("change", llmSync);
    llmCustom.addEventListener("input", llmSync);
    llmApply.addEventListener("click", async () => {
      const model = llmTarget();
      if (!confirm(`Switch the cloud AI model to ${model}? The server downloads and loads it — AI features pause for a few minutes.`)) return;
      llmApply.disabled = true;
      llmMsg.textContent = "Switching…";
      const r = await fetch("/api/cloud/llm", {method: "POST", headers: {"Content-Type": "application/json"},
                                               body: JSON.stringify({model})});
      const res = await r.json().catch(() => ({}));
      if (!r.ok) { llmMsg.textContent = res.error || `Failed (HTTP ${r.status})`; llmSync(); return; }
      llmCurrent = model;
      const t0 = Date.now();
      const tick = async () => {
        const secs = Math.round((Date.now() - t0) / 1000);
        const p = await fetch("/api/cloud/llm?probe=1").then(x => x.json()).catch(() => ({}));
        if (p.ready && secs > 20) { llmMsg.textContent = `✅ ${model} is live.`; return; }
        if (secs > 900) { llmMsg.textContent = "Still not answering after 15 min — the model may not fit. Pick another and Apply."; llmSync(); return; }
        llmMsg.textContent = `Loading on the server… ${Math.floor(secs / 60)}:${String(secs % 60).padStart(2, "0")}`;
        setTimeout(tick, 10000);
      };
      tick();
    });

    // Live status sits at the top of the Cloud group (not a separate card).
    const cloudGroup = slot.querySelector('.setting-group[data-group="Remote infra"]');
    if (cloudGroup) {
      panel.className = "cloud-inline";
      panel.firstChild.remove();                   // drop its own heading
      cloudGroup.insertBefore(panel, cloudGroup.children[1] || null);
    } else {
      slot.appendChild(panel);
    }

    function setState(state, itype) {
      dot.className = "cloud-dot " + (state || "");
      stateEl.textContent = state || "unavailable";
      typeEl.textContent = itype ? itype : "";
    }

    async function refresh() {
      try {
        const r = await fetch("/api/cloud/status");
        const s = await r.json();
        if (!r.ok || s.error) {
          setState("", null);
          msgEl.textContent = s.error || "Could not query instance.";
          return;
        }
        setState(s.state, s.instance_type);
        tunnelEl.textContent = s.state === "running"
          ? (s.tunnel ? "🔒 Tunnel connected" : "Tunnel not connected")
          : "";
        tunnelEl.className = "cloud-tunnel" + (s.tunnel ? " up" : "");
        stopBtn.disabled = s.state !== "running";
        startBtn.disabled = s.state !== "stopped";
        msgEl.textContent =
          s.state === "running"
            ? "Running \u2014 parks itself after an idle hour."
            : s.state === "stopped"
              ? "Parked \u2014 no compute charges. Dictating wakes it up."
              : s.state === "pending"
                ? "Starting up \u2014 ready in about 2 minutes."
                : s.state === "stopping"
                  ? "Shutting down\u2026"
                  : "";
      } catch (e) {
        setState("", null);
      }
    }

    async function control(action) {
      stopBtn.disabled = startBtn.disabled = true;
      setState(action === "stop" ? "stopping" : "pending", null);
      try {
        const r = await fetch("/api/cloud/control", {
          method: "POST",
          headers: {"Content-Type": "application/json"},
          body: JSON.stringify({action}),
        });
        const res = await r.json();
        if (!r.ok || res.error) {
          msgEl.textContent = res.error || "Request failed.";
        } else if (action === "start") {
          msgEl.textContent =
            "Starting \u2014 ready in about 2 minutes. VoiceClip reconnects on its own.";
        }
      } catch (e) {
        msgEl.textContent = "Request failed.";
      }
      // State transitions take a while; poll a few times.
      setTimeout(refresh, 2000);
      setTimeout(refresh, 10000);
      setTimeout(refresh, 30000);
    }

    stopBtn.addEventListener("click", () => {
      if (confirm("Park the cloud server now? Your next dictation wakes it (about 2 minutes)."))
        control("stop");
    });
    startBtn.addEventListener("click", () => control("start"));

    refresh();
  }

  // Booleans render as compact chips in a grid (label + switch), not rows.
  function renderToggleChip(key, schema, value) {
    const [label, desc] = SETTING_COPY[key] || [key, null];
    const chip = el("label", {class: "toggle-chip", title: desc || ""});
    chip.dataset.key = key;
    const cb = document.createElement("input");
    cb.type = "checkbox";
    cb.checked = Boolean(value);
    cb.addEventListener("change", () => commitSetting(key, cb.checked));
    chip.appendChild(el("span", {class: "chip-text"}, [
      el("span", {class: "chip-label"}, [label,
        schema.restart_required ? el("span", {class: "restart-mark", "aria-label": "restart required"}, "↻") : null]),
      desc ? el("span", {class: "chip-desc"}, desc) : null,
    ]));
    chip.appendChild(cb);
    chip.appendChild(el("span", {class: "setting-status", "aria-live": "polite"}, ""));
    return chip;
  }
  function renderSettingRow(key, schema, currentValue, data) {
    const [label, desc] = SETTING_COPY[key] || [key, null];
    const row = el("div", {class:"setting-row"});
    row.dataset.key = key;
    if (schema.type === "text_list" || key === "polish_prompt") row.classList.add("textarea-row");
    const labelEl = el("div", {class:"setting-label"}, [
      el("span", null, [
        label,
        schema.restart_required
          ? el("span", {class: "restart-mark", title: "Takes effect after restarting voiceclip",
                        "aria-label": "restart required"}, "↻")
          : null,
      ]),
      desc ? el("span", {class:"desc"}, desc) : null,
    ]);
    row.appendChild(labelEl);
    const inputWrap = el("div", {class:"setting-input"});
    const primary = buildSettingInput(key, schema, currentValue);
    inputWrap.appendChild(primary);
    const modeKey = key + "_mode";
    const modeSchema = data && data.schema[modeKey];
    if (modeSchema) {
      row.dataset.modeKey = modeKey;
      inputWrap.classList.add("paired");
      const modeInput = buildSettingInput(modeKey, modeSchema, data.values[modeKey]);
      modeInput.classList.add("mode-select");
      modeInput.setAttribute("aria-label", `${label} mode`);
      const syncMode = () => {
        const v = primary.value;
        modeInput.hidden = v === "__null__" || v === "" || v == null;
      };
      syncMode();
      primary.addEventListener("change", syncMode);
      inputWrap.appendChild(modeInput);
    }
    row.appendChild(inputWrap);
    row.appendChild(el("div", {class:"setting-status", "aria-live": "polite"}, ""));
    return row;
  }

  function buildAiModelPicker(key, value) {
    const sel = document.createElement("select");
    sel.className = "ai-model-select";
    const loading = document.createElement("option");
    loading.textContent = "Loading models…";
    sel.appendChild(loading);
    sel.disabled = true;
    let current = value || "none";
    fetch("/api/ai/models").then(r => r.json()).then(data => {
      sel.innerHTML = "";
      const off = document.createElement("option");
      off.value = "none"; off.textContent = "Off";
      sel.appendChild(off);
      const groups = [["cloud", "Your cloud · private"], ["local", "This Mac · private"],
                      ["openai", "Third-party"], ["anthropic", "Third-party"]];
      const made = {};
      for (const m of data.models || []) {
        const gl = (groups.find(g => g[0] === m.provider) || [m.provider, m.provider])[1];
        if (!made[gl]) { made[gl] = document.createElement("optgroup"); made[gl].label = gl; sel.appendChild(made[gl]); }
        const o = document.createElement("option");
        o.value = m.value;
        o.textContent = `${m.label}${m.note ? " — " + m.note : ""}`;
        o.disabled = !m.available && m.value !== current;
        made[gl].appendChild(o);
      }
      sel.value = current;
      sel.disabled = false;
    }).catch(() => {
      loading.textContent = "Couldn't load models";
    });
    sel.addEventListener("change", async () => {
      const next = sel.value;
      const prov = next.split(":")[0];
      const wasThirdParty = ["openai", "anthropic"].includes(current.split(":")[0]);
      if (["openai", "anthropic"].includes(prov) && !wasThirdParty) {
        const ok = await confirmCloudSwitch(key, prov === "openai" ? "OpenAI" : "Anthropic");
        if (!ok) { sel.value = current; return; }
      }
      current = next;
      commitSetting(key, next);
    });
    return sel;
  }
  // "If the cloud is down" = which LOCAL engine AND model to fall back to,
  // chosen in one list. Writes cloud.fallback_engine + model/parakeet_model.
  function buildFallbackPicker(key, value) {
    const cache = _settingsCache || {schema: {}, values: {}};
    const sel = document.createElement("select");
    const add = (val, label, parent) => {
      const o = document.createElement("option");
      o.value = val; o.textContent = label; (parent || sel).appendChild(o);
    };
    add("none", "Off — show an error");
    const wg = document.createElement("optgroup"); wg.label = "Whisper · this Mac"; sel.appendChild(wg);
    for (const m of (cache.schema.model || {}).choices || []) add(`whisper:${m}`, choiceLabel("model", m), wg);
    const pg = document.createElement("optgroup"); pg.label = "Parakeet · this Mac"; sel.appendChild(pg);
    for (const m of (cache.schema.parakeet_model || {}).choices || []) add(`parakeet:${m}`, choiceLabel("parakeet_model", m), pg);
    const v = cache.values;
    let current = "none";
    if (value === "whisper" || value === "whisper_cpp") current = `whisper:${v.model}`;
    else if (value === "parakeet") current = `parakeet:${v.parakeet_model}`;
    if (value === "whisper_cpp") add(current = `whisper_cpp:${v.model}`, `whisper.cpp · ${choiceLabel("model", v.model)}`);
    sel.value = current;
    sel.addEventListener("change", async () => {
      const [engine, model] = sel.value.split(/:(.*)/s);
      await commitSetting(key, engine);
      if (engine === "whisper" || engine === "whisper_cpp") await commitSetting("model", model);
      if (engine === "parakeet") await commitSetting("parakeet_model", model);
    });
    return sel;
  }
  function buildSettingInput(key, schema, value) {
    if (schema.type === "ai_model") return buildAiModelPicker(key, value);
    if (key === "cloud.fallback_engine") return buildFallbackPicker(key, value);
    if (schema.type === "bool") {
      const cb = document.createElement("input");
      cb.type = "checkbox";
      cb.checked = Boolean(value);
      cb.addEventListener("change", () => commitSetting(key, cb.checked));
      const lab = el("label", {class:"toggle"}, [cb, value ? "On" : "Off"]);
      cb.addEventListener("change", () => {
        lab.lastChild.textContent = cb.checked ? "On" : "Off";
      });
      return lab;
    }
    if (schema.type === "select" || schema.type === "select_or_none") {
      const sel = document.createElement("select");
      if (schema.type === "select_or_none") {
        const opt = document.createElement("option");
        opt.value = "__null__";
        opt.textContent = "Off";
        sel.appendChild(opt);
      }
      for (const choice of schema.choices) {
        const opt = document.createElement("option");
        opt.value = choice;
        opt.textContent = choiceLabel(key, choice);
        sel.appendChild(opt);
      }
      // Select the current value; '__null__' for off
      sel.value = (value === null || value === undefined) ? "__null__" : String(value);
      sel.addEventListener("change", async () => {
        const newVal = sel.value === "__null__" ? null : sel.value;
        const isCloudProviderFlip =
          schema.cloud_providers &&
          schema.cloud_providers.indexOf(newVal) >= 0 &&
          (!value || schema.cloud_providers.indexOf(value) < 0);
        if (isCloudProviderFlip) {
          const ok = await confirmCloudSwitch(key, newVal);
          if (!ok) {
            sel.value = (value === null || value === undefined) ? "__null__" : String(value);
            return;
          }
        }
        commitSetting(key, newVal);
      });
      return sel;
    }
    if (schema.type === "int") {
      const inp = document.createElement("input");
      inp.type = "number";
      if (schema.min !== undefined) inp.min = String(schema.min);
      if (schema.max !== undefined) inp.max = String(schema.max);
      inp.value = value != null ? String(value) : "";
      inp.addEventListener("change", () => {
        const n = parseInt(inp.value, 10);
        if (Number.isNaN(n)) return;
        commitSetting(key, n);
      });
      return inp;
    }
    if (schema.type === "text_list") {
      // Multi-line textarea — each non-empty line is one entry. Save on
      // blur (avoid hammering the server per keystroke). Server will
      // trim, dedupe, and validate length per entry.
      const ta = document.createElement("textarea");
      ta.rows = 6;
      ta.className = "setting-textarea";
      if (schema.placeholder) ta.placeholder = schema.placeholder;
      const items = Array.isArray(value) ? value : [];
      ta.value = items.join("\n");
      ta.addEventListener("blur", () => {
        const lines = ta.value.split("\n").map(s => s.trim()).filter(Boolean);
        commitSetting(key, lines);
      });
      return ta;
    }
    // text
    const inp = /** @type {HTMLInputElement & { _t?: number }} */ (document.createElement("input"));
    inp.type = "text";
    inp.value = value ? String(value) : "";
    if (schema.placeholder) inp.placeholder = schema.placeholder;
    // Debounce text input to avoid hammering on every keystroke.
    inp.addEventListener("input", () => {
      clearTimeout(inp._t);
      inp._t = setTimeout(() => commitSetting(key, inp.value), 600);
    });
    inp.addEventListener("blur", () => {
      clearTimeout(inp._t);
      commitSetting(key, inp.value);
    });

    // For *.local_model fields, surface a curated "Quick pick" dropdown
    // above the text input. Selecting an option fills the input (and
    // commits via the input's own handler). Free-form paste still works.
    if (key.endsWith(".local_model")) {
      const wrap = document.createElement("div");
      wrap.className = "local-model-input";
      const picker = buildRecommendedPicker(key, inp);
      wrap.appendChild(picker);
      wrap.appendChild(inp);
      return wrap;
    }
    return inp;
  }

  // Build the "Quick pick" dropdown for a *.local_model field.
  // Populates from GET /api/models/recommended?feature=<summaries|research|patterns>.
  // Rendered immediately with a loading placeholder; options swap in when
  // the fetch resolves. If the fetch fails we quietly hide the dropdown
  // rather than showing an error — manual text input still works.
  /**
   * @param {string} key
   * @param {HTMLInputElement & { _t?: number }} targetInput
   */
  function buildRecommendedPicker(key, targetInput) {
    const feature = key.split(".")[0];  // summaries.local_model -> summaries
    const wrapper = document.createElement("div");
    wrapper.className = "recommended-picker";

    const select = document.createElement("select");
    select.className = "recommended-picker-select";
    const placeholder = document.createElement("option");
    placeholder.value = "";
    placeholder.textContent = "Quick pick a model…";
    select.appendChild(placeholder);
    wrapper.appendChild(select);

    const note = el("div", {class: "recommended-picker-note"}, "");
    wrapper.appendChild(note);

    (async () => {
      try {
        const r = await fetch(`/api/models/recommended?feature=${feature}`);
        const data = await r.json();
        const models = data.models || [];
        if (!models.length) {
          wrapper.style.display = "none";
          return;
        }
        for (const m of models) {
          const opt = document.createElement("option");
          opt.value = m.id;
          opt.dataset.note = m.note || "";
          opt.textContent = `${m.label} · ${m.size_gb.toFixed(1)} GB · ${m.backend}`;
          select.appendChild(opt);
        }
        // If the current value matches a known model, highlight it
        if (targetInput.value) {
          const match = Array.from(select.options).find(
            o => o.value === targetInput.value,
          );
          if (match) {
            select.value = match.value;
            note.textContent = match.dataset.note || "";
          }
        }
      } catch(e) {
        wrapper.style.display = "none";
      }
    })();

    select.addEventListener("change", () => {
      const chosen = select.value;
      if (!chosen) return;  // placeholder row
      targetInput.value = chosen;
      // Update note hint
      const opt = select.options[select.selectedIndex];
      note.textContent = opt.dataset.note || "";
      // Trigger the text input's own commit path — same debounce as a
      // manual paste would get.
      clearTimeout(targetInput._t);
      targetInput._t = setTimeout(
        () => commitSetting(key, chosen), 50,
      );
    });

    return wrapper;
  }

  async function commitSetting(key, value) {
    try {
      const r = await fetch("/api/settings/update", {
        method: "POST",
        headers: {"Content-Type":"application/json"},
        body: JSON.stringify({[key]: value}),
      });
      const data = await r.json();
      const row = document.querySelector(
        `.setting-row[data-key="${key}"], .setting-row[data-mode-key="${key}"], .toggle-chip[data-key="${key}"]`);
      // querySelector returns Element; cast to HTMLElement so .style works.
      const status = /** @type {HTMLElement | null} */ (
        row ? row.querySelector(".setting-status") : null
      );
      if (!r.ok) {
        if (status) {
          status.textContent = data.error || "failed";
          status.style.color = "#c44";
          status.classList.add("shown");
          setTimeout(() => status.classList.remove("shown"), 2200);
        }
        return;
      }
      if (status) {
        status.textContent = "saved";
        status.style.color = "var(--accent)";
        status.classList.add("shown");
        setTimeout(() => status.classList.remove("shown"), 1500);
      }
      if (data.restart_required) showRestartBanner();
      // Refresh cache so subsequent cloud-flip detection uses the new value
      if (_settingsCache) {
        _settingsCache.values[key] = value;
        // If this field is a visibility condition for other fields (e.g.
        // "engine" controls whether model/parakeet_model are shown),
        // re-render so the correct fields appear/disappear.
        const isVisibilityTrigger = Object.values(_settingsCache.schema).some(
          s => visibilityKeys(s.visible_when).includes(key)
        );
        if (isVisibilityTrigger) renderSettings(_settingsCache);
      }
    } catch(e) {}
  }

  function showRestartBanner() {
    const banner = document.getElementById("settings_restart_banner");
    banner.style.display = "";
    banner.textContent =
      "One of the settings you just changed needs a restart to apply. " +
      "Quit voiceclip (Ctrl+C in the terminal) and run it again to pick it up.";
  }

  function renderSystemInfo(sys) {
    const body = document.getElementById("settings_system_body");
    body.innerHTML = "";
    const rows = [
      ["Database",            sys.db_path],
      ["DB size",             `${(sys.db_size_mb || 0).toFixed(2)} MB`],
      ["Config",              sys.config_path],
      ["HuggingFace cache",   `${(sys.huggingface_cache_gb || 0).toFixed(1)} GB`],
      ["Journal enabled",     sys.history_enabled ? "yes" : "no"],
    ];
    for (const [k, v] of rows) {
      body.appendChild(el("div", {class:"sys-row"}, [
        el("div", null, k),
        el("div", {class:"sys-val"}, String(v || "")),
      ]));
    }
  }

  // ---------- Model cache management ----------
  //
  // Shows every cached HuggingFace model with size, last-used date, and
  // a Delete button. Destructive but reversible — HF redownloads on the
  // next use if the model is still configured.

  async function loadModels() {
    const body = document.getElementById("settings_models_body");
    body.innerHTML = "";
    body.appendChild(el("div", {class:"empty", style:"padding:16px"}, "Scanning cache…"));
    try {
      const r = await fetch("/api/models");
      const data = await r.json();
      renderModels(data);
    } catch(e) {
      body.innerHTML = "";
      body.appendChild(el("div", {class:"empty", style:"padding:16px"},
        "Could not list cached models."));
    }
  }

  function renderModels(data) {
    const body = document.getElementById("settings_models_body");
    body.innerHTML = "";

    if (data.error) {
      body.appendChild(el("div", {class:"models-error"},
        `Could not scan cache: ${data.error}`));
      return;
    }
    const models = data.models || [];
    if (!models.length) {
      body.appendChild(el("div", {class:"empty", style:"padding:16px"},
        "No cached models. Pick a local provider in Summaries / Research / " +
        "Patterns and they'll download on first use."));
      return;
    }

    // Header with total size
    body.appendChild(el("div", {class:"models-header"}, [
      el("span", null, `${models.length} model${models.length === 1 ? "" : "s"} on disk`),
      el("span", {class:"models-total"},
        `${data.total_size_gb.toFixed(1)} GB total`),
    ]));

    for (const m of models) {
      body.appendChild(renderModelRow(m));
    }
  }

  function renderModelRow(model) {
    const row = el("div", {class:"model-row"});
    row.dataset.repoId = model.repo_id;

    const title = el("div", {class:"model-title"}, [
      el("span", {class:"model-id"}, model.repo_id),
      model.in_use_for
        ? el("span", {class:"model-badge"}, `in use · ${model.in_use_for}`)
        : null,
    ]);

    const meta = el("div", {class:"model-meta"},
      `${model.size_on_disk_str} · ${model.last_accessed_str} · ` +
      `${model.nb_files} file${model.nb_files === 1 ? "" : "s"}`);

    const deleteBtn = el("button", {
      class: "model-delete",
      onclick: ev => handleModelDelete(ev.target, row, model),
    }, "Delete");

    row.appendChild(title);
    row.appendChild(meta);
    row.appendChild(deleteBtn);
    return row;
  }

  // Two-click confirm: first click arms the button with a warning that
  // names the model + freed size; second click within 4s commits. Dictation
  // model comes back from the server with a hard refusal, so no arming
  // needed — we surface the 409 as an inline error.
  function handleModelDelete(btn, row, model) {
    const warning = model.in_use_for
      ? `Delete ${model.size_on_disk_str}? (${model.in_use_for} will re-download)`
      : `Delete ${model.size_on_disk_str}?`;
    twoClickConfirm(
      btn,
      () => deleteModel(btn, row, model),
      {timeoutMs: 4000, confirmLabel: warning},
    );
  }

  async function deleteModel(btn, row, model) {
    btn.disabled = true;
    btn.textContent = "Deleting…";
    btn.classList.remove("armed");
    try {
      const r = await fetch("/api/models/delete", {
        method: "POST",
        headers: {"Content-Type": "application/json"},
        body: JSON.stringify({repo_id: model.repo_id}),
      });
      const data = await r.json();
      if (!r.ok) {
        btn.disabled = false;
        btn.textContent = "Delete";
        // Surface a precise error inline so the user understands why
        // (e.g. "this is the dictation model currently in use").
        const err = el("div", {class:"model-error"}, data.error || "Delete failed");
        row.appendChild(err);
        setTimeout(() => err.remove(), 6000);
        return;
      }
      // Success — fade the row out, then refresh the whole list so the
      // header total updates too.
      row.classList.add("removing");
      setTimeout(() => loadModels(), 360);
    } catch(e) {
      btn.disabled = false;
      btn.textContent = "Delete";
    }
  }

  // ---------- Cloud-switch confirm modal ----------

  function confirmCloudSwitch(key, newProvider) {
    return new Promise((resolve) => {
      const body = CLOUD_DISCLOSURES[key] ||
        "This setting will send data to a cloud provider.";
      const root = document.getElementById("modal_root");
      const backdrop = el("div", {class:"modal-backdrop"});
      const modal = el("div", {class:"modal"}, [
        el("h3", null, "Switch to cloud provider?"),
        el("div", {class:"modal-body"}, [
          el("p", null, [
            "You're about to set ",
            el("strong", null, key),
            " to ",
            el("strong", null, newProvider),
            ".",
          ]),
          el("p", null, body),
          el("p", null, [
            "Providers typically retain API data for about 30 days for abuse detection. ",
            "API keys must come from environment variables (",
            el("code", null, "OPENAI_API_KEY"),
            " / ",
            el("code", null, "ANTHROPIC_API_KEY"),
            ").",
          ]),
        ]),
        el("div", {class:"modal-actions"}, [
          el("button", {onclick: () => { close(false); }}, "Cancel"),
          el("button", {class:"primary", onclick: () => { close(true); }}, "Yes, switch"),
        ]),
      ]);
      backdrop.appendChild(modal);
      root.appendChild(backdrop);

      function close(ok) {
        root.removeChild(backdrop);
        resolve(ok);
      }
      // Click-outside to cancel
      backdrop.addEventListener("click", (ev) => {
        if (ev.target === backdrop) close(false);
      });
    });
  }

  // ---------- Cloud-provider consent banner ----------
  // Shown when GET /api/consent returns any `pending` features. Stays
  // up until the user clicks "I understand, continue" — which POSTs to
  // /api/consent/ack to record the current cloud config as the new
  // baseline. Reappears automatically next load if the user flips
  // another feature to cloud.

  async function loadConsentBanner() {
    const slot = $id("consent_banner_slot");
    slot.innerHTML = "";
    let data;
    try {
      const r = await fetch("/api/consent");
      data = await r.json();
    } catch (e) { return; }  // silent — banner is non-critical

    const pending = data.pending || {};
    const features = Object.keys(pending);
    if (!features.length) return;

    const rows = features.sort().map(f => {
      const info = pending[f];
      return el("div", {class: "consent-row"}, [
        el("div", {class: "consent-feature"}, [
          el("strong", null, f),
          " · ",
          el("span", {class: "consent-provider"},
            `${info.provider} · ${info.model}`),
        ]),
        el("div", {class: "consent-data-sent"}, info.data_sent),
      ]);
    });

    const dismissBtn = el("button", {
      class: "primary",
      onclick: async () => {
        try {
          await fetch("/api/consent/ack", {
            method: "POST",
            headers: {"Content-Type": "application/json"},
            body: JSON.stringify({}),
          });
        } catch (e) { /* ignore */ }
        slot.innerHTML = "";
      },
    }, "I understand, continue");

    const banner = el("div", {class: "consent-banner"}, [
      el("div", {class: "consent-header"}, [
        el("strong", null, "⚠️ Cloud provider change detected"),
      ]),
      el("div", {class: "consent-body"}, rows),
      el("div", {class: "consent-note"},
        "When these features run, data leaves your Mac. Providers typically retain API data for ~30 days. " +
        "Flip the provider back to \"none\" in Settings if unintended."),
      el("div", {class: "consent-actions"}, dismissBtn),
    ]);
    slot.appendChild(banner);
  }

  load(state.date);
  loadConsentBanner();
})();
