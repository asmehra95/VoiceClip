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
    $id("journal_nav").style.visibility = view === "journal" ? "" : "hidden";
    // Clear stale search on tab-switch
    if (view !== "journal") {
      const si = $input("search_input");
      const sc = $id("search_clear");
      si.value = "";
      sc.style.display = "none";
    }
    if (view === "queue") loadQueue();
    else if (view === "patterns") loadPatterns();
    else if (view === "settings") loadSettings();
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
  const searchInput = $input("search_input");
  const searchClear = $id("search_clear");

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
    // setTimeout id stashed on the function object for the same reason as
    // the inp._t idiom — debounce state has to live somewhere stable.
    /** @type {any} */
    const ds = debounceSearch;
    clearTimeout(ds._t);
    ds._t = setTimeout(() => {
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

  // ---------- Settings ----------
  //
  // Loads the schema + current values from /api/settings, renders one input
  // per setting grouped by category, and commits changes on change (with
  // debounce for text inputs). Cloud-provider flips gate through a confirm
  // modal that mirrors the consent banner's language.

  // Plain-language labels and descriptions. Keyed by dotted config key.
  const SETTING_COPY = {
    "model":                   ["Whisper model", "Larger models are more accurate but slower. Turbo is the sweet spot on Apple Silicon. Tiny / base are fast; large-v3 is highest quality."],
    "hotkey":                  ["Hotkey", "Key to hold/press for dictation"],
    "hotkey_mode":             ["Hotkey mode", "Hold to record, or tap to toggle"],
    "english_only":            ["English only", "Faster and smaller if all your dictation is English"],
    "reflection_hotkey":       ["Reflection hotkey", "Separate key for saving a thought (not pasted)"],
    "reflection_hotkey_mode":  ["Reflection hotkey mode", "Hold or toggle for reflections specifically"],
    "window_title_capture":    ["Capture window titles", "Stored alongside each entry. Uses Accessibility."],
    "history":                 ["Save dictations to a journal", "Enables the Journal/Queue/Patterns tabs"],
    "summaries.provider":      ["Daily summary provider", "'local' runs on your Mac; 'openai'/'anthropic' send a day's entries"],
    "summaries.local_model":   ["Local model", "e.g. mlx-community/Qwen2.5-7B-Instruct-4bit"],
    "summaries.openai_model":  ["OpenAI model", null],
    "summaries.anthropic_model": ["Anthropic model", null],
    "summaries.style":         ["Summary style", "Descriptive (what you did) or Reflective (what you were thinking)"],
    "research.provider":       ["Research provider", "Local stays on your Mac (no web search); cloud can search the web per topic."],
    "research.local_model":    ["Local model", "Reuses mlx-lm. Any HuggingFace repo with -mlx or an MLX-compatible fork."],
    "research.openai_model":   ["OpenAI model", null],
    "research.anthropic_model":["Anthropic model", null],
    "patterns.provider":       ["Patterns provider", "Reads up to a week of entries. 'local' stays on your Mac."],
    "patterns.local_model":    ["Local model", null],
    "patterns.openai_model":   ["OpenAI model", null],
    "patterns.anthropic_model":["Anthropic model", null],
    "patterns.window_days":    ["Window (days)", "How many days of history to read"],
    "custom_vocabulary":       ["Custom vocabulary", "Words and phrases that bias dictation. One per line. Names, jargon, acronyms — anything Whisper keeps getting wrong."],
  };

  // Cloud-provider confirm copy, keyed by the dotted setting key.
  // Used by confirmCloudSwitch().
  const CLOUD_DISCLOSURES = {
    "summaries.provider": "Your entries for each day (every transcription and reflection) will be sent to this provider when a summary is generated.",
    "research.provider":  "The research topic you dictate or type will be sent to this provider. If the model uses its web search tool, that topic also goes to the search backend.",
    "patterns.provider":  "Up to a week of your reflections and daily summaries will be sent in a single prompt.",
  };

  let _settingsCache = null;

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

  function renderSettings(data) {
    const slot = document.getElementById("settings_slot");
    slot.innerHTML = "";
    document.getElementById("settings_status").textContent =
      "Changes save automatically. Some require restarting voiceclip.";

    // Group by schema.group
    const groups = {};
    for (const key in data.schema) {
      const g = data.schema[key].group;
      if (!groups[g]) groups[g] = [];
      groups[g].push(key);
    }

    const order = ["Dictation", "Reflections", "Journal", "Summaries", "Research", "Patterns"];
    for (const gname of order) {
      if (!groups[gname]) continue;
      const groupEl = el("div", {class:"setting-group"}, [
        el("h3", null, gname),
      ]);
      for (const key of groups[gname]) {
        groupEl.appendChild(renderSettingRow(key, data.schema[key], data.values[key]));
      }
      slot.appendChild(groupEl);
    }

    renderSystemInfo(data.system || {});
  }

  function renderSettingRow(key, schema, currentValue) {
    const [label, desc] = SETTING_COPY[key] || [key, null];
    const row = el("div", {class:"setting-row"});
    row.dataset.key = key;
    // Multi-line inputs get a stacked layout so the textarea can breathe.
    if (schema.type === "text_list") row.classList.add("textarea-row");

    const labelEl = el("div", {class:"setting-label"}, [
      el("span", null, [
        label,
        schema.restart_required
          ? el("span", {class:"restart-pill"}, "restart")
          : null,
      ]),
      desc ? el("span", {class:"desc"}, desc) : null,
    ]);
    row.appendChild(labelEl);

    const inputWrap = el("div", {class:"setting-input"});
    inputWrap.appendChild(buildSettingInput(key, schema, currentValue));
    row.appendChild(inputWrap);

    const status = el("div", {class:"setting-status"}, "");
    row.appendChild(status);
    return row;
  }

  function buildSettingInput(key, schema, value) {
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
        opt.textContent = "(off)";
        sel.appendChild(opt);
      }
      for (const choice of schema.choices) {
        const opt = document.createElement("option");
        opt.value = choice;
        opt.textContent = choice;
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
      const row = document.querySelector(`.setting-row[data-key="${key}"]`);
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
      if (_settingsCache) _settingsCache.values[key] = value;
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

  load(state.date);
})();
