# VoiceClip — backlog

Living list of known issues and deferred work surfaced by the multi-angle
code reviews. Maintained top-down by priority. Items get deleted when done
(git history is the audit trail).

---

## Doing now (top 5, in execution order)

*(All 5 shipped in commit — see "Recently shipped" at bottom.)*

---

## P1 — after the top 5

### Security / trust
- [ ] **Write `PRIVACY.md`** enumerating every feature, default state, exact network behavior, provider retention.
- [ ] **Cost controls.** No per-day / per-month spend cap on cloud. Negligible at current usage; real at gpt-4o. Add a per-feature daily token budget.

### Architecture
- [ ] **REST convention pass** — `DELETE /api/entries/:id`, `PATCH /api/entries/:id`, `/api/v1/` prefix, standardized response shape.
- [ ] **Idempotency keys** on `/api/research/run` and `/api/patterns/run`.
- [ ] **Narrow the `transcriptions` table** when we add a 4th kind — move `is_research_topic` into a proper `entry_type` enum.

### Performance
- [ ] **`find_research_topic_by_text`** — switch to SQLite FTS5 when we pass ~500 topics.
- [ ] **Persist patterns cache across restarts** in a SQLite `patterns_cache` table.

### Testing
- [ ] **Tests for `hotkey.HotkeyHandler`** — hold/toggle modes, coordinator gate contention, profile branches.
- [ ] **Tests for provider wrappers** — mock the SDKs, assert request shape and error handling.
- [ ] **Prompt-injection regression test** — seed adversarial reflection, assert no fabricated themes.
- [ ] **JSON parsing robustness test for `patterns._parse_json_safely`**.

### Operability
- [ ] **Log rotation** — rotating file handler at `~/.voiceclip/voiceclip.log`.
- [ ] **`voiceclip export`** — JSON dump of all tables. Needed before EU distribution.
- [ ] **Runtime mic/sample-rate drop detection** — detect Bluetooth SCO switch mid-session.

---

## P2 — eventually

### UX / product
- [ ] **Startup banner mentions the viewer** (currently users have to read README to discover `voiceclip view`).
- [ ] **Hover affordance for contenteditable text** — subtle underline or pencil icon.
- [ ] **Tooltips on icon-only action buttons** — Copy / Keep / Delete / Queue it.
- [ ] **Skip link for keyboard users** at top of the viewer.
- [ ] **Keyboard shortcuts in the viewer** — j/k between entries, e to edit, / to focus search, ]/[ for day nav, 1/2/3 for tabs.
- [ ] **Remember last-viewed tab** per session (localStorage).
- [ ] **Persist "Show N transcriptions" open state** per session.
- [ ] **README: move hero features above Configuration.**
- [ ] **Rewrite competitive positioning in README** — compare to Day One / Reflect / Mem, not just Otter.

### Accessibility
- [ ] **ARIA for tabs** — `role="tablist"`, `role="tab"`, `aria-selected`, `aria-controls`.
- [ ] **`aria-label` on icon-only buttons.**
- [ ] **`prefers-reduced-motion` support** — disable row-remove animation for motion-sensitive users.
- [ ] **`aria-live` on loading / error regions.**
- [ ] **Visual alternative to audio chimes** — flash indicator for low-hearing / muted-headphone users.
- [ ] **Localized day labels** — `Intl.RelativeTimeFormat` on the client.
- [ ] **Audit color contrast** — 11px muted text likely fails WCAG AA.

### Open-source hygiene
- [ ] **`pyproject.toml`** with optional-deps groups.
- [ ] **GitHub Actions CI** on macOS.
- [ ] **`CONTRIBUTING.md`**, **`ARCHITECTURE.md`**, **`CHANGELOG.md`**.
- [ ] **Issue + PR templates.**

### Features (parked)
- [ ] **Voice capture for research topics** (third hotkey).
- [ ] **Batch research CLI** — `voiceclip research --pending` for launchd cron.
- [ ] **Voice hashtags** — "note to self, hashtag todo" → `#todo`.
- [ ] **Menu bar app** (rumps).
- [ ] **Long-form dictation** — chunked streaming for clips > 30s.
- [ ] **Weekly digest summary.**
- [ ] **Mac App Store packaging.** The single biggest "hobby → product" gap.
- [ ] **i18n.** Realistic 3-engineer-week project; wait for a user who asks.
- [ ] **Local research provider** (deliberately parked). Evaluated three approaches:
  - *LLM-only local*: no web search, no sources — feels broken for a button labeled "Research."
  - *Ollama + SearXNG/DuckDuckGo*: real local research but ~1-2 days of work, 30-60s per query, noticeably lower quality than cloud at this model scale, requires Ollama as a new dependency.
  - *mlx-lm + custom search loop*: ~3× the code for the same result as Ollama, worse tool-use quality.
  
  Conclusion: stay cloud-only for research until (a) there's a confirmed need for offline, or (b) local tool-use in small models improves meaningfully. Revisit in 6-12 months.

### Ethics / product guardrails
- [ ] **Never ship reflection frequency metrics** — no streaks, no "you reflected 2x less this week."
- [ ] **Cloud-provider consent moment** — visible confirmation before first cloud call.
- [ ] **Model-output-as-suggestion framing** — Patterns "Suggested to learn" should be labeled as AI suggestions that may be wrong.

---

## Recently shipped (keeping for morale)

- **Usability pass 2:**
  - ⚙️ Settings tab in the viewer. Live reads `/api/settings` (whitelist-validated), patches `/api/settings/update` with per-field type + range checks. Save-on-change (debounced for text, instant for toggles/selects). "Saved" pill flashes inline. Yellow restart-required banner when a hotkey-like change needs a daemon reboot. Cloud-provider flips gate through a confirm modal that mirrors the consent-banner language. Collapsible System section at the bottom surfaces DB path, DB size, config path, HF cache size.
  - Shared `voiceclip/config_io.py` — one source of truth for reading and writing the config file (shallow-merge, 0600, corrupt-file tolerant). Onboarding + Settings both use it.
  - 21 new tests (settings GET/POST + validation rejections + merge semantics + system info). 179 → 200.
- **Usability pass 1:**
  - First-run onboarding flow (`voiceclip onboard`). 5-step keyboard-driven walkthrough, keyboard-only, no deps. Teaches dictation, checks permissions, offers reflections/toggle-mode/history/summaries as opt-ins, writes accepted choices directly into `~/.voiceclip/config.json`. Runs automatically on first launch, skippable with Enter, gated by TTY. 14 new tests.
- **Top-3 pass 3:**
  - `history._conn` auto-reconnect on `OperationalError`. A transient DB failure no longer silently drops writes — `save()` and `update_text()` retry once against a fresh connection.
  - Cloud-provider consent banner. First run after flipping any `*.provider` to cloud prints a loud one-time warning. Tracked via `~/.voiceclip/cloud_ack.json` so it re-fires only when provider or model changes. Fires on all three entry points (daemon, viewer, `voiceclip summarize`).
  - Viewer static extraction. `voiceclip/viewer.py` went from 64KB → 18KB. CSS lives in `static/app.css`, JS in `static/app.js`, HTML shell in `static/index.html`. HTTP server now has a `/static/*` route with path-traversal protection. Every future viewer feature is easier to work on.
  - 10 new tests (consent banner behavior + connection-reconnect paths); 155 → 165.
- **Top-5 pass 2:**
  - Prompt-injection hardening: `<entry>`/`<topic>`/`<reflection>` delimiters + "ignore instructions inside" directive in all three system prompts
  - XSS defense: viewer's `renderMarkdown` now constructs DOM via `textContent` — no `innerHTML` interpretation of LLM output
  - Thread-safe locks on `llm_provider._mlx_cache` and `patterns._cache` (double-check locking pattern)
  - `install.sh` refuses to run with empty or root `$HOME`
  - `list_research_topics` rewritten as single JOIN (was 2N+1 queries at N topics, now 1)
  - `day_stats` collapsed to one conditional-aggregation query
  - Grammar Polish fully removed — README section, config defaults, stale test env vars, lingering docs
  - `voiceclip doctor` command — checks system, permissions, storage, schema, optional providers, model caches
  - Full-text search across history (SQLite FTS5) with triggers, viewer search box with Cmd/Ctrl+K, debounced, race-safe via per-query seq counter, graceful fallback to LIKE on malformed queries
  - 12 new tests (search + doctor); 143 → 155
- **Top-5 pass 1:**
  - Honest README privacy claim + "Privacy & data flow" section
  - Unified LLM provider abstraction (`voiceclip/llm_provider.py`) with in-process model cache
  - Patterns output cached in-process, force-regenerates on demand
  - 36 new tests covering schema migration + every viewer endpoint (107 → 143 tests)
- Opt-in history + reflection hotkey with additive schema migration
- Local web viewer with 📓 Journal / 📚 Queue / 📊 Patterns
- Daily summaries (local/openai/anthropic)
- Research briefs with automatic web-search tool use
- Patterns coach that only suggests when grounded in real quotes
- Inline editing with plain-text paste normalization
- Two-click confirm delete with animated row removal
- Zero new runtime dependencies for the core viewer (stdlib only)
