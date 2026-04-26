# VoiceClip — backlog

Living list of known issues and deferred work surfaced by the multi-angle
code review. Maintained top-down by priority. Items get deleted when done
(git history is the audit trail).

---

## P0 — doing now

- [ ] **Fix README privacy claim.** The top-of-README "No cloud, no subscription, no data leaves your laptop" is false once summaries/research/patterns ship with cloud providers. Rewrite to "local by default; cloud opt-in per feature."
- [ ] **Unify LLM provider abstraction.** `summarizer.py`, `researcher.py`, `patterns.py` each re-implement local/openai/anthropic plumbing. Extract a `voiceclip/llm_provider.py` with one `complete()` entry point. Target: ~300 lines deduplicated.
- [ ] **Cache `mlx_lm.load()` output in-process.** Every call to summarizer/patterns reloads the model (5-15s). One module-level dict keyed by model_id fixes this.
- [ ] **Cache Patterns output** the way summaries are cached. Key: `(window_end_date, transcription_count + reflection_count)`. Regen on stale, serve from cache otherwise.
- [ ] **Integration tests for viewer endpoints + schema migration.** At minimum: one `urllib` round-trip per endpoint, plus migration from pre-feature schema with seeded old rows.

---

## P1 — next

### Security / trust
- [ ] **XSS hardening in `renderMarkdown` (viewer.py).** LLM output is rendered via `innerHTML`. Prompt injection through user reflections → HTML execution. Switch to DOM construction with `textContent` for each segment.
- [ ] **`install.sh` guard against empty `$HOME`.** Add `[[ -n "$HOME" ]]` at top to prevent `rm -rf /.voiceclip/voiceclip` if env is broken.
- [ ] **Write `PRIVACY.md`** that lists every feature, default state, exact network behavior, provider retention policies (OpenAI ~30d, Anthropic ~30d).
- [ ] **Config-time cloud-provider confirmation.** When a user edits config to turn on a cloud provider, the first `voiceclip view` or `voiceclip summarize` should print a one-time visible warning before proceeding.

### Architecture
- [ ] **Extract embedded HTML/CSS/JS from `viewer.py`.** Move `_PAGE_HTML` to `voiceclip/static/index.html` + `static/app.js` + `static/app.css`. Serve from `ThreadingHTTPServer` via a simple static handler. Makes JS debuggable and `viewer.py` half its current size.
- [ ] **Reconnect `history._conn` on failure.** Currently if the SQLite connection dies mid-session, all subsequent writes silently drop. Add a lazy-reconnect on `OperationalError`.
- [ ] **Narrow the `transcriptions` table.** `is_research_topic` on a row typed `transcription` is a smell. Longer term this wants a proper `entry_type` enum (clip, reflection, topic). Migration cost not justified yet; flag for when we add a 4th kind.

### Performance
- [ ] **Rewrite `list_research_topics` as one JOIN.** Currently runs 2 subqueries per topic row. At 1000 topics this becomes slow.
- [ ] **Cap recorder `frames` buffer.** Currently grows unbounded — a 10-minute recording is ~115MB RAM. Spill to disk after N seconds, or hard-cap at 120s with a "recording too long" stop.
- [ ] **`find_research_topic_by_text` will get quadratic.** Current O(N) Python matching over last 50 topics is fine; switch to SQL full-text or trigram once we hit ~500 topics.

### Testing
- [ ] **Tests for the schema migration.** Seed old-schema DB, run `init()`, assert new columns present, old rows backfilled, `user_version` doesn't regress.
- [ ] **Tests for `hotkey.HotkeyHandler`.** Mock Recorder/transcriber/beep/paste. Exercise hold-mode, toggle-mode, coordinator-gate contention, profile=reflection branches.
- [ ] **Tests for provider wrappers.** Mock the `openai`/`anthropic` SDKs, assert request shape + error handling (missing key, missing package, unsupported tool).
- [ ] **Test for JSON parsing robustness in `patterns._parse_json_safely`.** Feed it fenced output, prose-before, prose-after, invalid JSON.

### Operability
- [ ] **Log rotation.** Currently stdout only; a long-running daemon accumulates all logs in terminal scrollback. Add a rotating file handler at `~/.voiceclip/voiceclip.log`.
- [ ] **`voiceclip doctor` diagnostic command.** Checks: mic permission, accessibility permission, SQLite integrity, model cache path sizes, `mlx-lm/openai/anthropic` installed if provider configured. Print one-line status per check.
- [ ] **Uninstall script for clean removal.** Document and script: `~/.voiceclip/` + `~/.cache/huggingface/` (the big one, 3-10GB) + `~/.local/bin/voiceclip`. README only mentions the first.
- [ ] **Runtime mic/sample-rate drop detection.** Currently only warned at startup. Detect mid-session switch to Bluetooth SCO and notify.

---

## P2 — eventually

### UX / product
- [ ] **Startup banner mentions the viewer.** Currently users have to read README to discover `voiceclip view`.
- [ ] **Hover affordance for contenteditable text.** Subtle underline or pencil icon so users know they can click to edit.
- [ ] **Tooltips on icon-only action buttons.** Copy / Keep / Delete / Queue it all benefit from `title` attributes for the first few weeks.
- [ ] **Skip link for keyboard users** at top of the viewer for fast access to tab nav.
- [ ] **"Show N transcriptions" toggle state persistence.** Currently collapses on every render. Remember the user's preference per-session.
- [ ] **Remove or archive the Grammar Polish feature.** README marks it opt-in; code and config keys exist but it's not wired into the hotkey path anymore. Either re-integrate or delete cleanly.

### Accessibility
- [ ] **ARIA for tabs.** `role="tablist"`, `role="tab"`, `aria-selected`, `aria-controls`. One pass, half a day.
- [ ] **`aria-label` on icon-only buttons.** Copy/Keep/Delete/Queue it — screen readers need textual equivalents.
- [ ] **`prefers-reduced-motion` support.** Disable the 350ms row-remove animation for users with motion sensitivity.
- [ ] **`aria-live` on loading / error regions.** "Thinking…" and error messages are currently visual-only.
- [ ] **Visual alternative to audio chimes.** Flash the menu bar or a viewer indicator. Low-hearing users and muted-headphone users get zero feedback today.
- [ ] **Localized day labels.** "Today/Yesterday/Monday" in `viewer._friendly_day_label` are English-only. Switch to `Intl.RelativeTimeFormat` on the client if we want i18n.
- [ ] **Audit color contrast.** Small 11px muted text likely fails WCAG AA. Measure and bump if needed.

### Open-source hygiene
- [ ] **`pyproject.toml`** with proper optional-dependencies groups (`[local]` → mlx-lm, `[openai]` → openai, `[anthropic]` → anthropic).
- [ ] **GitHub Actions CI** running pytest on macOS (matrix: py3.10/3.11/3.12).
- [ ] **`CONTRIBUTING.md`** with the module layout, testing instructions, and conventions.
- [ ] **`ARCHITECTURE.md`** at repo root — one-paragraph-per-module overview.
- [ ] **`CHANGELOG.md`** auto-populated from git log, or manually maintained from here on.
- [ ] **Issue + PR templates.**

### Features (user-facing, parked)
- [ ] **Voice capture for research topics.** Third hotkey ("hold F7 to dictate a research topic") so users can queue research without opening the viewer.
- [ ] **Batch research CLI.** `voiceclip research --pending` to run all queued briefs. Can be hooked into `launchd` for morning digest.
- [ ] **Mark-as-read / archive for briefs.** Currently `status=ready` is terminal; no archiving.
- [ ] **Voice hashtags.** "Note to self, hashtag todo" → entry tagged `#todo`. Opens door to organization without typing.
- [ ] **Menu bar app.** Native rumps-based icon with "recent" dropdown and "open viewer" action. Big UX lever, medium effort.
- [ ] **Long-form dictation.** Chunked streaming transcription so clips > 30s stay usable.
- [ ] **Weekly digest summary** cached like daily summaries. Feeds Patterns without re-reading raw entries.

---

## Wins we shipped (keeping for morale)

- Opt-in history + reflection hotkey with additive schema migration
- Local web viewer with 📓 Journal / 📚 Queue / 📊 Patterns
- Daily summaries (local/openai/anthropic)
- Research briefs with automatic web-search tool use
- Patterns coach that only suggests when grounded in real quotes
- Inline editing with plain-text paste normalization
- Two-click confirm delete with animated row removal
- Auto-refresh today's page while viewing it
- Zero new runtime dependencies for the core viewer (stdlib only)
