"""Transcription & reflection history — local SQLite storage.

Opt-in via config.json: "history": true (default: false).
Stores every dictation when enabled. A dedicated reflection hotkey can
save "reflection" entries into the same table (discriminated by `kind`).

DB location: ~/.voiceclip/history.db (chmod 600)
"""

import logging
import os
import sqlite3
import threading
from datetime import datetime, timedelta

from voiceclip.config import CONFIG_DIR

log = logging.getLogger(__name__)

DB_PATH = os.path.join(CONFIG_DIR, "history.db")

_conn: sqlite3.Connection | None = None
_write_lock = threading.Lock()

VALID_KINDS = ("transcription", "reflection")


# ---------------------------------------------------------------------------
# Connection recovery
# ---------------------------------------------------------------------------

def _reconnect() -> bool:
    """Re-open the SQLite connection after a failure. Returns True on success.

    Called from `_with_retry` when a write raises OperationalError. We close
    the stale connection (best effort) and re-run init(); init is idempotent
    and will run migration again (also idempotent).
    """
    global _conn
    log.warning("Reconnecting history DB after error...")
    try:
        if _conn is not None:
            try:
                _conn.close()
            except Exception:
                pass
        _conn = None
        init()
        return _conn is not None
    except Exception as e:
        log.warning("Reconnect failed: %s", e)
        _conn = None
        return False


def _with_retry(operation, *args, **kwargs):
    """Run a write-path callable, reconnecting once on OperationalError.

    operation(*args, **kwargs) is expected to access `_conn` internally (via
    closure). We don't pass the connection — the caller owns that reference
    and will pick up the new _conn on retry.

    Returns whatever `operation` returns, or None if both attempts fail.
    """
    try:
        return operation(*args, **kwargs)
    except sqlite3.OperationalError as e:
        log.warning("History write failed (%s). Reconnecting and retrying once.", e)
        if not _reconnect():
            return None
        try:
            return operation(*args, **kwargs)
        except Exception as e2:
            log.warning("Retry also failed: %s. Giving up.", e2)
            return None


def init():
    """Initialize the history DB. Creates/migrates schema as needed."""
    global _conn
    try:
        os.makedirs(CONFIG_DIR, exist_ok=True)
        _conn = sqlite3.connect(DB_PATH, check_same_thread=False)
        # WAL mode for safe concurrent reads/writes from multiple threads
        _conn.execute("PRAGMA journal_mode=WAL")
        # Base table (legacy shape); additive migration below handles new columns.
        _conn.execute("""
            CREATE TABLE IF NOT EXISTS transcriptions (
                id INTEGER PRIMARY KEY,
                timestamp TEXT NOT NULL,
                raw_text TEXT NOT NULL,
                formatted_text TEXT NOT NULL,
                duration_seconds REAL,
                persona TEXT,
                model TEXT,
                word_count INTEGER
            )
        """)
        _conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_timestamp ON transcriptions(timestamp)"
        )
        _migrate(_conn)
        _conn.commit()
        os.chmod(DB_PATH, 0o600)
        log.info("History database ready at %s", DB_PATH)
    except Exception as e:
        log.warning("Could not initialize history database: %s", e)
        _conn = None


def _migrate(conn: sqlite3.Connection):
    """Additively add kind/app_name/window_title columns and their index.

    Idempotent: catches "duplicate column name" errors. Runs in the outer
    transaction opened by init().
    """
    # Add `kind` (default 'transcription' so existing rows are correctly backfilled)
    _add_column(conn, "kind", "TEXT NOT NULL DEFAULT 'transcription'")
    _add_column(conn, "app_name", "TEXT")
    _add_column(conn, "window_title", "TEXT")
    # Inline-edit tracking: nullable ISO timestamp, set only on edits
    _add_column(conn, "edited_at", "TEXT")
    # Research flag: entry is a topic for the research queue (not a normal clip)
    _add_column(conn, "is_research_topic", "INTEGER NOT NULL DEFAULT 0")
    # Backfill any pre-existing NULL kinds (shouldn't happen given the DEFAULT,
    # but harmless and explicit).
    conn.execute(
        "UPDATE transcriptions SET kind = 'transcription' WHERE kind IS NULL"
    )
    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_kind_timestamp "
        "ON transcriptions(kind, timestamp)"
    )
    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_research_topic "
        "ON transcriptions(is_research_topic, timestamp)"
    )
    # Daily summary cache (one row per day).
    conn.execute("""
        CREATE TABLE IF NOT EXISTS day_summaries (
            date TEXT PRIMARY KEY,
            summary TEXT NOT NULL,
            provider TEXT NOT NULL,
            model TEXT NOT NULL,
            style TEXT NOT NULL,
            generated_at TEXT NOT NULL,
            entry_count INTEGER NOT NULL
        )
    """)
    # Research briefs: zero-or-many per entry. Status lets us queue, mark
    # in-progress, record success/failure.
    conn.execute("""
        CREATE TABLE IF NOT EXISTS research_briefs (
            id INTEGER PRIMARY KEY,
            entry_id INTEGER NOT NULL,
            status TEXT NOT NULL,
            brief_text TEXT,
            sources_json TEXT,
            provider TEXT,
            model TEXT,
            used_web_search INTEGER NOT NULL DEFAULT 0,
            generated_at TEXT,
            error TEXT,
            FOREIGN KEY (entry_id) REFERENCES transcriptions(id) ON DELETE CASCADE
        )
    """)
    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_brief_entry ON research_briefs(entry_id)"
    )
    # Full-text search index (FTS5). Uses the `external content` pattern —
    # the FTS table references the real `transcriptions` column so there's no
    # data duplication, and triggers keep it in sync on insert/update/delete.
    _setup_fts(conn)


def _setup_fts(conn: sqlite3.Connection):
    """Create the FTS5 virtual table + sync triggers if missing, then
    backfill any rows that aren't indexed yet. Idempotent."""
    conn.execute("""
        CREATE VIRTUAL TABLE IF NOT EXISTS transcriptions_fts
        USING fts5(
            formatted_text,
            content='transcriptions',
            content_rowid='id',
            tokenize='porter unicode61'
        )
    """)
    conn.execute("""
        CREATE TRIGGER IF NOT EXISTS transcriptions_ai
        AFTER INSERT ON transcriptions
        BEGIN
            INSERT INTO transcriptions_fts(rowid, formatted_text)
            VALUES (new.id, new.formatted_text);
        END;
    """)
    conn.execute("""
        CREATE TRIGGER IF NOT EXISTS transcriptions_ad
        AFTER DELETE ON transcriptions
        BEGIN
            INSERT INTO transcriptions_fts(transcriptions_fts, rowid, formatted_text)
            VALUES ('delete', old.id, old.formatted_text);
        END;
    """)
    conn.execute("""
        CREATE TRIGGER IF NOT EXISTS transcriptions_au
        AFTER UPDATE OF formatted_text ON transcriptions
        BEGIN
            INSERT INTO transcriptions_fts(transcriptions_fts, rowid, formatted_text)
            VALUES ('delete', old.id, old.formatted_text);
            INSERT INTO transcriptions_fts(rowid, formatted_text)
            VALUES (new.id, new.formatted_text);
        END;
    """)
    # Backfill: if any base rows aren't indexed, rebuild the FTS table.
    # This covers the "old DB, first run with search" case.
    row = conn.execute(
        "SELECT (SELECT COUNT(*) FROM transcriptions) - "
        "(SELECT COUNT(*) FROM transcriptions_fts)"
    ).fetchone()
    if row and row[0] and row[0] > 0:
        conn.execute(
            "INSERT INTO transcriptions_fts(transcriptions_fts) VALUES('rebuild')"
        )


def _add_column(conn: sqlite3.Connection, name: str, ddl: str):
    try:
        conn.execute(f"ALTER TABLE transcriptions ADD COLUMN {name} {ddl}")
    except sqlite3.OperationalError as e:
        if "duplicate column name" not in str(e).lower():
            raise


def save(
    raw_text: str,
    formatted_text: str,
    duration: float = 0.0,
    *,
    kind: str = "transcription",
    app_name: str | None = None,
    window_title: str | None = None,
    is_research_topic: bool = False,
) -> int | None:
    """Save an entry to history. Thread-safe via write lock.

    Returns the new entry's id on success, None if history isn't initialized.
    Auto-reconnects once on sqlite3.OperationalError (e.g. WAL corruption,
    transient disk error) so a single dead connection doesn't silently drop
    the rest of the user's day.
    """
    if _conn is None:
        return None
    if kind not in VALID_KINDS:
        log.warning("Invalid kind '%s', defaulting to 'transcription'", kind)
        kind = "transcription"

    def _do():
        if _conn is None:
            return None
        with _write_lock:
            from voiceclip.config import PERSONA, MODEL
            cur = _conn.execute(
                "INSERT INTO transcriptions "
                "(timestamp, raw_text, formatted_text, duration_seconds, "
                " persona, model, word_count, kind, app_name, window_title, "
                " is_research_topic) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    datetime.now().isoformat(timespec="seconds"),
                    raw_text,
                    formatted_text,
                    round(duration, 1),
                    PERSONA,
                    MODEL,
                    len(formatted_text.split()),
                    kind,
                    app_name,
                    window_title,
                    1 if is_research_topic else 0,
                ),
            )
            _conn.commit()
            return cur.lastrowid

    return _with_retry(_do)


def cleanup(max_days: int, reflection_max_days: int = 0):
    """Delete old entries. Transcriptions and reflections are scoped separately.

    `max_days <= 0` skips transcription cleanup.
    `reflection_max_days <= 0` skips reflection cleanup (default — reflections
    are kept forever unless the user explicitly opts into a retention window).
    """
    if _conn is None:
        return
    if max_days > 0:
        _cleanup_kind("transcription", max_days)
    if reflection_max_days > 0:
        _cleanup_kind("reflection", reflection_max_days)


def _cleanup_kind(kind: str, max_days: int):
    try:
        cutoff = (datetime.now() - timedelta(days=max_days)).isoformat()
        cursor = _conn.execute(
            "DELETE FROM transcriptions WHERE kind = ? AND timestamp < ?",
            (kind, cutoff),
        )
        _conn.commit()
        if cursor.rowcount > 0:
            log.info("Cleaned up %d old %s entries", cursor.rowcount, kind)
    except Exception as e:
        log.warning("History cleanup (%s) failed: %s", kind, e)


def clear_all(force: bool = False):
    """Delete all entries (both kinds). Asks for confirmation unless force=True."""
    if _conn is None:
        print("History is not enabled.")
        return
    n = count()
    if n == 0:
        print("  History is already empty.")
        return
    if not force:
        confirm = input(f"  Delete all {n} entries? [y/N]: ").strip().lower()
        if confirm != "y":
            print("  Cancelled.")
            return
    _conn.execute("DELETE FROM transcriptions")
    _conn.commit()
    print(f"  Deleted {n} entries.")


def clear_kind(kind: str, force: bool = False):
    """Delete entries of a single kind. Asks for confirmation unless force=True."""
    if _conn is None:
        print("History is not enabled.")
        return
    if kind not in VALID_KINDS:
        print(f"  Unknown kind: {kind}")
        return
    n = count(kind=kind)
    if n == 0:
        print(f"  No {kind} entries to delete.")
        return
    if not force:
        confirm = input(f"  Delete all {n} {kind} entries? [y/N]: ").strip().lower()
        if confirm != "y":
            print("  Cancelled.")
            return
    _conn.execute("DELETE FROM transcriptions WHERE kind = ?", (kind,))
    _conn.commit()
    print(f"  Deleted {n} {kind} entries.")


def close():
    """Close the database connection. Call on shutdown."""
    global _conn
    if _conn is not None:
        try:
            _conn.close()
        except Exception:
            pass
        _conn = None


def count(kind: str | None = None) -> int:
    """Return total number of entries (optionally filtered by kind)."""
    if _conn is None:
        return 0
    if kind is None:
        row = _conn.execute("SELECT COUNT(*) FROM transcriptions").fetchone()
    else:
        row = _conn.execute(
            "SELECT COUNT(*) FROM transcriptions WHERE kind = ?", (kind,)
        ).fetchone()
    return row[0] if row else 0


# ---------------------------------------------------------------------------
# Query helpers
# ---------------------------------------------------------------------------

_KIND_MARKER = {"transcription": "📝", "reflection": "💭"}


def _format_rows(rows: list, show_id: bool = True) -> str:
    """Format query results for terminal display."""
    if not rows:
        return "  No entries found."
    lines = []
    for row in rows:
        id_, ts, text, dur, persona, wc, kind, app_name = row
        marker = _KIND_MARKER.get(kind, "  ")
        try:
            dt = datetime.fromisoformat(ts)
            time_str = dt.strftime("%Y-%m-%d %H:%M")
        except ValueError:
            time_str = ts[:16]
        prefix = f"  [{id_}]" if show_id else "  "
        meta = f"{dur}s, {wc} words, {persona}"
        if app_name:
            meta += f", app={app_name}"
        lines.append(f"{prefix} {marker} {time_str}  ({meta})")
        lines.append(f"       {text[:200]}")
        lines.append("")
    return "\n".join(lines)


_SELECT = (
    "SELECT id, timestamp, formatted_text, duration_seconds, "
    "persona, word_count, kind, app_name "
    "FROM transcriptions"
)


def _kind_clause(kind: str | None, prefix: str = " AND") -> tuple[str, tuple]:
    if kind is None:
        return "", ()
    return f"{prefix} kind = ?", (kind,)


def query_recent(limit: int = 10, kind: str | None = None) -> str:
    """Return the most recent entries (optionally filtered by kind)."""
    if _conn is None:
        return "History is not enabled. Set \"history\": true in config.json"
    where = "WHERE 1=1"
    kclause, kparams = _kind_clause(kind)
    rows = _conn.execute(
        f"{_SELECT} {where}{kclause} ORDER BY id DESC LIMIT ?",
        (*kparams, limit),
    ).fetchall()
    return _format_rows(rows)


def query_today(kind: str | None = None) -> str:
    """Return today's entries."""
    if _conn is None:
        return "History is not enabled."
    today = datetime.now().strftime("%Y-%m-%d")
    kclause, kparams = _kind_clause(kind)
    rows = _conn.execute(
        f"{_SELECT} WHERE timestamp >= ?{kclause} ORDER BY id ASC",
        (today, *kparams),
    ).fetchall()
    label = _list_label(kind, len(rows), "today")
    return f"  Today ({label}):\n" + _format_rows(rows)


def query_yesterday(kind: str | None = None) -> str:
    """Return yesterday's entries."""
    if _conn is None:
        return "History is not enabled."
    yesterday = (datetime.now() - timedelta(days=1)).strftime("%Y-%m-%d")
    today = datetime.now().strftime("%Y-%m-%d")
    kclause, kparams = _kind_clause(kind)
    rows = _conn.execute(
        f"{_SELECT} WHERE timestamp >= ? AND timestamp < ?{kclause} ORDER BY id ASC",
        (yesterday, today, *kparams),
    ).fetchall()
    label = _list_label(kind, len(rows), "yesterday")
    return f"  Yesterday ({label}):\n" + _format_rows(rows)


def query_search(term: str, limit: int = 20, kind: str | None = None) -> str:
    """Search entries by keyword (FTS5). Returns formatted CLI output."""
    if _conn is None:
        return "History is not enabled."
    rows = _search_rows(term, limit=limit, kind=kind)
    label = _list_label(kind, len(rows), f'matching "{term}"')
    return f"  Search ({label}):\n" + _format_rows(rows)


def _fts_query_for(term: str) -> str:
    """Turn a user-entered search string into a safe FTS5 query.

    FTS5 has its own query syntax (phrase, NEAR, etc). Rather than teach it to
    users, we tokenize the input on whitespace and AND the tokens together.
    Punctuation and FTS5-meta characters are stripped; quoted phrases are
    preserved as-is.
    """
    term = (term or "").strip()
    if not term:
        return ""
    # Strip FTS5 meta characters that could otherwise break the query
    # (":" starts a column filter, "-" is NOT, "*" is prefix, parens group).
    cleaned = []
    for ch in term:
        if ch.isalnum() or ch in ' "-':
            cleaned.append(ch)
        else:
            cleaned.append(" ")
    cleaned = "".join(cleaned).strip()
    if not cleaned:
        return ""
    # If the user used quotes, keep them. Otherwise split and AND.
    if '"' in cleaned:
        return cleaned
    tokens = [t for t in cleaned.split() if t and t != "-"]
    if not tokens:
        return ""
    # Each token gets suffix-prefix matching via * so "rec" matches "recording"
    return " AND ".join(f"{t}*" for t in tokens)


def _search_rows(term: str, limit: int = 20, kind: str | None = None) -> list:
    """FTS-backed search returning the same row shape as _SELECT."""
    if _conn is None:
        return []
    q = _fts_query_for(term)
    if not q:
        return []
    kclause, kparams = _kind_clause(kind)
    try:
        return _conn.execute(
            "SELECT t.id, t.timestamp, t.formatted_text, t.duration_seconds, "
            "       t.persona, t.word_count, t.kind, t.app_name "
            "FROM transcriptions_fts f "
            "JOIN transcriptions t ON t.id = f.rowid "
            "WHERE f.formatted_text MATCH ? "
            f"  AND t.is_research_topic = 0{kclause} "
            "ORDER BY f.rank, t.id DESC "
            "LIMIT ?",
            (q, *kparams, limit),
        ).fetchall()
    except sqlite3.OperationalError:
        # FTS query malformed (unclosed quote, etc) — fall back to LIKE so
        # the user still gets results instead of an error.
        safe_term = term.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")
        return _conn.execute(
            "SELECT id, timestamp, formatted_text, duration_seconds, "
            "       persona, word_count, kind, app_name "
            "FROM transcriptions "
            "WHERE formatted_text LIKE ? ESCAPE '\\' "
            f"  AND is_research_topic = 0{kclause} "
            "ORDER BY id DESC LIMIT ?",
            (f"%{safe_term}%", *kparams, limit),
        ).fetchall()


def search_entries(term: str, limit: int = 50, kind: str | None = None) -> list[dict]:
    """FTS search returning a list of dicts — used by the viewer search box."""
    rows = _search_rows(term, limit=limit, kind=kind)
    return [
        {
            "id": r[0],
            "timestamp": r[1],
            "text": r[2],
            "duration": r[3],
            "kind": r[6],
            "app_name": r[7],
            "word_count": r[5],
        }
        for r in rows
    ]


def _list_label(kind: str | None, n: int, suffix: str) -> str:
    noun = "entry" if n == 1 else "entries"
    if kind == "reflection":
        noun = "reflection" if n == 1 else "reflections"
    elif kind == "transcription":
        noun = "transcription" if n == 1 else "transcriptions"
    return f"{n} {noun} {suffix}"


def get_by_id(entry_id: int) -> str | None:
    """Get the formatted text of a specific entry by ID."""
    if _conn is None:
        return None
    row = _conn.execute(
        "SELECT formatted_text FROM transcriptions WHERE id = ?", (entry_id,)
    ).fetchone()
    return row[0] if row else None


def get_entry_full(entry_id: int) -> dict | None:
    """Get a full row as a dict (text + metadata incl. kind)."""
    if _conn is None:
        return None
    row = _conn.execute(
        "SELECT id, timestamp, formatted_text, kind, app_name "
        "FROM transcriptions WHERE id = ?",
        (entry_id,),
    ).fetchone()
    if not row:
        return None
    return {
        "id": row[0],
        "timestamp": row[1],
        "text": row[2],
        "kind": row[3],
        "app_name": row[4],
    }


def delete_entry(entry_id: int) -> dict | None:
    """Delete a single entry by ID. Returns the deleted row's summary on success.

    Used by the web viewer's per-entry delete button. This bypasses the
    interactive confirmation used by clear_all/clear_kind because the UI
    does its own two-step confirm.
    """
    if _conn is None:
        return None
    existing = _conn.execute(
        "SELECT id, timestamp, formatted_text, kind FROM transcriptions WHERE id = ?",
        (entry_id,),
    ).fetchone()
    if not existing:
        return None
    with _write_lock:
        try:
            _conn.execute("DELETE FROM transcriptions WHERE id = ?", (entry_id,))
            _conn.commit()
        except Exception as e:
            log.warning("Failed to delete entry %s: %s", entry_id, e)
            return None
    return {
        "id": existing[0],
        "timestamp": existing[1],
        "text": existing[2],
        "kind": existing[3],
    }


def promote_to_reflection(entry_id: int | None = None, last: bool = False) -> dict | None:
    """Promote a past transcription to a reflection.

    Pass `last=True` to promote the most recent transcription, or `entry_id=N`
    to target a specific entry. Only updates `kind`; all other fields are
    preserved byte-for-byte. Returns the promoted entry dict, or None if the
    target doesn't exist or is already a reflection.
    """
    if _conn is None:
        return None
    if last:
        row = _conn.execute(
            "SELECT id FROM transcriptions WHERE kind = 'transcription' "
            "ORDER BY id DESC LIMIT 1"
        ).fetchone()
        if not row:
            return None
        entry_id = row[0]
    if entry_id is None:
        return None

    existing = _conn.execute(
        "SELECT id, timestamp, formatted_text, kind FROM transcriptions WHERE id = ?",
        (entry_id,),
    ).fetchone()
    if not existing:
        return None
    if existing[3] == "reflection":
        return None

    _conn.execute(
        "UPDATE transcriptions SET kind = 'reflection' WHERE id = ?",
        (entry_id,),
    )
    _conn.commit()
    return {
        "id": existing[0],
        "timestamp": existing[1],
        "text": existing[2],
    }



# ---------------------------------------------------------------------------
# Day views — for the web viewer and summarizer
# ---------------------------------------------------------------------------

def list_days(limit: int = 90) -> list[str]:
    """Return distinct YYYY-MM-DD strings that have entries, newest first."""
    if _conn is None:
        return []
    rows = _conn.execute(
        "SELECT DISTINCT substr(timestamp, 1, 10) AS day "
        "FROM transcriptions ORDER BY day DESC LIMIT ?",
        (limit,),
    ).fetchall()
    return [r[0] for r in rows]


def entries_for_day(date: str, kind: str | None = None) -> list[dict]:
    """Return all entries for a YYYY-MM-DD day, oldest first (so the viewer
    can show the day in chronological order).

    Research-queue topics are excluded from the day view — they live on their
    own tab. If you want to include them, use `entries_for_day_all`.
    """
    if _conn is None:
        return []
    params: tuple = (f"{date}T", f"{date}T\uffff")
    kclause = ""
    if kind is not None:
        kclause = " AND kind = ?"
        params = (*params, kind)
    rows = _conn.execute(
        f"SELECT id, timestamp, formatted_text, duration_seconds, "
        f"       kind, app_name, word_count, edited_at "
        f"FROM transcriptions "
        f"WHERE timestamp >= ? AND timestamp < ?{kclause} "
        f"  AND is_research_topic = 0 "
        f"ORDER BY id ASC",
        params,
    ).fetchall()
    return [
        {
            "id": r[0],
            "timestamp": r[1],
            "text": r[2],
            "duration": r[3],
            "kind": r[4],
            "app_name": r[5],
            "word_count": r[6],
            "edited_at": r[7],
        }
        for r in rows
    ]


def day_stats(date: str) -> dict:
    """Return counts and top-apps for a single day. Excludes research topics.

    Collapses what used to be three separate queries (transcriptions,
    reflections, apps) into one pass via conditional aggregation + one
    GROUP BY for the apps bar. Fewer round-trips to SQLite, and the
    counts come from a single consistent read.
    """
    if _conn is None:
        return {"transcriptions": 0, "reflections": 0, "apps": []}

    # Single-query counts via conditional aggregation
    count_row = _conn.execute(
        "SELECT "
        "  SUM(CASE WHEN kind = 'transcription' THEN 1 ELSE 0 END) AS t_count, "
        "  SUM(CASE WHEN kind = 'reflection' THEN 1 ELSE 0 END) AS r_count "
        "FROM transcriptions "
        "WHERE timestamp >= ? AND timestamp < ? AND is_research_topic = 0",
        (f"{date}T", f"{date}T\uffff"),
    ).fetchone()
    t_count = (count_row[0] if count_row else 0) or 0
    r_count = (count_row[1] if count_row else 0) or 0

    app_rows = _conn.execute(
        "SELECT COALESCE(NULLIF(app_name, ''), 'unknown') AS app, COUNT(*) AS n "
        "FROM transcriptions "
        "WHERE timestamp >= ? AND timestamp < ? "
        "  AND is_research_topic = 0 "
        "GROUP BY app ORDER BY n DESC LIMIT 6",
        (f"{date}T", f"{date}T\uffff"),
    ).fetchall()
    return {
        "transcriptions": t_count,
        "reflections": r_count,
        "apps": [{"name": a[0], "count": a[1]} for a in app_rows],
    }


# ---------------------------------------------------------------------------
# Day summary cache (LLM-generated)
# ---------------------------------------------------------------------------

def get_day_summary(date: str) -> dict | None:
    """Return the cached summary for a day, or None."""
    if _conn is None:
        return None
    row = _conn.execute(
        "SELECT date, summary, provider, model, style, generated_at, entry_count "
        "FROM day_summaries WHERE date = ?",
        (date,),
    ).fetchone()
    if not row:
        return None
    return {
        "date": row[0],
        "summary": row[1],
        "provider": row[2],
        "model": row[3],
        "style": row[4],
        "generated_at": row[5],
        "entry_count": row[6],
    }


def save_day_summary(
    date: str, summary: str, *,
    provider: str, model: str, style: str, entry_count: int,
):
    """Upsert a cached day summary."""
    if _conn is None:
        return
    with _write_lock:
        try:
            _conn.execute(
                "INSERT INTO day_summaries "
                "(date, summary, provider, model, style, generated_at, entry_count) "
                "VALUES (?, ?, ?, ?, ?, ?, ?) "
                "ON CONFLICT(date) DO UPDATE SET "
                "  summary=excluded.summary, provider=excluded.provider, "
                "  model=excluded.model, style=excluded.style, "
                "  generated_at=excluded.generated_at, entry_count=excluded.entry_count",
                (
                    date, summary, provider, model, style,
                    datetime.now().isoformat(timespec="seconds"),
                    entry_count,
                ),
            )
            _conn.commit()
        except Exception as e:
            log.warning("Failed to save day summary: %s", e)


def count_for_day(date: str) -> int:
    """Return the total entry count for a YYYY-MM-DD day (both kinds)."""
    if _conn is None:
        return 0
    row = _conn.execute(
        "SELECT COUNT(*) FROM transcriptions WHERE timestamp >= ? AND timestamp < ?",
        (f"{date}T", f"{date}T\uffff"),
    ).fetchone()
    return row[0] if row else 0



# ---------------------------------------------------------------------------
# Inline editing
# ---------------------------------------------------------------------------

def update_text(entry_id: int, new_text: str) -> dict | None:
    """Update an entry's formatted_text. Sets edited_at timestamp.

    Returns {'id', 'text', 'edited_at', 'word_count'} on success, None if
    the entry doesn't exist or history isn't initialized.
    """
    if _conn is None:
        return None
    existing = _conn.execute(
        "SELECT id FROM transcriptions WHERE id = ?", (entry_id,)
    ).fetchone()
    if not existing:
        return None
    now_iso = datetime.now().isoformat(timespec="seconds")
    word_count = len(new_text.split())

    def _do():
        if _conn is None:
            return None
        with _write_lock:
            _conn.execute(
                "UPDATE transcriptions SET formatted_text = ?, "
                "word_count = ?, edited_at = ? WHERE id = ?",
                (new_text, word_count, now_iso, entry_id),
            )
            _conn.commit()
            return {
                "id": entry_id,
                "text": new_text,
                "edited_at": now_iso,
                "word_count": word_count,
            }

    return _with_retry(_do)


# ---------------------------------------------------------------------------
# Research queue
# ---------------------------------------------------------------------------

def create_research_topic(text: str, *, app_name: str | None = None) -> int | None:
    """Create a typed-in research topic as a new entry (kind=transcription,
    is_research_topic=True, zero duration). Returns new id."""
    if _conn is None:
        return None
    text = (text or "").strip()
    if not text:
        return None
    return save(
        text, text, 0.0,
        kind="transcription",
        app_name=app_name,
        is_research_topic=True,
    )


def list_research_topics(status_filter: str | None = None, limit: int = 100) -> list[dict]:
    """Return research topics with the status of their latest brief.

    status_filter: None = all; 'pending' = no brief yet OR failed;
                   'ready' = at least one completed brief; 'read' = TBD later.

    Uses a LEFT JOIN with a correlated MAX(id) to pull each topic's latest
    brief in a single query — cheaper than running subqueries per row at
    N topics (was 2N+1 queries, now 1).
    """
    if _conn is None:
        return []
    rows = _conn.execute(
        """
        SELECT
          t.id, t.timestamp, t.formatted_text, t.app_name,
          latest.status AS latest_status,
          COALESCE(done_counts.n, 0) AS brief_count
        FROM transcriptions t
        LEFT JOIN (
            SELECT entry_id, status
            FROM research_briefs
            WHERE id IN (
                SELECT MAX(id) FROM research_briefs GROUP BY entry_id
            )
        ) AS latest ON latest.entry_id = t.id
        LEFT JOIN (
            SELECT entry_id, COUNT(*) AS n
            FROM research_briefs
            WHERE status = 'done'
            GROUP BY entry_id
        ) AS done_counts ON done_counts.entry_id = t.id
        WHERE t.is_research_topic = 1
        ORDER BY t.id DESC
        LIMIT ?
        """,
        (limit,),
    ).fetchall()
    topics = []
    for r in rows:
        latest = r[4]
        brief_count = r[5] or 0
        if brief_count > 0:
            status = "ready"
        elif latest == "running":
            status = "running"
        elif latest == "failed":
            status = "failed"
        else:
            status = "pending"
        if status_filter and status != status_filter:
            continue
        topics.append({
            "id": r[0],
            "timestamp": r[1],
            "text": r[2],
            "app_name": r[3],
            "status": status,
            "brief_count": brief_count,
        })
    return topics


def latest_brief(entry_id: int) -> dict | None:
    """Return the most recent completed brief for an entry, or None."""
    if _conn is None:
        return None
    row = _conn.execute(
        "SELECT id, brief_text, sources_json, provider, model, "
        "       used_web_search, generated_at "
        "FROM research_briefs "
        "WHERE entry_id = ? AND status = 'done' "
        "ORDER BY id DESC LIMIT 1",
        (entry_id,),
    ).fetchone()
    if not row:
        return None
    import json as _json
    sources = []
    if row[2]:
        try:
            sources = _json.loads(row[2])
        except Exception:
            sources = []
    return {
        "id": row[0],
        "text": row[1],
        "sources": sources,
        "provider": row[3],
        "model": row[4],
        "used_web_search": bool(row[5]),
        "generated_at": row[6],
    }


def save_brief(
    entry_id: int,
    *,
    status: str,
    brief_text: str | None = None,
    sources: list | None = None,
    provider: str | None = None,
    model: str | None = None,
    used_web_search: bool = False,
    error: str | None = None,
) -> int | None:
    """Insert a research brief row. Returns the new brief id."""
    if _conn is None:
        return None
    import json as _json
    with _write_lock:
        try:
            cur = _conn.execute(
                "INSERT INTO research_briefs "
                "(entry_id, status, brief_text, sources_json, provider, model, "
                " used_web_search, generated_at, error) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    entry_id, status, brief_text,
                    _json.dumps(sources) if sources else None,
                    provider, model,
                    1 if used_web_search else 0,
                    datetime.now().isoformat(timespec="seconds"),
                    error,
                ),
            )
            _conn.commit()
            return cur.lastrowid
        except Exception as e:
            log.warning("Failed to save brief: %s", e)
            return None


def get_topic(entry_id: int) -> dict | None:
    """Return a research topic (entry_id must be a research topic)."""
    if _conn is None:
        return None
    row = _conn.execute(
        "SELECT id, timestamp, formatted_text, app_name, is_research_topic "
        "FROM transcriptions WHERE id = ?",
        (entry_id,),
    ).fetchone()
    if not row or not row[4]:
        return None
    return {
        "id": row[0],
        "timestamp": row[1],
        "text": row[2],
        "app_name": row[3],
    }



# ---------------------------------------------------------------------------
# Patterns view support
# ---------------------------------------------------------------------------

def find_research_topic_by_text(text: str) -> dict | None:
    """Return the most recent research topic whose formatted_text (first ~80
    chars, case-insensitive) matches the given text. Used by the Patterns
    'Queue it' button to avoid creating duplicate topics.
    """
    if _conn is None:
        return None
    needle = (text or "").strip().lower()
    if not needle:
        return None
    # Cheap substring / prefix match — we don't need exact equality because
    # "CRDTs" and "CRDTs " should be considered the same.
    rows = _conn.execute(
        "SELECT id, timestamp, formatted_text FROM transcriptions "
        "WHERE is_research_topic = 1 "
        "ORDER BY id DESC LIMIT 50"
    ).fetchall()
    for r in rows:
        existing = (r[2] or "").strip().lower()
        # Match if one is a prefix of the other, limited to first 80 chars.
        a, b = existing[:80], needle[:80]
        if a == b or a.startswith(b) or b.startswith(a):
            return {"id": r[0], "timestamp": r[1], "text": r[2]}
    return None


def entries_for_window(start_date: str, end_date: str, kind: str | None = None) -> list[dict]:
    """Return entries in [start_date, end_date) — oldest first. Excludes
    research topics. Used to feed the patterns prompt.
    """
    if _conn is None:
        return []
    params: tuple = (f"{start_date}T", f"{end_date}T")
    kclause = ""
    if kind is not None:
        kclause = " AND kind = ?"
        params = (*params, kind)
    rows = _conn.execute(
        f"SELECT id, timestamp, formatted_text, duration_seconds, "
        f"       kind, app_name, word_count "
        f"FROM transcriptions "
        f"WHERE timestamp >= ? AND timestamp < ? AND is_research_topic = 0{kclause} "
        f"ORDER BY id ASC",
        params,
    ).fetchall()
    return [
        {
            "id": r[0], "timestamp": r[1], "text": r[2],
            "duration": r[3], "kind": r[4],
            "app_name": r[5], "word_count": r[6],
        }
        for r in rows
    ]


def app_distribution_for_window(start_date: str, end_date: str, top_n: int = 8) -> list[dict]:
    """Return top apps by dictation count across the window."""
    if _conn is None:
        return []
    rows = _conn.execute(
        "SELECT COALESCE(NULLIF(app_name, ''), 'unknown') AS app, COUNT(*) AS n "
        "FROM transcriptions "
        "WHERE timestamp >= ? AND timestamp < ? AND is_research_topic = 0 "
        "GROUP BY app ORDER BY n DESC LIMIT ?",
        (f"{start_date}T", f"{end_date}T", top_n),
    ).fetchall()
    return [{"name": r[0], "count": r[1]} for r in rows]



def update_brief_text(brief_id: int, new_text: str) -> dict | None:
    """Update the text of a research brief in place. Used when the user
    edits a brief inline. Returns the updated brief row as a dict, or None
    if the brief doesn't exist / history isn't initialized.

    Note: sources and used_web_search are preserved — the edit is a textual
    cleanup, not a full regeneration. Re-research is the path for fresh
    sources.
    """
    if _conn is None:
        return None
    existing = _conn.execute(
        "SELECT id FROM research_briefs WHERE id = ?", (brief_id,)
    ).fetchone()
    if not existing:
        return None

    def _do():
        if _conn is None:
            return None
        with _write_lock:
            _conn.execute(
                "UPDATE research_briefs SET brief_text = ? WHERE id = ?",
                (new_text, brief_id),
            )
            _conn.commit()
            return {"id": brief_id, "text": new_text}

    return _with_retry(_do)
