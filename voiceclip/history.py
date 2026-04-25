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
    # Backfill any pre-existing NULL kinds (shouldn't happen given the DEFAULT,
    # but harmless and explicit).
    conn.execute(
        "UPDATE transcriptions SET kind = 'transcription' WHERE kind IS NULL"
    )
    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_kind_timestamp "
        "ON transcriptions(kind, timestamp)"
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
):
    """Save an entry to history. Thread-safe via write lock."""
    if _conn is None:
        return
    if kind not in VALID_KINDS:
        log.warning("Invalid kind '%s', defaulting to 'transcription'", kind)
        kind = "transcription"
    with _write_lock:
        try:
            from voiceclip.config import PERSONA, MODEL
            _conn.execute(
                "INSERT INTO transcriptions "
                "(timestamp, raw_text, formatted_text, duration_seconds, "
                " persona, model, word_count, kind, app_name, window_title) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
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
                ),
            )
            _conn.commit()
        except Exception as e:
            log.warning("Failed to save to history: %s", e)


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
    """Search entries by keyword."""
    if _conn is None:
        return "History is not enabled."
    safe_term = term.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")
    kclause, kparams = _kind_clause(kind)
    rows = _conn.execute(
        f"{_SELECT} WHERE formatted_text LIKE ? ESCAPE '\\'{kclause} "
        "ORDER BY id DESC LIMIT ?",
        (f"%{safe_term}%", *kparams, limit),
    ).fetchall()
    label = _list_label(kind, len(rows), f'matching "{term}"')
    return f"  Search ({label}):\n" + _format_rows(rows)


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
        f"       kind, app_name, word_count "
        f"FROM transcriptions "
        f"WHERE timestamp >= ? AND timestamp < ?{kclause} "
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
        }
        for r in rows
    ]


def day_stats(date: str) -> dict:
    """Return counts and top-apps for a single day."""
    if _conn is None:
        return {"transcriptions": 0, "reflections": 0, "apps": []}
    trow = _conn.execute(
        "SELECT COUNT(*) FROM transcriptions "
        "WHERE timestamp >= ? AND timestamp < ? AND kind = 'transcription'",
        (f"{date}T", f"{date}T\uffff"),
    ).fetchone()
    rrow = _conn.execute(
        "SELECT COUNT(*) FROM transcriptions "
        "WHERE timestamp >= ? AND timestamp < ? AND kind = 'reflection'",
        (f"{date}T", f"{date}T\uffff"),
    ).fetchone()
    app_rows = _conn.execute(
        "SELECT COALESCE(NULLIF(app_name, ''), 'unknown') AS app, COUNT(*) AS n "
        "FROM transcriptions "
        "WHERE timestamp >= ? AND timestamp < ? "
        "GROUP BY app ORDER BY n DESC LIMIT 6",
        (f"{date}T", f"{date}T\uffff"),
    ).fetchall()
    return {
        "transcriptions": trow[0] if trow else 0,
        "reflections": rrow[0] if rrow else 0,
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
