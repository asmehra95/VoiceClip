"""Transcription history — local SQLite storage.

Opt-in via config.json: "history": true (default: false).
Stores every transcription with timestamp, text, duration, persona, model.
Provides CLI search and retrieval via the `history` subcommand.

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


def init():
    """Initialize the history database. Creates the table and indexes if needed."""
    global _conn
    try:
        os.makedirs(CONFIG_DIR, exist_ok=True)
        _conn = sqlite3.connect(DB_PATH, check_same_thread=False)
        # WAL mode for safe concurrent reads/writes from multiple threads
        _conn.execute("PRAGMA journal_mode=WAL")
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
        # Index on timestamp for today/yesterday/cleanup queries
        _conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_timestamp ON transcriptions(timestamp)"
        )
        _conn.commit()
        os.chmod(DB_PATH, 0o600)
        log.info("History database ready at %s", DB_PATH)
    except Exception as e:
        log.warning("Could not initialize history database: %s", e)
        _conn = None


def save(raw_text: str, formatted_text: str, duration: float = 0.0):
    """Save a transcription to history. Thread-safe via write lock."""
    if _conn is None:
        return
    with _write_lock:
        try:
            from voiceclip.config import PERSONA, MODEL
            _conn.execute(
                "INSERT INTO transcriptions "
                "(timestamp, raw_text, formatted_text, duration_seconds, persona, model, word_count) "
                "VALUES (?, ?, ?, ?, ?, ?, ?)",
                (
                    datetime.now().isoformat(timespec="seconds"),
                    raw_text,
                    formatted_text,
                    round(duration, 1),
                    PERSONA,
                    MODEL,
                    len(formatted_text.split()),
                ),
            )
            _conn.commit()
        except Exception as e:
            log.warning("Failed to save to history: %s", e)


def cleanup(max_days: int):
    """Delete entries older than max_days."""
    if _conn is None or max_days <= 0:
        return
    try:
        cutoff = (datetime.now() - timedelta(days=max_days)).isoformat()
        cursor = _conn.execute(
            "DELETE FROM transcriptions WHERE timestamp < ?", (cutoff,)
        )
        _conn.commit()
        if cursor.rowcount > 0:
            log.info("Cleaned up %d old history entries", cursor.rowcount)
    except Exception as e:
        log.warning("History cleanup failed: %s", e)


def clear_all(force: bool = False):
    """Delete all history entries. Asks for confirmation unless force=True."""
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


def close():
    """Close the database connection. Call on shutdown."""
    global _conn
    if _conn is not None:
        try:
            _conn.close()
        except Exception:
            pass
        _conn = None


def count() -> int:
    """Return total number of entries."""
    if _conn is None:
        return 0
    row = _conn.execute("SELECT COUNT(*) FROM transcriptions").fetchone()
    return row[0] if row else 0


# ---------------------------------------------------------------------------
# Query helpers
# ---------------------------------------------------------------------------

def _format_rows(rows: list, show_id: bool = True) -> str:
    """Format query results for terminal display."""
    if not rows:
        return "  No transcriptions found."
    lines = []
    for row in rows:
        id_, ts, text, dur, persona, wc = row
        # Parse and format timestamp
        try:
            dt = datetime.fromisoformat(ts)
            time_str = dt.strftime("%Y-%m-%d %H:%M")
        except ValueError:
            time_str = ts[:16]
        prefix = f"  [{id_}]" if show_id else "  "
        lines.append(f"{prefix} {time_str}  ({dur}s, {wc} words, {persona})")
        lines.append(f"       {text[:200]}")
        lines.append("")
    return "\n".join(lines)


_SELECT = (
    "SELECT id, timestamp, formatted_text, duration_seconds, persona, word_count "
    "FROM transcriptions"
)


def query_recent(limit: int = 10) -> str:
    """Return the most recent transcriptions."""
    if _conn is None:
        return "History is not enabled. Set \"history\": true in config.json"
    rows = _conn.execute(
        f"{_SELECT} ORDER BY id DESC LIMIT ?", (limit,)
    ).fetchall()
    return _format_rows(rows)


def query_today() -> str:
    """Return today's transcriptions."""
    if _conn is None:
        return "History is not enabled."
    today = datetime.now().strftime("%Y-%m-%d")
    rows = _conn.execute(
        f"{_SELECT} WHERE timestamp >= ? ORDER BY id ASC", (today,)
    ).fetchall()
    header = f"  Today ({len(rows)} transcriptions):\n"
    return header + _format_rows(rows)


def query_yesterday() -> str:
    """Return yesterday's transcriptions."""
    if _conn is None:
        return "History is not enabled."
    yesterday = (datetime.now() - timedelta(days=1)).strftime("%Y-%m-%d")
    today = datetime.now().strftime("%Y-%m-%d")
    rows = _conn.execute(
        f"{_SELECT} WHERE timestamp >= ? AND timestamp < ? ORDER BY id ASC",
        (yesterday, today),
    ).fetchall()
    header = f"  Yesterday ({len(rows)} transcriptions):\n"
    return header + _format_rows(rows)


def query_search(term: str, limit: int = 20) -> str:
    """Search transcriptions by keyword."""
    if _conn is None:
        return "History is not enabled."
    # Escape SQL LIKE wildcards in the search term
    safe_term = term.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")
    rows = _conn.execute(
        f"{_SELECT} WHERE formatted_text LIKE ? ESCAPE '\\' ORDER BY id DESC LIMIT ?",
        (f"%{safe_term}%", limit),
    ).fetchall()
    n = len(rows)
    result_word = "result" if n == 1 else "results"
    header = f"  Search \"{term}\" ({n} {result_word}):\n"
    return header + _format_rows(rows)


def get_by_id(entry_id: int) -> str | None:
    """Get the formatted text of a specific entry by ID."""
    if _conn is None:
        return None
    row = _conn.execute(
        "SELECT formatted_text FROM transcriptions WHERE id = ?", (entry_id,)
    ).fetchone()
    return row[0] if row else None
