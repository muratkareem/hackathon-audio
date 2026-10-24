"""SQLite metadata; all media paths derive solely from validated UUIDs."""

from contextlib import contextmanager
import json
import sqlite3
from pathlib import Path
from uuid import UUID


class Store:
    def __init__(self, root: Path):
        self.root = root
        root.mkdir(parents=True, exist_ok=True, mode=0o700)
        self.db = root / "meetings.sqlite3"
        with self.connect() as con:
            con.execute(
                "CREATE TABLE IF NOT EXISTS meetings (id TEXT PRIMARY KEY, created TEXT DEFAULT CURRENT_TIMESTAMP, payload TEXT NOT NULL)"
            )

    @contextmanager
    def connect(self):
        con = sqlite3.connect(self.db, timeout=10)
        con.row_factory = sqlite3.Row
        try:
            with con:
                yield con
        finally:
            con.close()

    def directory(self, meeting_id):
        if str(UUID(meeting_id)) != meeting_id:
            raise ValueError("Invalid meeting id")
        return self.root / meeting_id

    def save(self, meeting):
        self.directory(meeting["id"])
        with self.connect() as con:
            con.execute(
                "INSERT INTO meetings(id,payload) VALUES (?,?) ON CONFLICT(id) DO UPDATE SET payload=excluded.payload",
                (meeting["id"], json.dumps(meeting, ensure_ascii=False)),
            )

    def get(self, meeting_id):
        self.directory(meeting_id)
        with self.connect() as con:
            row = con.execute(
                "SELECT payload FROM meetings WHERE id=?", (meeting_id,)
            ).fetchone()
        return json.loads(row["payload"]) if row else None

    def list(self):
        with self.connect() as con:
            rows = con.execute(
                "SELECT payload, created FROM meetings ORDER BY created DESC, rowid DESC"
            ).fetchall()
        return [
            {**json.loads(row["payload"]), "created": row["created"]} for row in rows
        ]

    def recover(self, retain_audio=False):
        for meeting in self.list():
            if meeting["status"] in ("queued", "processing"):
                meeting.update(
                    status="failed", stage="interrupted", error="interrupted"
                )
                self.save(meeting)
                if not retain_audio:
                    directory = self.directory(meeting["id"])
                    if not directory.is_symlink():
                        for audio in directory.glob("audio.*"):
                            audio.unlink(missing_ok=True)

    def delete(self, meeting_id):
        import shutil

        path = self.directory(meeting_id)
        if path.is_symlink():
            raise ValueError("Unsafe meeting directory")
        shutil.rmtree(path, ignore_errors=False) if path.exists() else None
        with self.connect() as con:
            con.execute("DELETE FROM meetings WHERE id=?", (meeting_id,))
