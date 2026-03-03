"""
Room Activity Logger
--------------------
Thread-safe, de-duplicated room activity logger.
Writes structured logs to datasets/logs.json with the format:

{
  "logs": {
    "YYYY-MM-DD": [
      "HH:MM:SS - Event description",
      ...
    ]
  }
}

Supported events:
  - "[Person] has entered the room"
  - "[Person] is present in the room"
  - "[Person] is interacting with [object]"
  - "[Person] has left the room"
"""

import json
import threading
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Set


class RoomActivityLogger:
    """Manages structured room activity logs with de-duplication and batched I/O."""

    # Frames a track must be absent before logging "has left the room"
    ABSENT_THRESHOLD = 90  # ~3 seconds at 30 fps

    # Minimum seconds between flushing to disk
    FLUSH_INTERVAL = 5.0

    def __init__(self, log_path: str | Path) -> None:
        self._path = Path(log_path)
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()

        # Persistent log data (loaded from / saved to JSON)
        self._data: Dict[str, List[str]] = self._load()

        # --- Per-session runtime state (never persisted) ---
        # track_id -> identity name currently in the room
        self._present: Dict[int, str] = {}
        # track_id -> last logged interaction object (to avoid repeats)
        self._last_interaction: Dict[int, str] = {}
        # track_id -> consecutive absent frame count
        self._absent_counter: Dict[int, int] = {}
        # Tracks that have already been logged as "left" (skip duplicates)
        self._left_logged: Set[int] = set()

        self._dirty = False
        self._last_flush = time.monotonic()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def update(self, detections: List[Dict[str, Any]]) -> List[str]:
        """Process one frame of detections and return any new log entries.

        Args:
            detections: list of dicts with keys ``track_id``, ``identity``,
                        and optionally ``behavior_status``.

        Returns:
            List of human-readable log strings appended this frame.
        """
        current_ids: Set[int] = set()
        new_entries: List[str] = []

        for det in detections:
            track_id: int = det.get("track_id", -1)
            if track_id < 0:
                continue

            identity: str = self._normalize_identity(det.get("identity", "Unknown"))
            behavior: str = det.get("behavior_status", "STATUS: NO INTERACTION")
            current_ids.add(track_id)

            # Reset absent counter since this track is visible
            self._absent_counter.pop(track_id, None)
            self._left_logged.discard(track_id)

            # --- Entry detection ---
            if track_id not in self._present:
                entry = self._add_entry(f"{identity} has entered the room")
                self._present[track_id] = identity
                self._last_interaction[track_id] = ""
                new_entries.append(entry)

            else:
                # Identity may update (e.g. Unknown -> recognized name)
                prev_name = self._present[track_id]
                if identity != "Unknown person" and prev_name == "Unknown person":
                    self._present[track_id] = identity
                    entry = self._add_entry(f"{identity} is present in the room")
                    new_entries.append(entry)

            # --- Interaction detection ---
            obj_name = self._parse_interaction(behavior)
            if obj_name:
                if self._last_interaction.get(track_id) != obj_name:
                    self._last_interaction[track_id] = obj_name
                    entry = self._add_entry(
                        f"{self._present[track_id]} is interacting with {obj_name}"
                    )
                    new_entries.append(entry)
            else:
                # Interaction ended — allow re-logging if they interact again
                if self._last_interaction.get(track_id):
                    self._last_interaction[track_id] = ""

        # --- Absence / left detection ---
        for tid in list(self._present):
            if tid not in current_ids:
                self._absent_counter[tid] = self._absent_counter.get(tid, 0) + 1
                if (
                    self._absent_counter[tid] >= self.ABSENT_THRESHOLD
                    and tid not in self._left_logged
                ):
                    name = self._present.pop(tid)
                    self._last_interaction.pop(tid, None)
                    self._left_logged.add(tid)
                    entry = self._add_entry(f"{name} has left the room")
                    new_entries.append(entry)

        # Periodic disk flush
        self._maybe_flush()

        return new_entries

    def flush(self) -> None:
        """Force-write pending data to disk."""
        with self._lock:
            self._save()
            self._dirty = False
            self._last_flush = time.monotonic()

    def get_logs(self, date: Optional[str] = None) -> Dict[str, List[str]]:
        """Return logs, optionally filtered to a single ISO date string."""
        with self._lock:
            if date:
                return {date: list(self._data.get(date, []))}
            return {k: list(v) for k, v in self._data.items()}

    def get_available_dates(self) -> List[str]:
        """Return sorted list of dates that have log entries."""
        with self._lock:
            return sorted(self._data.keys(), reverse=True)

    def close(self) -> None:
        """Mark all remaining people as left and flush to disk."""
        for tid in list(self._present):
            name = self._present.pop(tid)
            self._add_entry(f"{name} has left the room")
        self.flush()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _normalize_identity(self, raw: str) -> str:
        if not raw or raw.strip().lower() == "unknown":
            return "Unknown person"
        return raw.strip()

    @staticmethod
    def _parse_interaction(behavior: str) -> str:
        """Extract object name from behavior status, or return empty string."""
        if "INTERACTING WITH" in behavior:
            return behavior.split("INTERACTING WITH")[-1].strip().lower()
        return ""

    def _today(self) -> str:
        return datetime.now().strftime("%Y-%m-%d")

    def _timestamp(self) -> str:
        return datetime.now().strftime("%H:%M:%S")

    def _add_entry(self, message: str) -> str:
        """Append a timestamped entry for today and return the full string."""
        entry = f"{self._timestamp()} - {message}"
        date_key = self._today()
        with self._lock:
            self._data.setdefault(date_key, [])
            self._data[date_key].append(entry)
            self._dirty = True
        return entry

    def _maybe_flush(self) -> None:
        if self._dirty and (time.monotonic() - self._last_flush) >= self.FLUSH_INTERVAL:
            self.flush()

    def _load(self) -> Dict[str, List[str]]:
        if self._path.exists() and self._path.stat().st_size > 0:
            try:
                with open(self._path, "r", encoding="utf-8") as f:
                    raw = json.load(f)
                return raw.get("logs", {}) if isinstance(raw, dict) else {}
            except (json.JSONDecodeError, OSError):
                return {}
        return {}

    def _save(self) -> None:
        """Atomic write: write to .tmp then rename."""
        tmp = self._path.with_suffix(".tmp")
        try:
            with open(tmp, "w", encoding="utf-8") as f:
                json.dump({"logs": self._data}, f, indent=2, ensure_ascii=False)
            tmp.replace(self._path)
        except OSError:
            # Best-effort — don't crash the pipeline for a log write failure
            if tmp.exists():
                tmp.unlink(missing_ok=True)
