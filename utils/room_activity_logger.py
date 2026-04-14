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

Supported events (ONLY these — nothing else):
  - "[Person] has entered the room"       (one-time, on first detection)
  - "[Person] is present in the room"     (one-time, when Unknown upgrades to known name)
  - "[Person] is interacting with [obj]"  (once per interaction start; suppressed while ongoing)
  - "[Person] has left the room"          (one-time, after sustained absence)

Anti-spam guarantees:
  - Per-identity wall-clock debounce: same event for the same person is suppressed
    for DEBOUNCE_SECONDS after the last log.
  - "Unknown person" entries are coalesced: only ONE "Unknown person has entered"
    per session regardless of how many unknown track_ids appear.
  - Interaction flicker protection: interaction must drop for
    INTERACTION_DROP_FRAMES consecutive frames before the logger considers
    it "ended" and allows a re-log.
  - [repeated xN] suffixes from the pipeline are always stripped.
"""

import json
import re
import threading
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Set


class RoomActivityLogger:
    """Manages structured room activity logs with aggressive de-duplication."""

    # Frames a track must be absent before logging "has left the room"
    ABSENT_THRESHOLD = 90  # ~3 seconds at 30 fps

    # Frames of "no interaction" before resetting interaction state.
    # Must be long enough to ride out YOLO flicker AND the user's >5-second rule.
    INTERACTION_DROP_FRAMES = 150  # ~5 seconds at 30 fps

    # Wall-clock debounce: suppress identical (person + event_type) logs
    # that occur within this window.  Covers enter, present, interact, leave.
    DEBOUNCE_SECONDS = 10.0

    # Minimum seconds between flushing to disk
    FLUSH_INTERVAL = 5.0

    # Regex to strip "[repeated xN]" / "[REPEATED xN]" suffixes
    _REPEATED_RE = re.compile(r"\s*\[repeated\s+x\d+\]", re.IGNORECASE)

    def __init__(self, log_path: str | Path) -> None:
        self._path = Path(log_path)
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()

        # Persistent log data (loaded from / saved to JSON)
        self._data: Dict[str, List[str]] = self._load()

        # --- Per-session runtime state (never persisted) ---
        # track_id -> identity name currently in the room
        self._present: Dict[int, str] = {}
        # track_id -> current interaction object (clean, lowercase)
        self._last_interaction: Dict[int, str] = {}
        # track_id -> consecutive frames with no interaction (drop cooldown)
        self._interaction_drop_counter: Dict[int, int] = {}
        # track_id -> consecutive absent frame count
        self._absent_counter: Dict[int, int] = {}
        # Tracks already logged as "left" (skip duplicates)
        self._left_logged: Set[int] = set()

        # Wall-clock debounce: (identity, event_key) -> last log timestamp
        # event_key examples: "entered", "present", "interact:laptop", "left"
        self._last_log_time: Dict[tuple, float] = {}

        # Coalesce unknowns: have we already logged "Unknown person has entered"
        # in this session?  If so, suppress further unknown-enter logs.
        self._unknown_entered_logged: bool = False

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
        now = time.monotonic()

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
                self._present[track_id] = identity
                self._last_interaction[track_id] = ""

                # Coalesce unknowns: only log ONE "Unknown person has entered"
                if identity == "Unknown person":
                    if not self._unknown_entered_logged:
                        self._unknown_entered_logged = True
                        entry = self._debounced_log(identity, "entered",
                                                    f"{identity} has entered the room", now)
                        if entry:
                            new_entries.append(entry)
                    # else: silently suppress additional unknown entries
                else:
                    entry = self._debounced_log(identity, "entered",
                                                f"{identity} has entered the room", now)
                    if entry:
                        new_entries.append(entry)

            else:
                # Identity may upgrade (Unknown -> recognized name)
                prev_name = self._present[track_id]
                if identity != "Unknown person" and prev_name == "Unknown person":
                    self._present[track_id] = identity
                    entry = self._debounced_log(identity, "present",
                                                f"{identity} is present in the room", now)
                    if entry:
                        new_entries.append(entry)

            # --- Interaction detection ---
            obj_name = self._parse_interaction(behavior)
            if obj_name:
                # Active interaction — reset drop counter
                self._interaction_drop_counter.pop(track_id, None)

                if self._last_interaction.get(track_id) != obj_name:
                    self._last_interaction[track_id] = obj_name
                    person = self._present[track_id]
                    entry = self._debounced_log(
                        person, f"interact:{obj_name}",
                        f"{person} is interacting with {obj_name}", now,
                    )
                    if entry:
                        new_entries.append(entry)
            else:
                # No interaction this frame — use drop cooldown before resetting
                if self._last_interaction.get(track_id):
                    drop = self._interaction_drop_counter.get(track_id, 0) + 1
                    self._interaction_drop_counter[track_id] = drop
                    if drop >= self.INTERACTION_DROP_FRAMES:
                        self._last_interaction[track_id] = ""
                        self._interaction_drop_counter.pop(track_id, None)

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
                    self._interaction_drop_counter.pop(tid, None)
                    self._left_logged.add(tid)
                    entry = self._debounced_log(name, "left",
                                                f"{name} has left the room", now)
                    if entry:
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
        now = time.monotonic()
        for tid in list(self._present):
            name = self._present.pop(tid)
            # Force-log "left" on close (bypass debounce — session is ending)
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
        """Extract the clean object name from a behavior status string.

        Strips any ``[repeated xN]`` suffix so de-duplication always
        compares against the bare object name.
        """
        if "INTERACTING WITH" in behavior:
            raw = behavior.split("INTERACTING WITH")[-1].strip()
            clean = RoomActivityLogger._REPEATED_RE.sub("", raw).strip().lower()
            return clean
        return ""

    def _debounced_log(self, identity: str, event_key: str,
                       message: str, now: float) -> Optional[str]:
        """Log *message* only if (identity, event_key) hasn't been logged
        within the last ``DEBOUNCE_SECONDS``.  Returns the logged entry
        string, or ``None`` if suppressed."""
        key = (identity.lower(), event_key)
        last = self._last_log_time.get(key, 0.0)
        if (now - last) < self.DEBOUNCE_SECONDS:
            return None  # suppressed
        entry = self._add_entry(message)
        if not entry:
            return None  # suppressed by _add_entry (same-second dup)
        self._last_log_time[key] = now
        return entry

    def _today(self) -> str:
        return datetime.now().strftime("%Y-%m-%d")

    def _timestamp(self) -> str:
        return datetime.now().strftime("%H:%M:%S")

    def _add_entry(self, message: str) -> str:
        """Append a timestamped entry for today and return the full string.

        Safety-net rules applied here (last line of defence):
        1. Strip any ``[repeated xN]`` suffix that leaked through.
        2. Reject exact duplicate of the most recent entry for today
           (same timestamp + same message → skip).
        """
        # Safety-net: strip [repeated xN] from the final message
        message = self._REPEATED_RE.sub("", message).strip()

        ts = self._timestamp()
        entry = f"{ts} - {message}"
        date_key = self._today()

        with self._lock:
            today_entries = self._data.setdefault(date_key, [])
            # Reject same-second exact duplicate (scan last few entries)
            for prev in reversed(today_entries[-10:]):
                if prev == entry:
                    return ""  # suppressed — identical entry already exists
                # Stop scanning once we pass a different timestamp
                if not prev.startswith(ts):
                    break
            today_entries.append(entry)
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
            if tmp.exists():
                tmp.unlink(missing_ok=True)
