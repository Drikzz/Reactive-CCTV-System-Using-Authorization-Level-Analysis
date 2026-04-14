"""
Session Metrics Tracker
-----------------------
Collects system-level evaluation metrics during a CCTV monitoring session:
  • Detection Rate   — % of frames where YOLO found ≥1 person
  • Prediction Rate   — % of detected people successfully identified (not "Unknown")
  • Performance       — FPS, processing time, session duration, etc.

Usage:
    metrics = SessionMetrics()
    metrics.tick_frame(detections)   # call every processed frame
    report = metrics.report()        # get final summary dict
"""

import time
import threading
from typing import Any, Dict, List


class SessionMetrics:
    """Lightweight, thread-safe session-level metrics collector."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._start_time: float = time.time()
        self._last_tick: float = self._start_time

        # Frame counters
        self.total_frames: int = 0
        self.frames_with_people: int = 0      # frames where ≥1 person detected

        # Detection counters (per-detection, not per-frame)
        self.total_person_detections: int = 0  # every person bbox across all frames
        self.identified_detections: int = 0    # person bbox where identity ≠ "Unknown"
        self.unknown_detections: int = 0       # person bbox where identity == "Unknown"

        # Authorization breakdown
        self.auth_counts: Dict[str, int] = {
            "Authorized": 0,
            "Partially Authorized": 0,
            "Unauthorized": 0,
        }

        # Behavior / interaction
        self.total_interactions: int = 0       # detections with active behavior
        self.unauthorized_interactions: int = 0

        # Unique identities seen (set of lowercase names)
        self.unique_people: set = set()

        # FPS tracking (rolling window)
        self._fps_window: List[float] = []
        self._FPS_WINDOW_SIZE = 30

    # ------------------------------------------------------------------
    # Per-frame update
    # ------------------------------------------------------------------
    def tick_frame(self, detections: List[Dict[str, Any]]) -> None:
        """Call once per processed frame with the list of person detections."""
        now = time.time()
        with self._lock:
            self.total_frames += 1

            # FPS
            if self._last_tick:
                dt = now - self._last_tick
                if dt > 0:
                    self._fps_window.append(1.0 / dt)
                    if len(self._fps_window) > self._FPS_WINDOW_SIZE:
                        self._fps_window.pop(0)
            self._last_tick = now

            # Person-level stats
            people_in_frame = [d for d in detections if d.get("track_id", -1) >= 0 or d.get("identity")]
            if people_in_frame:
                self.frames_with_people += 1

            for det in people_in_frame:
                self.total_person_detections += 1
                identity = det.get("identity", "Unknown")
                auth = det.get("authorization", "Unauthorized")
                behavior = det.get("behavior_status", "STATUS: NO INTERACTION")

                if identity and identity != "Unknown":
                    self.identified_detections += 1
                    self.unique_people.add(identity.lower())
                else:
                    self.unknown_detections += 1

                if auth in self.auth_counts:
                    self.auth_counts[auth] += 1

                if behavior != "STATUS: NO INTERACTION":
                    self.total_interactions += 1
                    if auth in ("Unauthorized", "Partially Authorized"):
                        self.unauthorized_interactions += 1

    # ------------------------------------------------------------------
    # Queries (safe to call from UI thread)
    # ------------------------------------------------------------------
    @property
    def current_fps(self) -> float:
        with self._lock:
            if not self._fps_window:
                return 0.0
            return sum(self._fps_window) / len(self._fps_window)

    @property
    def elapsed_seconds(self) -> float:
        return time.time() - self._start_time

    @property
    def detection_rate(self) -> float:
        """% of frames that had ≥1 person detected."""
        with self._lock:
            if self.total_frames == 0:
                return 0.0
            return (self.frames_with_people / self.total_frames) * 100.0

    @property
    def prediction_rate(self) -> float:
        """% of person detections successfully identified (not Unknown)."""
        with self._lock:
            if self.total_person_detections == 0:
                return 0.0
            return (self.identified_detections / self.total_person_detections) * 100.0

    # ------------------------------------------------------------------
    # Snapshot for UI (lightweight dict)
    # ------------------------------------------------------------------
    def snapshot(self) -> Dict[str, Any]:
        """Return a small dict for live UI updates."""
        with self._lock:
            fps = (sum(self._fps_window) / len(self._fps_window)) if self._fps_window else 0.0
            det = (self.frames_with_people / self.total_frames * 100.0) if self.total_frames else 0.0
            pred = (self.identified_detections / self.total_person_detections * 100.0) if self.total_person_detections else 0.0
            return {
                "fps": round(fps, 1),
                "total_frames": self.total_frames,
                "detection_rate": round(det, 1),
                "prediction_rate": round(pred, 1),
                "unique_people": len(self.unique_people),
            }

    # ------------------------------------------------------------------
    # Full report (for session end)
    # ------------------------------------------------------------------
    def report(self) -> Dict[str, Any]:
        """Generate a complete evaluation report dict."""
        elapsed = self.elapsed_seconds
        mins, secs = divmod(int(elapsed), 60)
        hrs, mins = divmod(mins, 60)
        duration_str = f"{hrs:02d}:{mins:02d}:{secs:02d}"

        with self._lock:
            avg_fps = sum(self._fps_window) / len(self._fps_window) if self._fps_window else 0.0
            det_rate = (self.frames_with_people / self.total_frames * 100.0) if self.total_frames else 0.0
            pred_rate = (self.identified_detections / self.total_person_detections * 100.0) if self.total_person_detections else 0.0

            return {
                # Session
                "session_duration": duration_str,
                "session_seconds": round(elapsed, 1),

                # Performance
                "avg_fps": round(avg_fps, 1),
                "total_frames_processed": self.total_frames,

                # Detection rate
                "frames_with_people": self.frames_with_people,
                "detection_rate_pct": round(det_rate, 2),

                # Prediction rate
                "total_person_detections": self.total_person_detections,
                "identified_detections": self.identified_detections,
                "unknown_detections": self.unknown_detections,
                "prediction_rate_pct": round(pred_rate, 2),

                # Authorization
                "auth_authorized": self.auth_counts.get("Authorized", 0),
                "auth_partial": self.auth_counts.get("Partially Authorized", 0),
                "auth_unauthorized": self.auth_counts.get("Unauthorized", 0),

                # Behavior
                "total_interactions": self.total_interactions,
                "unauthorized_interactions": self.unauthorized_interactions,

                # People
                "unique_people_count": len(self.unique_people),
                "unique_people_list": sorted(self.unique_people),
            }

    def report_text(self) -> str:
        """Human-readable text version of the report."""
        r = self.report()
        lines = [
            "=" * 60,
            "SYSTEM EVALUATION REPORT",
            "=" * 60,
            "",
            "── Performance ──",
            f"  Session Duration     : {r['session_duration']}",
            f"  Average FPS          : {r['avg_fps']}",
            f"  Total Frames         : {r['total_frames_processed']}",
            "",
            "── Detection Rate ──",
            f"  Frames with People   : {r['frames_with_people']} / {r['total_frames_processed']}",
            f"  Detection Rate       : {r['detection_rate_pct']}%",
            "",
            "── Prediction Rate (Face Recognition) ──",
            f"  Total Person Detections : {r['total_person_detections']}",
            f"  Successfully Identified : {r['identified_detections']}",
            f"  Unknown / Unrecognized  : {r['unknown_detections']}",
            f"  Prediction Rate         : {r['prediction_rate_pct']}%",
            "",
            "── Authorization Breakdown ──",
            f"  Authorized              : {r['auth_authorized']}",
            f"  Partially Authorized    : {r['auth_partial']}",
            f"  Unauthorized            : {r['auth_unauthorized']}",
            "",
            "── Behavior / Interactions ──",
            f"  Total Interactions      : {r['total_interactions']}",
            f"  Unauthorized Actions    : {r['unauthorized_interactions']}",
            "",
            "── People Identified ──",
            f"  Unique People : {r['unique_people_count']}",
        ]
        if r["unique_people_list"]:
            for name in r["unique_people_list"]:
                lines.append(f"    • {name.title()}")
        else:
            lines.append("    (none)")
        lines.append("=" * 60)
        return "\n".join(lines)
