"""
Confirmation Manager — Temporal smoothing for YOLO + FaceNet detections.
------------------------------------------------------------------------
Sits between the pipeline output and the activity logger to ensure only
**stable, confirmed** detections are logged.

Technique: Per-track state machine with hysteresis.
  - A new track starts as CANDIDATE.
  - After CONFIRM_FRAMES consecutive frames with consistent identity,
    it transitions to CONFIRMED → only then is it safe to log.
  - If a track disappears, it enters a GRACE period (tolerating brief
    occlusion) before being declared "left."
  - Identity votes accumulate — a flicker between "Unknown" and "Aldrikz"
    for 2 frames won't override a stable "Aldrikz" seen for 15 frames.
  - Face confidence is smoothed with an exponential moving average (EMA).

Anti-spam guarantees:
  - No "Unknown person" entry unless the person is *still* unknown after
    the full confirmation window.
  - Name changes require the same confirmation window before updating.
  - Brief occlusion (< GONE_GRACE_FRAMES) does NOT reset state.
"""

from __future__ import annotations

import time
from collections import defaultdict
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Set, Tuple


class TrackPhase(Enum):
    """State machine phases for each tracked person."""
    CANDIDATE = "candidate"      # Just appeared — not yet confirmed
    CONFIRMED = "confirmed"      # Stable detection — safe to log/display
    GONE_GRACE = "gone_grace"    # Disappeared but within grace window


@dataclass
class TrackState:
    """Per-track confirmation state."""

    phase: TrackPhase = TrackPhase.CANDIDATE

    # Current candidate (what YOLO + FaceNet claim *right now*)
    candidate_identity: str = "Unknown"
    candidate_auth: str = "Unauthorized"

    # Confirmed (what we've verified as stable)
    confirmed_identity: str = "Unknown"
    confirmed_auth: str = "Unauthorized"

    # --- Counters ---
    consistent_frames: int = 0       # Consecutive frames with same identity
    gone_frames: int = 0             # Consecutive frames NOT seen
    total_seen: int = 0              # Total frames this track was visible

    # --- Identity voting (handles flicker between names) ---
    identity_votes: Dict[str, int] = field(default_factory=dict)

    # --- Confidence smoothing (EMA) ---
    smoothed_confidence: float = 0.0

    # --- Flags ---
    entry_logged: bool = False       # Has "entered the room" been logged?


class ConfirmationManager:
    """Temporal confirmation for YOLO track detections.

    Prevents spammy logs by requiring consistent detection over
    multiple frames before confirming identity/authorization.

    Usage::

        confirmer = ConfirmationManager()

        # Inside the per-frame loop:
        seen_ids = set()
        for det in detections:
            result = confirmer.update(det)
            seen_ids.add(det["track_id"])
            if result["confirmed"]:
                # Safe to use det for logging / display
                ...

        # After processing all detections, handle absences:
        confirmed_detections, left_events = confirmer.finish_frame(seen_ids)
    """

    def __init__(
        self,
        confirm_frames: int = 15,        # ~0.5s at 30 fps
        gone_grace_frames: int = 45,     # ~1.5s — occlusion tolerance
        name_lock_hits: int = 3,         # FaceNet must agree N times
        ema_alpha: float = 0.3,          # Confidence smoothing factor
    ) -> None:
        self.confirm_frames = confirm_frames
        self.gone_grace_frames = gone_grace_frames
        self.name_lock_hits = name_lock_hits
        self.ema_alpha = ema_alpha

        self._tracks: Dict[int, TrackState] = {}

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def update(self, detection: Dict[str, Any]) -> Dict[str, Any]:
        """Process one detection from the current frame.

        Args:
            detection: dict with keys ``track_id``, ``identity``,
                       ``authorization``, ``identity_conf``.

        Returns:
            Result dict with:
              - ``confirmed`` (bool): True if the track is stable enough to log.
              - ``identity`` (str): The confirmed (or best-candidate) name.
              - ``authorization`` (str): Confirmed auth level.
              - ``confidence`` (float): EMA-smoothed confidence.
              - ``phase`` (TrackPhase): Current state machine phase.
              - ``just_confirmed`` (bool): True on the *exact frame* the track
                transitions from CANDIDATE → CONFIRMED.
        """
        track_id: int = detection.get("track_id", -1)
        raw_identity: str = detection.get("identity", "Unknown")
        raw_auth: str = detection.get("authorization", "Unauthorized")
        raw_conf: float = float(detection.get("identity_conf", 0.0))

        if track_id < 0:
            # Untracked detection — pass through unconfirmed
            return self._unconfirmed_result(raw_identity, raw_auth, raw_conf)

        # Get or create state
        if track_id not in self._tracks:
            self._tracks[track_id] = TrackState()
        st = self._tracks[track_id]

        st.total_seen += 1
        st.gone_frames = 0  # We see this track right now

        # --- Confidence EMA ---
        if st.smoothed_confidence < 0.01:
            st.smoothed_confidence = raw_conf  # Seed on first frame
        else:
            st.smoothed_confidence = (
                self.ema_alpha * raw_conf
                + (1.0 - self.ema_alpha) * st.smoothed_confidence
            )

        # --- Identity voting ---
        st.identity_votes[raw_identity] = st.identity_votes.get(raw_identity, 0) + 1

        # Check consistency with current candidate
        if raw_identity == st.candidate_identity:
            st.consistent_frames += 1
        else:
            # Identity changed — check vote leader
            best_name = max(st.identity_votes, key=st.identity_votes.get)
            if st.identity_votes[best_name] >= self.name_lock_hits:
                st.candidate_identity = best_name
                st.candidate_auth = raw_auth
                # Give partial credit for accumulated votes
                st.consistent_frames = min(
                    st.consistent_frames, self.confirm_frames // 2
                )
            else:
                st.candidate_identity = raw_identity
                st.candidate_auth = raw_auth
                st.consistent_frames = 1

        # --- State machine transitions ---
        just_confirmed = False

        if st.phase == TrackPhase.CANDIDATE:
            if st.consistent_frames >= self.confirm_frames:
                st.phase = TrackPhase.CONFIRMED
                st.confirmed_identity = st.candidate_identity
                st.confirmed_auth = st.candidate_auth
                just_confirmed = True

        elif st.phase == TrackPhase.CONFIRMED:
            # Update confirmed identity if a *new* name stabilizes
            if (
                st.candidate_identity != st.confirmed_identity
                and st.consistent_frames >= self.confirm_frames
            ):
                st.confirmed_identity = st.candidate_identity
                st.confirmed_auth = st.candidate_auth
                just_confirmed = True

        elif st.phase == TrackPhase.GONE_GRACE:
            # Was gone, now back — re-enter CONFIRMED
            st.phase = TrackPhase.CONFIRMED

        is_confirmed = st.phase == TrackPhase.CONFIRMED
        return {
            "confirmed": is_confirmed,
            "identity": st.confirmed_identity if is_confirmed else st.candidate_identity,
            "authorization": st.confirmed_auth if is_confirmed else st.candidate_auth,
            "confidence": st.smoothed_confidence,
            "phase": st.phase,
            "just_confirmed": just_confirmed,
        }

    def finish_frame(self, seen_ids: Set[int]) -> List[Dict[str, Any]]:
        """Call once per frame AFTER processing all detections.

        Ticks absent counters for every track NOT in ``seen_ids``.

        Returns:
            A list of "left" event dicts for tracks whose grace expired.
            Each dict has keys: ``track_id``, ``identity``, ``authorization``.
        """
        left_events: List[Dict[str, Any]] = []

        for tid in list(self._tracks.keys()):
            if tid in seen_ids:
                continue  # Already updated via update()

            st = self._tracks[tid]
            st.gone_frames += 1

            if st.phase == TrackPhase.CONFIRMED and st.gone_frames == 1:
                st.phase = TrackPhase.GONE_GRACE

            if st.gone_frames >= self.gone_grace_frames:
                left_events.append({
                    "track_id": tid,
                    "identity": st.confirmed_identity,
                    "authorization": st.confirmed_auth,
                })
                del self._tracks[tid]

        return left_events

    def get_confirmed_detections(
        self, detections: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Convenience: filter a list of detections to only confirmed ones.

        Also patches each detection's identity/auth/confidence with the
        confirmed (smoothed) values.  Untracked detections (track_id < 0)
        are always excluded.
        """
        confirmed: List[Dict[str, Any]] = []

        for det in detections:
            tid = det.get("track_id", -1)
            if tid < 0:
                continue
            st = self._tracks.get(tid)
            if st is None or st.phase != TrackPhase.CONFIRMED:
                continue

            # Patch with confirmed values
            det = dict(det)  # Shallow copy so we don't mutate the original
            det["identity"] = st.confirmed_identity
            det["authorization"] = st.confirmed_auth
            det["identity_conf"] = st.smoothed_confidence
            confirmed.append(det)

        return confirmed

    def reset(self) -> None:
        """Clear all track states (e.g. on session restart)."""
        self._tracks.clear()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _unconfirmed_result(
        identity: str, auth: str, conf: float
    ) -> Dict[str, Any]:
        return {
            "confirmed": False,
            "identity": identity,
            "authorization": auth,
            "confidence": conf,
            "phase": TrackPhase.CANDIDATE,
            "just_confirmed": False,
        }
