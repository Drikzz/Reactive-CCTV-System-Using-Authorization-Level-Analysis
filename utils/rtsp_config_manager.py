"""
RTSP Config Manager
-------------------
Manages RTSP camera credentials dynamically (no hardcoding).
Stores camera configurations in a local JSON file that is gitignored,
with optional keyring integration for password security.

Usage:
    from utils.rtsp_config_manager import RTSPConfigManager

    mgr = RTSPConfigManager()
    mgr.add_camera("Cam1", ip="192.168.1.10", username="admin", password="pass")
    url = mgr.build_url("Cam1")
    cameras = mgr.list_cameras()
"""

import json
import base64
import threading
from pathlib import Path
from typing import Optional, Dict, List, Tuple, Any

# ---------------------------------------------------------------------------
# Optional keyring import – graceful fallback
# ---------------------------------------------------------------------------
try:
    import keyring as _keyring
    _HAS_KEYRING = True
except ImportError:
    _keyring = None
    _HAS_KEYRING = False

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
_DEFAULT_CONFIG_DIR = Path(__file__).resolve().parents[1] / "config"
_DEFAULT_CONFIG_FILE = _DEFAULT_CONFIG_DIR / "rtsp_cameras.json"
_KEYRING_SERVICE = "ReactiveCCTV_RTSP"

# Template for a new camera entry (no credentials stored in code)
_CAMERA_TEMPLATE: Dict[str, Any] = {
    "ip": "",
    "port": "554",
    "username": "",
    "password_b64": "",          # base64-obfuscated fallback
    "password_in_keyring": False, # True → read from OS keyring
    "stream": "stream2",
    "protocol": "rtsp",
    "enabled": True,
    "remember": True,            # persist credentials across sessions
}


class RTSPConfigManager:
    """Thread-safe manager for RTSP camera credentials stored locally."""

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------
    def __init__(self, config_path: Optional[Path] = None):
        self._path = Path(config_path) if config_path else _DEFAULT_CONFIG_FILE
        self._lock = threading.Lock()
        self._cameras: Dict[str, Dict[str, Any]] = {}
        self._load()

    # ------------------------------------------------------------------
    # Public API – CRUD
    # ------------------------------------------------------------------
    def add_camera(
        self,
        name: str,
        *,
        ip: str,
        port: str = "554",
        username: str = "",
        password: str = "",
        stream: str = "stream2",
        protocol: str = "rtsp",
        enabled: bool = True,
        remember: bool = True,
    ) -> None:
        """Add or update an RTSP camera entry."""
        with self._lock:
            entry = dict(_CAMERA_TEMPLATE)
            entry.update(
                ip=ip.strip(),
                port=port.strip(),
                username=username.strip(),
                stream=stream.strip(),
                protocol=protocol.strip(),
                enabled=enabled,
                remember=remember,
            )
            # Store password
            self._store_password(entry, name, password, remember)
            self._cameras[name] = entry
            if remember:
                self._save()

    def remove_camera(self, name: str) -> bool:
        """Delete a camera entry.  Returns True if it existed."""
        with self._lock:
            if name not in self._cameras:
                return False
            # Clean up keyring entry
            if self._cameras[name].get("password_in_keyring") and _HAS_KEYRING:
                try:
                    _keyring.delete_password(_KEYRING_SERVICE, name)
                except Exception:
                    pass
            del self._cameras[name]
            self._save()
            return True

    def update_camera(self, name: str, **kwargs) -> bool:
        """Partially update fields.  Returns False if camera doesn't exist."""
        with self._lock:
            if name not in self._cameras:
                return False
            entry = self._cameras[name]
            for key, value in kwargs.items():
                if key == "password":
                    self._store_password(entry, name, value, entry.get("remember", True))
                elif key in entry:
                    entry[key] = value
            if entry.get("remember", True):
                self._save()
            return True

    def get_camera(self, name: str) -> Optional[Dict[str, Any]]:
        """Return a *copy* of the camera config (password resolved)."""
        with self._lock:
            entry = self._cameras.get(name)
            if entry is None:
                return None
            out = dict(entry)
            out["password"] = self._resolve_password(entry, name)
            out.pop("password_b64", None)
            out.pop("password_in_keyring", None)
            return out

    def list_cameras(self) -> List[Tuple[str, str, bool]]:
        """Return [(name, display_label, enabled), ...]."""
        with self._lock:
            result = []
            for name, cfg in self._cameras.items():
                label = f"{cfg['ip']} ({cfg['stream']})"
                result.append((name, label, cfg.get("enabled", True)))
            return result

    def get_enabled_cameras(self) -> List[Tuple[str, str]]:
        """Return [(name, display_label)] for enabled cameras only."""
        return [(n, lbl) for n, lbl, en in self.list_cameras() if en]

    # ------------------------------------------------------------------
    # URL building
    # ------------------------------------------------------------------
    def build_url(self, name: str) -> Optional[str]:
        """Build a full RTSP URL for the named camera, or None."""
        cam = self.get_camera(name)
        if cam is None:
            return None
        if not cam.get("enabled", True):
            return None
        proto = cam.get("protocol", "rtsp")
        user = cam.get("username", "")
        pwd = cam.get("password", "")
        ip = cam["ip"]
        port = cam.get("port", "554")
        stream = cam.get("stream", "stream2")
        if user and pwd:
            return f"{proto}://{user}:{pwd}@{ip}:{port}/{stream}"
        elif user:
            return f"{proto}://{user}@{ip}:{port}/{stream}"
        else:
            return f"{proto}://{ip}:{port}/{stream}"

    # ------------------------------------------------------------------
    # Migration helper – import from old hardcoded dict
    # ------------------------------------------------------------------
    def migrate_from_dict(self, cameras_dict: Dict[str, Dict]) -> int:
        """
        Import cameras from the legacy RTSP_CAMERAS dict in
        camera_config_streamlit.py.  Skips entries that already exist.
        Returns the number of cameras imported.
        """
        count = 0
        for name, cfg in cameras_dict.items():
            if name in self._cameras:
                continue
            self.add_camera(
                name,
                ip=cfg.get("ip", ""),
                port=cfg.get("port", "554"),
                username=cfg.get("username", ""),
                password=cfg.get("password", ""),
                stream=cfg.get("stream", "stream2"),
                enabled=cfg.get("enabled", True),
                remember=True,
            )
            count += 1
        return count

    # ------------------------------------------------------------------
    # Persistence helpers
    # ------------------------------------------------------------------
    @property
    def has_keyring(self) -> bool:
        return _HAS_KEYRING

    @property
    def config_path(self) -> Path:
        return self._path

    @property
    def camera_count(self) -> int:
        return len(self._cameras)

    def reload(self) -> None:
        """Re-read config file from disk (useful if edited externally)."""
        with self._lock:
            self._load()

    # ------------------------------------------------------------------
    # Internal – password storage
    # ------------------------------------------------------------------
    def _store_password(
        self, entry: dict, cam_name: str, password: str, persist: bool
    ) -> None:
        """Store password in keyring (preferred) or base64 fallback."""
        if not password:
            entry["password_b64"] = ""
            entry["password_in_keyring"] = False
            return

        if _HAS_KEYRING and persist:
            try:
                _keyring.set_password(_KEYRING_SERVICE, cam_name, password)
                entry["password_in_keyring"] = True
                entry["password_b64"] = ""  # don't duplicate
                return
            except Exception:
                pass  # fall through to base64

        # Fallback: base64 obfuscation (NOT encryption – just avoids plaintext)
        entry["password_b64"] = base64.b64encode(password.encode()).decode()
        entry["password_in_keyring"] = False

    def _resolve_password(self, entry: dict, cam_name: str) -> str:
        """Read back the password from keyring or base64."""
        if entry.get("password_in_keyring") and _HAS_KEYRING:
            try:
                pwd = _keyring.get_password(_KEYRING_SERVICE, cam_name)
                if pwd is not None:
                    return pwd
            except Exception:
                pass

        b64 = entry.get("password_b64", "")
        if b64:
            try:
                return base64.b64decode(b64.encode()).decode()
            except Exception:
                return ""
        return ""

    # ------------------------------------------------------------------
    # Internal – file I/O
    # ------------------------------------------------------------------
    def _load(self) -> None:
        if not self._path.exists():
            self._cameras = {}
            return
        try:
            with open(self._path, "r", encoding="utf-8") as f:
                data = json.load(f)
            self._cameras = data.get("cameras", {})
        except (json.JSONDecodeError, IOError):
            self._cameras = {}

    def _save(self) -> None:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        # Only write cameras flagged with remember=True
        persisted = {
            k: v for k, v in self._cameras.items() if v.get("remember", True)
        }
        payload = {"cameras": persisted}
        tmp = self._path.with_suffix(".tmp")
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)
        tmp.replace(self._path)


# ---------------------------------------------------------------------------
# Module-level singleton for convenience
# ---------------------------------------------------------------------------
_singleton: Optional[RTSPConfigManager] = None
_singleton_lock = threading.Lock()


def get_manager(config_path: Optional[Path] = None) -> RTSPConfigManager:
    """Return a module-level singleton RTSPConfigManager."""
    global _singleton
    with _singleton_lock:
        if _singleton is None:
            _singleton = RTSPConfigManager(config_path)
        return _singleton
