"""
Local runtime settings for RESA_AI.

These settings are intentionally stored outside the codebase logic so the
frontend server and the analysis subprocess can share the same local config.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
from threading import Lock


ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
DATA_DIR.mkdir(parents=True, exist_ok=True)
SETTINGS_PATH = DATA_DIR / "backend_settings.json"

DEFAULT_SETTINGS = {
    "groq_api_key": "",
}

_SETTINGS_LOCK = Lock()


def load_backend_settings() -> dict:
    """Load persisted backend settings from disk."""
    with _SETTINGS_LOCK:
        if not SETTINGS_PATH.exists():
            return DEFAULT_SETTINGS.copy()

        try:
            raw = json.loads(SETTINGS_PATH.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return DEFAULT_SETTINGS.copy()

    if not isinstance(raw, dict):
        return DEFAULT_SETTINGS.copy()

    settings = DEFAULT_SETTINGS.copy()
    for key, value in raw.items():
        if key in settings and isinstance(value, str):
            settings[key] = value.strip()
    return settings


def save_backend_settings(updates: dict) -> dict:
    """Persist backend settings and return the merged result."""
    settings = load_backend_settings()
    for key, value in updates.items():
        if key in settings:
            settings[key] = str(value or "").strip()

    with _SETTINGS_LOCK:
        SETTINGS_PATH.write_text(
            json.dumps(settings, indent=2, ensure_ascii=True),
            encoding="utf-8",
        )

    return settings


def get_stored_groq_api_key() -> str:
    """Return the persisted Groq API key, if one exists."""
    return load_backend_settings().get("groq_api_key", "").strip()


def get_groq_api_key() -> str:
    """Resolve the active Groq API key."""
    env_key = os.environ.get("GROQ_API_KEY", "").strip()
    return env_key or get_stored_groq_api_key()


def set_groq_api_key(api_key: str) -> dict:
    """Save a Groq API key to the backend settings store."""
    return save_backend_settings({"groq_api_key": api_key})


def build_subprocess_env(base_env: dict | None = None) -> dict:
    """Build an environment for analysis subprocesses."""
    env = dict(base_env or os.environ)
    stored_key = get_stored_groq_api_key()

    if stored_key:
        env["GROQ_API_KEY"] = stored_key
    elif not env.get("GROQ_API_KEY"):
        env.pop("GROQ_API_KEY", None)

    return env


def mask_secret(secret: str) -> str:
    """Return a short masked version of a secret for UI status displays."""
    value = str(secret or "").strip()
    if not value:
        return ""
    if len(value) <= 8:
        return "*" * len(value)
    return f"{value[:4]}...{value[-4:]}"
