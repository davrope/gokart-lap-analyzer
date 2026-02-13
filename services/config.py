from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any


def _streamlit_secrets() -> dict[str, Any]:
    try:
        import streamlit as st

        return dict(st.secrets)
    except Exception:
        return {}


@dataclass(frozen=True)
class AppConfig:
    supabase_url: str
    supabase_anon_key: str
    app_base_url: str
    multi_attempt_mode: bool



def _to_bool(value: Any, default: bool = False) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    s = str(value).strip().lower()
    if s in {"1", "true", "yes", "on"}:
        return True
    if s in {"0", "false", "no", "off"}:
        return False
    return default



def _get(key: str, secrets: dict[str, Any], default: str = "") -> str:
    if key in os.environ:
        return str(os.environ.get(key, default))
    if key in secrets:
        return str(secrets.get(key, default))
    return default



def get_app_config() -> AppConfig:
    secrets = _streamlit_secrets()

    supabase_url = _get("SUPABASE_URL", secrets)
    supabase_anon_key = _get("SUPABASE_ANON_KEY", secrets)
    app_base_url = _get("APP_BASE_URL", secrets)

    mam_raw = os.environ.get("MULTI_ATTEMPT_MODE")
    if mam_raw is None and "MULTI_ATTEMPT_MODE" in secrets:
        mam_raw = str(secrets.get("MULTI_ATTEMPT_MODE"))

    return AppConfig(
        supabase_url=supabase_url,
        supabase_anon_key=supabase_anon_key,
        app_base_url=app_base_url,
        multi_attempt_mode=_to_bool(mam_raw, default=True),
    )



def supabase_configured(cfg: AppConfig | None = None) -> bool:
    cfg = cfg or get_app_config()
    return bool(cfg.supabase_url and cfg.supabase_anon_key)
