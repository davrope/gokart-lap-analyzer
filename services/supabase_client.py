from __future__ import annotations

from functools import lru_cache

from .config import get_app_config, supabase_configured


@lru_cache(maxsize=1)
def get_supabase_client():
    cfg = get_app_config()
    if not supabase_configured(cfg):
        raise RuntimeError("Supabase is not configured. Set SUPABASE_URL and SUPABASE_ANON_KEY.")

    try:
        from supabase import create_client
    except Exception as exc:
        raise RuntimeError(
            "Missing supabase dependency. Install with `pip install supabase`."
        ) from exc

    return create_client(cfg.supabase_url, cfg.supabase_anon_key)
