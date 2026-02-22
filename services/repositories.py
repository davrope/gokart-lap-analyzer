from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

import pandas as pd

from .supabase_client import get_supabase_client


@dataclass
class TrackRecord:
    id: str
    user_id: str
    name: str
    layout_direction: str
    layout_variant: str | None
    location: str | None
    created_at: str


@dataclass
class AttemptRecord:
    id: str
    user_id: str
    track_id: str
    attempt_name: str | None
    source_filename: str
    storage_bucket: str
    storage_path: str
    status: str
    method_name: str
    params_json: dict[str, Any]
    uploaded_at: str
    processed_at: str | None


class TrackRepository:
    def __init__(self, client: Any) -> None:
        self.client = client

    def list_tracks(self, user_id: str) -> list[dict[str, Any]]:
        resp = (
            self.client.table("tracks")
            .select("id,user_id,name,layout_direction,layout_variant,location,created_at")
            .eq("user_id", user_id)
            .order("created_at", desc=False)
            .execute()
        )
        return list(getattr(resp, "data", []) or [])

    def create_track(
        self,
        user_id: str,
        name: str,
        layout_direction: str,
        layout_variant: str | None,
        location: str | None,
    ) -> dict[str, Any]:
        payload = {
            "user_id": user_id,
            "name": name.strip(),
            "layout_direction": layout_direction,
            "layout_variant": (layout_variant or "").strip() or None,
            "location": (location or "").strip() or None,
        }
        resp = self.client.table("tracks").insert(payload).execute()
        data = getattr(resp, "data", []) or []
        if not data:
            raise RuntimeError("Track creation failed")
        return data[0]


class AttemptRepository:
    FALLBACK_ATTEMPT_NAME_KEY = "__attempt_name"

    def __init__(self, client: Any) -> None:
        self.client = client

    @staticmethod
    def _error_text(exc: Exception) -> str:
        return str(exc).lower()

    @classmethod
    def _is_missing_attempt_name_column_error(cls, exc: Exception) -> bool:
        text = cls._error_text(exc)
        return "attempt_name" in text and ("schema cache" in text or "pgrst204" in text)

    @staticmethod
    def _sanitize_attempt_name(attempt_name: str) -> str:
        clean_name = (attempt_name or "").strip()
        if not clean_name:
            raise ValueError("Attempt name cannot be empty.")
        return clean_name

    @classmethod
    def _params_json_with_attempt_name(cls, params_json: Any, attempt_name: str) -> dict[str, Any]:
        payload = dict(params_json) if isinstance(params_json, dict) else {}
        payload[cls.FALLBACK_ATTEMPT_NAME_KEY] = attempt_name
        return payload

    def create_attempt(self, payload: dict[str, Any]) -> dict[str, Any]:
        insert_payload = dict(payload)
        raw_attempt_name = insert_payload.get("attempt_name")
        clean_attempt_name = self._sanitize_attempt_name(raw_attempt_name) if raw_attempt_name else None
        try:
            resp = self.client.table("attempts").insert(insert_payload).execute()
        except Exception as exc:
            # Backward-compatible fallback when DB migration hasn't been applied yet.
            if clean_attempt_name and "attempt_name" in insert_payload and self._is_missing_attempt_name_column_error(exc):
                insert_payload.pop("attempt_name", None)
                insert_payload["params_json"] = self._params_json_with_attempt_name(
                    insert_payload.get("params_json"), clean_attempt_name
                )
                resp = self.client.table("attempts").insert(insert_payload).execute()
            else:
                raise
        data = getattr(resp, "data", []) or []
        if not data:
            raise RuntimeError("Attempt creation failed")
        return data[0]

    def update_attempt_processed(
        self,
        attempt_id: str,
        payload: dict[str, Any],
    ) -> dict[str, Any]:
        patch = dict(payload)
        patch["processed_at"] = datetime.now(timezone.utc).isoformat()
        resp = self.client.table("attempts").update(patch).eq("id", attempt_id).execute()
        data = getattr(resp, "data", []) or []
        if not data:
            raise RuntimeError("Attempt update failed")
        return data[0]

    def mark_failed(self, attempt_id: str, error: str) -> None:
        self.client.table("attempts").update(
            {
                "status": "failed",
                "processing_error": error[:5000],
                "processed_at": datetime.now(timezone.utc).isoformat(),
            }
        ).eq("id", attempt_id).execute()

    def list_attempts(self, user_id: str, track_id: str | None = None) -> list[dict[str, Any]]:
        q = (
            self.client.table("attempts")
            .select("*")
            .eq("user_id", user_id)
            .order("uploaded_at", desc=True)
        )
        if track_id:
            q = q.eq("track_id", track_id)
        resp = q.execute()
        return list(getattr(resp, "data", []) or [])

    def get_attempt(self, attempt_id: str) -> dict[str, Any] | None:
        resp = self.client.table("attempts").select("*").eq("id", attempt_id).limit(1).execute()
        data = getattr(resp, "data", []) or []
        return data[0] if data else None

    def update_attempt_name(self, attempt_id: str, attempt_name: str) -> dict[str, Any]:
        clean_name = self._sanitize_attempt_name(attempt_name)
        patch = {"attempt_name": clean_name}
        try:
            resp = self.client.table("attempts").update(patch).eq("id", attempt_id).execute()
        except Exception as exc:
            if not self._is_missing_attempt_name_column_error(exc):
                raise
            attempt = self.get_attempt(attempt_id)
            if attempt is None:
                raise RuntimeError("Attempt rename failed") from exc
            fallback_patch = {
                "params_json": self._params_json_with_attempt_name(attempt.get("params_json"), clean_name),
            }
            resp = self.client.table("attempts").update(fallback_patch).eq("id", attempt_id).execute()
        data = getattr(resp, "data", []) or []
        if not data:
            raise RuntimeError("Attempt rename failed")
        return data[0]

    def delete_attempt(self, attempt_id: str) -> bool:
        resp = self.client.table("attempts").delete().eq("id", attempt_id).execute()
        data = getattr(resp, "data", []) or []
        return bool(data)

    def replace_attempt_laps(self, attempt_id: str, lap_rows: list[dict[str, Any]]) -> None:
        self.client.table("attempt_laps").delete().eq("attempt_id", attempt_id).execute()
        if lap_rows:
            self.client.table("attempt_laps").insert(lap_rows).execute()

    def replace_attempt_curves(self, attempt_id: str, curve_rows: list[dict[str, Any]]) -> None:
        self.client.table("attempt_curves").delete().eq("attempt_id", attempt_id).execute()
        if curve_rows:
            self.client.table("attempt_curves").insert(curve_rows).execute()

    def list_attempt_laps(self, attempt_ids: list[str]) -> pd.DataFrame:
        if not attempt_ids:
            return pd.DataFrame()
        resp = self.client.table("attempt_laps").select("*").in_("attempt_id", attempt_ids).execute()
        return pd.DataFrame(getattr(resp, "data", []) or [])

    def list_attempt_curves(self, attempt_ids: list[str]) -> pd.DataFrame:
        if not attempt_ids:
            return pd.DataFrame()
        resp = self.client.table("attempt_curves").select("*").in_("attempt_id", attempt_ids).execute()
        return pd.DataFrame(getattr(resp, "data", []) or [])



def get_track_repository() -> TrackRepository:
    return TrackRepository(get_supabase_client())



def get_attempt_repository() -> AttemptRepository:
    return AttemptRepository(get_supabase_client())
