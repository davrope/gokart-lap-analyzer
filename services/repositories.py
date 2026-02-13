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
    def __init__(self, client: Any) -> None:
        self.client = client

    def create_attempt(self, payload: dict[str, Any]) -> dict[str, Any]:
        resp = self.client.table("attempts").insert(payload).execute()
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
