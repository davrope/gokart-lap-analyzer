from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

from .supabase_client import get_supabase_client


BUCKET = "fit-files"


@dataclass
class StorageService:
    client: Any

    def upload_fit(self, user_id: str, attempt_id: str, fit_bytes: bytes, filename: str) -> str:
        safe_name = re.sub(r"[^A-Za-z0-9._-]+", "_", filename or "session.fit")
        path = f"{user_id}/{attempt_id}/{safe_name}"
        self.client.storage.from_(BUCKET).upload(
            path,
            fit_bytes,
            file_options={"content-type": "application/octet-stream", "upsert": "false"},
        )
        return path

    def download_fit(self, storage_path: str) -> bytes:
        data = self.client.storage.from_(BUCKET).download(storage_path)
        if isinstance(data, bytes):
            return data
        if hasattr(data, "read"):
            return data.read()
        raise RuntimeError("Unexpected storage download response")

    def delete_fit(self, storage_path: str) -> None:
        self.client.storage.from_(BUCKET).remove([storage_path])



def get_storage_service() -> StorageService:
    return StorageService(get_supabase_client())
