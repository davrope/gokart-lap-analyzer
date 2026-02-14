from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .config import get_app_config
from .supabase_client import get_supabase_client


@dataclass
class AuthUser:
    id: str
    email: str | None


class AuthService:
    def __init__(self, client: Any, app_base_url: str) -> None:
        self.client = client
        self.app_base_url = app_base_url

    def send_magic_link(self, email: str) -> None:
        email = email.strip()
        if not email:
            raise ValueError("Email is required")
        self.client.auth.sign_in_with_otp(
            {
                "email": email,
                "options": {"email_redirect_to": self.app_base_url or None},
            }
        )

    def get_google_auth_url(self) -> str:
        result = self.client.auth.sign_in_with_oauth(
            {
                "provider": "google",
                "options": {"redirect_to": self.app_base_url or None},
            }
        )
        if isinstance(result, dict):
            direct = str(result.get("url", "")).strip()
            if direct:
                return direct
            data = result.get("data")
            if isinstance(data, dict):
                nested = str(data.get("url", "")).strip()
                if nested:
                    return nested
        direct_attr = str(getattr(result, "url", "")).strip()
        if direct_attr:
            return direct_attr
        data_attr = getattr(result, "data", None)
        if isinstance(data_attr, dict):
            nested_attr = str(data_attr.get("url", "")).strip()
            if nested_attr:
                return nested_attr
        raise RuntimeError("Supabase did not return an OAuth URL for Google login.")

    def consume_magic_link(self, query_params: dict[str, Any]) -> dict[str, Any] | None:
        code = str(query_params.get("code", "")).strip()
        if code:
            result = self.client.auth.exchange_code_for_session({"auth_code": code})
            session = self._extract_session(result)
            if session is None:
                return None
            return self._persist_and_dump_session(session)

        token_hash = str(query_params.get("token_hash", "")).strip()
        otp_type = str(query_params.get("type", "")).strip()
        if not token_hash or not otp_type:
            return None

        result = self.client.auth.verify_otp(
            {
                "token_hash": token_hash,
                "type": otp_type,
            }
        )
        session = self._extract_session(result)
        if session is None:
            return None

        return self._persist_and_dump_session(session)

    def _extract_session(self, result: Any) -> Any | None:
        session = getattr(result, "session", None)
        if session is None and isinstance(result, dict):
            session = result.get("session")
        return session

    def _persist_and_dump_session(self, session: Any) -> dict[str, Any]:
        access_token = getattr(session, "access_token", None)
        refresh_token = getattr(session, "refresh_token", None)
        if access_token and refresh_token:
            try:
                self.client.auth.set_session(access_token, refresh_token)
            except Exception:
                pass
        return {"access_token": access_token, "refresh_token": refresh_token}

    def current_user(self) -> AuthUser | None:
        result = self.client.auth.get_user()
        user = getattr(result, "user", None)
        if user is None and isinstance(result, dict):
            user = result.get("user")
        if not user:
            return None

        uid = getattr(user, "id", None) if not isinstance(user, dict) else user.get("id")
        email = getattr(user, "email", None) if not isinstance(user, dict) else user.get("email")
        if not uid:
            return None
        return AuthUser(id=str(uid), email=email)

    def set_session_tokens(self, access_token: str, refresh_token: str) -> None:
        self.client.auth.set_session(access_token, refresh_token)

    def sign_out(self) -> None:
        self.client.auth.sign_out()



def get_auth_service() -> AuthService:
    cfg = get_app_config()
    return AuthService(get_supabase_client(), app_base_url=cfg.app_base_url)



def get_current_user() -> AuthUser | None:
    return get_auth_service().current_user()



def require_auth() -> AuthUser:
    user = get_current_user()
    if user is None:
        raise RuntimeError("Authentication required")
    return user
