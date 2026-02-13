from __future__ import annotations

from typing import Any

import streamlit as st

from .auth_service import AuthUser, get_auth_service



def bootstrap_auth_session_from_query() -> str | None:
    svc = get_auth_service()
    query = dict(st.query_params)

    # Values can be list-like in some Streamlit versions
    qp: dict[str, Any] = {}
    for k, v in query.items():
        if isinstance(v, (list, tuple)) and v:
            qp[k] = v[0]
        else:
            qp[k] = v

    if "token_hash" not in qp or "type" not in qp:
        return None

    try:
        session = svc.consume_magic_link(qp)
        if session:
            st.session_state["auth_session"] = session
            # clear token params after successful consume
            for key in ("token_hash", "type", "next"):
                if key in st.query_params:
                    del st.query_params[key]
            return "Logged in successfully."
    except Exception as exc:
        return f"Login verification failed: {exc}"
    return None



def restore_auth_session() -> None:
    session = st.session_state.get("auth_session") or {}
    access = session.get("access_token")
    refresh = session.get("refresh_token")
    if access and refresh:
        try:
            get_auth_service().set_session_tokens(access, refresh)
        except Exception:
            st.session_state.pop("auth_session", None)



def get_authenticated_user() -> AuthUser | None:
    restore_auth_session()
    try:
        return get_auth_service().current_user()
    except Exception:
        return None



def sign_out_user() -> None:
    try:
        get_auth_service().sign_out()
    finally:
        st.session_state.pop("auth_session", None)
