from __future__ import annotations

import streamlit as st

from services import (
    get_app_config,
    get_attempt_repository,
    get_authenticated_user,
    get_storage_service,
    get_track_repository,
    run_attempt_analysis,
    sign_out_user,
    supabase_configured,
)
from ui import configure_page, render_advanced_analysis, render_overview_analysis, render_page_header, render_top_nav


@st.cache_data(show_spinner=False)
def _compute_runtime_bundle(
    attempt_id: str,
    storage_path: str,
    method_name: str,
    params_json: dict,
    processed_at: str,
) -> dict:
    del attempt_id, processed_at
    fit_bytes = get_storage_service().download_fit(storage_path)
    return run_attempt_analysis(fit_bytes=fit_bytes, method_name=method_name, params_json=params_json)


def _sign_out() -> None:
    sign_out_user()
    try:
        st.switch_page("app.py")
    except Exception:
        st.rerun()


def _attempt_title(attempt: dict) -> str:
    return (attempt.get("attempt_name") or attempt.get("source_filename") or "Untitled attempt").strip()


configure_page("Attempt Analysis")

cfg = get_app_config()
if not cfg.multi_attempt_mode:
    st.error("MULTI_ATTEMPT_MODE is disabled.")
    st.stop()
if not supabase_configured(cfg):
    st.error("Supabase is not configured.")
    st.stop()

user = get_authenticated_user()
if user is None:
    st.warning("Please log in from the landing page first.")
    st.stop()

render_top_nav(
    active_page="attempt",
    user_label=user.email or user.id,
    show_signout=True,
    on_signout=_sign_out,
)
render_page_header(
    "Attempt Analysis",
    "Choose a track and processed attempt, then inspect overview and advanced telemetry in one place.",
)

track_repo = get_track_repository()
attempt_repo = get_attempt_repository()
storage = get_storage_service()

tracks = track_repo.list_tracks(user.id)
if not tracks:
    st.info("No tracks yet. Upload an attempt first.")
    st.stop()

track_map = {
    f"{t['name']} • {t['layout_direction']} • {t.get('layout_variant') or 'default'}": t["id"] for t in tracks
}

initial_track = None
if st.session_state.get("selected_track_id"):
    for label, tid in track_map.items():
        if tid == st.session_state["selected_track_id"]:
            initial_track = label
            break

track_label = st.selectbox(
    "Track",
    options=list(track_map.keys()),
    index=list(track_map.keys()).index(initial_track) if initial_track in track_map else 0,
)
track_id = track_map[track_label]
st.session_state["selected_track_id"] = track_id

attempts = attempt_repo.list_attempts(user.id, track_id=track_id)
if not attempts:
    st.info("No attempts for this track yet.")
    st.stop()

show_processed_only = st.checkbox("Show only processed attempts", value=True)
if show_processed_only:
    attempts = [a for a in attempts if a.get("status") == "processed"]

if not attempts:
    st.info("No processed attempts available for this track.")
    st.stop()

attempt_by_id = {a["id"]: a for a in attempts}
attempt_ids = list(attempt_by_id.keys())

initial_attempt = st.session_state.get("selected_attempt_id")
initial_idx = 0
if initial_attempt:
    for i, attempt_id in enumerate(attempt_ids):
        if attempt_id == initial_attempt:
            initial_idx = i
            break

selected_attempt_id = st.selectbox(
    "Attempt",
    options=attempt_ids,
    index=initial_idx,
    format_func=lambda attempt_id: (
        f"{_attempt_title(attempt_by_id[attempt_id])} • "
        f"{attempt_by_id[attempt_id].get('uploaded_at', '')} • "
        f"{attempt_by_id[attempt_id].get('status', '')}"
    ),
)
attempt = attempt_by_id[selected_attempt_id]
st.session_state["selected_attempt_id"] = selected_attempt_id

with st.expander("Manage selected attempt"):
    st.caption("Deleting removes the attempt, derived analytics, and the uploaded FIT file.")
    confirm_delete = st.checkbox(
        "I understand this action is permanent.",
        key=f"delete_attempt_confirm.{attempt['id']}",
    )
    delete_clicked = st.button(
        "Delete attempt",
        use_container_width=True,
        disabled=not confirm_delete,
        key=f"delete_attempt_button.{attempt['id']}",
    )
    if delete_clicked:
        with st.spinner("Deleting attempt..."):
            try:
                deleted = attempt_repo.delete_attempt(attempt["id"])
                if not deleted:
                    st.warning("Attempt not found. It may have already been deleted.")
                    st.rerun()
                try:
                    storage.delete_fit(attempt["storage_path"])
                except Exception as storage_exc:
                    st.warning(f"Attempt deleted, but FIT cleanup failed: {storage_exc}")
                st.session_state.pop("selected_attempt_id", None)
                st.success("Attempt deleted.")
                st.rerun()
            except Exception as exc:
                st.error(f"Failed to delete attempt: {exc}")

if attempt.get("status") == "failed":
    st.error(f"Attempt failed: {attempt.get('processing_error') or 'Unknown error'}")
    st.stop()

with st.spinner("Loading attempt and computing runtime analysis..."):
    bundle = _compute_runtime_bundle(
        attempt_id=attempt["id"],
        storage_path=attempt["storage_path"],
        method_name=attempt["method_name"],
        params_json=attempt.get("params_json") or {},
        processed_at=str(attempt.get("processed_at") or ""),
    )

st.session_state["latest_analysis"] = {
    "method_name": bundle["method_name"],
    "params": bundle["params"],
    "result": bundle["result"],
}

st.caption(
    f"Attempt ID: {attempt['id']} • Name: {_attempt_title(attempt)} • Uploaded: {attempt.get('uploaded_at')} • Source: {attempt.get('source_filename')}"
)

tab_overview, tab_advanced = st.tabs(["Overview", "Advanced"])

with tab_overview:
    render_overview_analysis(
        result=bundle["result"],
        method_name=bundle["method_name"],
        params=bundle["params"],
        key_prefix=f"attempt.{attempt['id']}.overview",
    )

with tab_advanced:
    render_advanced_analysis(bundle["further"], key_prefix=f"attempt.{attempt['id']}.advanced")
