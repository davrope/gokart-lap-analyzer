from __future__ import annotations

import streamlit as st

from services import (
    get_app_config,
    get_attempt_repository,
    get_authenticated_user,
    get_storage_service,
    get_track_repository,
    run_attempt_analysis,
    supabase_configured,
)
from ui import render_advanced_analysis, render_overview_analysis


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


st.set_page_config(page_title="Attempt Analysis", layout="wide")
st.title("Attempt Analysis")

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

track_repo = get_track_repository()
attempt_repo = get_attempt_repository()

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

attempt_labels = {}
for a in attempts:
    label = f"{a.get('uploaded_at', '')} • {a.get('source_filename', '')} • {a.get('status', '')}"
    attempt_labels[label] = a

initial_attempt = st.session_state.get("selected_attempt_id")
initial_idx = 0
if initial_attempt:
    for i, a in enumerate(attempts):
        if a["id"] == initial_attempt:
            initial_idx = i
            break

selected_label = st.selectbox("Attempt", options=list(attempt_labels.keys()), index=initial_idx)
attempt = attempt_labels[selected_label]
st.session_state["selected_attempt_id"] = attempt["id"]

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
    f"Attempt ID: {attempt['id']} • Uploaded: {attempt.get('uploaded_at')} • Source: {attempt.get('source_filename')}"
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
