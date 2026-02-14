from __future__ import annotations

import uuid

import streamlit as st

from lap_methods import default_params, get_methods, params_to_dict
from lap_methods.ui import render_params_form
from services import (
    build_attempt_curve_rows,
    build_attempt_lap_rows,
    build_attempt_summary_payload,
    get_app_config,
    get_attempt_repository,
    get_authenticated_user,
    get_storage_service,
    get_track_repository,
    run_attempt_analysis,
    sign_out_user,
    supabase_configured,
)
from ui import configure_page, render_page_header, render_top_nav


ATTEMPT_PAGE = "pages/2_Attempt_Analysis.py"


def _go_attempt_page() -> None:
    try:
        st.switch_page(ATTEMPT_PAGE)
    except Exception:
        st.info("Use the top navigation buttons to open Attempt Analysis.")


def _sign_out() -> None:
    sign_out_user()
    try:
        st.switch_page("app.py")
    except Exception:
        st.rerun()


configure_page("Upload Attempt")

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
    active_page="upload",
    user_label=user.email or user.id,
    show_signout=True,
    on_signout=_sign_out,
)
render_page_header(
    "Upload Attempt",
    "Select or create a track, upload a FIT file, and persist a fully processed attempt.",
)

track_repo = get_track_repository()
attempt_repo = get_attempt_repository()
storage = get_storage_service()

tracks = track_repo.list_tracks(user.id)

st.subheader("Track")
track_mode = st.radio(
    "Track selection",
    ["Use existing track", "Create new track"],
    horizontal=True,
)

selected_track_id: str | None = None

if track_mode == "Use existing track":
    if not tracks:
        st.info("No tracks found yet. Create one below.")
    else:
        options = {f"{t['name']} • {t['layout_direction']} • {t.get('layout_variant') or 'default'}": t for t in tracks}
        label = st.selectbox("Choose track", list(options.keys()))
        selected_track_id = options[label]["id"]
else:
    with st.form("new_track_form"):
        t_name = st.text_input("Track name")
        direction_options = {
            "Unknown": "unknown",
            "Clockwise": "cw",
            "Counter - Clockwise": "ccw",
        }
        direction_label = st.selectbox("Layout direction", list(direction_options.keys()))
        t_direction = direction_options[direction_label]
        t_variant = st.text_input("Layout variant (optional)")
        t_location = st.text_input("Location (optional)")
        create_track = st.form_submit_button("Create track")

    if create_track:
        try:
            created = track_repo.create_track(
                user_id=user.id,
                name=t_name,
                layout_direction=t_direction,
                layout_variant=t_variant,
                location=t_location,
            )
            st.success(f"Track created: {created['name']}")
            st.session_state["selected_track_id"] = created["id"]
            st.rerun()
        except Exception as exc:
            st.error(f"Failed to create track: {exc}")

    selected_track_id = st.session_state.get("selected_track_id")

if selected_track_id is None and tracks:
    selected_track_id = tracks[0]["id"]

st.subheader("Method")
methods = get_methods()
method_name = st.selectbox("Lap detection method", list(methods.keys()), key="upload.method")
params = default_params(method_name)
params = render_params_form(params, title="Method parameters", key_prefix=f"upload.{method_name}")
params_json = params_to_dict(params)

st.subheader("FIT file")
uploaded = st.file_uploader("Upload Garmin .FIT", type=["fit"], key="upload.fit")

if st.button("Process and save attempt", type="primary", use_container_width=True):
    if selected_track_id is None:
        st.error("Please select or create a track first.")
        st.stop()
    if uploaded is None:
        st.error("Please upload a FIT file first.")
        st.stop()

    fit_bytes = uploaded.getvalue()
    filename = uploaded.name
    attempt_id = str(uuid.uuid4())

    with st.spinner("Uploading and processing attempt..."):
        try:
            storage_path = storage.upload_fit(user.id, attempt_id, fit_bytes, filename)

            attempt_payload = {
                "id": attempt_id,
                "user_id": user.id,
                "track_id": selected_track_id,
                "source_filename": filename,
                "storage_bucket": "fit-files",
                "storage_path": storage_path,
                "status": "uploaded",
                "method_name": method_name,
                "params_json": params_json,
            }
            attempt_repo.create_attempt(attempt_payload)

            analysis_bundle = run_attempt_analysis(
                fit_bytes=fit_bytes,
                method_name=method_name,
                params_json=params_json,
            )

            further = analysis_bundle["further"]
            summary_patch = build_attempt_summary_payload(analysis_bundle)
            attempt_repo.update_attempt_processed(attempt_id, summary_patch)

            lap_rows = build_attempt_lap_rows(attempt_id, further.lap_overview)
            curve_rows = build_attempt_curve_rows(attempt_id, further.curve_overview)
            attempt_repo.replace_attempt_laps(attempt_id, lap_rows)
            attempt_repo.replace_attempt_curves(attempt_id, curve_rows)

            st.session_state["latest_analysis"] = {
                "method_name": analysis_bundle["method_name"],
                "params": analysis_bundle["params"],
                "result": analysis_bundle["result"],
            }
            st.session_state["selected_attempt_id"] = attempt_id
            st.success("Attempt saved and processed.")

        except Exception as exc:
            try:
                attempt_repo.mark_failed(attempt_id, str(exc))
            except Exception:
                pass
            st.error(f"Attempt processing failed: {exc}")

if st.button("Open Attempt Analysis", use_container_width=True):
    _go_attempt_page()
