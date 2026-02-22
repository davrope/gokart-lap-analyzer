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
    params_json = attempt.get("params_json")
    fallback_name = params_json.get("__attempt_name") if isinstance(params_json, dict) else None
    return (attempt.get("attempt_name") or fallback_name or attempt.get("source_filename") or "Untitled attempt").strip()


def _set_manage_flash(kind: str, message: str) -> None:
    st.session_state["attempt.manage.flash"] = {"kind": kind, "message": message}


def _show_manage_flash() -> None:
    flash = st.session_state.pop("attempt.manage.flash", None)
    if not flash:
        return
    kind = str(flash.get("kind") or "").strip().lower()
    message = str(flash.get("message") or "").strip()
    if not message:
        return
    if kind == "success":
        st.success(message)
    elif kind == "warning":
        st.warning(message)
    else:
        st.error(message)


def _delete_attempt_with_storage(attempt_repo, storage, attempt: dict) -> tuple[bool, str | None]:
    deleted = attempt_repo.delete_attempt(attempt["id"])
    if not deleted:
        return False, None
    try:
        storage.delete_fit(attempt["storage_path"])
    except Exception as exc:
        return True, f"Attempt deleted, but FIT cleanup failed for '{_attempt_title(attempt)}': {exc}"
    return True, None


def _render_manage_attempts(attempts: list[dict], track_name_by_id: dict[str, str], attempt_repo, storage) -> None:
    if not attempts:
        st.info("No attempts available to manage.")
        return

    attempt_by_id = {a["id"]: a for a in attempts}

    h0, h1, h2, h3, h4, h5 = st.columns([1, 4, 3, 3, 1, 1])
    h0.caption("Select")
    h1.caption("Attempt")
    h2.caption("Uploaded")
    h3.caption("Track")
    h4.caption("Edit")
    h5.caption("Delete")

    for attempt in attempts:
        attempt_id = attempt["id"]
        select_key = f"attempt.manage.select.{attempt_id}"
        edit_key = f"attempt.manage.edit.{attempt_id}"
        delete_request_key = f"attempt.manage.delete.request.{attempt_id}"
        delete_confirm_key = f"attempt.manage.delete.confirm.{attempt_id}"
        delete_cancel_key = f"attempt.manage.delete.cancel.{attempt_id}"

        c0, c1, c2, c3, c4, c5 = st.columns([1, 4, 3, 3, 1, 1], vertical_alignment="center")
        c0.checkbox("Select", key=select_key, label_visibility="collapsed")
        c1.write(_attempt_title(attempt))
        c2.write(str(attempt.get("uploaded_at") or "—"))
        c3.write(track_name_by_id.get(str(attempt.get("track_id") or ""), "Unknown track"))

        edit_clicked = c4.button("", key=edit_key, help="Edit attempt name", icon=":material/edit:")
        delete_clicked = c5.button(
            "",
            key=delete_request_key,
            type="secondary",
            help="Delete attempt",
            icon=":material/delete:",
        )

        if edit_clicked:
            st.session_state["attempt.manage.editing_id"] = attempt_id
            st.rerun()
        if delete_clicked:
            st.session_state["attempt.manage.delete_confirm_id"] = attempt_id
            st.rerun()

        if st.session_state.get("attempt.manage.editing_id") == attempt_id:
            with st.form(f"attempt.manage.rename.form.{attempt_id}"):
                new_name = st.text_input("Attempt name", value=_attempt_title(attempt), max_chars=120)
                save_col, cancel_col = st.columns(2)
                save_clicked = save_col.form_submit_button("Save")
                cancel_clicked = cancel_col.form_submit_button("Cancel")

            if save_clicked:
                try:
                    attempt_repo.update_attempt_name(attempt_id, new_name)
                    st.session_state.pop("attempt.manage.editing_id", None)
                    _set_manage_flash("success", "Attempt name updated.")
                    st.rerun()
                except Exception as exc:
                    st.session_state.pop("attempt.manage.editing_id", None)
                    _set_manage_flash("error", f"Failed to update attempt name: {exc}")
                    st.rerun()
            if cancel_clicked:
                st.session_state.pop("attempt.manage.editing_id", None)
                st.rerun()

        if st.session_state.get("attempt.manage.delete_confirm_id") == attempt_id:
            st.warning(f"Delete '{_attempt_title(attempt)}'? This cannot be undone.")
            confirm_col, cancel_col = st.columns(2)
            if confirm_col.button("Confirm delete", key=delete_confirm_key, type="primary"):
                try:
                    deleted, warning = _delete_attempt_with_storage(attempt_repo, storage, attempt)
                    st.session_state.pop("attempt.manage.delete_confirm_id", None)
                    st.session_state.pop(select_key, None)
                    if st.session_state.get("selected_attempt_id") == attempt_id:
                        st.session_state.pop("selected_attempt_id", None)
                    if not deleted:
                        _set_manage_flash("warning", "Attempt not found. It may have already been deleted.")
                    elif warning:
                        _set_manage_flash("warning", warning)
                    else:
                        _set_manage_flash("success", "Attempt deleted.")
                    st.rerun()
                except Exception as exc:
                    st.session_state.pop("attempt.manage.delete_confirm_id", None)
                    _set_manage_flash("error", f"Failed to delete attempt: {exc}")
                    st.rerun()
            if cancel_col.button("Cancel", key=delete_cancel_key):
                st.session_state.pop("attempt.manage.delete_confirm_id", None)
                st.rerun()

    selected_ids = [attempt_id for attempt_id in attempt_by_id if st.session_state.get(f"attempt.manage.select.{attempt_id}")]
    st.caption(f"{len(selected_ids)} selected for bulk delete.")

    bulk_confirm = st.checkbox(
        "I understand selected attempts will be permanently deleted.",
        key="attempt.manage.bulk_confirm",
        disabled=not selected_ids,
    )
    if st.button(
        f"Delete selected ({len(selected_ids)})",
        key="attempt.manage.bulk_delete",
        type="primary",
        disabled=not selected_ids or not bulk_confirm,
    ):
        deleted_count = 0
        missing_count = 0
        cleanup_warning_count = 0

        for attempt_id in selected_ids:
            attempt = attempt_by_id.get(attempt_id)
            if not attempt:
                missing_count += 1
                continue
            deleted, warning = _delete_attempt_with_storage(attempt_repo, storage, attempt)
            if not deleted:
                missing_count += 1
                continue
            deleted_count += 1
            if warning:
                cleanup_warning_count += 1
            st.session_state.pop(f"attempt.manage.select.{attempt_id}", None)
            if st.session_state.get("selected_attempt_id") == attempt_id:
                st.session_state.pop("selected_attempt_id", None)

        st.session_state.pop("attempt.manage.bulk_confirm", None)
        st.session_state.pop("attempt.manage.delete_confirm_id", None)
        st.session_state.pop("attempt.manage.editing_id", None)

        if deleted_count == 0 and missing_count > 0:
            _set_manage_flash("warning", "Selected attempts were not found.")
        else:
            message = f"Deleted {deleted_count} attempt(s)."
            if missing_count:
                message += f" {missing_count} already missing."
            if cleanup_warning_count:
                message += f" FIT cleanup failed for {cleanup_warning_count} attempt(s)."
            _set_manage_flash("warning" if missing_count or cleanup_warning_count else "success", message)
        st.rerun()


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
_show_manage_flash()

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

all_attempts = attempt_repo.list_attempts(user.id)
track_name_by_id = {t["id"]: str(t.get("name") or "Unknown track") for t in tracks}

manage_open = bool(st.session_state.get("attempt.manage.open", False))
manage_button_label = "Hide Manage Attempts" if manage_open else "Manage Attempts"
if st.button(manage_button_label, key="attempt.manage.toggle"):
    st.session_state["attempt.manage.open"] = not manage_open
    if manage_open:
        st.session_state.pop("attempt.manage.editing_id", None)
        st.session_state.pop("attempt.manage.delete_confirm_id", None)
        st.session_state.pop("attempt.manage.bulk_confirm", None)
    st.rerun()

if st.session_state.get("attempt.manage.open", False):
    st.subheader("Manage Attempts")
    _render_manage_attempts(all_attempts, track_name_by_id, attempt_repo, storage)

attempts = [a for a in all_attempts if a.get("track_id") == track_id]
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
