from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

from analysis import generate_track_history_summary
from services import (
    build_track_history,
    get_app_config,
    get_attempt_repository,
    get_authenticated_user,
    get_track_repository,
    sign_out_user,
    supabase_configured,
)
from ui import configure_page, render_page_header, render_top_nav


def _sign_out() -> None:
    sign_out_user()
    try:
        st.switch_page("app.py")
    except Exception:
        st.rerun()


configure_page("Track History")

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
    active_page="history",
    user_label=user.email or user.id,
    show_signout=True,
    on_signout=_sign_out,
)
render_page_header(
    "Track History",
    "Compare attempts over time, monitor consistency, and identify the curves with highest time-loss potential.",
)

track_repo = get_track_repository()
attempt_repo = get_attempt_repository()

tracks = track_repo.list_tracks(user.id)
if not tracks:
    st.info("No tracks available yet.")
    st.stop()

track_map = {
    f"{t['name']} • {t['layout_direction']} • {t.get('layout_variant') or 'default'}": t["id"] for t in tracks
}
track_label = st.selectbox("Track", options=list(track_map.keys()))
track_id = track_map[track_label]

attempts_raw = [a for a in attempt_repo.list_attempts(user.id, track_id=track_id) if a.get("status") == "processed"]
if not attempts_raw:
    st.info("No processed attempts for this track yet.")
    st.stop()

attempts_df = pd.DataFrame(attempts_raw)
attempt_ids = attempts_df["id"].tolist()

laps_df = attempt_repo.list_attempt_laps(attempt_ids)
curves_df = attempt_repo.list_attempt_curves(attempt_ids)

history = build_track_history(attempts_df, laps_df, curves_df)

k1, k2, k3 = st.columns(3)
k1.metric("Attempts", f"{len(history.attempts)}")

best_vals = history.attempts["best_lap_s"].dropna().astype(float) if "best_lap_s" in history.attempts.columns else pd.Series(dtype=float)
k2.metric("Best lap", f"{best_vals.min():.2f}s" if not best_vals.empty else "—")

latest_best = np.nan
if not history.attempts.empty and "best_lap_s" in history.attempts.columns:
    latest_best = float(history.attempts.sort_values("uploaded_at").iloc[-1]["best_lap_s"])
k3.metric("Latest best lap", f"{latest_best:.2f}s" if np.isfinite(latest_best) else "—")

st.subheader("Attempt timeline")
if not history.trend.empty:
    fig_t = px.line(
        history.trend,
        x="uploaded_at",
        y=["best_lap_s", "rolling_median_best_lap_s"],
        markers=True,
        labels={"value": "Lap time (s)", "uploaded_at": "Attempt date", "variable": "Series"},
    )
    st.plotly_chart(fig_t, use_container_width=True)
else:
    st.info("No trend data available.")

st.subheader("Consistency trend")
if not history.consistency.empty:
    fig_c = px.line(history.consistency, x="uploaded_at", y="lap_time_cv", markers=True)
    fig_c.update_layout(template="plotly_white", yaxis_title="Lap time CV", xaxis_title="Attempt date")
    st.plotly_chart(fig_c, use_container_width=True)
else:
    st.info("No consistency data available.")

st.subheader("Curve improvement heatmap")
if not history.heatmap.empty:
    hdf = history.heatmap.copy()
    hdf = hdf.sort_index()
    fig_h = px.imshow(
        hdf,
        labels=dict(x="Curve", y="Attempt", color="Mean time loss (s)"),
        aspect="auto",
        color_continuous_scale="RdYlGn_r",
    )
    fig_h.update_layout(template="plotly_white")
    st.plotly_chart(fig_h, use_container_width=True)
else:
    st.info("No curve heatmap data available.")

st.subheader("Latest vs personal best")
lvb = history.latest_vs_best
if lvb:
    a, b, c, d = st.columns(4)
    a.metric("Latest best lap", f"{lvb.get('latest_best_lap_s', np.nan):.2f}s" if np.isfinite(lvb.get("latest_best_lap_s", np.nan)) else "—")
    b.metric("Personal best lap", f"{lvb.get('best_best_lap_s', np.nan):.2f}s" if np.isfinite(lvb.get("best_best_lap_s", np.nan)) else "—")
    c.metric("Delta best lap", f"{lvb.get('delta_best_lap_s', np.nan):+.2f}s" if np.isfinite(lvb.get("delta_best_lap_s", np.nan)) else "—")
    d.metric("Delta avg lap", f"{lvb.get('delta_avg_lap_s', np.nan):+.2f}s" if np.isfinite(lvb.get("delta_avg_lap_s", np.nan)) else "—")

    if not curves_df.empty and "attempt_id" in curves_df.columns:
        latest_id = lvb.get("latest_attempt_id")
        best_id = lvb.get("best_attempt_id")
        latest_curve = curves_df[curves_df["attempt_id"] == latest_id].groupby("curve_id", as_index=False)["time_loss_vs_best_s"].mean()
        best_curve = curves_df[curves_df["attempt_id"] == best_id].groupby("curve_id", as_index=False)["time_loss_vs_best_s"].mean()
        cmp_df = latest_curve.merge(best_curve, on="curve_id", how="outer", suffixes=("_latest", "_best")).fillna(0.0)
        cmp_df["delta_latest_minus_best"] = cmp_df["time_loss_vs_best_s_latest"] - cmp_df["time_loss_vs_best_s_best"]

        fig_cmp = px.bar(cmp_df, x="curve_id", y="delta_latest_minus_best", labels={"delta_latest_minus_best": "Delta time loss (s)", "curve_id": "Curve"})
        fig_cmp.update_layout(template="plotly_white", title="Per-curve delta (latest vs personal best attempt)")
        fig_cmp.add_hline(y=0.0, line_color="#374151")
        st.plotly_chart(fig_cmp, use_container_width=True)
else:
    st.info("Insufficient data for latest vs personal best comparison.")

st.subheader("Top opportunities")
if not history.opportunities.empty:
    st.dataframe(history.opportunities, use_container_width=True)
else:
    st.info("No opportunity ranking available yet.")

summary = generate_track_history_summary(
    {
        "attempts_count": len(history.attempts),
        "curves_count": int(curves_df["curve_id"].nunique()) if not curves_df.empty and "curve_id" in curves_df.columns else 0,
        "latest_vs_best_delta_s": history.latest_vs_best.get("delta_best_lap_s") if history.latest_vs_best else None,
    }
)

st.subheader("AI Coaching Summary (Architecture Stub)")
st.write(f"**{summary.title}**")
for b in summary.bullets:
    st.write(f"- {b}")
st.write("Opportunities")
for o in summary.opportunities:
    st.write(f"- {o}")
st.caption(f"Summary engine: {summary.engine}")
