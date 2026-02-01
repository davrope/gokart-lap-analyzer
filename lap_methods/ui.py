from __future__ import annotations

from dataclasses import fields, is_dataclass
from typing import Any, get_args, get_origin, Optional

import streamlit as st


def _unwrap_optional(t):
    """If type is Optional[T], return T; else return t."""
    origin = get_origin(t)
    if origin is None:
        return t
    if origin is list or origin is dict or origin is tuple:
        return t
    if origin is Optional or origin is type(Optional[int]):  # defensive
        args = [a for a in get_args(t) if a is not type(None)]
        return args[0] if args else t
    if origin is type(None):
        return t
    # handle Union[T, None] (Optional)
    if origin is getattr(__import__("typing"), "Union"):
        args = [a for a in get_args(t) if a is not type(None)]
        return args[0] if args else t
    return t


def render_params_form(params: Any, *, title: str = "Parameters", key_prefix: str = "p") -> Any:
    """
    Auto-render Streamlit widgets for a dataclass instance, returning an updated instance.
    Supports bool/int/float/str plus metadata:
      - min, max, step
      - options (for selectbox)
      - help
      - widget: "slider"|"number_input"|"text_input"|"selectbox"|"checkbox"
      - format: e.g. "%.2f"
    """
    if not is_dataclass(params):
        raise TypeError("render_params_form expects a dataclass instance")

    st.sidebar.subheader(title)

    for f in fields(params):
        name = f.name
        t = _unwrap_optional(f.type)
        value = getattr(params, name)
        meta = dict(f.metadata) if f.metadata else {}

        label = meta.get("label", name.replace("_", " ").title())
        help_text = meta.get("help", None)
        widget = meta.get("widget", None)
        options = meta.get("options", None)

        # Keys must be stable across reruns
        widget_key = f"{key_prefix}.{name}"

        # --- bool ---
        if t is bool or isinstance(value, bool) or widget == "checkbox":
            new_val = st.sidebar.checkbox(label, value=bool(value), help=help_text, key=widget_key)

        # --- selectbox by options ---
        elif options is not None or widget == "selectbox":
            if options is None:
                options = []
            # ensure current value is in options if possible
            if value is not None and value not in options:
                options = [value] + list(options)
            idx = options.index(value) if value in options else 0
            new_val = st.sidebar.selectbox(label, options=options, index=idx, help=help_text, key=widget_key)

        # --- int ---
        elif t is int or isinstance(value, int):
            vmin = meta.get("min", 0)
            vmax = meta.get("max", max(vmin + 10, vmin))
            step = meta.get("step", 1)

            # If metadata provides bounds, prefer slider; else number_input
            if widget == "slider" or ("min" in meta and "max" in meta):
                new_val = st.sidebar.slider(label, int(vmin), int(vmax), int(value), int(step), help=help_text, key=widget_key)
            else:
                new_val = st.sidebar.number_input(label, value=int(value), step=int(step), help=help_text, key=widget_key)

        # --- float ---
        elif t is float or isinstance(value, float):
            vmin = meta.get("min", 0.0)
            vmax = meta.get("max", float(vmin + 10.0))
            step = meta.get("step", 0.5)
            fmt = meta.get("format", None)

            if widget == "slider" or ("min" in meta and "max" in meta):
                # Streamlit slider supports floats
                new_val = st.sidebar.slider(label, float(vmin), float(vmax), float(value), float(step), help=help_text, key=widget_key)
            else:
                new_val = st.sidebar.number_input(label, value=float(value), step=float(step), format=fmt, help=help_text, key=widget_key)

        # --- str fallback ---
        else:
            new_val = st.sidebar.text_input(label, value="" if value is None else str(value), help=help_text, key=widget_key)

        setattr(params, name, new_val)

    return params
