import json
import math
import re
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st


# ---------------------------------------------------------------------------
# Page setup and robust helpers
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Housing Valuation & Underwriting Tool",
    page_icon="🏠",
    layout="wide",
)

BASE_DIR = Path(__file__).resolve().parent


def _normalise_key(value):
    """Normalise key names so variants like ROC_AUC, roc-auc and ROC AUC match."""
    return re.sub(r"[^a-z0-9]", "", str(value).lower().replace("²", "2"))


def _get_ci(mapping, wanted_key, default=None):
    """Case/punctuation-insensitive dictionary lookup."""
    if not isinstance(mapping, dict):
        return default
    wanted = _normalise_key(wanted_key)
    for key, value in mapping.items():
        if _normalise_key(key) == wanted:
            return value
    return default


def _find_section(obj, section_aliases):
    """Find a nested dictionary section without assuming a fixed nesting level."""
    aliases = {_normalise_key(x) for x in section_aliases}
    if isinstance(obj, dict):
        for key, value in obj.items():
            if _normalise_key(key) in aliases and isinstance(value, dict):
                return value
        for value in obj.values():
            found = _find_section(value, section_aliases)
            if found is not None:
                return found
    elif isinstance(obj, (list, tuple)):
        for value in obj:
            found = _find_section(value, section_aliases)
            if found is not None:
                return found
    return None


def _find_scalar(obj, aliases):
    """Recursively find a scalar field matching one of the supplied aliases."""
    wanted = {_normalise_key(x) for x in aliases}
    if isinstance(obj, dict):
        for key, value in obj.items():
            if _normalise_key(key) in wanted and not isinstance(value, (dict, list, tuple)):
                try:
                    number = float(value)
                    if math.isfinite(number):
                        return number
                except (TypeError, ValueError):
                    pass
        for value in obj.values():
            found = _find_scalar(value, aliases)
            if found is not None:
                return found
    elif isinstance(obj, (list, tuple)):
        for value in obj:
            found = _find_scalar(value, aliases)
            if found is not None:
                return found
    return None


def _artifact_path(filename, required=True):
    """Locate files beside app.py or inside common artifact folders."""
    candidates = [
        BASE_DIR / filename,
        BASE_DIR / "models" / filename,
        BASE_DIR / "artifacts" / filename,
    ]
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    if required:
        checked = "\n".join(str(p) for p in candidates)
        raise FileNotFoundError(
            f"Required file '{filename}' was not found. Locations checked:\n{checked}"
