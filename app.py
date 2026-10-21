from pathlib import Path
import math
import re

import joblib
import numpy as np
import pandas as pd
import streamlit as st


st.set_page_config(
    page_title="Housing Price Prediction & Bank Underwriting Tool",
    page_icon="🏠",
    layout="wide",
)

BASE_DIR = Path(__file__).resolve().parent


def normalise_key(value):
    """Make dictionary key lookup insensitive to case, spaces and punctuation."""
    return re.sub(r"[^a-z0-9]", "", str(value).lower().replace("²", "2"))


def get_key(mapping, aliases, default=None):
    """Return the first matching key from a dictionary, ignoring key formatting."""
    if not isinstance(mapping, dict):
        return default
    if isinstance(aliases, str):
        aliases = [aliases]
    wanted = {normalise_key(alias) for alias in aliases}
    for key, value in mapping.items():
        if normalise_key(key) in wanted:
            return value
    return default


def artifact_path(filename, required=True):
    """Find an artifact in the same directory as app.py or common subfolders."""
    candidates = [
        BASE_DIR / filename,
        BASE_DIR / "models" / filename,
        BASE_DIR / "artifacts" / filename,
    ]
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    if required:
        locations = ", ".join(str(path) for path in candidates)
        raise FileNotFoundError(f"Missing required file '{filename}'. Checked: {locations}")
    return None


def load_artifact(filename, required=True):
    path = artifact_path(filename, required=required)
    if path is None:
        return None
    return joblib.load(path)


def unwrap_model(package, filename):
    """Accept the supplied {model: estimator, metadata: ...} package or an estimator."""
    if isinstance(package, dict):
        model = get_key(package, "model")
        if model is not None and hasattr(model, "predict"):
            return model
    if hasattr(package, "predict"):
        return package
    raise RuntimeError(f"'{filename}' does not contain a usable prediction model.")


def get_metadata_bundle(linear_package, logistic_package):
    """Prefer metadata embedded in the supplied model files, then try the separate file."""
    packages = [linear_package, logistic_package]
    for package in packages:
        if not isinstance(package, dict):
            continue
        bundle = get_key(package, "metadata")
        if not isinstance(bundle, dict):
            continue
        numeric_ranges = get_key(bundle, "numeric_ranges")
        categorical_options = get_key(bundle, "categorical_options")
        metadata = get_key(bundle, "metadata")
        if isinstance(numeric_ranges, dict) and isinstance(categorical_options, dict) and isinstance(metadata, dict):
            return {
                "numeric_ranges": numeric_ranges,
                "categorical_options": categorical_options,
                "metadata": metadata,
            }

    separate = load_artifact("model_metadata.sav", required=False)
    if isinstance(separate, dict):
        numeric_ranges = get_key(separate, "numeric_ranges")
        categorical_options = get_key(separate, "categorical_options")
        metadata = get_key(separate, "metadata")
        if isinstance(numeric_ranges, dict) and isinstance(categorical_options, dict) and isinstance(metadata, dict):
