import streamlit as st
import pandas as pd
import joblib
from pathlib import Path

st.set_page_config(
    page_title="Housing Valuation & Underwriting Tool",
    page_icon="🏠",
    layout="wide"
)

BASE_DIR = Path(__file__).resolve().parent

def artifact_path(filename):
    """Find a model file in the app folder or common subfolders."""
    candidates = [
        BASE_DIR / filename,
        BASE_DIR / "models" / filename,
        BASE_DIR / "artifacts" / filename,
    ]
    for path in candidates:
        if path.is_file():
            return path
    raise FileNotFoundError(
        f"Could not find {filename}. Expected it in the repository root, "
        f"models/, or artifacts/. App directory: {BASE_DIR}"
    )

@st.cache_resource
def load_artifacts():
    # Load the trained models. Both model files also contain the UI metadata,
    # so the app can start even if model_metadata.sav was not committed.
    linear_artifact = joblib.load(artifact_path("linear_regression_model.sav"))
    logistic_artifact = joblib.load(artifact_path("logistic_regression_model.sav"))

    metadata_artifact = linear_artifact.get("metadata")
    if not isinstance(metadata_artifact, dict):
        metadata_artifact = logistic_artifact.get("metadata")

    # Backward-compatible fallback for older model files whose metadata is separate.
    if not isinstance(metadata_artifact, dict):
        try:
            metadata_artifact = joblib.load(artifact_path("model_metadata.sav"))
        except FileNotFoundError:
            metadata_artifact = None

    if not isinstance(metadata_artifact, dict):
        raise RuntimeError(
            "Could not load app metadata. Ensure the trained model files are the supplied "
            "linear_regression_model.sav and logistic_regression_model.sav files."
        )

    required_keys = {"metadata", "numeric_ranges", "categorical_options"}
    if not required_keys.issubset(metadata_artifact.keys()):
        raise RuntimeError(
            "The metadata inside the model file is incomplete. Re-upload the supplied model files."
        )

    return (
        linear_artifact["model"],
        logistic_artifact["model"],
        metadata_artifact,
    )

linear_model, logistic_model, metadata_bundle = load_artifacts()
meta = metadata_bundle["metadata"]
ranges = metadata_bundle["numeric_ranges"]
options = metadata_bundle["categorical_options"]


def _normalise_key(value):
    """Normalise metadata names so R2/r2 and ROC_AUC/roc_auc both work."""
    value = str(value).replace("²", "2")
    return "".join(char.lower() for char in value if char.isalnum())


def _read_nested_metric(metadata, section_names, metric_names):
    """Read a metric while tolerating capitalization and punctuation differences."""
    section_names = {_normalise_key(name) for name in section_names}
    metric_names = {_normalise_key(name) for name in metric_names}

    section = {}
    for key, value in metadata.items():
        if _normalise_key(key) in section_names and isinstance(value, dict):
            section = value
            break

    for key, value in section.items():
        if _normalise_key(key) in metric_names:
            try:
                return float(value)
            except (TypeError, ValueError):
                return None
    return None

