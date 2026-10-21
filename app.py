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

st.title("🏠 Housing Price Prediction & Bank Underwriting Tool")
st.caption(
    "Academic predictive tool: estimates property value from property characteristics, "
    "then calculates LTV and optional FOIR using user-entered lending inputs."
)

with st.expander("Model information", expanded=False):
    c1, c2, c3 = st.columns(3)
    c1.metric("Training rows", f'{meta["training_rows"]:,}')
    c2.metric("Linear validation R²", f'{meta["linear_validation_metrics"]["r2"]:.3f}')
    c3.metric("Logistic validation ROC-AUC", f'{meta["logistic_validation_metrics"]["roc_auc"]:.3f}')
    st.write(
        f'Logistic target: property value ≥ ₹{meta["high_value_threshold_lakhs"]:.0f} lakh '
        '(₹1 crore). This classification target is derived from the available property-value '
        'data because the supplied datasets contain no borrower or loan-sanction outcome.'
    )

st.subheader("1. Property details")

with st.form("valuation_form"):
    col1, col2, col3 = st.columns(3)

    with col1:
        city = st.selectbox("City", options["City"])
        locality_type = st.selectbox("Locality type", options["Locality_Type"])
        property_type = st.selectbox("Property type", options["Property_Type"])
        furnishing = st.selectbox("Furnishing status", options["Furnishing_Status"])
