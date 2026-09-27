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
    linear_artifact = joblib.load(artifact_path("linear_regression_model.sav"))
    logistic_artifact = joblib.load(artifact_path("logistic_regression_model.sav"))
    metadata_artifact = joblib.load(artifact_path("model_metadata.sav"))
    return (
        linear_artifact["model"],
        logistic_artifact["model"],
        metadata_artifact
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
    c2.metric("Linear validation R²", f'{meta["linear_validation_metrics"]["R2"]:.3f}')
    c3.metric("Logistic validation ROC-AUC", f'{meta["logistic_validation_metrics"]["ROC_AUC"]:.3f}')
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

    with col2:
        bhk = st.number_input(
            "BHK", min_value=int(ranges["BHK"]["min"]),
            max_value=int(ranges["BHK"]["max"]),
            value=int(ranges["BHK"]["median"]), step=1
        )
        bathrooms = st.number_input(
            "Bathrooms", min_value=int(ranges["Bathrooms"]["min"]),
            max_value=int(ranges["Bathrooms"]["max"]),
            value=int(ranges["Bathrooms"]["median"]), step=1
        )
        super_area = st.number_input(
            "Super area (sq ft)", min_value=float(ranges["Super_Area_SqFt"]["min"]),
            max_value=float(ranges["Super_Area_SqFt"]["max"]),
            value=float(ranges["Super_Area_SqFt"]["median"]), step=10.0
        )
        carpet_area = st.number_input(
            "Carpet area (sq ft)", min_value=float(ranges["Carpet_Area_SqFt"]["min"]),
            max_value=float(ranges["Carpet_Area_SqFt"]["max"]),
            value=float(ranges["Carpet_Area_SqFt"]["median"]), step=10.0
        )

    with col3:
        floor = st.number_input(
            "Floor number", min_value=int(ranges["Floor_Number"]["min"]),
            max_value=int(ranges["Floor_Number"]["max"]),
            value=int(ranges["Floor_Number"]["median"]), step=1
        )
        total_floors = st.number_input(
            "Total floors", min_value=int(ranges["Total_Floors"]["min"]),
            max_value=int(ranges["Total_Floors"]["max"]),
            value=int(ranges["Total_Floors"]["median"]), step=1
        )
        age = st.number_input(
            "Age of property (years)", min_value=int(ranges["Age_of_Property"]["min"]),
            max_value=int(ranges["Age_of_Property"]["max"]),
            value=int(ranges["Age_of_Property"]["median"]), step=1
        )
        distance_metro = st.number_input(
            "Distance to metro (km)", min_value=float(ranges["Distance_to_Metro_km"]["min"]),
            max_value=float(ranges["Distance_to_Metro_km"]["max"]),
            value=float(ranges["Distance_to_Metro_km"]["median"]), step=0.1
        )
        distance_center = st.number_input(
            "Distance to city center (km)", min_value=float(ranges["Distance_to_City_Center_km"]["min"]),
            max_value=float(ranges["Distance_to_City_Center_km"]["max"]),
            value=float(ranges["Distance_to_City_Center_km"]["median"]), step=0.1
        )

    st.markdown("**Amenities**")
    a1, a2, a3 = st.columns(3)
    with a1:
        parking = st.selectbox("Parking available", [0, 1], format_func=lambda x: "Yes" if x else "No")
    with a2:
        lift = st.selectbox("Lift available", [0, 1], index=1, format_func=lambda x: "Yes" if x else "No")
    with a3:
        gated = st.selectbox("Gated community", [0, 1], format_func=lambda x: "Yes" if x else "No")

    st.subheader("2. Lending inputs")
    l1, l2, l3 = st.columns(3)
    with l1:
        loan_amount = st.number_input(
            "Proposed loan amount (₹ lakh)", min_value=0.0, value=70.0, step=1.0
        )
        ltv_policy = st.number_input(
            "Illustrative maximum LTV policy (%)", min_value=1.0, max_value=100.0,
            value=75.0, step=1.0
        )
    with l2:
        monthly_income = st.number_input(
            "Gross monthly income (₹)", min_value=0.0, value=150000.0, step=5000.0
        )
        existing_obligations = st.number_input(
            "Existing monthly obligations (₹)", min_value=0.0, value=25000.0, step=1000.0
        )
    with l3:
        proposed_emi = st.number_input(
            "Proposed EMI (₹)", min_value=0.0, value=50000.0, step=1000.0
        )
        foir_policy = st.number_input(
            "Illustrative maximum FOIR (%)", min_value=1.0, max_value=100.0,
            value=50.0, step=1.0
        )

    submitted = st.form_submit_button("Estimate value & underwriting ratios", use_container_width=True)

if submitted:
    if floor > total_floors:
        st.error("Floor number cannot be greater than total floors.")
        st.stop()

    input_df = pd.DataFrame([{
        "City": city,
        "Locality_Type": locality_type,
        "Property_Type": property_type,
        "BHK": bhk,
        "Bathrooms": bathrooms,
        "Super_Area_SqFt": super_area,
        "Carpet_Area_SqFt": carpet_area,
        "Floor_Number": floor,
        "Total_Floors": total_floors,
        "Age_of_Property": age,
        "Furnishing_Status": furnishing,
        "Parking": parking,
        "Lift_Available": lift,
        "Gated_Community": gated,
        "Distance_to_Metro_km": distance_metro,
        "Distance_to_City_Center_km": distance_center
    }])

    estimated_value = max(0.0, float(linear_model.predict(input_df)[0]))
    high_value_prob = float(logistic_model.predict_proba(input_df)[0, 1])

    ltv = (loan_amount / estimated_value * 100.0) if estimated_value > 0 else None
    total_monthly_burden = existing_obligations + proposed_emi
    foir = (total_monthly_burden / monthly_income * 100.0) if monthly_income > 0 else None

    st.subheader("3. Valuation result")
    m1, m2, m3 = st.columns(3)
    m1.metric("Estimated property value", f"₹{estimated_value:,.2f} lakh")
    m2.metric("Estimated value", f"₹{estimated_value/100:.2f} crore")
    m3.metric(
        "Probability property is ≥ ₹1 crore",
        f"{high_value_prob*100:.1f}%"
    )

    st.subheader("4. Underwriting ratios")
    u1, u2, u3 = st.columns(3)

    if ltv is not None:
        with u1:
            st.metric("LTV", f"{ltv:.1f}%")
            if ltv <= ltv_policy:
                st.success(f"Within illustrative {ltv_policy:.0f}% LTV policy")
            else:
                st.warning(f"Above illustrative {ltv_policy:.0f}% LTV policy")

    with u2:
        if foir is not None:
            st.metric("FOIR", f"{foir:.1f}%")
            if foir <= foir_policy:
                st.success(f"Within illustrative {foir_policy:.0f}% FOIR policy")
            else:
                st.warning(f"Above illustrative {foir_policy:.0f}% FOIR policy")
        else:
            st.metric("FOIR", "N/A")
            st.info("Enter monthly income above zero to calculate FOIR.")

    with u3:
        st.metric("Loan amount", f"₹{loan_amount:,.2f} lakh")
        st.metric("Valuation buffer", f"₹{estimated_value-loan_amount:,.2f} lakh")

    st.info(
        "This is an academic valuation and ratio-calculation tool. LTV/FOIR policy thresholds "
        "shown in the app are user-entered illustrative values and are not a bank's official credit policy."
    )
