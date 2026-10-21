    st.caption("Both saved models passed startup prediction checks.")

st.subheader("1. Property details")

with st.form("valuation_form"):
    col1, col2, col3 = st.columns(3)

    with col1:
        city_options = get_key(categorical_options, "City", ["Ahmedabad"])
        locality_options = get_key(categorical_options, "Locality_Type", ["Affordable"])
        property_options = get_key(categorical_options, "Property_Type", ["Apartment"])
        furnishing_options = get_key(categorical_options, "Furnishing_Status", ["Semi-Furnished"])
        city = st.selectbox("City", city_options)
        locality_type = st.selectbox("Locality type", locality_options)
        property_type = st.selectbox("Property type", property_options)
        furnishing = st.selectbox("Furnishing status", furnishing_options)

    with col2:
        bhk = numeric_input("BHK", "BHK", integer=True)
        bathrooms = numeric_input("Bathrooms", "Bathrooms", integer=True)
        super_area = numeric_input("Super area (sq ft)", "Super_Area_SqFt", step=10.0)
        carpet_area = numeric_input("Carpet area (sq ft)", "Carpet_Area_SqFt", step=10.0)

    with col3:
        floor_number = numeric_input("Floor number", "Floor_Number", integer=True)
        total_floors = numeric_input("Total floors", "Total_Floors", integer=True)
        age = numeric_input("Age of property (years)", "Age_of_Property", integer=True)
        distance_metro = numeric_input("Distance to metro (km)", "Distance_to_Metro_km", step=0.1)
        distance_center = numeric_input("Distance to city center (km)", "Distance_to_City_Center_km", step=0.1)

    st.markdown("**Amenities**")
    amenity1, amenity2, amenity3 = st.columns(3)
    with amenity1:
        parking = st.selectbox("Parking available", [0, 1], format_func=lambda value: "Yes" if value else "No")
    with amenity2:
        lift = st.selectbox("Lift available", [0, 1], index=1, format_func=lambda value: "Yes" if value else "No")
    with amenity3:
        gated = st.selectbox("Gated community", [0, 1], format_func=lambda value: "Yes" if value else "No")

    st.subheader("2. Lending inputs")
    loan_col, income_col, policy_col = st.columns(3)
    with loan_col:
        loan_amount = st.number_input("Proposed loan amount (₹ lakh)", min_value=0.0, value=70.0, step=1.0)
        ltv_policy = st.number_input("Illustrative maximum LTV policy (%)", min_value=1.0, max_value=100.0, value=75.0, step=1.0)
    with income_col:
        monthly_income = st.number_input("Gross monthly income (₹)", min_value=0.0, value=150000.0, step=5000.0)
        existing_obligations = st.number_input("Existing monthly obligations (₹)", min_value=0.0, value=25000.0, step=1000.0)
    with policy_col:
        proposed_emi = st.number_input("Proposed EMI (₹)", min_value=0.0, value=50000.0, step=1000.0)
        foir_policy = st.number_input("Illustrative maximum FOIR (%)", min_value=1.0, max_value=100.0, value=50.0, step=1.0)

    submitted = st.form_submit_button("Estimate value & underwriting ratios", use_container_width=True)

if submitted:
    if floor_number > total_floors:
        st.error("Floor number cannot be greater than total floors.")
        st.stop()

    input_values = {
        "City": city,
        "Locality_Type": locality_type,
        "Property_Type": property_type,
        "BHK": bhk,
        "Bathrooms": bathrooms,
        "Super_Area_SqFt": super_area,
        "Carpet_Area_SqFt": carpet_area,
        "Floor_Number": floor_number,
        "Total_Floors": total_floors,
        "Age_of_Property": age,
        "Furnishing_Status": furnishing,
        "Parking": parking,
        "Lift_Available": lift,
        "Gated_Community": gated,
        "Distance_to_Metro_km": distance_metro,
        "Distance_to_City_Center_km": distance_center,
    }
    input_frame = pd.DataFrame([input_values], columns=feature_columns)

    try:
        estimated_value_lakhs = max(0.0, float(np.asarray(linear_model.predict(input_frame)).reshape(-1)[0]))
        probability_values = np.asarray(logistic_model.predict_proba(input_frame), dtype=float)[0]
        class_values = list(getattr(logistic_model, "classes_", [0, 1]))
        positive_index = class_values.index(1) if 1 in class_values else min(1, len(probability_values) - 1)
        high_value_probability = float(probability_values[positive_index])
    except Exception as error:
        st.error("Prediction failed. Check that the .sav files match this app.py. Details: " + str(error))
        st.stop()

    ltv = loan_amount / estimated_value_lakhs * 100.0 if estimated_value_lakhs > 0 else None
    total_monthly_burden = existing_obligations + proposed_emi
    foir = total_monthly_burden / monthly_income * 100.0 if monthly_income > 0 else None

    st.subheader("3. Valuation result")
    result1, result2, result3 = st.columns(3)
    result1.metric("Estimated property value", f"₹{estimated_value_lakhs:,.2f} lakh")
    result2.metric("Estimated value", f"₹{estimated_value_lakhs / 100.0:.2f} crore")
    result3.metric("Probability property is ≥ ₹1 crore", f"{high_value_probability * 100.0:.1f}%")

    st.subheader("4. Underwriting ratios")
    ratio1, ratio2, ratio3 = st.columns(3)
    with ratio1:
        if ltv is None:
            st.metric("LTV", "N/A")
            st.info("Estimated value is zero, so LTV cannot be calculated.")
        else:
            st.metric("LTV", f"{ltv:.1f}%")
            if ltv <= ltv_policy:
                st.success(f"Within illustrative {ltv_policy:.0f}% LTV policy")
            else:
                st.warning(f"Above illustrative {ltv_policy:.0f}% LTV policy")

    with ratio2:
        if foir is None:
            st.metric("FOIR", "N/A")
            st.info("Enter monthly income above zero to calculate FOIR.")
        else:
            st.metric("FOIR", f"{foir:.1f}%")
            if foir <= foir_policy:
                st.success(f"Within illustrative {foir_policy:.0f}% FOIR policy")
            else:
                st.warning(f"Above illustrative {foir_policy:.0f}% FOIR policy")

    with ratio3:
        st.metric("Proposed loan amount", f"₹{loan_amount:,.2f} lakh")
        st.metric("Valuation buffer", f"₹{estimated_value_lakhs - loan_amount:,.2f} lakh")

    st.info(
        "Academic valuation tool only. LTV and FOIR thresholds are illustrative inputs, not official bank policy. "
