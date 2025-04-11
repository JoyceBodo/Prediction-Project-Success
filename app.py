
# app.py
import streamlit as st
import pandas as pd
import joblib

# Load trained model and encoders
model = joblib.load("random_forest_model.pkl")
funding_encoder = joblib.load("funding_source_encoder.pkl")
sector_encoder = joblib.load("mtef_sector_encoder.pkl")
agency_encoder = joblib.load("implementing_agency_encoder.pkl")

st.set_page_config(page_title="Donor-Funded Project Predictor", layout="centered")
st.title("🎯 Donor-Funded Project Success Predictor in Kenya")
st.write("This tool predicts the success probability of a donor-funded project based on historical trends.")

# User Input
st.subheader("📋 Enter Project Details")

cost = st.number_input("💰 Total Project Cost (KES)", min_value=0.0, step=100000.0)
duration = st.number_input("⏱️ Project Duration (Months)", min_value=0)
funding_source = st.selectbox("🏦 Funding Source", funding_encoder.classes_)
mtef_sector = st.selectbox("📊 MTEF Sector", sector_encoder.classes_)
agency = st.selectbox("🏗️ Implementing Agency", agency_encoder.classes_)

# Helper
def encode_label(val, encoder):
    try:
        return encoder.transform([val])[0]
    except Exception as e:
        st.warning(f"Encoding failed for {val}. Using 0.")
        return 0

# Prepare input
input_df = pd.DataFrame([{
    'total_project_cost_kes': cost,
    'duration_months': duration,
    'funding_source': encode_label(funding_source, funding_encoder),
    'mtef_sector': encode_label(mtef_sector, sector_encoder),
    'implementing_agency': encode_label(agency, agency_encoder)
}])

# Prediction
if st.button("🔍 Predict Success"):
    try:
        prediction = model.predict(input_df)[0]
        confidence = model.predict_proba(input_df)[0][prediction]
        message = "✅ Project is likely to succeed" if prediction == 1 else "⚠️ Project may fail or stall"
        st.success(f"{message} (Confidence: {confidence:.2%})")
    except Exception as e:
        st.error(f"Prediction failed: {e}")
