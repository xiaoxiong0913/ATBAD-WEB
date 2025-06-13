import streamlit as st
import pandas as pd
import pickle
from sklearn.preprocessing import StandardScaler
import warnings

# Ignore sklearn version warnings
warnings.filterwarnings("ignore", category=UserWarning, module="sklearn")

# Load model and scaler
model_path = "svm_model.pkl"
scaler_path = "scaler.pkl"

try:
    with open(model_path, 'rb') as model_file, open(scaler_path, 'rb') as scaler_file:
        model = pickle.load(model_file)
        scaler = pickle.load(scaler_file)
except FileNotFoundError as e:
    st.error(f"Model or scaler file not found: {e}")
    st.stop()

# Feature names and ranges
feature_ranges = {
    "age": list(range(20, 101)),  # Age range
    "HR": list(range(30, 181)),  # Heart rate range
    "BUN": [round(x * 0.1, 1) for x in range(10, 501)],  # BUN range 1.0-50.0
    "coronary heart disease": ["No", "Yes"],  # Coronary heart disease
    "HGB": list(range(50, 201)),  # Hemoglobin range
    "hospitalization": list(range(1, 101)),  # Hospitalization days
    "renal insufficiency": ["No", "Yes"]  # Renal insufficiency
}

normal_ranges = {
    "HR": (60, 100),  # Normal heart rate
    "BUN": (2.9, 8.2),  # Normal BUN range
    "HGB": (120, 160)  # Normal hemoglobin range
}

# Page layout: two columns
col1, col2 = st.columns([1, 2])  # Left column 1 part, right column 2 parts

with col1:
    # Left: Selection panel
    st.markdown("<h4 style='margin-bottom:10px;'>Parameter Selection Panel</h4>", unsafe_allow_html=True)
    st.markdown("<p style='font-size:14px;'>Enter patient parameters</p>", unsafe_allow_html=True)

    # Using form
    with st.form("selection_form"):
        age = st.selectbox('Age (years)', options=feature_ranges["age"], index=30)
        hr = st.selectbox('Heart Rate (HR, bpm)', options=feature_ranges["HR"], index=40)
        hgb = st.selectbox('Hemoglobin (HGB, g/L)', options=feature_ranges["HGB"], index=70)
        hospitalization_days = st.selectbox('Hospitalization Days', options=feature_ranges["hospitalization"], index=9)
        bun = st.selectbox('Blood Urea Nitrogen (BUN, mmol/L)', options=feature_ranges["BUN"], index=30)
        coronary = st.selectbox('Coronary Heart Disease', options=feature_ranges["coronary heart disease"], index=0)
        renal = st.selectbox('Renal Insufficiency', options=feature_ranges["renal insufficiency"], index=0)

        # Submit button
        submit_button = st.form_submit_button("Predict")

with col2:
    # Right: Title and description
    st.markdown("<h3 style='margin-bottom:10px;'>3-Year Mortality Prediction for Acute Type B Aortic Dissection</h3>", unsafe_allow_html=True)
    st.markdown("<h4 style='margin-bottom:10px;'>Model Overview</h4>", unsafe_allow_html=True)
    st.markdown("""
    <p style='font-size:14px;'>
    This predictive tool uses an SVM machine learning model (AUC: 0.94, Accuracy: 88.8%) to estimate 3-year mortality risk in patients with acute Type B aortic dissection.
    </p>
    """, unsafe_allow_html=True)

    # Prediction results
    if submit_button:
        # Convert categorical variables to numerical
        coronary_binary = 1 if coronary == "Yes" else 0
        renal_binary = 1 if renal == "Yes" else 0
        
        # Collect input data - using EXACT feature names from model training
        data = {
            "age": age,
            "HR": hr,
            "HGB": hgb,
            "hospitalization": hospitalization_days,
            "BUN": bun,
            "coronary heart disease": coronary_binary,
            "renal insufficiency": renal_binary
        }

        try:
            # Convert to DataFrame with correct feature names
            feature_order = ['age', 'HR', 'HGB', 'hospitalization', 'BUN', 'coronary heart disease', 'renal insufficiency']
            data_df = pd.DataFrame([data], columns=feature_order)
            
            # Standardize data
            data_scaled = scaler.transform(data_df)

            # Predict
            prediction = model.predict_proba(data_scaled)[:, 1][0]

            # Display prediction results with color coding and styling
            if prediction >= 0.207:  # High risk
                st.markdown(
                    f"<div style='background-color:#ffcccc; padding:15px; border-radius:10px; margin-bottom:20px;'>"
                    f"<span style='color:#d32f2f; font-size:20px; font-weight:bold;'>Predicted Mortality Risk: <b>{prediction * 100:.2f}%</b> (High Risk)</span><br>"
                    f"<span style='color:#d32f2f; font-size:16px;'>Elevated risk of mortality within 3 years. Proactive intervention recommended.</span>"
                    f"</div>",
                    unsafe_allow_html=True
                )
            else:  # Low risk
                st.markdown(
                    f"<div style='background-color:#e6ffe6; padding:15px; border-radius:10px; margin-bottom:20px;'>"
                    f"<span style='color:#388e3c; font-size:20px; font-weight:bold;'>Predicted Mortality Risk: <b>{prediction * 100:.2f}%</b> (Low Risk)</span><br>"
                    f"<span style='color:#388e3c; font-size:16px;'>Lower risk of mortality within 3 years. Regular monitoring advised.</span>"
                    f"</div>",
                    unsafe_allow_html=True
                )

            # Personalized medical recommendations
            st.subheader("Personalized Medical Recommendations")
            st.markdown("<div style='background-color:#f8f9fa; padding:15px; border-radius:10px;'>", unsafe_allow_html=True)
            
            # Heart rate recommendations
            if hr < 60:
                st.markdown(f"<div style='margin-bottom:10px;'>❤️ <b>Heart Rate</b> ({hr} bpm): Below normal range (60-100 bpm). Consider: Adjusting antihypertensive medications, evaluating for conduction disorders</div>", unsafe_allow_html=True)
            elif hr > 100:
                st.markdown(f"<div style='margin-bottom:10px;'>❤️ <b>Heart Rate</b> ({hr} bpm): Above normal range (60-100 bpm). Consider: Aggressive heart rate control with beta-blockers</div>", unsafe_allow_html=True)
            else:
                st.markdown(f"<div style='margin-bottom:10px;'>✅ <b>Heart Rate</b> ({hr} bpm): Within normal range</div>", unsafe_allow_html=True)
            
            # BUN recommendations
            if bun < 2.9:
                st.markdown(f"<div style='margin-bottom:10px;'>🩸 <b>Blood Urea Nitrogen</b> ({bun} mmol/L): Below normal range (2.9-8.2 mmol/L). Consider: Nutritional assessment and liver function evaluation</div>", unsafe_allow_html=True)
            elif bun > 8.2:
                st.markdown(f"<div style='margin-bottom:10px;'>🩸 <b>Blood Urea Nitrogen</b> ({bun} mmol/L): Above normal range (2.9-8.2 mmol/L). Consider: Renal function assessment, protein restriction, hydration optimization</div>", unsafe_allow_html=True)
            else:
                st.markdown(f"<div style='margin-bottom:10px;'>✅ <b>Blood Urea Nitrogen</b> ({bun} mmol/L): Within normal range</div>", unsafe_allow_html=True)
            
            # Hemoglobin recommendations
            if hgb < 120:
                st.markdown(f"<div style='margin-bottom:10px;'>🔴 <b>Hemoglobin</b> ({hgb} g/L): Below normal range (120-160 g/L). Consider: Anemia workup, iron supplementation, or erythropoietin therapy</div>", unsafe_allow_html=True)
            elif hgb > 160:
                st.markdown(f"<div style='margin-bottom:10px;'>🔴 <b>Hemoglobin</b> ({hgb} g/L): Above normal range (120-160 g/L). Consider: Monitoring for hyperviscosity syndrome, ensuring adequate hydration</div>", unsafe_allow_html=True)
            else:
                st.markdown(f"<div style='margin-bottom:10px;'>✅ <b>Hemoglobin</b> ({hgb} g/L): Within normal range</div>", unsafe_allow_html=True)
            
            # Coronary heart disease recommendations
            if coronary_binary == 1:
                st.markdown(f"<div style='margin-bottom:10px;'>⚠️ <b>Coronary Heart Disease</b>: Present. Consider: Optimizing antiplatelet therapy, statins, and evaluation for revascularization</div>", unsafe_allow_html=True)
            
            # Renal insufficiency recommendations
            if renal_binary == 1:
                st.markdown(f"<div style='margin-bottom:10px;'>⚠️ <b>Renal Insufficiency</b>: Present. Consider: Nephrology consultation, avoiding nephrotoxic agents, BP target <130/80 mmHg</div>", unsafe_allow_html=True)
            
            # Hospitalization recommendations
            if hospitalization_days > 14:
                st.markdown(f"<div style='margin-bottom:10px;'>⏱️ <b>Hospitalization Duration</b> ({hospitalization_days} days): Extended stay. Consider: Comprehensive complication assessment and rehabilitation planning</div>", unsafe_allow_html=True)
            
            # General management recommendations
            st.markdown("""
            <div style='margin-top:20px;'>
            <b>Comprehensive Management Recommendations:</b>
            <ul style='margin-top:10px;'>
                <li>Strict blood pressure control (target SBP &lt;120 mmHg)</li>
                <li>Regular imaging surveillance (CTA every 6-12 months)</li>
                <li>Smoking cessation and lipid management</li>
                <li>Avoidance of strenuous physical activity</li>
                <li>Immediate evaluation for recurrent chest/back pain</li>
                <li>Consideration of TEVAR for appropriate candidates</li>
                <li>Annual cardiology follow-up</li>
            </ul>
            </div>
            """, unsafe_allow_html=True)
            
            st.markdown("</div>", unsafe_allow_html=True)  # Close recommendation container
                
        except Exception as e:
            st.error(f"Prediction error: {e}")
