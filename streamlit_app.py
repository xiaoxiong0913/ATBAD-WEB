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
    "BUN": [round(x * 0.1, 1) for x in range(10, 501)],  # BUN range 1.0–50.0
    "coronary heart disease": ["No", "Yes"],  # Coronary heart disease
    "HGB": list(range(50, 201)),  # Hemoglobin range
    "hospitalization": list(range(1, 101)),  # Hospitalization days
    "renal insufficiency": ["No", "Yes"]  # UI label
}

normal_ranges = {
    "HR": (60, 100),     # Normal heart rate
    "BUN": (2.9, 8.2),   # Normal BUN
    "HGB": (120, 160)    # Normal hemoglobin
}

# Page layout: two columns
col1, col2 = st.columns([1, 2])

with col1:
    st.markdown("<h4 style='margin-bottom:10px;'>Parameter Selection Panel</h4>", unsafe_allow_html=True)
    st.markdown("<p style='font-size:14px;'>Enter patient parameters</p>", unsafe_allow_html=True)

    with st.form("selection_form"):
        age = st.selectbox('Age (years)', feature_ranges["age"], index=30)
        hr = st.selectbox('Heart Rate (HR, bpm)', feature_ranges["HR"], index=40)
        hgb = st.selectbox('Hemoglobin (HGB, g/L)', feature_ranges["HGB"], index=70)
        hospitalization_days = st.selectbox('Hospitalization Days', feature_ranges["hospitalization"], index=9)
        bun = st.selectbox('Blood Urea Nitrogen (BUN, mmol/L)', feature_ranges["BUN"], index=30)
        coronary = st.selectbox('Coronary Heart Disease', feature_ranges["coronary heart disease"], index=0)
        renal = st.selectbox('Renal Insufficiency', feature_ranges["renal insufficiency"], index=0)

        submit_button = st.form_submit_button("Predict")

with col2:
    st.markdown("<h3 style='margin-bottom:10px;'>3-Year Mortality Prediction for Acute Type B Aortic Dissection</h3>", unsafe_allow_html=True)
    st.markdown("<h4 style='margin-bottom:10px;'>Model Overview</h4>", unsafe_allow_html=True)
    st.markdown("""
    <p style='font-size:14px;'>
    This predictive tool uses an SVM machine learning model (AUC: 0.94, Accuracy: 88.8%) to estimate 3-year mortality risk in patients with acute Type B aortic dissection.
    </p>
    """, unsafe_allow_html=True)

    if submit_button:
        coronary_binary = 1 if coronary == "Yes" else 0
        renal_binary = 1 if renal == "Yes" else 0

        data = {
            "age": age,
            "HR": hr,
            "BUN": bun,
            "HGB": hgb,
            "hospitalization": hospitalization_days,
            "coronary heart disease": coronary_binary,
            "renal dysfunction": renal_binary  # match model's feature name
        }

        try:
            # Ensure the exact same order used during model.fit
            feature_order = [
                'age',
                'HR',
                'BUN',
                'HGB',
                'hospitalization',
                'coronary heart disease',
                'renal dysfunction'
            ]
            data_df = pd.DataFrame([data], columns=feature_order)

            data_scaled = scaler.transform(data_df)
            prediction = model.predict_proba(data_scaled)[:, 1][0]

            # Display risk
            if prediction >= 0.207:
                st.markdown(
                    f"<div style='background-color:#ffcccc; padding:15px; border-radius:10px; margin-bottom:20px;'>"
                    f"<span style='color:#d32f2f; font-size:20px; font-weight:bold;'>Predicted Mortality Risk: <b>{prediction * 100:.2f}%</b> (High Risk)</span><br>"
                    f"<span style='color:#d32f2f; font-size:16px;'>Elevated risk of mortality within 3 years. Proactive intervention recommended.</span>"
                    f"</div>",
                    unsafe_allow_html=True
                )
            else:
                st.markdown(
                    f"<div style='background-color:#e6ffe6; padding:15px; border-radius:10px; margin-bottom:20px;'>"
                    f"<span style='color:#388e3c; font-size:20px; font-weight:bold;'>Predicted Mortality Risk: <b>{prediction * 100:.2f}%</b> (Low Risk)</span><br>"
                    f"<span style='color:#388e3c; font-size:16px;'>Lower risk of mortality within 3 years. Regular monitoring advised.</span>"
                    f"</div>",
                    unsafe_allow_html=True
                )

            # Recommendations panel
            st.subheader("Personalized Medical Recommendations")
            st.markdown("<div style='background-color:#f8f9fa; padding:15px; border-radius:10px;'>", unsafe_allow_html=True)

            # Heart rate
            if hr < normal_ranges["HR"][0]:
                st.markdown(f"<div style='margin-bottom:10px;'>❤️ <b>Heart Rate</b> ({hr} bpm): Below normal (60–100). Consider: Adjust meds, evaluate conduction</div>", unsafe_allow_html=True)
            elif hr > normal_ranges["HR"][1]:
                st.markdown(f"<div style='margin-bottom:10px;'>❤️ <b>Heart Rate</b> ({hr} bpm): Above normal (60–100). Consider: Beta-blockers for control</div>", unsafe_allow_html=True)
            else:
                st.markdown(f"<div style='margin-bottom:10px;'>✅ <b>Heart Rate</b> ({hr} bpm): Within normal range</div>", unsafe_allow_html=True)

            # BUN
            if bun < normal_ranges["BUN"][0]:
                st.markdown(f"<div style='margin-bottom:10px;'>🩸 <b>BUN</b> ({bun}): Below normal (2.9–8.2). Consider: Nutrition, liver eval</div>", unsafe_allow_html=True)
            elif bun > normal_ranges["BUN"][1]:
                st.markdown(f"<div style='margin-bottom:10px;'>🩸 <b>BUN</b> ({bun}): Above normal (2.9–8.2). Consider: Renal assessment, hydration</div>", unsafe_allow_html=True)
            else:
                st.markdown(f"<div style='margin-bottom:10px;'>✅ <b>BUN</b> ({bun}): Within normal range</div>", unsafe_allow_html=True)

            # Hemoglobin
            if hgb < normal_ranges["HGB"][0]:
                st.markdown(f"<div style='margin-bottom:10px;'>🔴 <b>HGB</b> ({hgb}): Below normal (120–160). Consider: Anemia workup</div>", unsafe_allow_html=True)
            elif hgb > normal_ranges["HGB"][1]:
                st.markdown(f"<div style='margin-bottom:10px;'>🔴 <b>HGB</b> ({hgb}): Above normal (120–160). Consider: Monitor viscosity</div>", unsafe_allow_html=True)
            else:
                st.markdown(f"<div style='margin-bottom:10px;'>✅ <b>HGB</b> ({hgb}): Within normal range</div>", unsafe_allow_html=True)

            # Coronary heart disease
            if coronary_binary:
                st.markdown(f"<div style='margin-bottom:10px;'>⚠️ <b>Coronary Heart Disease</b>: Present. Consider: Antiplatelets, statins, revascularization eval</div>", unsafe_allow_html=True)

            # Renal insufficiency
            if renal_binary:
                st.markdown(f"<div style='margin-bottom:10px;'>⚠️ <b>Renal Insufficiency</b>: Present. Consider: Nephrology consult, avoid nephrotoxins</div>", unsafe_allow_html=True)

            # Hospitalization days
            if hospitalization_days > 14:
                st.markdown(f"<div style='margin-bottom:10px;'>⏱️ <b>Hospital Stay</b> ({hospitalization_days} days): Extended. Consider: Complication review, rehab planning</div>", unsafe_allow_html=True)

            # General management
            st.markdown("""
            <div style='margin-top:20px;'>
            <b>Comprehensive Management Recommendations:</b>
            <ul style='margin-top:10px;'>
                <li>Strict BP control (SBP &lt;120 mmHg)</li>
                <li>CTA surveillance every 6–12 months</li>
                <li>Smoking cessation & lipid management</li>
                <li>Avoid strenuous activity</li>
                <li>Evaluate recurrent chest/back pain immediately</li>
                <li>Consider TEVAR where indicated</li>
                <li>Annual cardiology follow-up</li>
            </ul>
            </div>
            """, unsafe_allow_html=True)

            st.markdown("</div>", unsafe_allow_html=True)

        except Exception as e:
            st.error(f"Prediction error: {e}")
