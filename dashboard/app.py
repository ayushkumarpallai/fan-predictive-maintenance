"""Streamlit Web Dashboard for Fan Predictive Maintenance System.

Provides:
- Live Telemetry & Health Monitoring
- Real-time Anomaly Severity Score & RUL Countdown
- Explainable AI (SHAP) Feature Attribution
- Prescriptive Maintenance Recommendations
- Interactive Fault Injection & Telemetry Stream Playback
"""

import sys
import os
import time
import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

# Append project paths
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT_DIR)
sys.path.insert(0, os.path.join(ROOT_DIR, 'src'))

from src.data_preprocessing import DataPreprocessor
from src.feature_engineering import FeatureEngineer
from src.anomaly_detection import AnomalyDetector
from src.rul_estimation import RULEstimator
from src.explainability import SHAPExplainer
from src.alert_generation import AlertGenerator
from src.sensor_stream_simulator import SensorStreamSimulator

# Streamlit Page Config
st.set_page_config(
    page_title="Fan Predictive Maintenance System",
    page_icon="⚙️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for Industrial Dashboard Look
st.markdown("""
<style>
    .metric-card {
        background-color: #1e2430;
        border: 1px solid #2e384d;
        border-radius: 10px;
        padding: 15px;
        margin-bottom: 10px;
        text-align: center;
    }
    .status-healthy {
        background: linear-gradient(135deg, #11998e, #38ef7d);
        color: white;
        padding: 12px;
        border-radius: 8px;
        font-weight: bold;
        text-align: center;
        font-size: 20px;
    }
    .status-alert {
        background: linear-gradient(135deg, #f39c12, #f1c40f);
        color: black;
        padding: 12px;
        border-radius: 8px;
        font-weight: bold;
        text-align: center;
        font-size: 20px;
    }
    .status-critical {
        background: linear-gradient(135deg, #cb2d3e, #ef473a);
        color: white;
        padding: 12px;
        border-radius: 8px;
        font-weight: bold;
        text-align: center;
        font-size: 20px;
    }
</style>
""", unsafe_allow_html=True)

# -------------------------------------------------------------
# Cache ML Pipeline Models
# -------------------------------------------------------------
@st.cache_resource
def load_system_pipeline():
    config_path = os.path.join(ROOT_DIR, "config", "config.yaml")
    data_path = os.path.join(ROOT_DIR, "data", "predictive_maintenance_dataset.csv")
    models_dir = os.path.join(ROOT_DIR, "models")
    
    preprocessor = DataPreprocessor(config_path=config_path)
    if os.path.exists(data_path):
        preprocessor.load_data(data_path)
        baseline = preprocessor.extract_baseline()
    else:
        baseline = None

    engineer = FeatureEngineer(config_path=config_path)
    if baseline is not None:
        baseline_eng, feature_names = engineer.engineer_features(baseline)
        X_baseline = baseline_eng[feature_names].values
    else:
        feature_names = ['TotalVibration', 'Temp', 'Voltage', 'Current']
        X_baseline = np.zeros((10, len(feature_names)))

    detector = AnomalyDetector(config_path=config_path)
    iso_path = os.path.join(models_dir, "isolation_forest_model.pkl")
    if os.path.exists(iso_path):
        detector.load(iso_path)
    elif X_baseline is not None:
        detector.train(X_baseline)

    rul_model = RULEstimator(config_path=config_path)
    rul_path = os.path.join(models_dir, "rul_estimator_model.pkl")
    if os.path.exists(rul_path):
        rul_model.load(rul_path)

    explainer = SHAPExplainer(detector.model, X_baseline, feature_names)
    alert_gen = AlertGenerator(config_path=config_path)
    simulator = SensorStreamSimulator(csv_path=data_path)

    return preprocessor, engineer, detector, rul_model, explainer, alert_gen, simulator, feature_names

try:
    preprocessor, engineer, detector, rul_model, explainer, alert_gen, simulator, feature_names = load_system_pipeline()
    pipeline_ready = True
except Exception as e:
    st.error(f"Error loading ML pipeline: {e}")
    pipeline_ready = False

# -------------------------------------------------------------
# Header
# -------------------------------------------------------------
col_h1, col_h2 = st.columns([3, 1])
with col_h1:
    st.title("🏭 Fan Predictive Maintenance System")
    st.caption("ESP32 IoT + Edge AI Monitoring • Vibration • Thermal • Electrical Analytics")
with col_h2:
    st.metric("System Clock", time.strftime("%H:%M:%S UTC"), "Real-Time Telemetry")

st.divider()

# -------------------------------------------------------------
# Sidebar Configuration & Controls
# -------------------------------------------------------------
st.sidebar.header("🕹️ Control & Simulation Panel")
mode = st.sidebar.radio("Operating Mode", ["Manual What-If Slider", "Preset Fault Injection", "Live Stream Playback"])

# Initialize values
vib_val = 9.673
temp_val = 29.50
volt_val = 238.00
curr_val = 0.501

if mode == "Preset Fault Injection":
    preset = st.sidebar.selectbox("Select Fault Scenario", [
        "Normal Operation (Healthy)",
        "Bearing Wear (High Vibration)",
        "Motor Overheating (Cooling Loss)",
        "Overcurrent (Mechanical Jam / Short)",
        "Line Voltage Surge / Drop",
        "Multiple Compound Failures"
    ])
    if preset == "Normal Operation (Healthy)":
        vib_val, temp_val, volt_val, curr_val = 9.673, 29.50, 238.0, 0.501
    elif preset == "Bearing Wear (High Vibration)":
        vib_val, temp_val, volt_val, curr_val = 14.85, 30.10, 237.5, 0.531
    elif preset == "Motor Overheating (Cooling Loss)":
        vib_val, temp_val, volt_val, curr_val = 9.85, 64.20, 238.1, 0.536
    elif preset == "Overcurrent (Mechanical Jam / Short)":
        vib_val, temp_val, volt_val, curr_val = 10.09, 32.40, 233.5, 0.694
    elif preset == "Line Voltage Surge / Drop":
        vib_val, temp_val, volt_val, curr_val = 10.11, 33.67, 251.7, 0.458
    elif preset == "Multiple Compound Failures":
        vib_val, temp_val, volt_val, curr_val = 15.12, 66.50, 232.2, 0.703

elif mode == "Manual What-If Slider":
    st.sidebar.subheader("Adjust Sensor Telemetry")
    vib_val = st.sidebar.slider("Total Vibration (g)", min_value=8.0, max_value=20.0, value=9.673, step=0.01)
    temp_val = st.sidebar.slider("Motor Temperature (°C)", min_value=20.0, max_value=90.0, value=29.5, step=0.5)
    volt_val = st.sidebar.slider("Line Voltage (V)", min_value=180.0, max_value=280.0, value=238.0, step=1.0)
    curr_val = st.sidebar.slider("Phase Current (A)", min_value=0.10, max_value=2.00, value=0.501, step=0.01)

elif mode == "Live Stream Playback":
    st.sidebar.info("Streaming live historical/synthetic sensor frames...")
    if st.sidebar.button("Step to Next Telemetry Frame"):
        sample = simulator.generate_synthetic_sample(np.random.choice(["Healthy", "Healthy", "High_Vibration", "Overheating"]))
        vib_val, temp_val, volt_val, curr_val = sample['TotalVibration'], sample['Temp'], sample['Voltage'], sample['Current']

# -------------------------------------------------------------
# Compute Real-Time Model Inference
# -------------------------------------------------------------
if pipeline_ready:
    # Construct input dataframe
    input_df = pd.DataFrame([{
        'Timestamp': pd.Timestamp.now(),
        'TotalVibration': vib_val,
        'Temp': temp_val,
        'Voltage': volt_val,
        'Current': curr_val,
        'Status': 'Normal',
        'Condition': 'Healthy',
        'Recommendation': 'None'
    }])

    # Engineer features
    input_eng, _ = engineer.engineer_features(input_df)
    X_input = input_eng[feature_names].values

    # Anomaly Score
    anomaly_score = float(detector.predict_proba(X_input)[0])

    # RUL Prediction
    try:
        rul_days = float(rul_model.predict(X_input)[0])
    except Exception:
        rul_days = max(0.0, 30.0 * (1.0 - anomaly_score))

    # SHAP Explanation
    explanation = explainer.explain_prediction(X_input[0])

    # Alert Generation
    features_dict = {
        'TotalVibration': vib_val,
        'Temp': temp_val,
        'Voltage': volt_val,
        'Current': curr_val
    }
    alert = alert_gen.generate_alert(features_dict, anomaly_score, rul_days, explanation)

    # -------------------------------------------------------------
    # KPI & Status Row
    # -------------------------------------------------------------
    col_kpi1, col_kpi2, col_kpi3, col_kpi4 = st.columns(4)

    with col_kpi1:
        st.subheader("System Status")
        status_label = alert['status']
        if status_label == "Normal":
            st.markdown('<div class="status-healthy">🟢 HEALTHY / NORMAL</div>', unsafe_allow_html=True)
        elif status_label == "Alert":
            st.markdown('<div class="status-alert">🟡 WARNING / ALERT</div>', unsafe_allow_html=True)
        else:
            st.markdown('<div class="status-critical">🔴 CRITICAL FAILURE</div>', unsafe_allow_html=True)

    with col_kpi2:
        st.subheader("Fault Condition")
        st.metric("Diagnosed State", alert['condition'].replace("_", " "))

    with col_kpi3:
        st.subheader("Anomaly Severity")
        st.metric("Anomaly Score", f"{anomaly_score:.3f}", f"Threshold: 0.600")
        st.progress(min(1.0, max(0.0, anomaly_score)))

    with col_kpi4:
        st.subheader("Prognostics (RUL)")
        rul_color = "normal" if rul_days > 7 else "inverse"
        st.metric("Remaining Useful Life", f"{rul_days:.1f} Days", f"Status: {alert.get('rul_status', 'Normal')}")
        st.progress(min(1.0, max(0.0, rul_days / 30.0)))

    st.divider()

    # -------------------------------------------------------------
    # Live Gauges / Metrics
    # -------------------------------------------------------------
    st.subheader("📊 Live Sensor Telemetry Gauges")
    g1, g2, g3, g4, g5 = st.columns(5)

    with g1:
        st.metric("Total Vibration", f"{vib_val:.3f} g", f"{vib_val - 9.673:+.3f} g vs Base", delta_color="inverse")
    with g2:
        st.metric("Motor Temperature", f"{temp_val:.1f} °C", f"{temp_val - 29.5:+.1f} °C vs Base", delta_color="inverse")
    with g3:
        st.metric("Line Voltage", f"{volt_val:.1f} V", f"{volt_val - 238.0:+.1f} V vs Base", delta_color="normal")
    with g4:
        st.metric("Phase Current", f"{curr_val:.4f} A", f"{curr_val - 0.501:+.4f} A vs Base", delta_color="inverse")
    with g5:
        p_val = volt_val * curr_val
        st.metric("Active Power", f"{p_val:.1f} W", f"{p_val - 119.2:+.1f} W vs Base")

    # -------------------------------------------------------------
    # Explainable AI (SHAP) & Prescriptive Maintenance
    # -------------------------------------------------------------
    st.divider()
    c_left, c_right = st.columns([1, 1])

    with c_left:
        st.subheader("🔍 Explainable AI (SHAP Contribution)")
        st.caption("Which sensor deviations contributed to this anomaly decision?")
        
        top_factors = explanation.get('top_features', [])[:6]
        if top_factors:
            df_shap = pd.DataFrame(top_factors)
            fig_shap = px.bar(
                df_shap,
                x='shap_value',
                y='feature',
                orientation='h',
                color='shap_value',
                color_continuous_scale='Turbo',
                title="SHAP Feature Attribution Score (Local Explanation)",
                labels={'shap_value': 'SHAP Impact (Towards Anomaly)', 'feature': 'Engineered Feature'}
            )
            fig_shap.update_layout(height=300, yaxis={'categoryorder': 'total ascending'})
            st.plotly_chart(fig_shap, width='stretch')
        else:
            st.info("Feature contributions within normal baseline variance.")

    with c_right:
        st.subheader("🛠️ Prescriptive Maintenance Dispatch")
        st.caption("Automated Decision Support for Plant Operators")
        
        if alert['status'] == "Normal":
            st.success("✅ **Routine Operation:** All physical and electrical parameters are within standard operating thresholds.")
        elif alert['status'] == "Alert":
            st.warning(f"⚠️ **Scheduled Action Required:**\n\n**Action:** {alert['recommendation']}\n\n**Urgency:** Within 24-48 hours")
        else:
            st.error(f"🚨 **EMERGENCY INTERVENTION REQUIRED:**\n\n**Action:** {alert['recommendation']}\n\n**Urgency:** IMMEDIATE SHUTDOWN & INSPECTION")

        st.json({
            "Timestamp": alert['timestamp'],
            "Status": alert['status'],
            "Fault_Classification": alert['condition'],
            "Predicted_RUL_Days": round(rul_days, 1),
            "Recommended_Action": alert['recommendation']
        })

    # -------------------------------------------------------------
    # Historical Dataset Explorer Tab
    # -------------------------------------------------------------
    st.divider()
    with st.expander("📁 View Historical Sensor Dataset & Analytics"):
        if preprocessor.data is not None:
            st.dataframe(preprocessor.data.head(50), width='stretch')
            st.caption(f"Total Dataset Records: {len(preprocessor.data)} samples")
else:
    st.warning("Please train models first by running `python src/train_all_models.py`")
