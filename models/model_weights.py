"""
HUMAN-READABLE PURE PYTHON MODEL WEIGHTS & PARAMETERS
Generated from trained models. No pickle or binary files needed!
"""

# Feature Names used by the system
FEATURE_NAMES = ['TotalVibration', 'Temp', 'Voltage', 'Current', 'TotalVibration_roll_mean', 'Temp_roll_mean', 'Voltage_roll_mean', 'Current_roll_mean', 'TotalVibration_roll_std', 'Temp_roll_std', 'Voltage_roll_std', 'Current_roll_std', 'TotalVibration_delta', 'Temp_delta', 'Voltage_delta', 'Current_delta', 'Power']

# Healthy Baseline Mean Reference Values
BASELINE_HEALTHY_MEANS = [9.674935, 29.595975, 238.112675, 0.502515, 9.674939, 29.595798, 238.112973, 0.502512, 0.002838, 0.109444, 0.52797, 0.001266, -6e-06, 0.000225, -0.0002, 4e-06, 119.655183]

# Feature Scaler (Mean and Standard Deviation Scale)
SCALER_MEAN = [9.674935, 29.595975, 238.112675, 0.502515, 9.674939, 29.595798, 238.112973, 0.502512, 0.002838, 0.109444, 0.52797, 0.001266, -6e-06, 0.000225, -0.0002, 4e-06, 119.655183]
SCALER_SCALE = [0.003021, 0.123981, 0.571093, 0.001446, 0.001665, 0.071628, 0.322335, 0.000844, 0.001208, 0.058586, 0.233692, 0.000686, 0.004399, 0.175291, 0.819522, 0.002018, 0.445983]

# RUL (Remaining Useful Life) Linear Regression Equation:
# RUL_Days = Intercept + Sum(Coeff_i * Normalized_Degradation_i)
RUL_INTERCEPT = 10.252077
RUL_COEFFICIENTS = [-0.035564, -0.045771, -0.048023, 0.082541, 0.011131, 0.003996, -0.062499, -0.003221, -0.03569, -0.108707, -0.061592, -0.091423, -0.014281, -0.040318, -0.059814, -0.029756, -0.11111]
RUL_SCALER_MEAN = [1.126851, 6.991138, 4.438135, 0.11254, 1.123305, 6.941075, 3.578945, 0.110416, 1.502358, 8.588155, 5.693999, 0.150096, 1.830614, 10.521406, 7.592495, 0.185082, 26.432836]
RUL_SCALER_SCALE = [2.08327, 11.478485, 7.851318, 0.207843, 1.191494, 6.718184, 4.132833, 0.11775, 1.457761, 7.429292, 6.207268, 0.146962, 2.360463, 12.261581, 9.913572, 0.239716, 48.811173]

# Physical Thresholds and Classification Boundaries
THRESHOLDS = {
    "normal_max_score": 0.5,
    "alert_min_score": 0.6,
    "critical_min_score": 0.8,
    "vibration_high_g": 11.0,
    "temperature_high_c": 40.0,
    "current_high_a": 0.65,
    "voltage_nominal_v": 238.0,
    "voltage_tolerance_v": 25.0,
}

ANOMALY_ACTIONS = {
    "High_Vibration": "Inspect bearing, shaft alignment and mounting",
    "Overheating": "Check cooling, ventilation, friction and motor temperature",
    "Overcurrent": "Check mechanical load, motor condition and electrical connections",
    "Voltage_Anomaly": "Inspect supply voltage, wiring and voltage regulation",
    "Multiple_Anomalies": "Stop/inspect machine; check vibration, temperature, electrical load and supply",
    "Healthy": "No action required - normal operation"
}
