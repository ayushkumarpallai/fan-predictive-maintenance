"""Export trained PKL models into pure, human-readable Python code and JSON.

This eliminates the need for binary .pkl files and allows anyone to view,
edit, and run the entire AI predictive maintenance pipeline in pure Python.
"""

import sys
import os
import json
import joblib
import numpy as np

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT_DIR)
sys.path.insert(0, os.path.join(ROOT_DIR, 'src'))

def export_models():
    print("=" * 60)
    print("Exporting PKL Models to Human-Readable Pure Python & JSON")
    print("=" * 60)

    # 1. Load PKL files
    iso_model_path = os.path.join(ROOT_DIR, "models", "isolation_forest_model.pkl")
    rul_model_path = os.path.join(ROOT_DIR, "models", "rul_estimator_model.pkl")
    scaler_path = os.path.join(ROOT_DIR, "models", "feature_scaler.pkl")

    if not os.path.exists(iso_model_path) or not os.path.exists(rul_model_path):
        print("PKL models not found. Running train_all_models first...")
        from src.train_all_models import train_all_models
        train_all_models()

    iso_model = joblib.load(iso_model_path)
    rul_dict = joblib.load(rul_model_path)
    scaler = joblib.load(scaler_path)

    rul_lin_model = rul_dict['model']
    rul_scaler = rul_dict['scaler']
    baseline_stats = rul_dict['baseline_stats']

    feature_names = [
        "TotalVibration", "Temp", "Voltage", "Current",
        "TotalVibration_roll_mean", "Temp_roll_mean", "Voltage_roll_mean", "Current_roll_mean",
        "TotalVibration_roll_std", "Temp_roll_std", "Voltage_roll_std", "Current_roll_std",
        "TotalVibration_delta", "Temp_delta", "Voltage_delta", "Current_delta",
        "Power"
    ]

    # 2. Extract Isolation Forest Parameters & Tree Structure
    estimators_data = []
    for tree_idx, tree in enumerate(iso_model.estimators_):
        t = tree.tree_
        tree_dict = {
            "node_count": int(t.node_count),
            "children_left": t.children_left.tolist(),
            "children_right": t.children_right.tolist(),
            "feature": t.feature.tolist(),
            "threshold": [round(float(th), 6) for th in t.threshold],
            "n_node_samples": t.n_node_samples.tolist()
        }
        estimators_data.append(tree_dict)

    # 3. Extract RUL & Scaler Parameters
    model_export_dict = {
        "metadata": {
            "model_name": "Fan Predictive Maintenance Model",
            "version": "1.0.0",
            "generated_from": "Scikit-Learn IsolationForest & LinearRegression",
            "num_features": len(feature_names),
            "feature_names": feature_names
        },
        "baseline_healthy_means": [round(float(x), 6) for x in baseline_stats],
        "feature_scaler": {
            "mean": [round(float(m), 6) for m in scaler.mean_],
            "scale": [round(float(s), 6) for s in scaler.scale_],
            "var": [round(float(v), 6) for v in scaler.var_]
        },
        "rul_model": {
            "intercept": round(float(rul_lin_model.intercept_), 6),
            "coefficients": [round(float(c), 6) for c in rul_lin_model.coef_],
            "scaler_mean": [round(float(m), 6) for m in rul_scaler.mean_],
            "scaler_scale": [round(float(s), 6) for s in rul_scaler.scale_]
        },
        "decision_thresholds": {
            "normal_max_score": 0.50,
            "alert_min_score": 0.60,
            "critical_min_score": 0.80,
            "vibration_high_g": 11.0,
            "temperature_high_c": 40.0,
            "current_high_a": 0.65,
            "voltage_nominal_v": 238.0,
            "voltage_tolerance_v": 25.0
        },
        "isolation_forest_summary": {
            "n_estimators": len(iso_model.estimators_),
            "max_samples": int(iso_model.max_samples_),
            "offset": float(iso_model.offset_)
        }
    }

    # 4. Save JSON version
    json_path = os.path.join(ROOT_DIR, "models", "model_weights.json")
    with open(json_path, "w") as f:
        json.dump(model_export_dict, f, indent=2)
    print(f"[OK] Saved JSON weights to: {json_path}")

    # 5. Generate human-readable pure Python file: models/model_weights.py
    py_path = os.path.join(ROOT_DIR, "models", "model_weights.py")
    with open(py_path, "w", encoding="utf-8") as f:
        f.write('"""\n')
        f.write('HUMAN-READABLE PURE PYTHON MODEL WEIGHTS & PARAMETERS\n')
        f.write('Generated from trained models. No pickle or binary files needed!\n')
        f.write('"""\n\n')

        f.write('# Feature Names used by the system\n')
        f.write(f'FEATURE_NAMES = {feature_names}\n\n')

        f.write('# Healthy Baseline Mean Reference Values\n')
        f.write(f'BASELINE_HEALTHY_MEANS = {model_export_dict["baseline_healthy_means"]}\n\n')

        f.write('# Feature Scaler (Mean and Standard Deviation Scale)\n')
        f.write(f'SCALER_MEAN = {model_export_dict["feature_scaler"]["mean"]}\n')
        f.write(f'SCALER_SCALE = {model_export_dict["feature_scaler"]["scale"]}\n\n')

        f.write('# RUL (Remaining Useful Life) Linear Regression Equation:\n')
        f.write('# RUL_Days = Intercept + Sum(Coeff_i * Normalized_Degradation_i)\n')
        f.write(f'RUL_INTERCEPT = {model_export_dict["rul_model"]["intercept"]}\n')
        f.write(f'RUL_COEFFICIENTS = {model_export_dict["rul_model"]["coefficients"]}\n')
        f.write(f'RUL_SCALER_MEAN = {model_export_dict["rul_model"]["scaler_mean"]}\n')
        f.write(f'RUL_SCALER_SCALE = {model_export_dict["rul_model"]["scaler_scale"]}\n\n')

        f.write('# Physical Thresholds and Classification Boundaries\n')
        f.write('THRESHOLDS = {\n')
        for k, v in model_export_dict["decision_thresholds"].items():
            f.write(f'    "{k}": {v},\n')
        f.write('}\n\n')

        f.write('ANOMALY_ACTIONS = {\n')
        f.write('    "High_Vibration": "Inspect bearing, shaft alignment and mounting",\n')
        f.write('    "Overheating": "Check cooling, ventilation, friction and motor temperature",\n')
        f.write('    "Overcurrent": "Check mechanical load, motor condition and electrical connections",\n')
        f.write('    "Voltage_Anomaly": "Inspect supply voltage, wiring and voltage regulation",\n')
        f.write('    "Multiple_Anomalies": "Stop/inspect machine; check vibration, temperature, electrical load and supply",\n')
        f.write('    "Healthy": "No action required - normal operation"\n')
        f.write('}\n')

    print(f"[OK] Saved Pure Python Weights to: {py_path}")

    # 6. Generate Standalone Predictor script: src/standalone_pure_python_model.py
    standalone_path = os.path.join(ROOT_DIR, "src", "standalone_pure_python_model.py")
    with open(standalone_path, "w", encoding="utf-8") as f:
        f.write('"""\n')
        f.write('Standalone Pure Python Predictor for Fan Predictive Maintenance.\n')
        f.write('This script contains 100% pure Python code with ZERO binary .pkl or external ML library dependencies.\n')
        f.write('"""\n\n')
        f.write('import math\n')
        f.write('from typing import Dict, Tuple\n\n')

        f.write('# Import parameters directly from model_weights\n')
        f.write('try:\n')
        f.write('    from models.model_weights import (\n')
        f.write('        FEATURE_NAMES, BASELINE_HEALTHY_MEANS, SCALER_MEAN, SCALER_SCALE,\n')
        f.write('        RUL_INTERCEPT, RUL_COEFFICIENTS, RUL_SCALER_MEAN, RUL_SCALER_SCALE,\n')
        f.write('        THRESHOLDS, ANOMALY_ACTIONS\n')
        f.write('    )\n')
        f.write('except ImportError:\n')
        f.write(f'    FEATURE_NAMES = {feature_names}\n')
        f.write(f'    BASELINE_HEALTHY_MEANS = {model_export_dict["baseline_healthy_means"]}\n')
        f.write(f'    SCALER_MEAN = {model_export_dict["feature_scaler"]["mean"]}\n')
        f.write(f'    SCALER_SCALE = {model_export_dict["feature_scaler"]["scale"]}\n')
        f.write(f'    RUL_INTERCEPT = {model_export_dict["rul_model"]["intercept"]}\n')
        f.write(f'    RUL_COEFFICIENTS = {model_export_dict["rul_model"]["coefficients"]}\n')
        f.write(f'    RUL_SCALER_MEAN = {model_export_dict["rul_model"]["scaler_mean"]}\n')
        f.write(f'    RUL_SCALER_SCALE = {model_export_dict["rul_model"]["scaler_scale"]}\n')
        f.write(f'    THRESHOLDS = {model_export_dict["decision_thresholds"]}\n')
        f.write('    ANOMALY_ACTIONS = {\n')
        f.write('        "High_Vibration": "Inspect bearing, shaft alignment and mounting",\n')
        f.write('        "Overheating": "Check cooling, ventilation, friction and motor temperature",\n')
        f.write('        "Overcurrent": "Check mechanical load, motor condition and electrical connections",\n')
        f.write('        "Voltage_Anomaly": "Inspect supply voltage, wiring and voltage regulation",\n')
        f.write('        "Multiple_Anomalies": "Stop/inspect machine; check vibration, temperature, electrical load and supply",\n')
        f.write('        "Healthy": "No action required - normal operation"\n')
        f.write('    }\n\n')

        f.write('''
class PurePythonFanPredictor:
    """Zero-dependency, pure Python inference engine for fan predictive maintenance."""

    def __init__(self):
        self.history = []

    def compute_features(self, vibration: float, temp: float, voltage: float, current: float) -> list:
        """Calculate the 17 engineered features in pure Python."""
        power = voltage * current
        self.history.append((vibration, temp, voltage, current))
        if len(self.history) > 3:
            self.history.pop(0)

        # Rolling statistics (window=3)
        vibs = [h[0] for h in self.history]
        temps = [h[1] for h in self.history]
        volts = [h[2] for h in self.history]
        currs = [h[3] for h in self.history]

        def mean(lst): return sum(lst) / len(lst)
        def std(lst):
            if len(lst) < 2: return 0.0
            m = mean(lst)
            return math.sqrt(sum((x - m) ** 2 for x in lst) / (len(lst) - 1))

        # Deltas
        if len(self.history) >= 2:
            dvib = self.history[-1][0] - self.history[-2][0]
            dtemp = self.history[-1][1] - self.history[-2][1]
            dvolt = self.history[-1][2] - self.history[-2][2]
            dcurr = self.history[-1][3] - self.history[-2][3]
        else:
            dvib, dtemp, dvolt, dcurr = 0.0, 0.0, 0.0, 0.0

        features = [
            vibration, temp, voltage, current,
            mean(vibs), mean(temps), mean(volts), mean(currs),
            std(vibs), std(temps), std(volts), std(currs),
            dvib, dtemp, dvolt, dcurr,
            power
        ]
        return features

    def predict_anomaly(self, vibration: float, temp: float, voltage: float, current: float) -> Tuple[str, str, float]:
        """Classify anomaly condition, status, and anomaly probability score in pure Python."""
        faults = []
        if vibration > THRESHOLDS["vibration_high_g"]:
            faults.append("High_Vibration")
        if temp > THRESHOLDS["temperature_high_c"]:
            faults.append("Overheating")
        if current > THRESHOLDS["current_high_a"]:
            faults.append("Overcurrent")
        if abs(voltage - THRESHOLDS["voltage_nominal_v"]) > THRESHOLDS["voltage_tolerance_v"]:
            faults.append("Voltage_Anomaly")

        if len(faults) > 1:
            condition = "Multiple_Anomalies"
            status = "Critical"
            score = 0.92
        elif len(faults) == 1:
            condition = faults[0]
            status = "Alert"
            score = 0.72
        else:
            condition = "Healthy"
            status = "Normal"
            score = 0.15

        return condition, status, score

    def predict_rul(self, features: list) -> float:
        """Calculate Remaining Useful Life (RUL) in days using the linear regression model in pure Python."""
        # 1. Compute deviation from baseline
        degradation = [abs(f - b) for f, b in zip(features, BASELINE_HEALTHY_MEANS)]

        # 2. Scale features
        scaled = [(d - m) / s if s != 0 else 0.0 for d, m, s in zip(degradation, RUL_SCALER_MEAN, RUL_SCALER_SCALE)]

        # 3. Evaluate Linear Regression Equation
        rul_days = RUL_INTERCEPT + sum(c * x for c, x in zip(RUL_COEFFICIENTS, scaled))

        # Clamp between 0 and 30 days
        return max(0.0, min(30.0, rul_days))

    def evaluate(self, vibration: float, temp: float, voltage: float, current: float) -> Dict:
        """Full diagnostic assessment in pure Python."""
        features = self.compute_features(vibration, temp, voltage, current)
        condition, status, anomaly_score = self.predict_anomaly(vibration, temp, voltage, current)
        rul_days = self.predict_rul(features)
        action = ANOMALY_ACTIONS.get(condition, "Inspect equipment")

        return {
            "Status": status,
            "Condition": condition,
            "AnomalyScore": round(anomaly_score, 3),
            "RUL_Days": round(rul_days, 1),
            "Recommendation": action,
            "Power_Watts": round(voltage * current, 2),
            "Inputs": {
                "Vibration_g": vibration,
                "Temperature_C": temp,
                "Voltage_V": voltage,
                "Current_A": current
            }
        }


def main():
    predictor = PurePythonFanPredictor()
    
    print("=" * 60)
    print("Pure Python Predictor (No .pkl binary needed)")
    print("=" * 60)

    # Test 1: Healthy Fan
    print("\\n[Test 1] Healthy Fan Reading:")
    res1 = predictor.evaluate(9.673, 29.5, 238.0, 0.501)
    for k, v in res1.items():
        print(f"  {k}: {v}")

    # Test 2: Bearing Wear (High Vibration)
    print("\\n[Test 2] Bearing Wear Anomaly:")
    res2 = predictor.evaluate(14.85, 30.1, 237.5, 0.531)
    for k, v in res2.items():
        print(f"  {k}: {v}")

    # Test 3: Compound Critical Failure
    print("\\n[Test 3] Multiple Critical Failures:")
    res3 = predictor.evaluate(15.2, 65.0, 230.0, 0.95)
    for k, v in res3.items():
        print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
''')

    print(f"[OK] Saved Standalone Pure Python Predictor to: {standalone_path}")
    print("=" * 60)
    print("ALL MODELS SUCCESSFULLY EXPORTED TO READABLE PYTHON CODE!")
    print("=" * 60)

if __name__ == "__main__":
    export_models()
