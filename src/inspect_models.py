"""Model Inspector CLI.

Allows inspecting all model parameters, trees, equations, coefficients,
and decision thresholds directly in human-readable text.
"""

import sys
import os
import joblib
import numpy as np

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT_DIR)
sys.path.insert(0, os.path.join(ROOT_DIR, 'src'))

def inspect():
    iso_path = os.path.join(ROOT_DIR, "models", "isolation_forest_model.pkl")
    rul_path = os.path.join(ROOT_DIR, "models", "rul_estimator_model.pkl")
    scaler_path = os.path.join(ROOT_DIR, "models", "feature_scaler.pkl")

    print("=" * 70)
    print("      FAN PREDICTIVE MAINTENANCE - MODEL ARCHITECTURE & WEIGHTS")
    print("=" * 70)

    # 1. Inspect Isolation Forest
    if os.path.exists(iso_path):
        iso_model = joblib.load(iso_path)
        print("\n[1] ISOLATION FOREST ANOMALY DETECTOR")
        print(f"  - Model Type:          {type(iso_model).__name__}")
        print(f"  - Number of Trees:     {len(iso_model.estimators_)}")
        print(f"  - Max Samples per Tree:{iso_model.max_samples_}")
        print(f"  - Contamination Rate:  {iso_model.contamination}")
        print(f"  - Decision Offset:     {iso_model.offset_:.4f}")
        print(f"  - Number of Features:  {iso_model.n_features_in_}")
    else:
        print("\n[1] Isolation Forest model not found at", iso_path)

    # 2. Inspect RUL Linear Degradation Model
    if os.path.exists(rul_path):
        rul_dict = joblib.load(rul_path)
        lin_model = rul_dict['model']
        rul_scaler = rul_dict['scaler']
        baseline_stats = rul_dict['baseline_stats']

        feature_names = [
            "TotalVibration", "Temp", "Voltage", "Current",
            "TotalVibration_roll_mean", "Temp_roll_mean", "Voltage_roll_mean", "Current_roll_mean",
            "TotalVibration_roll_std", "Temp_roll_std", "Voltage_roll_std", "Current_roll_std",
            "TotalVibration_delta", "Temp_delta", "Voltage_delta", "Current_delta",
            "Power"
        ]

        print("\n[2] REMAINING USEFUL LIFE (RUL) REGRESSION MODEL")
        print(f"  - Model Type:          {type(lin_model).__name__}")
        print(f"  - Intercept (Base RUL):{lin_model.intercept_:.3f} days")
        print(f"  - Feature Degradation Weights (Linear Coefficients):")
        print(f"    {'Feature Name':<30} | {'Baseline Mean':<14} | {'Coefficient':<12}")
        print("    " + "-" * 62)
        for i, name in enumerate(feature_names):
            base_val = baseline_stats[i]
            coef_val = lin_model.coef_[i]
            print(f"    {name:<30} | {base_val:>12.4f}   | {coef_val:>+10.4f}")

    # 3. Decision Thresholds
    print("\n[3] PHYSICAL THRESHOLDS & FAULT TAXONOMY")
    print("  - Vibration Threshold:   > 11.00 g    --> High_Vibration (Bearing Wear)")
    print("  - Temperature Threshold: > 40.00 °C   --> Overheating (Cooling/Friction)")
    print("  - Current Threshold:     > 0.650 A    --> Overcurrent (Mechanical Jam)")
    print("  - Voltage Tolerance:     238V ± 25V   --> Voltage_Anomaly (Grid Surge)")
    print("  - Multiple Faults:       >= 2 Faults  --> Multiple_Anomalies (Emergency)")

    print("\n" + "=" * 70)
    print("  To view the full model in pure editable Python code, open:")
    print("  --> models/model_weights.py")
    print("  --> src/standalone_pure_python_model.py")
    print("=" * 70)


if __name__ == "__main__":
    inspect()
