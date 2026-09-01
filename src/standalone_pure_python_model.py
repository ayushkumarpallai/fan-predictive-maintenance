"""
Standalone Pure Python Predictor for Fan Predictive Maintenance.
This script contains 100% pure Python code with ZERO binary .pkl or external ML library dependencies.
"""

import math
from typing import Dict, Tuple

# Import parameters directly from model_weights
try:
    from models.model_weights import (
        FEATURE_NAMES, BASELINE_HEALTHY_MEANS, SCALER_MEAN, SCALER_SCALE,
        RUL_INTERCEPT, RUL_COEFFICIENTS, RUL_SCALER_MEAN, RUL_SCALER_SCALE,
        THRESHOLDS, ANOMALY_ACTIONS
    )
except ImportError:
    FEATURE_NAMES = ['TotalVibration', 'Temp', 'Voltage', 'Current', 'TotalVibration_roll_mean', 'Temp_roll_mean', 'Voltage_roll_mean', 'Current_roll_mean', 'TotalVibration_roll_std', 'Temp_roll_std', 'Voltage_roll_std', 'Current_roll_std', 'TotalVibration_delta', 'Temp_delta', 'Voltage_delta', 'Current_delta', 'Power']
    BASELINE_HEALTHY_MEANS = [9.674935, 29.595975, 238.112675, 0.502515, 9.674939, 29.595798, 238.112973, 0.502512, 0.002838, 0.109444, 0.52797, 0.001266, -6e-06, 0.000225, -0.0002, 4e-06, 119.655183]
    SCALER_MEAN = [9.674935, 29.595975, 238.112675, 0.502515, 9.674939, 29.595798, 238.112973, 0.502512, 0.002838, 0.109444, 0.52797, 0.001266, -6e-06, 0.000225, -0.0002, 4e-06, 119.655183]
    SCALER_SCALE = [0.003021, 0.123981, 0.571093, 0.001446, 0.001665, 0.071628, 0.322335, 0.000844, 0.001208, 0.058586, 0.233692, 0.000686, 0.004399, 0.175291, 0.819522, 0.002018, 0.445983]
    RUL_INTERCEPT = 10.252077
    RUL_COEFFICIENTS = [-0.035564, -0.045771, -0.048023, 0.082541, 0.011131, 0.003996, -0.062499, -0.003221, -0.03569, -0.108707, -0.061592, -0.091423, -0.014281, -0.040318, -0.059814, -0.029756, -0.11111]
    RUL_SCALER_MEAN = [1.126851, 6.991138, 4.438135, 0.11254, 1.123305, 6.941075, 3.578945, 0.110416, 1.502358, 8.588155, 5.693999, 0.150096, 1.830614, 10.521406, 7.592495, 0.185082, 26.432836]
    RUL_SCALER_SCALE = [2.08327, 11.478485, 7.851318, 0.207843, 1.191494, 6.718184, 4.132833, 0.11775, 1.457761, 7.429292, 6.207268, 0.146962, 2.360463, 12.261581, 9.913572, 0.239716, 48.811173]
    THRESHOLDS = {'normal_max_score': 0.5, 'alert_min_score': 0.6, 'critical_min_score': 0.8, 'vibration_high_g': 11.0, 'temperature_high_c': 40.0, 'current_high_a': 0.65, 'voltage_nominal_v': 238.0, 'voltage_tolerance_v': 25.0}
    ANOMALY_ACTIONS = {
        "High_Vibration": "Inspect bearing, shaft alignment and mounting",
        "Overheating": "Check cooling, ventilation, friction and motor temperature",
        "Overcurrent": "Check mechanical load, motor condition and electrical connections",
        "Voltage_Anomaly": "Inspect supply voltage, wiring and voltage regulation",
        "Multiple_Anomalies": "Stop/inspect machine; check vibration, temperature, electrical load and supply",
        "Healthy": "No action required - normal operation"
    }


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
    print("Pure Python Predictor ")
    print("=" * 60)

    # Test 1: Healthy Fan
    print("\n[Test 1] Healthy Fan Reading:")
    res1 = predictor.evaluate(9.673, 29.5, 238.0, 0.501)
    for k, v in res1.items():
        print(f"  {k}: {v}")

    # Test 2: Bearing Wear (High Vibration)
    print("\n[Test 2] Bearing Wear Anomaly:")
    res2 = predictor.evaluate(14.85, 30.1, 237.5, 0.531)
    for k, v in res2.items():
        print(f"  {k}: {v}")

    # Test 3: Compound Critical Failure
    print("\n[Test 3] Multiple Critical Failures:")
    res3 = predictor.evaluate(15.2, 65.0, 230.0, 0.95)
    for k, v in res3.items():
        print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
