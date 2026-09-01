"""Unit test for zero-dependency standalone pure Python model."""

import pytest
from src.standalone_pure_python_model import PurePythonFanPredictor

def test_pure_python_healthy_prediction():
    predictor = PurePythonFanPredictor()
    res = predictor.evaluate(9.673, 29.5, 238.0, 0.501)
    assert res['Status'] == 'Normal'
    assert res['Condition'] == 'Healthy'
    assert res['AnomalyScore'] < 0.5
    assert 0 <= res['RUL_Days'] <= 30

def test_pure_python_high_vibration_prediction():
    predictor = PurePythonFanPredictor()
    res = predictor.evaluate(14.85, 30.1, 237.5, 0.531)
    assert res['Status'] == 'Alert'
    assert res['Condition'] == 'High_Vibration'
    assert res['AnomalyScore'] >= 0.6
    assert "bearing" in res['Recommendation'].lower()

def test_pure_python_critical_compound_fault():
    predictor = PurePythonFanPredictor()
    res = predictor.evaluate(15.2, 65.0, 230.0, 0.95)
    assert res['Status'] == 'Critical'
    assert res['Condition'] == 'Multiple_Anomalies'
    assert res['AnomalyScore'] >= 0.8
    assert "stop" in res['Recommendation'].lower()

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
