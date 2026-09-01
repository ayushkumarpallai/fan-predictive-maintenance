"""Unit tests for anomaly detection."""

import pytest
import numpy as np
from src.data_preprocessing import DataPreprocessor
from src.feature_engineering import FeatureEngineer
from src.anomaly_detection import AnomalyDetector

def test_anomaly_detector_initialization():
    """Test detector initialization."""
    detector = AnomalyDetector()
    assert detector.model is not None

def test_train_anomaly_detector():
    """Test detector training."""
    # Load data
    preprocessor = DataPreprocessor()
    preprocessor.load_data()
    baseline = preprocessor.extract_baseline()
    
    # Engineer features
    engineer = FeatureEngineer()
    baseline_engineered, feature_names = engineer.engineer_features(baseline)
    X_baseline = baseline_engineered[feature_names].values
    
    # Train
    detector = AnomalyDetector()
    detector.train(X_baseline)
    
    assert detector.model is not None

def test_predict_anomalies():
    """Test anomaly prediction."""
    # Load data
    preprocessor = DataPreprocessor()
    preprocessor.load_data()
    baseline = preprocessor.extract_baseline()
    
    # Engineer features
    engineer = FeatureEngineer()
    baseline_engineered, feature_names = engineer.engineer_features(baseline)
    X_baseline = baseline_engineered[feature_names].values
    
    data_engineered, _ = engineer.engineer_features(preprocessor.data)
    X_all = data_engineered[feature_names].values
    
    # Train and predict
    detector = AnomalyDetector()
    detector.train(X_baseline)
    predictions = detector.predict(X_all)
    
    assert len(predictions) == len(X_all)
    assert all((predictions == 1) | (predictions == -1))

def test_anomaly_scores():
    """Test anomaly score prediction."""
    # Load data
    preprocessor = DataPreprocessor()
    preprocessor.load_data()
    baseline = preprocessor.extract_baseline()
    
    # Engineer features
    engineer = FeatureEngineer()
    baseline_engineered, feature_names = engineer.engineer_features(baseline)
    X_baseline = baseline_engineered[feature_names].values
    
    data_engineered, _ = engineer.engineer_features(preprocessor.data)
    X_all = data_engineered[feature_names].values
    
    # Train and get scores
    detector = AnomalyDetector()
    detector.train(X_baseline)
    scores = detector.predict_proba(X_all)
    
    assert len(scores) == len(X_all)
    assert all((scores >= 0) & (scores <= 1))

def test_evaluate_detector():
    """Test detector evaluation."""
    # Load data
    preprocessor = DataPreprocessor()
    preprocessor.load_data()
    baseline = preprocessor.extract_baseline()
    
    # Engineer features
    engineer = FeatureEngineer()
    baseline_engineered, feature_names = engineer.engineer_features(baseline)
    X_baseline = baseline_engineered[feature_names].values
    
    data_engineered, _ = engineer.engineer_features(preprocessor.data)
    X_all = data_engineered[feature_names].values
    y_all = preprocessor.get_labels()
    
    # Train and evaluate
    detector = AnomalyDetector()
    detector.train(X_baseline)
    metrics = detector.evaluate(X_all, y_all)
    
    assert 'auc' in metrics
    assert 'precision' in metrics
    assert 'recall' in metrics
    assert metrics['auc'] > 0.9  # Should have high AUC

if __name__ == "__main__":
    pytest.main([__file__, "-v"])