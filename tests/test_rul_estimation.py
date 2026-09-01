"""Unit tests for RUL estimation."""

import pytest
import numpy as np
from src.data_preprocessing import DataPreprocessor
from src.feature_engineering import FeatureEngineer
from src.anomaly_detection import AnomalyDetector
from src.rul_estimation import RULEstimator

def test_rul_estimator_initialization():
    """Test RUL estimator initialization."""
    rul = RULEstimator()
    assert rul.model is None  # Not trained yet

def test_train_rul_model():
    """Test RUL model training."""
    # Load data
    preprocessor = DataPreprocessor()
    preprocessor.load_data()
    baseline = preprocessor.extract_baseline()
    
    # Engineer features
    engineer = FeatureEngineer()
    baseline_engineered, feature_names = engineer.engineer_features(baseline)
    X_baseline = baseline_engineered[feature_names].values
    baseline_mean = X_baseline.mean(axis=0)
    
    data_engineered, _ = engineer.engineer_features(preprocessor.data)
    X_all = data_engineered[feature_names].values
    
    # Get anomaly scores
    detector = AnomalyDetector()
    detector.train(X_baseline)
    anomaly_scores = detector.predict_proba(X_all)
    
    # Train RUL
    rul_model = RULEstimator()
    rul_model.train(X_all, baseline_mean, anomaly_scores)
    
    assert rul_model.model is not None
    assert rul_model.baseline_stats is not None

def test_rul_predictions():
    """Test RUL predictions."""
    # Load data
    preprocessor = DataPreprocessor()
    preprocessor.load_data()
    baseline = preprocessor.extract_baseline()
    
    # Engineer features
    engineer = FeatureEngineer()
    baseline_engineered, feature_names = engineer.engineer_features(baseline)
    X_baseline = baseline_engineered[feature_names].values
    baseline_mean = X_baseline.mean(axis=0)
    
    data_engineered, _ = engineer.engineer_features(preprocessor.data)
    X_all = data_engineered[feature_names].values
    
    # Get anomaly scores
    detector = AnomalyDetector()
    detector.train(X_baseline)
    anomaly_scores = detector.predict_proba(X_all)
    
    # Train and predict
    rul_model = RULEstimator()
    rul_model.train(X_all, baseline_mean, anomaly_scores)
    predictions = rul_model.predict(X_all)
    
    assert len(predictions) == len(X_all)
    assert all((predictions >= 0) & (predictions <= 30))

if __name__ == "__main__":
    pytest.main([__file__, "-v"])