"""Unit tests for edge inference."""

import pytest
import numpy as np
import tempfile
import os
from src.data_preprocessing import DataPreprocessor
from src.feature_engineering import FeatureEngineer
from src.anomaly_detection import AnomalyDetector
from src.rul_estimation import RULEstimator
from src.edge_inference import EdgeInference

def test_edge_inference_initialization():
    """Test edge inference initialization."""
    # Create temporary model files
    with tempfile.TemporaryDirectory() as tmpdir:
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
        
        # Train models
        detector = AnomalyDetector()
        detector.train(X_baseline)
        model_path = os.path.join(tmpdir, 'model.pkl')
        detector.save(model_path)
        
        # Initialize edge inference
        edge = EdgeInference(anomaly_model_path=model_path)
        assert edge is not None

def test_edge_anomaly_prediction():
    """Test edge anomaly prediction."""
    # Create temporary model files
    with tempfile.TemporaryDirectory() as tmpdir:
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
        
        # Train model
        detector = AnomalyDetector()
        detector.train(X_baseline)
        model_path = os.path.join(tmpdir, 'model.pkl')
        detector.save(model_path)
        
        # Edge inference
        edge = EdgeInference(anomaly_model_path=model_path)
        prediction, score = edge.predict_anomaly(X_all[0])
        
        assert prediction in [1, -1]
        assert 0 <= score <= 1

if __name__ == "__main__":
    pytest.main([__file__, "-v"])