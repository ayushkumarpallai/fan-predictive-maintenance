"""Edge inference for ESP32 deployment.

Optimized for:
- Low memory footprint
- Fast inference (<5ms)
- Offline operation
- Real-time predictions
"""

import numpy as np
import joblib
from typing import Tuple, Optional
import logging

logger = logging.getLogger(__name__)

class EdgeInference:
    """Lightweight inference for edge devices (ESP32)."""
    
    def __init__(self,
                 anomaly_model_path: str = "models/isolation_forest_model.pkl",
                 rul_model_path: Optional[str] = None,
                 scaler_path: Optional[str] = None):
        """Initialize edge inference.
        
        Args:
            anomaly_model_path: Path to anomaly detection model
            rul_model_path: Path to RUL model (optional)
            scaler_path: Path to feature scaler (optional)
        """
        self.logger = logging.getLogger(__name__)
        
        # Load anomaly model
        self.anomaly_model = joblib.load(anomaly_model_path)
        self.logger.info(f"Loaded anomaly model from {anomaly_model_path}")
        
        # Load RUL model if provided
        self.rul_model = None
        if rul_model_path:
            try:
                rul_dict = joblib.load(rul_model_path)
                self.rul_model = rul_dict.get('model')
                self.scaler = rul_dict.get('scaler')
                self.baseline_stats = rul_dict.get('baseline_stats')
                self.logger.info(f"Loaded RUL model from {rul_model_path}")
            except Exception as e:
                self.logger.warning(f"Could not load RUL model: {e}")
        
        self.inference_count = 0
    
    def predict_anomaly(self, features: np.ndarray) -> Tuple[int, float]:
        """Predict anomaly (fast inference).
        
        Args:
            features: Feature vector (1D array)
            
        Returns:
            (prediction, anomaly_score) where prediction is 1 (normal) or -1 (anomaly)
        """
        # Ensure 2D
        if features.ndim == 1:
            features = features.reshape(1, -1)
        
        # Predict
        prediction = self.anomaly_model.predict(features)[0]
        
        # Get anomaly score
        score = self.anomaly_model.score_samples(features)[0]
        anomaly_score = 1 / (1 + np.exp(score))  # Sigmoid normalization
        
        self.inference_count += 1
        
        return int(prediction), float(anomaly_score)
    
    def predict_rul(self, features: np.ndarray) -> Optional[float]:
        """Predict RUL (if model available).
        
        Args:
            features: Feature vector (1D array)
            
        Returns:
            RUL in days, or None if model not available
        """
        if self.rul_model is None:
            return None
        
        # Ensure 2D
        if features.ndim == 1:
            features = features.reshape(1, -1)
        
        # Create degradation features
        degradation = np.abs(features - self.baseline_stats)
        
        # Scale
        degradation_scaled = self.scaler.transform(degradation)
        
        # Predict
        rul = self.rul_model.predict(degradation_scaled)[0]
        rul = np.clip(rul, 0, 30)  # Clamp to [0, 30]
        
        return float(rul)
    
    def predict(self, features: np.ndarray) -> Tuple[float, Optional[float]]:
        """Complete inference: anomaly score + RUL.
        
        Args:
            features: Feature vector (1D array)
            
        Returns:
            (anomaly_score, rul_days)
        """
        _, anomaly_score = self.predict_anomaly(features)
        rul = self.predict_rul(features)
        
        return anomaly_score, rul
    
    def batch_predict(self, features_list: list) -> list:
        """Predict for multiple samples (for testing).
        
        Args:
            features_list: List of feature vectors
            
        Returns:
            List of (anomaly_score, rul) tuples
        """
        results = []
        for features in features_list:
            anomaly_score, rul = self.predict(np.array(features))
            results.append((anomaly_score, rul))
        
        return results
    
    def get_stats(self) -> dict:
        """Get inference statistics.
        
        Returns:
            Dictionary with inference stats
        """
        return {
            'total_inferences': self.inference_count,
            'has_rul_model': self.rul_model is not None,
        }


class EdgeInferenceBuffer:
    """Buffered inference for ESP32 with rolling statistics."""
    
    def __init__(self, buffer_size: int = 10):
        """Initialize buffer.
        
        Args:
            buffer_size: Number of samples to keep in buffer
        """
        self.buffer_size = buffer_size
        self.buffer = []
        self.logger = logging.getLogger(__name__)
    
    def add_sample(self, features: np.ndarray) -> Optional[Tuple[float, Optional[float]]]:
        """Add sample to buffer and predict when full.
        
        Args:
            features: Feature vector
            
        Returns:
            Prediction tuple if buffer is full, None otherwise
        """
        self.buffer.append(features)
        
        if len(self.buffer) >= self.buffer_size:
            # Compute rolling statistics
            buffer_array = np.array(self.buffer)
            rolling_mean = buffer_array.mean(axis=0)
            rolling_std = buffer_array.std(axis=0)
            
            # Combine for prediction
            combined_features = np.concatenate([rolling_mean, rolling_std])
            
            # Clear buffer
            self.buffer = []
            
            return combined_features
        
        return None


def main():
    """Example usage."""
    logging.basicConfig(level=logging.INFO)
    
    from data_preprocessing import DataPreprocessor
    from feature_engineering import FeatureEngineer
    from anomaly_detection import AnomalyDetector
    from rul_estimation import RULEstimator
    
    # Prepare data and models
    preprocessor = DataPreprocessor()
    preprocessor.load_data()
    baseline = preprocessor.extract_baseline()
    
    engineer = FeatureEngineer()
    baseline_engineered, feature_names = engineer.engineer_features(baseline)
    X_baseline = baseline_engineered[feature_names].values
    baseline_mean = X_baseline.mean(axis=0)
    
    data_engineered, _ = engineer.engineer_features(preprocessor.data)
    X_all = data_engineered[feature_names].values
    
    # Train and save models
    detector = AnomalyDetector()
    detector.train(X_baseline)
    detector.save('models/isolation_forest_model.pkl')
    
    anomaly_scores = detector.predict_proba(X_all)
    rul_model = RULEstimator()
    rul_model.train(X_all, baseline_mean, anomaly_scores)
    rul_model.save('models/rul_estimator_model.pkl')
    
    # Test edge inference
    print("\nTesting Edge Inference:")
    edge = EdgeInference(
        anomaly_model_path='models/isolation_forest_model.pkl',
        rul_model_path='models/rul_estimator_model.pkl'
    )
    
    # Test on a few samples
    test_indices = [0, 100, 200]
    for idx in test_indices:
        anomaly_score, rul = edge.predict(X_all[idx])
        print(f"Sample {idx}: Anomaly={anomaly_score:.3f}, RUL={rul:.1f} days")
    
    print(f"\nInference Stats: {edge.get_stats()}")


if __name__ == "__main__":
    main()