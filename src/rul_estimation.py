"""RUL (Remaining Useful Life) estimation model.

Hybrid approach:
- Uses anomaly severity as primary RUL indicator
- Combines with feature degradation rates
- Predicts days until critical failure
"""

import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
import joblib
import logging
import yaml
from typing import Optional, Tuple

logger = logging.getLogger(__name__)

class RULEstimator:
    """Estimate remaining useful life of equipment."""
    
    def __init__(self, config_path: str = "config/config.yaml"):
        """Initialize RUL estimator.
        
        Args:
            config_path: Path to YAML configuration
        """
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.model = None
        self.scaler = StandardScaler()
        self.logger = logging.getLogger(__name__)
        
        # Baseline statistics for degradation calculation
        self.baseline_stats = None
    
    def create_degradation_features(self, X: np.ndarray, baseline_mean: np.ndarray) -> np.ndarray:
        """Create degradation features (deviation from baseline).
        
        Args:
            X: Feature matrix
            baseline_mean: Mean values of healthy baseline
            
        Returns:
            Degradation feature matrix
        """
        # Absolute deviation from baseline
        degradation = np.abs(X - baseline_mean)
        return degradation
    
    def train(self, X: np.ndarray, baseline_mean: np.ndarray, 
              degradation_scores: Optional[np.ndarray] = None) -> 'RULEstimator':
        """Train RUL model.
        
        Args:
            X: Feature matrix from all data
            baseline_mean: Mean features of healthy baseline
            degradation_scores: Optional anomaly scores (0-1) as RUL proxy
            
        Returns:
            Self for chaining
        """
        self.baseline_stats = baseline_mean
        
        # Create degradation features
        X_degradation = self.create_degradation_features(X, baseline_mean)
        
        # If degradation_scores provided, use as RUL target
        if degradation_scores is not None:
            # Map anomaly score to RUL (higher anomaly = lower RUL)
            # Assume: score 0 = 30 days RUL, score 1 = 0 days RUL
            y_rul = 30 * (1 - degradation_scores)
        else:
            # Default: use simple degradation magnitude
            y_rul = 30 / (1 + np.sum(X_degradation, axis=1) / X_degradation.shape[1])
        
        # Normalize features
        X_scaled = self.scaler.fit_transform(X_degradation)
        
        # Train linear model
        self.model = LinearRegression()
        self.model.fit(X_scaled, y_rul)
        
        self.logger.info(f"RUL model trained. R² score: {self.model.score(X_scaled, y_rul):.4f}")
        return self
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict RUL in days.
        
        Args:
            X: Feature matrix
            
        Returns:
            RUL predictions (days)
        """
        if self.model is None or self.baseline_stats is None:
            raise ValueError("Model not trained. Call train() first.")
        
        # Create degradation features
        X_degradation = self.create_degradation_features(X, self.baseline_stats)
        
        # Scale
        X_scaled = self.scaler.transform(X_degradation)
        
        # Predict
        rul = self.model.predict(X_scaled)
        
        # Clamp to [0, 30] days
        rul = np.clip(rul, 0, 30)
        
        return rul
    
    def save(self, path: str):
        """Save model to disk.
        
        Args:
            path: Path to save model
        """
        model_dict = {
            'model': self.model,
            'scaler': self.scaler,
            'baseline_stats': self.baseline_stats,
        }
        joblib.dump(model_dict, path)
        self.logger.info(f"RUL model saved to {path}")
    
    def load(self, path: str) -> 'RULEstimator':
        """Load model from disk.
        
        Args:
            path: Path to model file
            
        Returns:
            Self for chaining
        """
        model_dict = joblib.load(path)
        self.model = model_dict['model']
        self.scaler = model_dict['scaler']
        self.baseline_stats = model_dict['baseline_stats']
        self.logger.info(f"RUL model loaded from {path}")
        return self


def main():
    """Example usage."""
    logging.basicConfig(level=logging.INFO)
    
    from data_preprocessing import DataPreprocessor
    from feature_engineering import FeatureEngineer
    from anomaly_detection import AnomalyDetector
    
    # Load and preprocess
    preprocessor = DataPreprocessor()
    preprocessor.load_data()
    baseline = preprocessor.extract_baseline()
    
    # Engineer features
    engineer = FeatureEngineer()
    baseline_engineered, feature_names = engineer.engineer_features(baseline)
    X_baseline = baseline_engineered[feature_names].values
    baseline_mean = X_baseline.mean(axis=0)
    
    # Get all data
    data_engineered, _ = engineer.engineer_features(preprocessor.data)
    X_all = data_engineered[feature_names].values
    
    # Get anomaly scores
    detector = AnomalyDetector()
    detector.train(X_baseline)
    anomaly_scores = detector.predict_proba(X_all)
    
    # Train RUL model
    rul_model = RULEstimator()
    rul_model.train(X_all, baseline_mean, anomaly_scores)
    
    # Predict
    rul_predictions = rul_model.predict(X_all)
    print(f"\nRUL Predictions (days):")
    print(f"  Mean: {rul_predictions.mean():.2f}")
    print(f"  Min: {rul_predictions.min():.2f}")
    print(f"  Max: {rul_predictions.max():.2f}")
    
    # Save
    rul_model.save('models/rul_estimator_model.pkl')


if __name__ == "__main__":
    main()