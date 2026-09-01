"""Anomaly detection using Isolation Forest.

Trained on healthy baseline data to identify:
- High vibration
- Overheating
- Overcurrent
- Voltage anomalies
- Multiple simultaneous failures
"""

import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.metrics import roc_auc_score, confusion_matrix, classification_report
import joblib
import logging
import yaml
from typing import Optional, Tuple

logger = logging.getLogger(__name__)

class AnomalyDetector:
    """Isolation Forest based anomaly detection."""
    
    def __init__(self, config_path: str = "config/config.yaml"):
        """Initialize anomaly detector.
        
        Args:
            config_path: Path to YAML configuration
        """
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.model = None
        self.logger = logging.getLogger(__name__)
        self._init_model()
    
    def _init_model(self):
        """Initialize Isolation Forest model."""
        cfg = self.config['anomaly_detection']
        self.model = IsolationForest(
            n_estimators=cfg['n_estimators'],
            max_samples=cfg['max_samples'],
            contamination=cfg['contamination'],
            random_state=cfg['random_state'],
            verbose=cfg['verbose']
        )
    
    def train(self, X_baseline: np.ndarray) -> 'AnomalyDetector':
        """Train on healthy baseline data.
        
        Args:
            X_baseline: Feature matrix of healthy samples
            
        Returns:
            Self for chaining
        """
        self.logger.info(f"Training Isolation Forest on {len(X_baseline)} healthy samples")
        self.model.fit(X_baseline)
        self.logger.info("Training complete")
        return self
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict anomaly labels.
        
        Args:
            X: Feature matrix
            
        Returns:
            Predictions (1=normal, -1=anomaly)
        """
        return self.model.predict(X)
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Get anomaly scores (0=normal, 1=anomaly).
        
        Args:
            X: Feature matrix
            
        Returns:
            Anomaly scores between 0 and 1
        """
        # Get decision scores and normalize to [0, 1]
        scores = self.model.score_samples(X)
        # Normalize: more negative = more normal, less negative = more anomalous
        scores_normalized = 1 / (1 + np.exp(scores))  # Sigmoid normalization
        return scores_normalized
    
    def evaluate(self, X: np.ndarray, y_true: np.ndarray) -> dict:
        """Evaluate anomaly detector.
        
        Args:
            X: Feature matrix
            y_true: True labels (0=healthy, 1=anomaly)
            
        Returns:
            Dictionary with metrics
        """
        y_pred = self.predict(X)
        y_scores = self.predict_proba(X)
        
        # Convert predictions to binary (1=normal, -1=anomaly) -> (0=normal, 1=anomaly)
        y_pred_binary = (y_pred == -1).astype(int)
        
        # Calculate AUC
        auc = roc_auc_score(y_true, y_scores)
        
        # Confusion matrix
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred_binary).ravel()
        
        metrics = {
            'auc': auc,
            'tp': int(tp),
            'tn': int(tn),
            'fp': int(fp),
            'fn': int(fn),
            'precision': tp / (tp + fp) if (tp + fp) > 0 else 0,
            'recall': tp / (tp + fn) if (tp + fn) > 0 else 0,
            'f1': 2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) > 0 else 0,
        }
        
        self.logger.info(f"Evaluation results: AUC={metrics['auc']:.4f}, Precision={metrics['precision']:.4f}, Recall={metrics['recall']:.4f}")
        
        return metrics
    
    def save(self, path: str):
        """Save model to disk.
        
        Args:
            path: Path to save model
        """
        joblib.dump(self.model, path)
        self.logger.info(f"Model saved to {path}")
    
    def load(self, path: str) -> 'AnomalyDetector':
        """Load model from disk.
        
        Args:
            path: Path to model file
            
        Returns:
            Self for chaining
        """
        self.model = joblib.load(path)
        self.logger.info(f"Model loaded from {path}")
        return self


def main():
    """Example usage."""
    logging.basicConfig(level=logging.INFO)
    
    from data_preprocessing import DataPreprocessor
    from feature_engineering import FeatureEngineer
    
    # Load and preprocess data
    preprocessor = DataPreprocessor()
    preprocessor.load_data()
    preprocessor.validate_data()
    baseline = preprocessor.extract_baseline()
    
    # Engineer features
    engineer = FeatureEngineer()
    baseline_engineered, feature_names = engineer.engineer_features(baseline)
    X_baseline = baseline_engineered[feature_names].values
    
    # Train anomaly detector
    detector = AnomalyDetector()
    detector.train(X_baseline)
    
    # Get all data features for evaluation
    data_engineered, _ = engineer.engineer_features(preprocessor.data)
    X_all = data_engineered[feature_names].values
    y_all = preprocessor.get_labels()
    
    # Evaluate
    metrics = detector.evaluate(X_all, y_all)
    print(f"\nAnomalcy Detection Results:")
    for key, value in metrics.items():
        print(f"  {key}: {value}")
    
    # Save model
    detector.save('models/isolation_forest_model.pkl')


if __name__ == "__main__":
    main()