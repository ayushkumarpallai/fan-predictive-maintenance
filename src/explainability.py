"""SHAP-based explainability for anomaly detection.

Generates human-readable explanations for:
- Why a sample was flagged as anomaly
- Which features contributed most to the decision
- Force plots and summary plots for visualization
"""

import numpy as np
import shap
from sklearn.ensemble import IsolationForest
import logging
from typing import Dict, List, Tuple

logger = logging.getLogger(__name__)

class SHAPExplainer:
    """Generate SHAP explanations for anomaly detection."""
    
    def __init__(self, model: IsolationForest, X_background: np.ndarray, feature_names: List[str]):
        """Initialize SHAP explainer.
        
        Args:
            model: Trained Isolation Forest model
            X_background: Background data for SHAP (typically training data)
            feature_names: Names of features
        """
        self.model = model
        self.X_background = X_background
        self.feature_names = feature_names
        self.logger = logging.getLogger(__name__)
        
        # Initialize TreeExplainer for Isolation Forest
        self.explainer = shap.TreeExplainer(model)
    
    def explain_prediction(self, sample: np.ndarray) -> Dict:
        """Generate explanation for a single prediction.
        
        Args:
            sample: Feature vector (1D array)
            
        Returns:
            Dictionary with explanation
        """
        sample = sample.reshape(1, -1)
        
        # Get SHAP values
        shap_values = self.explainer.shap_values(sample)
        base_value = self.explainer.expected_value
        
        # Get prediction
        prediction = self.model.predict(sample)[0]  # 1 = normal, -1 = anomaly
        anomaly_score = 1 / (1 + np.exp(self.model.score_samples(sample)[0]))
        
        # Create feature importance ranking
        feature_importance = []
        for i, feature_name in enumerate(self.feature_names):
            importance = abs(shap_values[0][i])
            contribution = shap_values[0][i]
            feature_importance.append({
                'feature': feature_name,
                'shap_value': float(contribution),
                'abs_shap_value': float(importance),
                'value': float(sample[0][i]),
            })
        
        # Sort by absolute SHAP value
        feature_importance.sort(key=lambda x: x['abs_shap_value'], reverse=True)
        
        explanation = {
            'prediction': 'Anomaly' if prediction == -1 else 'Normal',
            'anomaly_score': float(anomaly_score),
            'base_value': float(base_value),
            'top_features': feature_importance[:5],  # Top 5 contributors
            'all_features': feature_importance,
        }
        
        return explanation
    
    def explain_batch(self, X: np.ndarray, indices: List[int] = None) -> List[Dict]:
        """Generate explanations for multiple samples.
        
        Args:
            X: Feature matrix
            indices: Specific indices to explain (all if None)
            
        Returns:
            List of explanations
        """
        if indices is None:
            indices = range(len(X))
        
        explanations = []
        for idx in indices:
            try:
                exp = self.explain_prediction(X[idx])
                explanations.append(exp)
            except Exception as e:
                self.logger.error(f"Error explaining sample {idx}: {e}")
        
        return explanations
    
    def get_feature_importance_summary(self, X: np.ndarray) -> Dict[str, float]:
        """Get global feature importance from SHAP values.
        
        Args:
            X: Feature matrix
            
        Returns:
            Dictionary of feature importance scores
        """
        shap_values = self.explainer.shap_values(X)
        
        # Mean absolute SHAP values
        importance = np.abs(shap_values).mean(axis=0)
        
        importance_dict = {}
        for i, name in enumerate(self.feature_names):
            importance_dict[name] = float(importance[i])
        
        return importance_dict


def create_feature_summary_text(explanation: Dict) -> str:
    """Convert SHAP explanation to readable text.
    
    Args:
        explanation: SHAP explanation dictionary
        
    Returns:
        Human-readable explanation text
    """
    lines = []
    lines.append(f"Status: {explanation['prediction']}")
    lines.append(f"Anomaly Score: {explanation['anomaly_score']:.3f}")
    lines.append(f"\nTop Contributing Factors:")
    
    for i, feature in enumerate(explanation['top_features'], 1):
        direction = "↑" if feature['shap_value'] > 0 else "↓"
        lines.append(f"  {i}. {feature['feature']}: {direction} {abs(feature['shap_value']):.4f} (value: {feature['value']:.3f})")
    
    return "\n".join(lines)


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
    
    # Get all data
    data_engineered, _ = engineer.engineer_features(preprocessor.data)
    X_all = data_engineered[feature_names].values
    
    # Train anomaly detector
    detector = AnomalyDetector()
    detector.train(X_baseline)
    
    # Create SHAP explainer
    explainer = SHAPExplainer(detector.model, X_baseline, feature_names)
    
    # Explain some anomalies
    y_all = preprocessor.get_labels()
    anomaly_indices = np.where(y_all == 1)[0][:5]  # First 5 anomalies
    
    print("\nAnomalies and their explanations:")
    for idx in anomaly_indices:
        exp = explainer.explain_prediction(X_all[idx])
        print(f"\nSample {idx}:")
        print(create_feature_summary_text(exp))


if __name__ == "__main__":
    main()