"""Alert generation and recommendation system.

Combines:
- Anomaly detection scores
- RUL predictions
- SHAP explanations
- Maintenance recommendations
"""

import numpy as np
import pandas as pd
from typing import Dict, Optional, List
import logging
import yaml
from datetime import datetime

logger = logging.getLogger(__name__)

class AlertGenerator:
    """Generate actionable alerts and recommendations."""
    
    def __init__(self, config_path: str = "config/config.yaml"):
        """Initialize alert generator.
        
        Args:
            config_path: Path to YAML configuration
        """
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.logger = logging.getLogger(__name__)
    
    def _get_status(self, anomaly_score: float) -> str:
        """Determine status based on anomaly score.
        
        Args:
            anomaly_score: Score between 0 and 1
            
        Returns:
            Status string
        """
        thresholds = self.config['alert_thresholds']
        
        if anomaly_score <= thresholds['normal_max_score']:
            return 'Normal'
        elif anomaly_score <= thresholds['alert_min_score']:
            return 'Normal'  # Below alert threshold
        elif anomaly_score <= thresholds['critical_min_score']:
            return 'Alert'
        else:
            return 'Critical'
    
    def _get_condition(self, features: Dict[str, float]) -> str:
        """Identify specific anomaly type.
        
        Args:
            features: Feature values {name: value}
            
        Returns:
            Condition string
        """
        vibration = features.get('TotalVibration', 0)
        temp = features.get('Temp', 0)
        voltage = features.get('Voltage', 238)
        current = features.get('Current', 0.5)
        
        # Thresholds learned from data
        anomalies = []
        
        if vibration > 11:
            anomalies.append('High_Vibration')
        if temp > 40:
            anomalies.append('Overheating')
        if current > 0.65:
            anomalies.append('Overcurrent')
        if abs(voltage - 238) > 25:
            anomalies.append('Voltage_Anomaly')
        
        if len(anomalies) > 1:
            return 'Multiple_Anomalies'
        elif anomalies:
            return anomalies[0]
        else:
            return 'Healthy'
    
    def _get_recommendation(self, condition: str) -> str:
        """Get maintenance recommendation for condition.
        
        Args:
            condition: Anomaly condition
            
        Returns:
            Recommendation text
        """
        recommendations = self.config.get('anomalies', {})
        
        for k, v in recommendations.items():
            if isinstance(v, dict):
                if v.get('condition') == condition or k == condition.lower() or k == condition.lower().replace('-', '_'):
                    return v.get('action', 'Inspect machine')
            elif k == condition:
                return str(v)
                
        return "No action required"
    
    def generate_alert(self, 
                      features: Dict[str, float],
                      anomaly_score: float,
                      rul_days: Optional[float] = None,
                      explanation: Optional[Dict] = None) -> Dict:
        """Generate complete alert with recommendations.
        
        Args:
            features: Dictionary of sensor features {name: value}
            anomaly_score: Anomaly score (0-1)
            rul_days: Remaining useful life (optional)
            explanation: SHAP explanation (optional)
            
        Returns:
            Alert dictionary
        """
        status = self._get_status(anomaly_score)
        condition = self._get_condition(features)
        recommendation = self._get_recommendation(condition)
        
        alert = {
            'timestamp': datetime.now().isoformat(),
            'status': status,
            'condition': condition,
            'anomaly_score': float(anomaly_score),
            'features': features,
            'recommendation': recommendation,
        }
        
        # Add RUL if provided
        if rul_days is not None:
            alert['rul_days'] = float(rul_days)
            if rul_days < self.config['alert_thresholds']['rul_critical_days']:
                alert['rul_status'] = 'Critical'
            elif rul_days < self.config['alert_thresholds']['rul_warning_days']:
                alert['rul_status'] = 'Warning'
            else:
                alert['rul_status'] = 'Healthy'
        
        # Add explanation if provided
        if explanation:
            alert['explanation'] = {
                'top_factors': explanation['top_features'],
                'summary': self._create_explanation_summary(explanation),
            }
        
        return alert
    
    def _create_explanation_summary(self, explanation: Dict) -> str:
        """Create text summary of explanation.
        
        Args:
            explanation: SHAP explanation dictionary
            
        Returns:
            Text summary
        """
        lines = []
        for i, feature in enumerate(explanation['top_features'][:3], 1):
            direction = "increased" if feature['shap_value'] > 0 else "decreased"
            lines.append(f"{i}. {feature['feature']} {direction} ({feature['value']:.2f})")
        return "; ".join(lines)
    
    def generate_alerts_batch(self, 
                             features_list: List[Dict],
                             anomaly_scores: np.ndarray,
                             rul_predictions: Optional[np.ndarray] = None,
                             explanations: Optional[List[Dict]] = None) -> List[Dict]:
        """Generate alerts for multiple samples.
        
        Args:
            features_list: List of feature dictionaries
            anomaly_scores: Array of anomaly scores
            rul_predictions: Optional array of RUL predictions
            explanations: Optional list of SHAP explanations
            
        Returns:
            List of alerts
        """
        alerts = []
        
        for i, features in enumerate(features_list):
            rul = rul_predictions[i] if rul_predictions is not None else None
            exp = explanations[i] if explanations is not None else None
            
            alert = self.generate_alert(features, anomaly_scores[i], rul, exp)
            alerts.append(alert)
        
        return alerts
    
    def format_alert_for_display(self, alert: Dict) -> str:
        """Format alert for display to technician.
        
        Args:
            alert: Alert dictionary
            
        Returns:
            Formatted string
        """
        lines = []
        lines.append(f"\n{'='*60}")
        lines.append(f"ALERT - {alert['timestamp']}")
        lines.append(f"{'='*60}")
        lines.append(f"Status:      {alert['status']}")
        lines.append(f"Condition:   {alert['condition']}")
        lines.append(f"Score:       {alert['anomaly_score']:.3f}")
        
        if 'rul_days' in alert:
            lines.append(f"RUL:         {alert['rul_days']:.1f} days ({alert['rul_status']})")
        
        lines.append(f"\nRECOMMENDATION:")
        lines.append(f"{alert['recommendation']}")
        
        if 'explanation' in alert:
            lines.append(f"\nTOP CONTRIBUTING FACTORS:")
            lines.append(alert['explanation']['summary'])
        
        lines.append(f"\n{'='*60}")
        
        return "\n".join(lines)


def main():
    """Example usage."""
    logging.basicConfig(level=logging.INFO)
    
    # Example alert
    alert_gen = AlertGenerator()
    
    # Healthy sample
    features_healthy = {
        'TotalVibration': 9.67,
        'Temp': 29.5,
        'Voltage': 238.0,
        'Current': 0.501,
    }
    alert = alert_gen.generate_alert(features_healthy, anomaly_score=0.2, rul_days=20.0)
    print("\nHealthy Sample Alert:")
    print(alert_gen.format_alert_for_display(alert))
    
    # Anomalous sample
    features_anomaly = {
        'TotalVibration': 14.5,
        'Temp': 62.0,
        'Voltage': 210.0,
        'Current': 0.95,
    }
    alert = alert_gen.generate_alert(features_anomaly, anomaly_score=0.88, rul_days=2.5)
    print("\nAnomalous Sample Alert:")
    print(alert_gen.format_alert_for_display(alert))


if __name__ == "__main__":
    main()