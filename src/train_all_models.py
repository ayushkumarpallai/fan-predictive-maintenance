"""Main training pipeline.

Runs all steps:
1. Load and preprocess data
2. Engineer features
3. Train anomaly detector
4. Train RUL model
5. Evaluate and save models
"""

import logging
import sys
import os
import joblib

# Ensure src is in sys.path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data_preprocessing import DataPreprocessor
from feature_engineering import FeatureEngineer
from anomaly_detection import AnomalyDetector
from rul_estimation import RULEstimator
from alert_generation import AlertGenerator
from explainability import SHAPExplainer

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def train_all_models():
    """Train complete predictive maintenance pipeline."""
    
    logger.info("="*60)
    logger.info("PREDICTIVE MAINTENANCE PIPELINE - TRAINING")
    logger.info("="*60)
    
    # Step 1: Load and preprocess data
    logger.info("\n[Step 1/5] Loading and preprocessing data...")
    preprocessor = DataPreprocessor()
    preprocessor.load_data()
    preprocessor.validate_data()
    
    baseline = preprocessor.extract_baseline()
    logger.info(f"  ✓ Loaded {len(preprocessor.data)} total samples")
    logger.info(f"  ✓ Extracted {len(baseline)} healthy baseline samples")
    
    baseline_stats = preprocessor.get_baseline_stats()
    for feature, stats in baseline_stats.items():
        logger.info(f"    {feature}: {stats['mean']:.3f} ± {stats['std']:.3f}")
    
    # Step 2: Engineer features
    logger.info("\n[Step 2/5] Engineering features...")
    engineer = FeatureEngineer()
    
    baseline_engineered, feature_names = engineer.engineer_features(baseline)
    X_baseline = baseline_engineered[feature_names].values
    logger.info(f"  ✓ Created {len(feature_names)} engineered features")
    logger.info(f"    Features: {', '.join(feature_names[:5])}...")
    
    data_engineered, _ = engineer.engineer_features(preprocessor.data)
    X_all = data_engineered[feature_names].values
    y_all = preprocessor.get_labels()
    logger.info(f"  ✓ Feature matrix shape: {X_all.shape}")
    
    # Save processed data and feature scaler
    preprocessor.save_processed_data('data/processed_data.pkl')
    engineer.fit_scaler(X_baseline)
    joblib.dump(engineer.scaler, 'models/feature_scaler.pkl')
    logger.info("  ✓ Saved models/feature_scaler.pkl and data/processed_data.pkl")
    
    # Step 3: Train anomaly detector
    logger.info("\n[Step 3/5] Training anomaly detector...")
    detector = AnomalyDetector()
    detector.train(X_baseline)
    detector.save('models/isolation_forest_model.pkl')
    logger.info(f"  ✓ Trained Isolation Forest on {len(X_baseline)} samples")
    
    # Evaluate
    metrics = detector.evaluate(X_all, y_all)
    logger.info(f"  ✓ Evaluation Results:")
    logger.info(f"    - AUC: {metrics['auc']:.4f}")
    logger.info(f"    - Precision: {metrics['precision']:.4f}")
    logger.info(f"    - Recall: {metrics['recall']:.4f}")
    logger.info(f"    - F1: {metrics['f1']:.4f}")
    
    # Step 4: Train RUL model
    logger.info("\n[Step 4/5] Training RUL model...")
    anomaly_scores = detector.predict_proba(X_all)
    
    rul_model = RULEstimator()
    baseline_mean = X_baseline.mean(axis=0)
    rul_model.train(X_all, baseline_mean, anomaly_scores)
    rul_model.save('models/rul_estimator_model.pkl')
    logger.info(f"  ✓ Trained RUL estimator")
    
    rul_predictions = rul_model.predict(X_all)
    logger.info(f"  ✓ RUL Predictions:")
    logger.info(f"    - Mean: {rul_predictions.mean():.2f} days")
    logger.info(f"    - Min: {rul_predictions.min():.2f} days")
    logger.info(f"    - Max: {rul_predictions.max():.2f} days")
    
    # Step 5: Generate sample alerts
    logger.info("\n[Step 5/5] Generating sample alerts...")
    alert_gen = AlertGenerator()
    
    # Find anomalies
    anomaly_indices = [i for i, score in enumerate(anomaly_scores) if score > 0.7]
    
    if anomaly_indices:
        sample_idx = anomaly_indices[0]
        features_dict = {
            'TotalVibration': float(X_all[sample_idx, 0]),
            'Temp': float(X_all[sample_idx, 1]),
            'Voltage': float(X_all[sample_idx, 2]),
            'Current': float(X_all[sample_idx, 3]),
        }
        
        alert = alert_gen.generate_alert(
            features=features_dict,
            anomaly_score=float(anomaly_scores[sample_idx]),
            rul_days=float(rul_predictions[sample_idx])
        )
        
        logger.info(f"  ✓ Sample alert generated:")
        logger.info(f"    - Status: {alert['status']}")
        logger.info(f"    - Condition: {alert['condition']}")
        logger.info(f"    - RUL: {alert.get('rul_days', 'N/A')} days")
    
    logger.info("\n" + "="*60)
    logger.info("TRAINING COMPLETE ✓")
    logger.info("="*60)
    logger.info("\nModels saved:")
    logger.info("  - models/isolation_forest_model.pkl")
    logger.info("  - models/rul_estimator_model.pkl")
    logger.info("  - models/feature_scaler.pkl")
    logger.info("  - data/processed_data.pkl")
    logger.info("\nNext steps:")
    logger.info("  1. Deploy to ESP32 using edge_inference.py")
    logger.info("  2. Run inference on live sensor streams")
    logger.info("  3. Monitor alerts and maintenance recommendations")

if __name__ == "__main__":
    try:
        train_all_models()
    except Exception as e:
        logger.error(f"Training failed: {e}", exc_info=True)
        sys.exit(1)