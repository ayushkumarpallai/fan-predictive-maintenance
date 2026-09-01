"""Data preprocessing module for fan sensor data.

Handles:
- Loading and cleaning CSV data
- Extracting healthy baseline
- Data validation and quality checks
- Feature extraction from raw sensor readings
"""

import pandas as pd
import numpy as np
from typing import Tuple, Optional
import logging
import yaml

logger = logging.getLogger(__name__)

class DataPreprocessor:
    """Preprocess fan sensor data from CSV."""
    
    def __init__(self, config_path: str = "config/config.yaml"):
        """Initialize preprocessor with configuration.
        
        Args:
            config_path: Path to YAML configuration file
        """
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.data = None
        self.baseline = None
        self.logger = logging.getLogger(__name__)
    
    def load_data(self, csv_path: Optional[str] = None) -> pd.DataFrame:
        """Load sensor data from CSV.
        
        Args:
            csv_path: Path to CSV file (uses config if None)
            
        Returns:
            DataFrame with sensor readings
        """
        csv_path = csv_path or self.config['data']['csv_path']
        self.logger.info(f"Loading data from {csv_path}")
        
        self.data = pd.read_csv(csv_path)
        self.logger.info(f"Loaded {len(self.data)} samples with columns: {list(self.data.columns)}")
        
        return self.data
    
    def extract_baseline(self) -> pd.DataFrame:
        """Extract healthy baseline samples.
        
        Returns:
            DataFrame containing only healthy samples
        """
        if self.data is None:
            raise ValueError("Data not loaded. Call load_data() first.")
        
        health_condition = self.config['data']['baseline_health_condition']
        health_status = self.config['data']['baseline_health_status']
        
        # Filter for healthy samples
        baseline = self.data[
            (self.data['Condition'] == health_condition) & 
            (self.data['Status'] == health_status)
        ].copy()
        
        self.logger.info(f"Extracted {len(baseline)} healthy baseline samples")
        self.baseline = baseline
        
        return baseline
    
    def get_baseline_stats(self) -> dict:
        """Get statistics of healthy baseline.
        
        Returns:
            Dictionary with mean, std, min, max for each feature
        """
        if self.baseline is None:
            self.extract_baseline()
        
        features = self.config['feature_engineering']['features_to_use']
        stats = {}
        
        for feature in features:
            stats[feature] = {
                'mean': self.baseline[feature].mean(),
                'std': self.baseline[feature].std(),
                'min': self.baseline[feature].min(),
                'max': self.baseline[feature].max(),
            }
        
        return stats
    
    def validate_data(self) -> bool:
        """Validate data quality.
        
        Returns:
            True if data passes validation
        """
        if self.data is None:
            raise ValueError("Data not loaded.")
        
        # Check for missing values
        missing = self.data.isnull().sum()
        if missing.any():
            self.logger.warning(f"Missing values found: {missing[missing > 0].to_dict()}")
        
        # Check for required columns
        required = ['Timestamp', 'TotalVibration', 'Temp', 'Voltage', 'Current', 'Status', 'Condition']
        if not all(col in self.data.columns for col in required):
            raise ValueError(f"Missing required columns. Need: {required}")
        
        self.logger.info("Data validation passed.")
        return True
    
    def extract_features(self, data: Optional[pd.DataFrame] = None) -> np.ndarray:
        """Extract feature matrix from sensor data.
        
        Args:
            data: DataFrame to extract features from (uses all data if None)
            
        Returns:
            Feature matrix (n_samples, n_features)
        """
        if data is None:
            data = self.data
        
        features_to_use = self.config['feature_engineering']['features_to_use']
        X = data[features_to_use].values
        
        self.logger.info(f"Extracted {X.shape[0]} samples with {X.shape[1]} features")
        return X
    
    def get_labels(self, data: Optional[pd.DataFrame] = None) -> np.ndarray:
        """Extract labels (0=healthy, 1=anomaly) from data.
        
        Args:
            data: DataFrame to extract labels from (uses all data if None)
            
        Returns:
            Label array
        """
        if data is None:
            data = self.data
        
        # 0 = Healthy (Normal), 1 = Any anomaly
        labels = (data['Status'] != 'Normal').astype(int).values
        
        n_healthy = (labels == 0).sum()
        n_anomaly = (labels == 1).sum()
        self.logger.info(f"Labels: {n_healthy} healthy, {n_anomaly} anomalies")
        
        return labels
    
    def split_data(self, test_size: Optional[float] = None) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Split data into train/test sets.
        
        Args:
            test_size: Proportion for test set (uses config if None)
            
        Returns:
            X_train, X_test, y_train, y_test
        """
        if self.data is None:
            raise ValueError("Data not loaded.")
        
        test_size = test_size or self.config['data']['test_split']
        
        X = self.extract_features()
        y = self.get_labels()
        
        # Simple train/test split (could use StratifiedKFold for better validation)
        n = len(X)
        n_test = int(n * test_size)
        
        idx = np.arange(n)
        np.random.shuffle(idx)
        
        test_idx = idx[:n_test]
        train_idx = idx[n_test:]
        
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        
        self.logger.info(f"Train: {len(X_train)} samples, Test: {len(X_test)} samples")
        
        return X_train, X_test, y_train, y_test
    
    def save_processed_data(self, output_path: str = "data/processed_data.pkl"):
        """Save processed data to pickle.
        
        Args:
            output_path: Path to save pickle file
        """
        import joblib
        
        processed = {
            'baseline': self.baseline,
            'features': self.extract_features(),
            'labels': self.get_labels(),
            'baseline_stats': self.get_baseline_stats(),
        }
        
        joblib.dump(processed, output_path)
        self.logger.info(f"Saved processed data to {output_path}")
        
        return processed


def main():
    """Example usage of DataPreprocessor."""
    logging.basicConfig(level=logging.INFO)
    
    preprocessor = DataPreprocessor()
    preprocessor.load_data()
    preprocessor.validate_data()
    
    baseline = preprocessor.extract_baseline()
    print(f"\nBaseline statistics:")
    stats = preprocessor.get_baseline_stats()
    for feature, stat in stats.items():
        print(f"  {feature}: {stat['mean']:.3f} ± {stat['std']:.3f}")
    
    X_train, X_test, y_train, y_test = preprocessor.split_data()
    print(f"\nTrain shape: {X_train.shape}, Test shape: {X_test.shape}")
    
    preprocessor.save_processed_data()


if __name__ == "__main__":
    main()