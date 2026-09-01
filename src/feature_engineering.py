"""Feature engineering module for temporal and statistical features.

Creates:
- Rolling mean/std of sensor readings
- Rate-of-change features
- Power consumption features (Voltage × Current)
- Normalized and scaled features
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from typing import Tuple
import logging
import yaml

logger = logging.getLogger(__name__)

class FeatureEngineer:
    """Create advanced features from raw sensor data."""
    
    def __init__(self, config_path: str = "config/config.yaml"):
        """Initialize feature engineer.
        
        Args:
            config_path: Path to YAML configuration
        """
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.scaler = StandardScaler()
        self.logger = logging.getLogger(__name__)
    
    def create_rolling_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Create rolling mean and std features.
        
        Args:
            data: DataFrame with sensor readings
            
        Returns:
            DataFrame with added rolling features
        """
        features = self.config['feature_engineering']['features_to_use']
        window = self.config['feature_engineering']['rolling_window']
        
        result = data.copy()
        
        if self.config['feature_engineering']['create_rolling_mean']:
            for feature in features:
                result[f'{feature}_roll_mean'] = result[feature].rolling(window=window, min_periods=1).mean()
        
        if self.config['feature_engineering']['create_rolling_std']:
            for feature in features:
                result[f'{feature}_roll_std'] = result[feature].rolling(window=window, min_periods=1).std().fillna(0)
        
        self.logger.info(f"Created rolling features (window={window})")
        return result
    
    def create_rate_of_change(self, data: pd.DataFrame) -> pd.DataFrame:
        """Create rate-of-change features.
        
        Args:
            data: DataFrame with sensor readings
            
        Returns:
            DataFrame with added rate-of-change features
        """
        features = self.config['feature_engineering']['features_to_use']
        
        result = data.copy()
        
        if self.config['feature_engineering']['create_rate_of_change']:
            for feature in features:
                result[f'{feature}_delta'] = result[feature].diff().fillna(0)
        
        self.logger.info("Created rate-of-change features")
        return result
    
    def create_power_feature(self, data: pd.DataFrame) -> pd.DataFrame:
        """Create power consumption feature (Voltage × Current).
        
        Args:
            data: DataFrame with Voltage and Current columns
            
        Returns:
            DataFrame with added Power feature
        """
        result = data.copy()
        
        if self.config['feature_engineering']['create_power_feature']:
            result['Power'] = result['Voltage'] * result['Current']
            self.logger.info("Created Power feature (Voltage × Current)")
        
        return result
    
    def engineer_features(self, data: pd.DataFrame) -> Tuple[pd.DataFrame, list]:
        """Create all engineered features.
        
        Args:
            data: Raw sensor data
            
        Returns:
            DataFrame with engineered features, list of feature names
        """
        # Create rolling features
        data = self.create_rolling_features(data)
        
        # Create rate-of-change
        data = self.create_rate_of_change(data)
        
        # Create power
        data = self.create_power_feature(data)
        
        # Determine final features
        if self.config['feature_engineering']['drop_original_features']:
            features = [col for col in data.columns if col not in 
                       ['Timestamp', 'Status', 'Condition', 'Recommendation']]
        else:
            features = [col for col in data.columns if col not in 
                       ['Timestamp', 'Status', 'Condition', 'Recommendation']]
        
        self.logger.info(f"Total engineered features: {len(features)}")
        return data, features
    
    def fit_scaler(self, X: np.ndarray) -> 'FeatureEngineer':
        """Fit feature scaler on training data.
        
        Args:
            X: Training feature matrix
            
        Returns:
            Self for chaining
        """
        self.scaler.fit(X)
        self.logger.info("Scaler fitted on training data")
        return self
    
    def scale_features(self, X: np.ndarray) -> np.ndarray:
        """Scale features using fitted scaler.
        
        Args:
            X: Feature matrix to scale
            
        Returns:
            Scaled feature matrix
        """
        return self.scaler.transform(X)
    
    def scale_features_fit_transform(self, X: np.ndarray) -> np.ndarray:
        """Fit and transform in one step.
        
        Args:
            X: Feature matrix
            
        Returns:
            Scaled feature matrix
        """
        return self.scaler.fit_transform(X)


def main():
    """Example usage."""
    logging.basicConfig(level=logging.INFO)
    
    from data_preprocessing import DataPreprocessor
    
    # Load data
    preprocessor = DataPreprocessor()
    preprocessor.load_data()
    data = preprocessor.data
    
    # Engineer features
    engineer = FeatureEngineer()
    data_engineered, feature_names = engineer.engineer_features(data)
    
    print(f"\nOriginal features: {list(preprocessor.config['feature_engineering']['features_to_use'])}")
    print(f"\nEngineered features ({len(feature_names)}):")
    print(feature_names)
    
    X = data_engineered[feature_names].values
    print(f"\nFeature matrix shape: {X.shape}")


if __name__ == "__main__":
    main()