"""Unit tests for data preprocessing."""

import pytest
import numpy as np
import pandas as pd
from src.data_preprocessing import DataPreprocessor

def test_load_data():
    """Test data loading."""
    preprocessor = DataPreprocessor()
    data = preprocessor.load_data()
    
    assert data is not None
    assert len(data) > 0
    assert 'TotalVibration' in data.columns
    assert 'Temp' in data.columns
    assert 'Voltage' in data.columns
    assert 'Current' in data.columns

def test_extract_baseline():
    """Test baseline extraction."""
    preprocessor = DataPreprocessor()
    preprocessor.load_data()
    baseline = preprocessor.extract_baseline()
    
    assert len(baseline) > 0
    assert all(baseline['Status'] == 'Normal')
    assert all(baseline['Condition'] == 'Healthy')

def test_validate_data():
    """Test data validation."""
    preprocessor = DataPreprocessor()
    preprocessor.load_data()
    
    assert preprocessor.validate_data() == True

def test_extract_features():
    """Test feature extraction."""
    preprocessor = DataPreprocessor()
    preprocessor.load_data()
    
    X = preprocessor.extract_features()
    
    assert X.shape[0] > 0
    assert X.shape[1] == 4  # 4 features
    assert not np.isnan(X).any()

def test_get_labels():
    """Test label extraction."""
    preprocessor = DataPreprocessor()
    preprocessor.load_data()
    
    y = preprocessor.get_labels()
    
    assert len(y) == len(preprocessor.data)
    assert all((y == 0) | (y == 1))

if __name__ == "__main__":
    pytest.main([__file__, "-v"])