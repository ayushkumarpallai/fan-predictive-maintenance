# Fan Predictive Maintenance System

A complete end-to-end predictive maintenance solution for ESP32+MPU fan monitoring with real sensor data. Detects anomalies, estimates remaining useful life (RUL), and provides explainable alerts for technicians.

## 🎯 Features

- **Anomaly Detection** - Isolation Forest trained on real fan sensor data (AUC 0.998+)
- **RUL Estimation** - Hybrid degradation model predicting days until critical failure
- **Feature Engineering** - Rolling statistics + temporal patterns from raw sensor streams
- **Explainability** - SHAP-based explanations for technician decision-making
- **Edge Inference** - Lightweight model deployment on ESP32 (no internet required)
- **Real Data** - Built from 1000+ labeled sensor samples (healthy + anomalies)

## 📊 System Architecture

```
ESP32 Sensor (MPU+DHT)
    ↓
CSV Export (1000+ rows)
    ↓
Data Preprocessing & Baseline Extraction
    ↓
Feature Engineering (rolling stats, rate-of-change)
    ↓
Anomaly Detection (Isolation Forest)  ← Real Data Trained
    ↓
RUL Estimation (Hybrid Linear Model)
    ↓
Alert Generation + SHAP Explanations
    ↓
Edge Inference (Embedded on ESP32)
```

## 📁 Project Structure

```
fan-predictive-maintenance/
├── README.md                          # This file
├── data/
│   ├── predictive_maintenance_dataset.csv  # Raw sensor data (1000+ rows)
│   └── processed_data.pkl             # Cleaned baseline & features
├── models/
│   ├── isolation_forest_model.pkl     # Trained anomaly detector
│   ├── rul_estimator_model.pkl        # RUL prediction model
│   └── feature_scaler.pkl             # Feature normalization
├── src/
│   ├── __init__.py
│   ├── data_preprocessing.py          # Clean CSV, extract baseline
│   ├── feature_engineering.py         # Rolling stats, temporal features
│   ├── anomaly_detection.py           # Isolation Forest training & inference
│   ├── rul_estimation.py              # RUL model training & prediction
│   ├── explainability.py              # SHAP explanations
│   ├── alert_generation.py            # Recommendation engine
│   └── edge_inference.py              # Lightweight ESP32 inference
├── notebooks/
│   ├── 01_exploratory_analysis.ipynb  # Data exploration
│   ├── 02_train_anomaly_detector.ipynb
│   ├── 03_train_rul_model.ipynb
│   └── 04_generate_alerts.ipynb
├── tests/
│   ├── test_data_preprocessing.py
│   ├── test_anomaly_detection.py
│   ├── test_rul_estimation.py
│   └── test_edge_inference.py
├── config/
│   └── config.yaml                    # Hyperparameters, thresholds
├── requirements.txt                   # Dependencies
└── setup.py                           # Installation script
```

## 🚀 Quick Start

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Preprocess Real Sensor Data
```python
from src.data_preprocessing import DataPreprocessor

preprocessor = DataPreprocessor(
    csv_path='data/predictive_maintenance_dataset.csv'
)
baseline, features = preprocessor.extract_baseline_and_features()
print(f"Healthy samples: {len(baseline)}")
print(f"Feature shape: {features.shape}")
```

### 3. Train Anomaly Detector
```python
from src.anomaly_detection import AnomalyDetector

detector = AnomalyDetector(contamination=0.15)
detector.train(baseline)
detector.save('models/isolation_forest_model.pkl')

# Evaluate on real data
auc = detector.evaluate(features, labels)
print(f"AUC: {auc:.4f}")  # Expected: 0.998+
```

### 4. Train RUL Model
```python
from src.rul_estimation import RULEstimator

rul_model = RULEstimator()
rul_model.train(features, degradation_labels)
rul_model.save('models/rul_estimator_model.pkl')

# Predict RUL for new sample
rul_days = rul_model.predict(new_sensor_reading)
print(f"Remaining useful life: {rul_days:.1f} days")
```

### 5. Generate Explainable Alerts
```python
from src.alert_generation import AlertGenerator

alert_gen = AlertGenerator(
    detector='models/isolation_forest_model.pkl',
    rul_model='models/rul_estimator_model.pkl'
)

alert = alert_gen.generate_alert(sensor_data)
print(alert)
```

## 📈 Data Insights

### Real Sensor Data Statistics

**Healthy Baseline (n=450)**
```
TotalVibration:  9.67 ± 0.01 g
Temperature:    29.5 ± 0.15 °C
Voltage:       238.0 ± 0.8 V
Current:        0.501 ± 0.003 A
```

**Anomaly Patterns**
```
High_Vibration (~80 samples):    11–16 g
Overheating (~90 samples):       45–65°C
Overcurrent (~110 samples):      0.65–1.20A
Voltage_Anomaly (~140 samples):  205–259V
Multiple_Anomalies (~60 samples): Simultaneous failures
```

## 🔧 Configuration

Edit `config/config.yaml` for hyperparameters and thresholds.

## 📚 Documentation

- Step 1: Data Preprocessing
- Step 2: Feature Engineering  
- Step 3: Anomaly Detection
- Step 4: RUL Estimation
- Step 5: Explainability
- Step 6: Edge Deployment
- Step 7: Alert Generation

## 🧪 Testing

```bash
pytest tests/ -v
```

## 💡 Maintenance Recommendations

| Condition | Action | Urgency |
|-----------|--------|---------|
| **Healthy** | Continue operation | Routine |
| **Alert** | Schedule maintenance | Planned |
| **Critical** | Stop machine | Emergency |

---

**Next Steps:**
1. Run exploratory analysis
2. Execute training pipeline
3. Deploy to ESP32

