"""Sensor stream simulator for fan telemetry.

Simulates real-time ESP32 sensor output by streaming rows from the CSV dataset
or generating synthetic sensor streams with optional anomaly injection.
"""

import time
import json
import logging
from typing import Generator, Dict, Optional
import pandas as pd
import numpy as np

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SensorStreamSimulator:
    """Stream fan sensor data sequentially or synthetically."""

    def __init__(self, csv_path: str = "data/predictive_maintenance_dataset.csv"):
        """Initialize simulator.
        
        Args:
            csv_path: Path to sensor dataset
        """
        self.csv_path = csv_path
        try:
            self.df = pd.read_csv(csv_path)
            logger.info(f"Loaded {len(self.df)} samples for telemetry streaming.")
        except Exception as e:
            logger.warning(f"Could not load dataset from {csv_path}: {e}")
            self.df = None

    def stream_dataset(self, 
                       delay_seconds: float = 1.0, 
                       loop: bool = True,
                       filter_condition: Optional[str] = None) -> Generator[Dict, None, None]:
        """Stream sensor readings sequentially from dataset.
        
        Args:
            delay_seconds: Interval between samples
            loop: Whether to loop indefinitely
            filter_condition: Optional condition filter ('Healthy', 'High_Vibration', etc.)
            
        Yields:
            Dictionary containing sensor telemetry
        """
        if self.df is None:
            raise ValueError("Dataset not loaded.")

        data = self.df
        if filter_condition:
            data = data[data['Condition'] == filter_condition]

        while True:
            for idx, row in data.iterrows():
                sample = {
                    "timestamp": pd.Timestamp.now().isoformat(),
                    "TotalVibration": float(row['TotalVibration']),
                    "Temp": float(row['Temp']),
                    "Voltage": float(row['Voltage']),
                    "Current": float(row['Current']),
                    "Status": str(row['Status']),
                    "Condition": str(row['Condition']),
                    "Recommendation": str(row['Recommendation'])
                }
                yield sample
                if delay_seconds > 0:
                    time.sleep(delay_seconds)

            if not loop:
                break

    def generate_synthetic_sample(self, fault_type: str = "Healthy") -> Dict:
        """Generate a synthetic sensor reading with realistic noise.
        
        Args:
            fault_type: One of 'Healthy', 'High_Vibration', 'Overheating', 
                        'Overcurrent', 'Voltage_Anomaly', 'Multiple_Anomalies'
                        
        Returns:
            Dictionary with sensor readings
        """
        # Baseline healthy values
        vib = np.random.normal(9.673, 0.015)
        temp = np.random.normal(29.50, 0.25)
        volt = np.random.normal(238.00, 0.80)
        curr = np.random.normal(0.501, 0.005)

        # Inject faults based on type
        if fault_type == "High_Vibration":
            vib = np.random.uniform(12.0, 16.5)
        elif fault_type == "Overheating":
            temp = np.random.uniform(45.0, 68.0)
        elif fault_type == "Overcurrent":
            curr = np.random.uniform(0.70, 1.25)
        elif fault_type == "Voltage_Anomaly":
            volt = np.random.choice([np.random.uniform(195, 212), np.random.uniform(252, 265)])
        elif fault_type == "Multiple_Anomalies":
            vib = np.random.uniform(12.5, 15.5)
            temp = np.random.uniform(50.0, 65.0)
            curr = np.random.uniform(0.75, 1.10)

        return {
            "timestamp": pd.Timestamp.now().isoformat(),
            "TotalVibration": round(float(vib), 3),
            "Temp": round(float(temp), 2),
            "Voltage": round(float(volt), 2),
            "Current": round(float(curr), 4),
            "Power": round(float(volt * curr), 2),
            "Condition": fault_type
        }


if __name__ == "__main__":
    sim = SensorStreamSimulator()
    print("Testing 3 synthetic samples:")
    for fault in ["Healthy", "High_Vibration", "Multiple_Anomalies"]:
        print(sim.generate_synthetic_sample(fault))
