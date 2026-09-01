# ESP32 Fan Predictive Maintenance Firmware

Autonomous edge monitoring and on-device anomaly detection firmware for industrial fans.

---

## 📌 Hardware Pinout & Wiring

| Component | ESP32 GPIO | Description |
|-----------|------------|-------------|
| **MPU6050 SDA** | `GPIO 21` | I2C Data line for vibration sensor |
| **MPU6050 SCL** | `GPIO 22` | I2C Clock line for vibration sensor |
| **MPU6050 VCC/GND** | `3.3V / GND` | Power supply |
| **DHT22 Data** | `GPIO 4` | Digital 1-wire temperature sensor |
| **ACS712 OUT** | `GPIO 35` (ADC1_CH7) | Analog current sensor output |
| **ZMPT101B OUT** | `GPIO 34` (ADC1_CH6) | Analog voltage sensor output |
| **Green LED** | `GPIO 18` | Status: Normal / Healthy |
| **Yellow LED** | `GPIO 19` | Status: Alert / Warning |
| **Red LED & Buzzer** | `GPIO 23` | Status: Critical Fault |

---

## 🛠️ Arduino IDE Setup

1. **Install ESP32 Board Support**:
   - In Arduino IDE, go to `File` > `Preferences` > `Additional Boards Manager URLs`.
   - Add: `https://raw.githubusercontent.com/espressif/arduino-esp32/gh-pages/package_esp32_index.json`.
   - Install `esp32 by Espressif Systems` via Tools > Board Manager.

2. **Required Libraries** (Install via Library Manager):
   - `DHT sensor library` by Adafruit
   - `Adafruit Unified Sensor` by Adafruit
   - `PubSubClient` by Nick O'Leary (for MQTT)

3. **Compilation & Upload**:
   - Board: `ESP32 Dev Module`
   - Upload Speed: `921600` / `115200`
   - Flash Frequency: `80MHz`
   - Select the appropriate COM Port and click **Upload**.

---

## ⚙️ Edge Features
- **Offline Inference**: Executes embedded C algorithm in [`edge_model.h`](edge_model.h) in under $1\text{ms}$ with zero dynamic allocations.
- **Sensor Fusion**: Computes RMS acceleration vector magnitude ($\sqrt{a_x^2 + a_y^2 + a_z^2}$) and active power ($V \times I$).
- **Multi-protocol Output**: Simultaneous 115200 Baud Serial monitoring and JSON MQTT telemetry over WiFi.
