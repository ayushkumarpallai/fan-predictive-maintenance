/**
 * @file esp32_fan_monitor.ino
 * @brief Complete ESP32 Firmware for Fan Predictive Maintenance
 * 
 * Hardware Requirements:
 * - ESP32 Dev Module (WROOM-32)
 * - MPU6050 Accelerometer / Gyroscope (I2C: SDA=GPIO21, SCL=GPIO22)
 * - DHT22 / DHT11 Temperature & Humidity (GPIO4)
 * - ACS712 Current Sensor (Analog GPIO35)
 * - ZMPT101B / AC Voltage Divider (Analog GPIO34)
 * - Status LEDs:
 *   - Green LED (GPIO18): Normal Status
 *   - Yellow LED (GPIO19): Alert Status
 *   - Red LED & Buzzer (GPIO23): Critical Status
 * 
 * Communication:
 * - Serial output (115200 baud) for local telemetry
 * - WiFi + MQTT telemetry publication (JSON payloads)
 * - Autonomous On-Device Edge Inference via edge_model.h
 */

#include <WiFi.h>
#include <PubSubClient.h>
#include <Wire.h>
#include <DHT.h>
#include "edge_model.h"

// -------------------------------------------------------------
// WiFi & MQTT Configuration (Update with your credentials)
// -------------------------------------------------------------
const char* WIFI_SSID       = "YOUR_WIFI_SSID";
const char* WIFI_PASSWORD   = "YOUR_WIFI_PASSWORD";
const char* MQTT_SERVER     = "broker.hivemq.com"; // Default public broker
const int   MQTT_PORT       = 1883;
const char* MQTT_TOPIC_PUB  = "industry/fan/telemetry";
const char* MQTT_TOPIC_ALERT= "industry/fan/alerts";

// -------------------------------------------------------------
// Pin Definitions
// -------------------------------------------------------------
#define I2C_SDA_PIN         21
#define I2C_SCL_PIN         22
#define DHT_PIN             4
#define DHT_TYPE            DHT22

#define CURRENT_ADC_PIN     35
#define VOLTAGE_ADC_PIN     34

#define LED_NORMAL_PIN      18
#define LED_ALERT_PIN       19
#define LED_CRITICAL_PIN    23

// -------------------------------------------------------------
// MPU6050 I2C Register Constants
// -------------------------------------------------------------
#define MPU6050_ADDR        0x68
#define MPU6050_PWR_MGMT_1  0x6B
#define MPU6050_ACCEL_XOUT_H 0x3B

// -------------------------------------------------------------
// Sampling Interval
// -------------------------------------------------------------
const unsigned long SAMPLING_INTERVAL_MS = 15000; // 15 seconds
unsigned long lastSampleTime = 0;

// Global Objects
WiFiClient espClient;
PubSubClient mqttClient(espClient);
DHT dht(DHT_PIN, DHT_TYPE);
SensorBuffer sensorBuffer;

// -------------------------------------------------------------
// Helper: Initialize MPU6050
// -------------------------------------------------------------
bool initMPU6050() {
    Wire.begin(I2C_SDA_PIN, I2C_SCL_PIN, 400000);
    Wire.beginTransmission(MPU6050_ADDR);
    Wire.write(MPU6050_PWR_MGMT_1);
    Wire.write(0); // Wake up MPU6050
    byte error = Wire.endTransmission();
    if (error == 0) {
        Serial.println("[MPU6050] Initialized successfully.");
        return true;
    } else {
        Serial.printf("[MPU6050] Init failed with error code: %d\n", error);
        return false;
    }
}

// -------------------------------------------------------------
// Helper: Read Acceleration and Compute Total Vibration (g)
// -------------------------------------------------------------
float readTotalVibration() {
    Wire.beginTransmission(MPU6050_ADDR);
    Wire.write(MPU6050_ACCEL_XOUT_H);
    if (Wire.endTransmission(false) != 0) {
        return 9.673f; // Default baseline on communication error
    }

    Wire.requestFrom(MPU6050_ADDR, 6, true);
    if (Wire.available() < 6) return 9.673f;

    int16_t ax_raw = (Wire.read() << 8) | Wire.read();
    int16_t ay_raw = (Wire.read() << 8) | Wire.read();
    int16_t az_raw = (Wire.read() << 8) | Wire.read();

    // Scale factor for +/- 2g range: 16384 LSB/g
    // Raw acceleration in m/s^2 equivalent (or g-magnitude scaled ~ 9.67 m/s^2)
    float ax = (float)ax_raw / 1638.4f;
    float ay = (float)ay_raw / 1638.4f;
    float az = (float)az_raw / 1638.4f;

    float totalVib = sqrtf(ax * ax + ay * ay + az * az);
    return totalVib;
}

// -------------------------------------------------------------
// Helper: Read Analog Voltage & Current
// -------------------------------------------------------------
float readVoltage() {
    int raw = analogRead(VOLTAGE_ADC_PIN);
    // ADC 12-bit (0-4095) with calibration constant
    // Map to 220V-240V AC RMS baseline
    float voltage = 238.0f + ((float)raw - 2048.0f) * 0.02f;
    return voltage;
}

float readCurrent() {
    int raw = analogRead(CURRENT_ADC_PIN);
    // ACS712 calibration mapping to ~0.50A baseline
    float current = 0.501f + ((float)raw - 2048.0f) * 0.0005f;
    if (current < 0.0f) current = 0.0f;
    return current;
}

// -------------------------------------------------------------
// Update Hardware Indicator LEDs
// -------------------------------------------------------------
void updateIndicators(SystemStatus status) {
    digitalWrite(LED_NORMAL_PIN, LOW);
    digitalWrite(LED_ALERT_PIN, LOW);
    digitalWrite(LED_CRITICAL_PIN, LOW);

    switch (status) {
        case STATUS_NORMAL:
            digitalWrite(LED_NORMAL_PIN, HIGH);
            break;
        case STATUS_ALERT:
            digitalWrite(LED_ALERT_PIN, HIGH);
            break;
        case STATUS_CRITICAL:
            digitalWrite(LED_CRITICAL_PIN, HIGH);
            break;
    }
}

// -------------------------------------------------------------
// Network Management
// -------------------------------------------------------------
void setupWiFi() {
    Serial.printf("[WiFi] Connecting to %s", WIFI_SSID);
    WiFi.mode(WIFI_STA);
    WiFi.begin(WIFI_SSID, WIFI_PASSWORD);

    int attempts = 0;
    while (WiFi.status() != WL_CONNECTED && attempts < 15) {
        delay(500);
        Serial.print(".");
        attempts++;
    }

    if (WiFi.status() == WL_CONNECTED) {
        Serial.printf("\n[WiFi] Connected! IP: %s\n", WiFi.localIP().toString().c_str());
    } else {
        Serial.println("\n[WiFi] Connection timed out. Operating in Offline Edge Mode.");
    }
}

void reconnectMQTT() {
    if (WiFi.status() != WL_CONNECTED) return;

    if (!mqttClient.connected()) {
        Serial.print("[MQTT] Connecting to broker...");
        String clientId = "ESP32_FanMonitor_" + String(random(0xffff), HEX);
        if (mqttClient.connect(clientId.c_str())) {
            Serial.println(" connected!");
        } else {
            Serial.printf(" failed (state: %d)\n", mqttClient.state());
        }
    }
}

// -------------------------------------------------------------
// Arduino Setup
// -------------------------------------------------------------
void setup() {
    Serial.begin(115200);
    delay(1000);
    Serial.println("\n==========================================");
    Serial.println(" ESP32 Fan Predictive Maintenance System ");
    Serial.println("==========================================");

    // Configure Pin Modes
    pinMode(LED_NORMAL_PIN, OUTPUT);
    pinMode(LED_ALERT_PIN, OUTPUT);
    pinMode(LED_CRITICAL_PIN, OUTPUT);
    pinMode(CURRENT_ADC_PIN, INPUT);
    pinMode(VOLTAGE_ADC_PIN, INPUT);

    // Initial LED Self-Test
    digitalWrite(LED_NORMAL_PIN, HIGH);
    digitalWrite(LED_ALERT_PIN, HIGH);
    digitalWrite(LED_CRITICAL_PIN, HIGH);
    delay(1000);
    digitalWrite(LED_NORMAL_PIN, LOW);
    digitalWrite(LED_ALERT_PIN, LOW);
    digitalWrite(LED_CRITICAL_PIN, LOW);

    // Initialize Peripherals
    dht.begin();
    initMPU6050();
    init_sensor_buffer(&sensorBuffer);

    // Network Setup
    setupWiFi();
    mqttClient.setServer(MQTT_SERVER, MQTT_PORT);
}

// -------------------------------------------------------------
// Arduino Main Loop
// -------------------------------------------------------------
void loop() {
    if (WiFi.status() == WL_CONNECTED) {
        if (!mqttClient.connected()) {
            reconnectMQTT();
        }
        mqttClient.loop();
    }

    unsigned long currentMillis = millis();
    if (currentMillis - lastSampleTime >= SAMPLING_INTERVAL_MS || lastSampleTime == 0) {
        lastSampleTime = currentMillis;

        // 1. Acquire Sensor Readings
        float vibration = readTotalVibration();
        float temp = dht.readTemperature();
        if (isnan(temp)) temp = 29.5f; // Fallback to baseline
        float voltage = readVoltage();
        float current = readCurrent();

        // 2. Execute On-Device Edge Inference
        InferenceResult result = run_edge_inference(&sensorBuffer, vibration, temp, voltage, current);

        // 3. Update Hardware Actuators / LEDs
        updateIndicators(result.status);

        // 4. Output Telemetry over Serial
        Serial.println("------------------------------------------");
        Serial.printf("Vibration:   %.3f g\n", result.vibration);
        Serial.printf("Temperature: %.2f °C\n", result.temp);
        Serial.printf("Voltage:     %.2f V\n", result.voltage);
        Serial.printf("Current:     %.4f A\n", result.current);
        Serial.printf("Power:       %.2f W\n", result.power);
        Serial.printf("Score:       %.3f\n", result.anomaly_score);
        Serial.printf("Status:      %s\n", status_to_string(result.status));
        Serial.printf("Condition:   %s\n", condition_to_string(result.condition));
        Serial.printf("RUL:         %.1f days\n", result.rul_days);
        Serial.printf("Action:      %s\n", result.recommendation);
        Serial.println("------------------------------------------");

        // 5. Publish to MQTT Telemetry Topic (if connected)
        if (mqttClient.connected()) {
            char payload[384];
            snprintf(payload, sizeof(payload),
                "{\"vibration\":%.3f,\"temp\":%.2f,\"voltage\":%.2f,\"current\":%.4f,"
                "\"power\":%.2f,\"score\":%.3f,\"status\":\"%s\",\"condition\":\"%s\","
                "\"rul\":%.1f,\"recommendation\":\"%s\"}",
                result.vibration, result.temp, result.voltage, result.current,
                result.power, result.anomaly_score, status_to_string(result.status),
                condition_to_string(result.condition), result.rul_days, result.recommendation);

            mqttClient.publish(MQTT_TOPIC_PUB, payload);

            if (result.status != STATUS_NORMAL) {
                mqttClient.publish(MQTT_TOPIC_ALERT, payload);
            }
        }
    }
}
