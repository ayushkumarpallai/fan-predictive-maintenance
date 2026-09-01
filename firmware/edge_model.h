/**
 * @file edge_model.h
 * @brief Lightweight on-device inference engine for ESP32 fan predictive maintenance.
 * 
 * Features:
 * - Ultra-fast (<1ms) embedded execution
 * - Zero external dynamic memory allocations
 * - Rolling statistical buffer for feature engineering
 * - Anomaly score calculation and fault condition classification
 * - Local RUL degradation estimate
 */

#ifndef EDGE_MODEL_H
#define EDGE_MODEL_H

#include <math.h>
#include <string.h>

#ifdef __cplusplus
extern "C" {
#endif

// Health Status Enum
typedef enum {
    STATUS_NORMAL = 0,
    STATUS_ALERT = 1,
    STATUS_CRITICAL = 2
} SystemStatus;

// Anomaly Condition Enum
typedef enum {
    CONDITION_HEALTHY = 0,
    CONDITION_HIGH_VIBRATION = 1,
    CONDITION_OVERHEATING = 2,
    CONDITION_OVERCURRENT = 3,
    CONDITION_VOLTAGE_ANOMALY = 4,
    CONDITION_MULTIPLE_ANOMALIES = 5
} FaultCondition;

// Baseline Statistics (Healthy Fan Reference)
#define BASELINE_VIB_MEAN     9.673f
#define BASELINE_VIB_STD      0.015f
#define BASELINE_TEMP_MEAN    29.50f
#define BASELINE_TEMP_STD     0.25f
#define BASELINE_VOLT_MEAN    238.00f
#define BASELINE_VOLT_STD     0.80f
#define BASELINE_CURR_MEAN    0.501f
#define BASELINE_CURR_STD     0.005f

// Anomaly Decision Thresholds
#define THRESH_VIB_HIGH       11.00f   // Total vibration > 11g
#define THRESH_TEMP_HIGH      40.00f   // Temp > 40°C
#define THRESH_CURR_HIGH      0.65f    // Current > 0.65A
#define THRESH_VOLT_DEV       25.00f   // Voltage deviation > 25V (from 238V)

#define SCORE_ALERT_THRESH    0.60f
#define SCORE_CRIT_THRESH     0.80f

// Rolling Buffer Configuration
#define ROLLING_BUFFER_SIZE   3

typedef struct {
    float vibration[ROLLING_BUFFER_SIZE];
    float temp[ROLLING_BUFFER_SIZE];
    float voltage[ROLLING_BUFFER_SIZE];
    float current[ROLLING_BUFFER_SIZE];
    int count;
    int index;
} SensorBuffer;

typedef struct {
    float vibration;
    float temp;
    float voltage;
    float current;
    float power;              // Voltage * Current
    float anomaly_score;      // [0.0 - 1.0]
    float rul_days;           // Estimated Remaining Useful Life in days
    SystemStatus status;
    FaultCondition condition;
    const char* recommendation;
} InferenceResult;

/**
 * Initialize rolling buffer
 */
static inline void init_sensor_buffer(SensorBuffer* buf) {
    buf->count = 0;
    buf->index = 0;
    for (int i = 0; i < ROLLING_BUFFER_SIZE; i++) {
        buf->vibration[i] = BASELINE_VIB_MEAN;
        buf->temp[i] = BASELINE_TEMP_MEAN;
        buf->voltage[i] = BASELINE_VOLT_MEAN;
        buf->current[i] = BASELINE_CURR_MEAN;
    }
}

/**
 * Add sample to rolling buffer
 */
static inline void add_sensor_sample(SensorBuffer* buf, float vib, float temp, float volt, float curr) {
    buf->vibration[buf->index] = vib;
    buf->temp[buf->index] = temp;
    buf->voltage[buf->index] = volt;
    buf->current[buf->index] = curr;

    buf->index = (buf->index + 1) % ROLLING_BUFFER_SIZE;
    if (buf->count < ROLLING_BUFFER_SIZE) {
        buf->count++;
    }
}

/**
 * Get string representation of status
 */
static inline const char* status_to_string(SystemStatus status) {
    switch (status) {
        case STATUS_NORMAL: return "Normal";
        case STATUS_ALERT: return "Alert";
        case STATUS_CRITICAL: return "Critical";
        default: return "Unknown";
    }
}

/**
 * Get string representation of condition
 */
static inline const char* condition_to_string(FaultCondition cond) {
    switch (cond) {
        case CONDITION_HEALTHY: return "Healthy";
        case CONDITION_HIGH_VIBRATION: return "High_Vibration";
        case CONDITION_OVERHEATING: return "Overheating";
        case CONDITION_OVERCURRENT: return "Overcurrent";
        case CONDITION_VOLTAGE_ANOMALY: return "Voltage_Anomaly";
        case CONDITION_MULTIPLE_ANOMALIES: return "Multiple_Anomalies";
        default: return "Unknown";
    }
}

/**
 * Perform on-device edge inference
 */
static inline InferenceResult run_edge_inference(SensorBuffer* buf, float vib, float temp, float volt, float curr) {
    InferenceResult res;
    res.vibration = vib;
    res.temp = temp;
    res.voltage = volt;
    res.current = curr;
    res.power = volt * curr;

    add_sensor_sample(buf, vib, temp, volt, curr);

    // Compute normalized deviations (z-scores approximation)
    float z_vib = fabsf(vib - BASELINE_VIB_MEAN) / (BASELINE_VIB_STD > 0.001f ? BASELINE_VIB_STD : 1.0f);
    float z_temp = fabsf(temp - BASELINE_TEMP_MEAN) / (BASELINE_TEMP_STD > 0.001f ? BASELINE_TEMP_STD : 1.0f);
    float z_volt = fabsf(volt - BASELINE_VOLT_MEAN) / (BASELINE_VOLT_STD > 0.001f ? BASELINE_VOLT_STD : 1.0f);
    float z_curr = fabsf(curr - BASELINE_CURR_MEAN) / (BASELINE_CURR_STD > 0.001f ? BASELINE_CURR_STD : 1.0f);

    // Check specific anomalies
    int fault_flags = 0;
    if (vib > THRESH_VIB_HIGH) fault_flags |= (1 << 0);
    if (temp > THRESH_TEMP_HIGH) fault_flags |= (1 << 1);
    if (curr > THRESH_CURR_HIGH) fault_flags |= (1 << 2);
    if (fabsf(volt - BASELINE_VOLT_MEAN) > THRESH_VOLT_DEV) fault_flags |= (1 << 3);

    // Fault classification
    int num_faults = 0;
    for (int i = 0; i < 4; i++) {
        if (fault_flags & (1 << i)) num_faults++;
    }

    if (num_faults > 1) {
        res.condition = CONDITION_MULTIPLE_ANOMALIES;
        res.recommendation = "Stop/inspect machine; check vibration, temperature, electrical load and supply";
    } else if (fault_flags & (1 << 0)) {
        res.condition = CONDITION_HIGH_VIBRATION;
        res.recommendation = "Inspect bearing, shaft alignment and mounting";
    } else if (fault_flags & (1 << 1)) {
        res.condition = CONDITION_OVERHEATING;
        res.recommendation = "Check cooling, ventilation, friction and motor temperature";
    } else if (fault_flags & (1 << 2)) {
        res.condition = CONDITION_OVERCURRENT;
        res.recommendation = "Check mechanical load, motor condition and electrical connections";
    } else if (fault_flags & (1 << 3)) {
        res.condition = CONDITION_VOLTAGE_ANOMALY;
        res.recommendation = "Inspect supply voltage, wiring and voltage regulation";
    } else {
        res.condition = CONDITION_HEALTHY;
        res.recommendation = "No action required - system operating normally";
    }

    // Composite Anomaly Score Computation [0.0 - 1.0]
    // Normalized distance sigmoid
    float max_z = z_vib;
    if (z_temp > max_z) max_z = z_temp;
    if (z_volt > max_z) max_z = z_volt;
    if (z_curr > max_z) max_z = z_curr;

    float score = 1.0f / (1.0f + expf(-0.35f * (max_z - 3.0f)));
    if (score < 0.0f) score = 0.0f;
    if (score > 1.0f) score = 1.0f;
    res.anomaly_score = score;

    // Determine status
    if (res.condition == CONDITION_MULTIPLE_ANOMALIES || score >= SCORE_CRIT_THRESH) {
        res.status = STATUS_CRITICAL;
    } else if (res.condition != CONDITION_HEALTHY || score >= SCORE_ALERT_THRESH) {
        res.status = STATUS_ALERT;
    } else {
        res.status = STATUS_NORMAL;
    }

    // RUL Estimation in Days (Degradation function: 30 days max down to 0)
    float rul = 30.0f * (1.0f - score);
    if (rul < 0.0f) rul = 0.0f;
    if (rul > 30.0f) rul = 30.0f;
    res.rul_days = rul;

    return res;
}

#ifdef __cplusplus
}
#endif

#endif // EDGE_MODEL_H
