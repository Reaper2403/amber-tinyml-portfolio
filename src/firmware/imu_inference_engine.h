#pragma once

#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/schema/schema_generated.h"

#include <stdint.h>

struct ImuPrediction {
  bool valid = false;
  uint32_t t_ms = 0;
  uint32_t sample_count = 0;
  int label_index = -1;
  float confidence = 0.0f;
  int64_t invoke_us = 0;
  int64_t cadence_us = 0;
  float accel_mag_mean = 0.0f;
  float accel_mag_std = 0.0f;
  float gyro_mag_mean = 0.0f;
  float gyro_mag_std = 0.0f;
};

class ImuInferenceEngine {
 public:
  bool begin();
  bool tick();
  ImuPrediction latest_prediction() const;
  const char* label_name(int index) const;

 private:
  struct SensorSample {
    float accel_x = 0.0f;
    float accel_y = 0.0f;
    float accel_z = 0.0f;
    float gyro_x = 0.0f;
    float gyro_y = 0.0f;
    float gyro_z = 0.0f;
  };

  bool init_sensor();
  bool init_model();
  bool read_register(uint8_t reg, uint8_t* buffer, uint8_t length) const;
  bool write_register(uint8_t reg, uint8_t value) const;
  bool sensor_data_ready(bool* ready) const;
  bool read_sensor_data(SensorSample* sample) const;
  void push_sample(const SensorSample& sample);
  void update_motion_aggregates(const SensorSample& sample);
  void reset_motion_aggregates();
  bool run_inference(ImuPrediction* prediction_out);
  float compute_stddev(float sum, float sq_sum, int count) const;
  float magnitude3(float x, float y, float z) const;
  int8_t quantize_to_int8(float value, float scale, int zero_point) const;

  bool ready_ = false;
  uint8_t sensor_address_ = 0;
  uint16_t accel_lsb_div_ = 4096;
  uint16_t gyro_lsb_div_ = 64;
  uint32_t next_sample_at_ms_ = 0;
  float imu_window_[150][6] = {};
  int sample_count_ = 0;
  uint64_t inference_count_ = 0;
  int64_t previous_inference_cycle_start_us_ = 0;
  int64_t last_inference_us_ = 0;
  int64_t last_cadence_us_ = 0;
  int ble_motion_samples_ = 0;
  float ble_accel_mag_sum_ = 0.0f;
  float ble_accel_mag_sq_sum_ = 0.0f;
  float ble_gyro_mag_sum_ = 0.0f;
  float ble_gyro_mag_sq_sum_ = 0.0f;

  const tflite::Model* model_ = nullptr;
  tflite::MicroInterpreter* interpreter_ = nullptr;
  TfLiteTensor* input_ = nullptr;
  TfLiteTensor* output_ = nullptr;
  uint8_t* tensor_arena_ = nullptr;
  ImuPrediction latest_prediction_ = {};
};
