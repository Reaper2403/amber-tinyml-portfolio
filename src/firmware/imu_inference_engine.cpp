#include "imu_inference_engine.h"

#include <algorithm>
#include <cmath>
#include <cstring>

#include <Arduino.h>

extern "C" {
#include "esp_heap_caps.h"
#include "esp_log.h"
#include "esp_timer.h"
}

#include "esp32-hal-i2c.h"

#include "models/stage1_classifier_keras_int8.h"
#include "models/stage1_classifier_labels.h"

#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"
#include "tensorflow/lite/micro/system_setup.h"

namespace {

constexpr char kTag[] = "IMU_CTX";

constexpr uint8_t kQmiAddressLow = 0x6A;
constexpr uint8_t kQmiAddressHigh = 0x6B;
constexpr uint8_t kQmiWhoAmIRegister = 0x00;
constexpr uint8_t kQmiExpectedWhoAmI = 0x05;
constexpr uint8_t kQmiCtrl1 = 0x02;
constexpr uint8_t kQmiCtrl2 = 0x03;
constexpr uint8_t kQmiCtrl3 = 0x04;
constexpr uint8_t kQmiCtrl7 = 0x08;
constexpr uint8_t kQmiStatus0 = 0x2E;
constexpr uint8_t kQmiAxLow = 0x35;

constexpr uint8_t kQmiEnableAccel = 0x01;
constexpr uint8_t kQmiEnableGyro = 0x02;
constexpr uint8_t kQmiAccelRange8G = 0x02;
constexpr uint8_t kQmiGyroRange512Dps = 0x04;
constexpr uint8_t kQmiAccelOdr62_5Hz = 0x07;
constexpr uint8_t kQmiGyroOdr62_5Hz = 0x07;

constexpr int kImuAxes = 6;
constexpr int kWindowSamples = 150;
constexpr int kImuPeriodMs = 20;
constexpr int kPublishEverySamples = 10;
constexpr int kTensorArenaSize = 2000 * 1024;
constexpr float kOneG = 9.807f;
constexpr float kPi = 3.14159265358979323846f;

inline int round_to_nearest_int(float value) {
  return static_cast<int>(value >= 0.0f ? value + 0.5f : value - 0.5f);
}

inline float fast_exp(float x) {
  union {
    uint32_t i;
    float f;
  } value{};
  value.i = static_cast<uint32_t>(12102203.0f * x + 1064866805.0f);
  return value.f;
}

}  // namespace

bool ImuInferenceEngine::begin() {
  if (!init_sensor()) {
    return false;
  }
  if (!init_model()) {
    return false;
  }
  next_sample_at_ms_ = millis();
  ready_ = true;
  ESP_LOGI(kTag,
           "IMU engine ready | arena=%d model_bytes=%u labels=%d window=%dx%d",
           kTensorArenaSize, stage1_classifier_keras_int8_tflite_len,
           kStage1Classes, kWindowSamples, kImuAxes);
  return true;
}

bool ImuInferenceEngine::tick() {
  if (!ready_) {
    return false;
  }

  const uint32_t now_ms = millis();
  if (static_cast<int32_t>(now_ms - next_sample_at_ms_) < 0) {
    return false;
  }
  next_sample_at_ms_ = now_ms + kImuPeriodMs;

  bool data_ready = false;
  if (!sensor_data_ready(&data_ready) || !data_ready) {
    return false;
  }

  SensorSample sample{};
  if (!read_sensor_data(&sample)) {
    ESP_LOGW(kTag, "Failed to read IMU sample");
    return false;
  }

  push_sample(sample);
  ++sample_count_;
  update_motion_aggregates(sample);

  if (sample_count_ < kWindowSamples) {
    return false;
  }

  ImuPrediction prediction{};
  if (!run_inference(&prediction)) {
    return false;
  }

  if ((sample_count_ % kPublishEverySamples) != 0) {
    return false;
  }

  latest_prediction_ = prediction;

  reset_motion_aggregates();
  return true;
}

ImuPrediction ImuInferenceEngine::latest_prediction() const {
  return latest_prediction_;
}

const char* ImuInferenceEngine::label_name(int index) const {
  if (index >= 0 && index < kStage1Classes) {
    return kStage1Labels[index];
  }
  return "unknown";
}

bool ImuInferenceEngine::init_sensor() {
  uint8_t who_am_i = 0;
  sensor_address_ = kQmiAddressHigh;
  if ((!read_register(kQmiWhoAmIRegister, &who_am_i, 1) ||
       who_am_i != kQmiExpectedWhoAmI)) {
    sensor_address_ = kQmiAddressLow;
    if (!read_register(kQmiWhoAmIRegister, &who_am_i, 1) ||
        who_am_i != kQmiExpectedWhoAmI) {
      ESP_LOGE(kTag, "QMI8658 WHO_AM_I failed on both addresses");
      return false;
    }
  }

  if (!write_register(kQmiCtrl1, 0x60)) {
    ESP_LOGE(kTag, "Failed to initialize QMI8658 CTRL1");
    return false;
  }
  if (!write_register(kQmiCtrl2,
                      static_cast<uint8_t>((kQmiAccelRange8G << 4) |
                                           kQmiAccelOdr62_5Hz))) {
    return false;
  }
  if (!write_register(kQmiCtrl3,
                      static_cast<uint8_t>((kQmiGyroRange512Dps << 4) |
                                           kQmiGyroOdr62_5Hz))) {
    return false;
  }
  if (!write_register(kQmiCtrl7, kQmiEnableAccel | kQmiEnableGyro)) {
    return false;
  }

  ESP_LOGI(kTag, "QMI8658 ready at 0x%02X", sensor_address_);
  return true;
}

bool ImuInferenceEngine::init_model() {
  tflite::InitializeTarget();

  model_ = tflite::GetModel(stage1_classifier_keras_int8_tflite);
  if (model_ == nullptr || model_->version() != TFLITE_SCHEMA_VERSION) {
    ESP_LOGE(kTag, "IMU model schema mismatch");
    return false;
  }

  tensor_arena_ = static_cast<uint8_t*>(
      heap_caps_malloc(kTensorArenaSize, MALLOC_CAP_SPIRAM | MALLOC_CAP_8BIT));
  if (tensor_arena_ == nullptr) {
    tensor_arena_ =
        static_cast<uint8_t*>(heap_caps_malloc(kTensorArenaSize, MALLOC_CAP_8BIT));
  }
  if (tensor_arena_ == nullptr) {
    ESP_LOGE(kTag, "Failed to allocate IMU tensor arena");
    return false;
  }

  static tflite::MicroMutableOpResolver<8> resolver;
  static bool resolver_initialized = false;
  if (!resolver_initialized) {
    if (resolver.AddConv2D() != kTfLiteOk ||
        resolver.AddDepthwiseConv2D() != kTfLiteOk ||
        resolver.AddFullyConnected() != kTfLiteOk ||
        resolver.AddReshape() != kTfLiteOk ||
        resolver.AddMaxPool2D() != kTfLiteOk ||
        resolver.AddExpandDims() != kTfLiteOk ||
        resolver.AddMean() != kTfLiteOk) {
      ESP_LOGE(kTag, "IMU op resolver init failed");
      return false;
    }
    resolver_initialized = true;
  }

  static tflite::MicroInterpreter static_interpreter(model_, resolver, tensor_arena_,
                                                     kTensorArenaSize);
  interpreter_ = &static_interpreter;
  if (interpreter_->AllocateTensors() != kTfLiteOk) {
    ESP_LOGE(kTag, "IMU AllocateTensors failed");
    return false;
  }

  input_ = interpreter_->input(0);
  output_ = interpreter_->output(0);
  if (input_ == nullptr || output_ == nullptr) {
    ESP_LOGE(kTag, "IMU model tensors missing");
    return false;
  }
  return true;
}

bool ImuInferenceEngine::read_register(uint8_t reg, uint8_t* buffer,
                                       uint8_t length) const {
  size_t read_count = 0;
  const esp_err_t err = i2cWriteReadNonStop(0, sensor_address_, &reg, 1, buffer,
                                            length, 1000, &read_count);
  return err == ESP_OK && read_count == length;
}

bool ImuInferenceEngine::write_register(uint8_t reg, uint8_t value) const {
  const uint8_t payload[2] = {reg, value};
  return i2cWrite(0, sensor_address_, payload, sizeof(payload), 1000) == ESP_OK;
}

bool ImuInferenceEngine::sensor_data_ready(bool* ready) const {
  if (ready == nullptr) {
    return false;
  }
  uint8_t status = 0;
  if (!read_register(kQmiStatus0, &status, 1)) {
    return false;
  }
  *ready = (status & 0x03) != 0;
  return true;
}

bool ImuInferenceEngine::read_sensor_data(SensorSample* sample) const {
  if (sample == nullptr) {
    return false;
  }
  uint8_t buffer[12] = {};
  if (!read_register(kQmiAxLow, buffer, sizeof(buffer))) {
    return false;
  }

  const int16_t raw_ax = static_cast<int16_t>((buffer[1] << 8) | buffer[0]);
  const int16_t raw_ay = static_cast<int16_t>((buffer[3] << 8) | buffer[2]);
  const int16_t raw_az = static_cast<int16_t>((buffer[5] << 8) | buffer[4]);
  const int16_t raw_gx = static_cast<int16_t>((buffer[7] << 8) | buffer[6]);
  const int16_t raw_gy = static_cast<int16_t>((buffer[9] << 8) | buffer[8]);
  const int16_t raw_gz = static_cast<int16_t>((buffer[11] << 8) | buffer[10]);

  sample->accel_x = (raw_ax * kOneG) / accel_lsb_div_;
  sample->accel_y = (raw_ay * kOneG) / accel_lsb_div_;
  sample->accel_z = (raw_az * kOneG) / accel_lsb_div_;
  sample->gyro_x = (raw_gx * kPi / 180.0f) / gyro_lsb_div_;
  sample->gyro_y = (raw_gy * kPi / 180.0f) / gyro_lsb_div_;
  sample->gyro_z = (raw_gz * kPi / 180.0f) / gyro_lsb_div_;
  return true;
}

void ImuInferenceEngine::push_sample(const SensorSample& sample) {
  memmove(&imu_window_[0][0], &imu_window_[1][0],
          sizeof(float) * (kWindowSamples - 1) * kImuAxes);

  imu_window_[kWindowSamples - 1][0] = sample.accel_x;
  imu_window_[kWindowSamples - 1][1] = sample.accel_y;
  imu_window_[kWindowSamples - 1][2] = sample.accel_z;
  imu_window_[kWindowSamples - 1][3] = sample.gyro_x;
  imu_window_[kWindowSamples - 1][4] = sample.gyro_y;
  imu_window_[kWindowSamples - 1][5] = sample.gyro_z;
}

void ImuInferenceEngine::update_motion_aggregates(const SensorSample& sample) {
  const float accel_mag =
      magnitude3(sample.accel_x, sample.accel_y, sample.accel_z);
  const float gyro_mag = magnitude3(sample.gyro_x, sample.gyro_y, sample.gyro_z);
  ble_motion_samples_++;
  ble_accel_mag_sum_ += accel_mag;
  ble_accel_mag_sq_sum_ += accel_mag * accel_mag;
  ble_gyro_mag_sum_ += gyro_mag;
  ble_gyro_mag_sq_sum_ += gyro_mag * gyro_mag;
}

void ImuInferenceEngine::reset_motion_aggregates() {
  ble_motion_samples_ = 0;
  ble_accel_mag_sum_ = 0.0f;
  ble_accel_mag_sq_sum_ = 0.0f;
  ble_gyro_mag_sum_ = 0.0f;
  ble_gyro_mag_sq_sum_ = 0.0f;
}

bool ImuInferenceEngine::run_inference(ImuPrediction* prediction_out) {
  if (prediction_out == nullptr || input_ == nullptr || output_ == nullptr) {
    return false;
  }

  if (input_->type == kTfLiteInt8) {
    const float input_scale = input_->params.scale;
    const int input_zero_point = input_->params.zero_point;
    for (int sample = 0; sample < kWindowSamples; ++sample) {
      for (int axis = 0; axis < kImuAxes; ++axis) {
        const int flat_index = sample * kImuAxes + axis;
        input_->data.int8[flat_index] =
            quantize_to_int8(imu_window_[sample][axis], input_scale,
                             input_zero_point);
      }
    }
  } else {
    for (int sample = 0; sample < kWindowSamples; ++sample) {
      for (int axis = 0; axis < kImuAxes; ++axis) {
        input_->data.f[sample * kImuAxes + axis] = imu_window_[sample][axis];
      }
    }
  }

  const int64_t inference_cycle_start_us = esp_timer_get_time();
  if (interpreter_->Invoke() != kTfLiteOk) {
    ESP_LOGE(kTag, "Stage1 IMU inference failed");
    return false;
  }
  const int64_t inference_latency_us =
      esp_timer_get_time() - inference_cycle_start_us;

  float logits[kStage1Classes] = {};
  if (output_->type == kTfLiteInt8) {
    const float output_scale = output_->params.scale;
    const int output_zero_point = output_->params.zero_point;
    for (int i = 0; i < kStage1Classes; ++i) {
      logits[i] =
          (static_cast<int>(output_->data.int8[i]) - output_zero_point) *
          output_scale;
    }
  } else {
    for (int i = 0; i < kStage1Classes; ++i) {
      logits[i] = output_->data.f[i];
    }
  }

  float max_logit = logits[0];
  for (int i = 1; i < kStage1Classes; ++i) {
    if (logits[i] > max_logit) {
      max_logit = logits[i];
    }
  }

  float sum_exp = 0.0f;
  int best_index = 0;
  float best_probability = 0.0f;
  for (int i = 0; i < kStage1Classes; ++i) {
    const float probability = fast_exp(logits[i] - max_logit);
    logits[i] = probability;
    sum_exp += probability;
  }
  const float inv_sum = 1.0f / std::max(sum_exp, 1e-6f);
  for (int i = 0; i < kStage1Classes; ++i) {
    const float probability = logits[i] * inv_sum;
    if (probability > best_probability) {
      best_probability = probability;
      best_index = i;
    }
  }

  ++inference_count_;
  last_inference_us_ = inference_latency_us;
  if (previous_inference_cycle_start_us_ != 0) {
    last_cadence_us_ =
        inference_cycle_start_us - previous_inference_cycle_start_us_;
  } else {
    last_cadence_us_ = 0;
  }
  previous_inference_cycle_start_us_ = inference_cycle_start_us;

  prediction_out->valid = true;
  prediction_out->t_ms = static_cast<uint32_t>(esp_log_timestamp());
  prediction_out->sample_count = sample_count_;
  prediction_out->label_index = best_index;
  prediction_out->confidence = best_probability;
  prediction_out->invoke_us = last_inference_us_;
  prediction_out->cadence_us = last_cadence_us_;
  prediction_out->accel_mag_mean =
      ble_motion_samples_ > 0
          ? (ble_accel_mag_sum_ / static_cast<float>(ble_motion_samples_))
          : 0.0f;
  prediction_out->accel_mag_std = compute_stddev(
      ble_accel_mag_sum_, ble_accel_mag_sq_sum_, ble_motion_samples_);
  prediction_out->gyro_mag_mean =
      ble_motion_samples_ > 0
          ? (ble_gyro_mag_sum_ / static_cast<float>(ble_motion_samples_))
          : 0.0f;
  prediction_out->gyro_mag_std = compute_stddev(
      ble_gyro_mag_sum_, ble_gyro_mag_sq_sum_, ble_motion_samples_);
  return true;
}

float ImuInferenceEngine::compute_stddev(float sum, float sq_sum,
                                         int count) const {
  if (count <= 0) {
    return 0.0f;
  }
  const float mean = sum / static_cast<float>(count);
  float variance = (sq_sum / static_cast<float>(count)) - (mean * mean);
  if (variance < 0.0f) {
    variance = 0.0f;
  }
  return sqrtf(variance);
}

float ImuInferenceEngine::magnitude3(float x, float y, float z) const {
  return sqrtf(x * x + y * y + z * z);
}

int8_t ImuInferenceEngine::quantize_to_int8(float value, float scale,
                                            int zero_point) const {
  const float inv_scale = (scale > 0.0f) ? (1.0f / scale) : 0.0f;
  int quantized = round_to_nearest_int(value * inv_scale) + zero_point;
  if (quantized < -128) {
    quantized = -128;
  } else if (quantized > 127) {
    quantized = 127;
  }
  return static_cast<int8_t>(quantized);
}
