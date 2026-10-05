#include <Arduino.h>
#include <math.h>

extern "C" {
#include "amber_ble.h"
#include "esp_log.h"
#include "freertos/FreeRTOS.h"
#include "freertos/task.h"
}

#include "audio_context_engine.h"
#include "imu_inference_engine.h"

namespace {

AudioContextEngine g_audio_engine;
ImuInferenceEngine g_imu_engine;
TaskHandle_t g_audio_task_handle = nullptr;
uint32_t g_ble_publish_sequence = 0;
int g_last_logged_imu_label = -1;
uint32_t g_last_logged_audio_seq = 0;
uint32_t g_last_state_log_ms = 0;

constexpr char kBleDeviceName[] = "amber-stage1";
constexpr uint32_t kAudioStartupDelayMs = 5000;
constexpr uint32_t kAudioBurstIntervalMs = 60 * 1000;
constexpr uint32_t kStateLogIntervalMs = 10 * 1000;
constexpr size_t kBlePayloadBufferSize = 128;

uint16_t confidence_to_milli(float confidence) {
  int scaled = static_cast<int>(lroundf(confidence * 1000.0f));
  if (scaled < 0) {
    scaled = 0;
  }
  if (scaled > 1000) {
    scaled = 1000;
  }
  return static_cast<uint16_t>(scaled);
}

void maybe_log_state(const ImuPrediction& imu_prediction,
                     const AudioBurstSummary& audio_summary) {
  const uint32_t now_ms = millis();
  const bool imu_changed = imu_prediction.label_index != g_last_logged_imu_label;
  const bool audio_changed = audio_summary.burst_sequence != g_last_logged_audio_seq;
  const bool periodic = (now_ms - g_last_state_log_ms) >= kStateLogIntervalMs;
  if (!imu_changed && !audio_changed && !periodic) {
    return;
  }

  const uint32_t audio_age_ms =
      audio_summary.valid && audio_summary.finished_at_ms <= now_ms
          ? (now_ms - audio_summary.finished_at_ms)
          : 0;

  Serial.printf(
      "[state] imu=%s c=%.2f | audio=%s c=%.2f burst=%lu support=%d/5 age=%lus active=%d\n",
      g_imu_engine.label_name(imu_prediction.label_index),
      static_cast<double>(imu_prediction.confidence),
      audio_summary.valid ? g_audio_engine.label_name(audio_summary.majority_label_index)
                          : "na",
      static_cast<double>(audio_summary.valid
                              ? audio_summary.majority_average_confidence
                              : 0.0f),
      static_cast<unsigned long>(audio_summary.burst_sequence),
      audio_summary.valid ? audio_summary.majority_count : 0,
      static_cast<unsigned long>(audio_age_ms / 1000),
      audio_summary.active ? 1 : 0);

  g_last_logged_imu_label = imu_prediction.label_index;
  g_last_logged_audio_seq = audio_summary.burst_sequence;
  g_last_state_log_ms = now_ms;
}

void publish_combined_status(const ImuPrediction& imu_prediction) {
  const AudioBurstSummary audio_summary = g_audio_engine.latest_summary();
  const uint32_t now_ms = millis();
  const uint32_t audio_age_ms =
      audio_summary.valid && audio_summary.finished_at_ms <= now_ms
          ? (now_ms - audio_summary.finished_at_ms)
          : 0;

  char payload[kBlePayloadBufferSize];
  snprintf(
      payload, sizeof(payload), "A2|%lu|%lu|%d|%u|%lu|%d|%u|%d|%lu|%d",
      static_cast<unsigned long>(++g_ble_publish_sequence),
      static_cast<unsigned long>(imu_prediction.t_ms),
      imu_prediction.label_index, confidence_to_milli(imu_prediction.confidence),
      static_cast<unsigned long>(audio_summary.burst_sequence),
      audio_summary.valid ? audio_summary.majority_label_index : -1,
      confidence_to_milli(audio_summary.valid
                              ? audio_summary.majority_average_confidence
                              : 0.0f),
      audio_summary.valid ? audio_summary.majority_count : 0,
      static_cast<unsigned long>(audio_age_ms), audio_summary.active ? 1 : 0);

  amber_ble_publish_status(payload);
  maybe_log_state(imu_prediction, audio_summary);
}

void audio_task(void* param) {
  (void)param;

  vTaskDelay(pdMS_TO_TICKS(kAudioStartupDelayMs));
  TickType_t last_burst_start = xTaskGetTickCount();

  while (true) {
    AudioBurstSummary burst_summary{};
    if (!g_audio_engine.run_burst(&burst_summary)) {
      Serial.println("[audio] burst failed");
    }
    vTaskDelayUntil(&last_burst_start, pdMS_TO_TICKS(kAudioBurstIntervalMs));
  }
}

}  // namespace

void setup() {
  Serial.begin(921600);
  const uint32_t serial_wait_start = millis();
  while (!Serial && (millis() - serial_wait_start) < 4000) {
    delay(10);
  }
  delay(300);

  esp_log_level_set("AMBER_BLE", ESP_LOG_WARN);
  esp_log_level_set("IMU_CTX", ESP_LOG_WARN);
  esp_log_level_set("AUDIO_CTX", ESP_LOG_WARN);

  Serial.println("[amber] boot");

  if (!g_audio_engine.begin()) {
    Serial.println("[FATAL] Audio context engine init failed");
    while (true) {
      delay(1000);
    }
  }

  if (!g_imu_engine.begin()) {
    Serial.println("[FATAL] IMU inference engine init failed");
    while (true) {
      delay(1000);
    }
  }

  if (!amber_ble_init(kBleDeviceName)) {
    Serial.println("[FATAL] BLE init failed");
    while (true) {
      delay(1000);
    }
  }

  if (xTaskCreate(audio_task, "audio_burst_task", 12288, nullptr, 1,
                  &g_audio_task_handle) != pdPASS) {
    Serial.println("[FATAL] Audio burst task create failed");
    while (true) {
      delay(1000);
    }
  }

  Serial.printf(
      "[amber] ready ble=%s | audio=60s cycle,10s active,5x2s | imu=20ms sample\n",
      kBleDeviceName);
}

void loop() {
  if (g_imu_engine.tick()) {
    publish_combined_status(g_imu_engine.latest_prediction());
  }
  delay(1);
}
