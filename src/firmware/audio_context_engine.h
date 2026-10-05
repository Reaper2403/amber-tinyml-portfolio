#pragma once

#include "freertos/FreeRTOS.h"
#include "freertos/semphr.h"

#include <stdint.h>

struct AudioWindowPrediction {
  int label_index = -1;
  int second_label_index = -1;
  float confidence = 0.0f;
  float second_confidence = 0.0f;
  float clip_rms = 0.0f;
  float clip_peak = 0.0f;
  float inference_ms = 0.0f;
};

struct AudioBurstSummary {
  static constexpr int kMaxWindows = 5;

  bool valid = false;
  bool active = false;
  uint32_t burst_sequence = 0;
  uint32_t started_at_ms = 0;
  uint32_t finished_at_ms = 0;
  int window_count = 0;
  AudioWindowPrediction windows[kMaxWindows] = {};
  int majority_label_index = -1;
  int majority_count = 0;
  float majority_average_confidence = 0.0f;
  int last_label_index = -1;
  float last_confidence = 0.0f;
};

class AudioContextEngine {
 public:
  bool begin();
  void tick();
  bool run_burst(AudioBurstSummary* summary_out);
  AudioBurstSummary latest_summary() const;
  const char* label_name(int index) const;

 private:
  bool init_mic();
  bool init_model();
  bool init_frontend();
  bool allocate_work_buffers();
  bool capture_samples(int16_t* destination, int sample_count);
  bool infer_current_clip(AudioWindowPrediction* prediction_out);
  void run_probe_cycle();
  bool should_label_quiet(float probe_rms, float probe_peak) const;
  bool should_run_inference(float probe_rms, float probe_peak) const;
  void update_baseline(float probe_rms, float probe_peak, bool quiet_like);
  bool extract_logmel_features_to_model_input(const int16_t* audio_samples);
  void emit_quiet_label(float probe_rms, float probe_peak);
  void emit_hold_label(float probe_rms, float probe_peak);
  void run_model_inference(float probe_rms, float probe_peak);
  void mark_summary_active(bool active, uint32_t burst_sequence);
  void store_latest_summary(const AudioBurstSummary& summary);
  void log_boot_summary() const;
  float compute_rms(const int16_t* audio_samples, int sample_count) const;
  float compute_peak_abs(const int16_t* audio_samples, int sample_count) const;

  bool ready_ = false;
  bool baseline_initialized_ = false;
  bool have_last_nonquiet_label_ = false;
  uint32_t next_probe_at_ms_ = 0;
  uint32_t probe_cycle_count_ = 0;
  int stable_nonquiet_skip_count_ = 0;
  int last_label_index_ = -1;
  float baseline_rms_ = 0.0f;
  float baseline_peak_ = 0.0f;
  uint32_t burst_sequence_ = 0;
  mutable SemaphoreHandle_t summary_mutex_ = nullptr;
  AudioBurstSummary latest_summary_ = {};
};
