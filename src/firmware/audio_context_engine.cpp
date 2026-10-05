#include "audio_context_engine.h"

#include <algorithm>
#include <cmath>
#include <cstring>

#include <Arduino.h>
#include <ESP_I2S.h>
#include <Wire.h>

extern "C" {
#include "esp_check.h"
#include "esp_heap_caps.h"
#include "esp_log.h"
#include "esp_timer.h"
}

#include "es8311.h"
#include "pin_config.h"

#include "models/audio_context_labels.h"
#include "models/audio_context_stage1_pretrain_int8.h"
#include "models/audio_context_stage2_finetune_int8.h"

#include "signal/src/complex.h"
#include "signal/src/rfft.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"
#include "tensorflow/lite/schema/schema_generated.h"

namespace {

constexpr char kTag[] = "AUDIO_CTX";

I2SClass g_i2s;

constexpr int kSampleRateHz = 16000;
constexpr int kClipDurationMs = 2000;
constexpr int kClipSamples = (kSampleRateHz * kClipDurationMs) / 1000;
constexpr int kProbeDurationMs = 250;
constexpr int kProbeSamples = (kSampleRateHz * kProbeDurationMs) / 1000;
constexpr int kBurstWindowCount = AudioBurstSummary::kMaxWindows;
constexpr int kFrameLengthSamples = (kSampleRateHz * 30) / 1000;
constexpr int kFrameStepSamples = (kSampleRateHz * 20) / 1000;
constexpr int kFftLength = 512;
constexpr int kPadSamples = kFftLength / 2;
constexpr int kPaddedSamples = kClipSamples + (2 * kPadSamples);
constexpr int kFftBins = (kFftLength / 2) + 1;
constexpr int kNumMelBins = 40;
constexpr int kFrameCount = 101;
constexpr uint32_t kProbeIntervalMs = 30000;
constexpr uint32_t kStartupWarmupMs = 5000;
constexpr bool kContinuousInferenceMode = true;
constexpr bool kLogAudioWindows = false;
constexpr es8311_mic_gain_t kMicGain = ES8311_MIC_GAIN_36DB;
constexpr int kTensorArenaSize = 2000 * 1024;
constexpr bool kUseFineTunedModel = true;
constexpr int kMaxStableNonquietSkips = 1;
// Keep the synthetic quiet gate as close to "near-silence only" as possible
// so we can observe the model's own behavior without the heuristic dominating.
constexpr float kAbsoluteQuietRms = 25.0f;
constexpr float kAbsoluteQuietPeak = 120.0f;
constexpr float kQuietBaselineMultiplier = 0.0f;
constexpr float kQuietPeakMultiplier = 0.0f;
constexpr float kChangeRmsFloor = 180.0f;
constexpr float kChangePeakFloor = 700.0f;
constexpr float kChangeMultiplier = 0.30f;
constexpr float kBaselineQuietAlpha = 0.25f;
constexpr float kBaselineActiveAlpha = 0.05f;
constexpr float kLowerEdgeHz = 20.0f;
constexpr float kUpperEdgeHz = 7600.0f;
constexpr float kLogClipMinDb = -80.0f;
constexpr float kLogClipMaxDb = 0.0f;
constexpr float kPi = 3.14159265358979323846f;
constexpr float kPcmScale = 1.0f / 32768.0f;

const char kQuietLabel[] = "quiet";

struct EngineRuntime {
  const tflite::Model* model = nullptr;
  tflite::MicroInterpreter* interpreter = nullptr;
  TfLiteTensor* input = nullptr;
  TfLiteTensor* output = nullptr;
  uint8_t* tensor_arena = nullptr;
  int16_t* clip_buffer = nullptr;
  int16_t* padded_audio = nullptr;
  float* hann_window = nullptr;
  float* mel_weights = nullptr;
  float* fft_input = nullptr;
  Complex<float>* fft_output = nullptr;
  float* power_spectrum = nullptr;
  float* mel_scratch = nullptr;
  void* rfft_state = nullptr;
  uint8_t* rfft_state_buffer = nullptr;
  size_t rfft_state_size = 0;
};

EngineRuntime g_runtime;

static esp_err_t es8311_codec_init(void) {
  es8311_handle_t es_handle = es8311_create(0, ES8311_ADDRRES_0);
  ESP_RETURN_ON_FALSE(es_handle, ESP_FAIL, kTag, "es8311 create failed");

  const es8311_clock_config_t es_clk = {
      .mclk_inverted = false,
      .sclk_inverted = false,
      .mclk_from_mclk_pin = true,
      .mclk_frequency = kSampleRateHz * 256,
      .sample_frequency = kSampleRateHz,
  };

  ESP_ERROR_CHECK(
      es8311_init(es_handle, &es_clk, ES8311_RESOLUTION_16, ES8311_RESOLUTION_16));
  ESP_ERROR_CHECK(es8311_microphone_config(es_handle, false));
  ESP_ERROR_CHECK(es8311_microphone_gain_set(es_handle, kMicGain));

  return ESP_OK;
}

inline float hz_to_mel(float hz) {
  return 2595.0f * log10f(1.0f + (hz / 700.0f));
}

inline float mel_to_hz(float mel) {
  return 700.0f * (powf(10.0f, mel / 2595.0f) - 1.0f);
}

inline int8_t quantize_feature(float normalized, const TfLiteTensor* input) {
  const float clamped = std::max(0.0f, std::min(1.0f, normalized));
  const float scale = input->params.scale;
  const int zero_point = input->params.zero_point;
  if (input->type == kTfLiteInt8) {
    int quantized = static_cast<int>(lroundf(clamped / scale)) + zero_point;
    quantized = std::max(-128, std::min(127, quantized));
    return static_cast<int8_t>(quantized);
  }
  return static_cast<int8_t>(clamped * 127.0f);
}

inline const unsigned char* selected_model_data() {
  return kUseFineTunedModel ? audio_context_stage2_finetune_int8_tflite
                            : audio_context_stage1_pretrain_int8_tflite;
}

inline const char* selected_model_name() {
  return kUseFineTunedModel ? "stage2_finetune" : "stage1_pretrain";
}

}  // namespace

bool AudioContextEngine::begin() {
  if (summary_mutex_ == nullptr) {
    summary_mutex_ = xSemaphoreCreateMutex();
    if (summary_mutex_ == nullptr) {
      Serial.println("[FATAL] Audio summary mutex create failed");
      return false;
    }
  }
  if (!init_mic()) {
    return false;
  }
  if (!allocate_work_buffers()) {
    return false;
  }
  if (!init_frontend()) {
    return false;
  }
  if (!init_model()) {
    return false;
  }
  next_probe_at_ms_ = millis() + kStartupWarmupMs;
  ready_ = true;
  log_boot_summary();
  return true;
}

void AudioContextEngine::tick() {
  if (!ready_) {
    delay(250);
    return;
  }
  if (kContinuousInferenceMode) {
    run_probe_cycle();
    return;
  }
  const uint32_t now = millis();
  if (static_cast<int32_t>(now - next_probe_at_ms_) < 0) {
    delay(20);
    return;
  }
  run_probe_cycle();
  next_probe_at_ms_ = millis() + kProbeIntervalMs;
}

bool AudioContextEngine::run_burst(AudioBurstSummary* summary_out) {
  if (!ready_) {
    return false;
  }

  AudioBurstSummary summary{};
  summary.active = true;
  summary.burst_sequence = ++burst_sequence_;
  summary.started_at_ms = millis();
  mark_summary_active(true, summary.burst_sequence);

  Serial.printf("[audio] burst=%lu start windows=%d clip_ms=%d\n",
                static_cast<unsigned long>(summary.burst_sequence),
                kBurstWindowCount, kClipDurationMs);

  int label_counts[kAudioContextClasses] = {};
  float label_confidence_sums[kAudioContextClasses] = {};

  for (int window_index = 0; window_index < kBurstWindowCount; ++window_index) {
    if (!capture_samples(g_runtime.clip_buffer, kClipSamples)) {
      Serial.printf("[audio] burst=%lu capture failed at window=%d\n",
                    static_cast<unsigned long>(summary.burst_sequence),
                    window_index);
      mark_summary_active(false, summary.burst_sequence);
      return false;
    }

    AudioWindowPrediction prediction{};
    if (!infer_current_clip(&prediction)) {
      Serial.printf("[audio] burst=%lu inference failed at window=%d\n",
                    static_cast<unsigned long>(summary.burst_sequence),
                    window_index);
      mark_summary_active(false, summary.burst_sequence);
      return false;
    }

    summary.windows[window_index] = prediction;
    summary.window_count++;
    if (prediction.label_index >= 0 &&
        prediction.label_index < kAudioContextClasses) {
      label_counts[prediction.label_index]++;
      label_confidence_sums[prediction.label_index] += prediction.confidence;
    }
    summary.last_label_index = prediction.label_index;
    summary.last_confidence = prediction.confidence;

    if (kLogAudioWindows) {
      Serial.printf(
          "[audio] burst=%lu win=%d/%d label=%s p=%.2f alt=%s p2=%.2f rms=%.1f peak=%.1f infer=%.1fms\n",
          static_cast<unsigned long>(summary.burst_sequence), window_index + 1,
          kBurstWindowCount, label_name(prediction.label_index),
          prediction.confidence, label_name(prediction.second_label_index),
          prediction.second_confidence, prediction.clip_rms,
          prediction.clip_peak, prediction.inference_ms);
    }
  }

  summary.finished_at_ms = millis();
  summary.valid = true;
  summary.active = false;

  int majority_index = 0;
  for (int i = 1; i < kAudioContextClasses; ++i) {
    if (label_counts[i] > label_counts[majority_index]) {
      majority_index = i;
    }
  }
  summary.majority_label_index = majority_index;
  summary.majority_count = label_counts[majority_index];
  if (summary.majority_count > 0) {
    summary.majority_average_confidence =
        label_confidence_sums[majority_index] /
        static_cast<float>(summary.majority_count);
  }

  store_latest_summary(summary);

  Serial.printf(
      "[audio] burst=%lu done label=%s conf=%.2f support=%d/%d dur=%lums\n",
      static_cast<unsigned long>(summary.burst_sequence),
      label_name(summary.majority_label_index),
      summary.majority_average_confidence, summary.majority_count,
      summary.window_count,
      static_cast<unsigned long>(summary.finished_at_ms - summary.started_at_ms));

  if (summary_out != nullptr) {
    *summary_out = summary;
  }
  return true;
}

AudioBurstSummary AudioContextEngine::latest_summary() const {
  AudioBurstSummary copy{};
  if (summary_mutex_ != nullptr &&
      xSemaphoreTake(summary_mutex_, pdMS_TO_TICKS(5)) == pdTRUE) {
    copy = latest_summary_;
    xSemaphoreGive(summary_mutex_);
  }
  return copy;
}

const char* AudioContextEngine::label_name(int index) const {
  if (index >= 0 && index < kAudioContextClasses) {
    return kAudioContextLabels[index];
  }
  return "unknown";
}

bool AudioContextEngine::init_mic() {
  Wire.begin(IIC_SDA, IIC_SCL);
  Wire.setClock(400000);

  pinMode(PA, OUTPUT);
  digitalWrite(PA, LOW);

  g_i2s.setPins(BCLKPIN, WSPIN, DIPIN, DOPIN, MCLKPIN);
  if (!g_i2s.begin(I2S_MODE_STD, kSampleRateHz, I2S_DATA_BIT_WIDTH_16BIT,
                   I2S_SLOT_MODE_MONO, I2S_STD_SLOT_LEFT)) {
    Serial.println("[FATAL] I2S init failed");
    return false;
  }

  if (es8311_codec_init() != ESP_OK) {
    Serial.println("[FATAL] ES8311 init failed");
    return false;
  }

  return true;
}

bool AudioContextEngine::allocate_work_buffers() {
  auto alloc_bytes = [](size_t bytes) -> uint8_t* {
    uint8_t* ptr = static_cast<uint8_t*>(
        heap_caps_malloc(bytes, MALLOC_CAP_SPIRAM | MALLOC_CAP_8BIT));
    if (ptr == nullptr) {
      ptr = static_cast<uint8_t*>(heap_caps_malloc(bytes, MALLOC_CAP_8BIT));
    }
    return ptr;
  };

  g_runtime.clip_buffer = reinterpret_cast<int16_t*>(
      alloc_bytes(sizeof(int16_t) * kClipSamples));
  g_runtime.padded_audio = reinterpret_cast<int16_t*>(
      alloc_bytes(sizeof(int16_t) * kPaddedSamples));
  g_runtime.hann_window =
      reinterpret_cast<float*>(alloc_bytes(sizeof(float) * kFrameLengthSamples));
  g_runtime.mel_weights =
      reinterpret_cast<float*>(alloc_bytes(sizeof(float) * kNumMelBins * kFftBins));
  g_runtime.fft_input =
      reinterpret_cast<float*>(alloc_bytes(sizeof(float) * kFftLength));
  g_runtime.fft_output = reinterpret_cast<Complex<float>*>(
      alloc_bytes(sizeof(Complex<float>) * kFftBins));
  g_runtime.power_spectrum =
      reinterpret_cast<float*>(alloc_bytes(sizeof(float) * kFftBins));
  g_runtime.mel_scratch =
      reinterpret_cast<float*>(alloc_bytes(sizeof(float) * kNumMelBins));

  g_runtime.rfft_state_size = tflm_signal::RfftFloatGetNeededMemory(kFftLength);
  g_runtime.rfft_state_buffer = alloc_bytes(g_runtime.rfft_state_size);

  if (g_runtime.clip_buffer == nullptr || g_runtime.padded_audio == nullptr ||
      g_runtime.hann_window == nullptr || g_runtime.mel_weights == nullptr ||
      g_runtime.fft_input == nullptr || g_runtime.fft_output == nullptr ||
      g_runtime.power_spectrum == nullptr || g_runtime.mel_scratch == nullptr ||
      g_runtime.rfft_state_buffer == nullptr) {
    Serial.println("[FATAL] Audio work buffer allocation failed");
    return false;
  }

  g_runtime.rfft_state =
      tflm_signal::RfftFloatInit(kFftLength, g_runtime.rfft_state_buffer,
                                 g_runtime.rfft_state_size);
  if (g_runtime.rfft_state == nullptr) {
    Serial.println("[FATAL] RFFT init failed");
    return false;
  }
  return true;
}

bool AudioContextEngine::init_frontend() {
  for (int i = 0; i < kFrameLengthSamples; ++i) {
    g_runtime.hann_window[i] =
        0.5f - (0.5f * cosf((2.0f * kPi * static_cast<float>(i)) /
                            static_cast<float>(kFrameLengthSamples)));
  }

  const float mel_min = hz_to_mel(kLowerEdgeHz);
  const float mel_max = hz_to_mel(kUpperEdgeHz);
  float mel_edges[kNumMelBins + 2];
  float hz_edges[kNumMelBins + 2];
  for (int i = 0; i < kNumMelBins + 2; ++i) {
    mel_edges[i] = mel_min +
                   (static_cast<float>(i) * (mel_max - mel_min) /
                    static_cast<float>(kNumMelBins + 1));
    hz_edges[i] = mel_to_hz(mel_edges[i]);
  }

  for (int mel = 0; mel < kNumMelBins; ++mel) {
    const float lower = hz_edges[mel];
    const float center = hz_edges[mel + 1];
    const float upper = hz_edges[mel + 2];
    for (int bin = 0; bin < kFftBins; ++bin) {
      const float hz =
          (static_cast<float>(bin) * static_cast<float>(kSampleRateHz)) /
          static_cast<float>(kFftLength);
      float weight = 0.0f;
      if (hz > lower && hz <= center) {
        weight = (hz - lower) / std::max(1e-6f, center - lower);
      } else if (hz > center && hz < upper) {
        weight = (upper - hz) / std::max(1e-6f, upper - center);
      }
      g_runtime.mel_weights[(mel * kFftBins) + bin] = weight;
    }
  }

  return true;
}

bool AudioContextEngine::init_model() {
  g_runtime.model = tflite::GetModel(selected_model_data());
  if (g_runtime.model == nullptr ||
      g_runtime.model->version() != TFLITE_SCHEMA_VERSION) {
    Serial.println("[FATAL] Audio model schema mismatch");
    return false;
  }

  g_runtime.tensor_arena = static_cast<uint8_t*>(
      heap_caps_malloc(kTensorArenaSize, MALLOC_CAP_SPIRAM | MALLOC_CAP_8BIT));
  if (g_runtime.tensor_arena == nullptr) {
    g_runtime.tensor_arena =
        static_cast<uint8_t*>(heap_caps_malloc(kTensorArenaSize, MALLOC_CAP_8BIT));
  }
  if (g_runtime.tensor_arena == nullptr) {
    Serial.println("[FATAL] Tensor arena allocation failed");
    return false;
  }

  static tflite::MicroMutableOpResolver<5> resolver;
  static bool resolver_initialized = false;
  if (!resolver_initialized) {
    if (resolver.AddConv2D() != kTfLiteOk ||
        resolver.AddDepthwiseConv2D() != kTfLiteOk ||
        resolver.AddMean() != kTfLiteOk ||
        resolver.AddPad() != kTfLiteOk ||
        resolver.AddFullyConnected() != kTfLiteOk) {
      Serial.println("[FATAL] Audio op resolver init failed");
      return false;
    }
    resolver_initialized = true;
  }
  static tflite::MicroInterpreter static_interpreter(
      g_runtime.model, resolver, g_runtime.tensor_arena, kTensorArenaSize);
  g_runtime.interpreter = &static_interpreter;

  if (g_runtime.interpreter->AllocateTensors() != kTfLiteOk) {
    Serial.println("[FATAL] AllocateTensors failed");
    return false;
  }

  g_runtime.input = g_runtime.interpreter->input(0);
  g_runtime.output = g_runtime.interpreter->output(0);
  if (g_runtime.input == nullptr || g_runtime.output == nullptr) {
    Serial.println("[FATAL] Model tensors missing");
    return false;
  }

  if (g_runtime.input->dims == nullptr || g_runtime.input->dims->size != 4 ||
      g_runtime.input->dims->data[1] != kNumMelBins ||
      g_runtime.input->dims->data[2] != kFrameCount) {
    Serial.println("[FATAL] Unexpected audio model input shape");
    return false;
  }

  return true;
}

bool AudioContextEngine::capture_samples(int16_t* destination, int sample_count) {
  int collected = 0;
  while (collected < sample_count) {
    const int sample = g_i2s.read();
    if (sample < 0) {
      continue;
    }
    destination[collected++] = static_cast<int16_t>(sample);
  }
  return true;
}

bool AudioContextEngine::infer_current_clip(AudioWindowPrediction* prediction_out) {
  if (prediction_out == nullptr) {
    return false;
  }
  if (!extract_logmel_features_to_model_input(g_runtime.clip_buffer)) {
    Serial.println("[audio] feature extraction failed");
    return false;
  }

  const int64_t start_us = esp_timer_get_time();
  if (g_runtime.interpreter->Invoke() != kTfLiteOk) {
    Serial.println("[audio] Invoke failed");
    return false;
  }
  const int64_t inference_us = esp_timer_get_time() - start_us;

  float logits[kAudioContextClasses] = {};
  int best_index = 0;
  float max_logit = -INFINITY;
  if (g_runtime.output->type == kTfLiteInt8) {
    for (int i = 0; i < kAudioContextClasses; ++i) {
      const int8_t q = g_runtime.output->data.int8[i];
      logits[i] = (static_cast<int>(q) - g_runtime.output->params.zero_point) *
                  g_runtime.output->params.scale;
      if (logits[i] > max_logit) {
        max_logit = logits[i];
        best_index = i;
      }
    }
  } else if (g_runtime.output->type == kTfLiteFloat32) {
    for (int i = 0; i < kAudioContextClasses; ++i) {
      logits[i] = g_runtime.output->data.f[i];
      if (logits[i] > max_logit) {
        max_logit = logits[i];
        best_index = i;
      }
    }
  } else {
    Serial.println("[audio] Unsupported output tensor type");
    return false;
  }

  float exp_sum = 0.0f;
  float probabilities[kAudioContextClasses] = {};
  for (int i = 0; i < kAudioContextClasses; ++i) {
    probabilities[i] = expf(logits[i] - max_logit);
    exp_sum += probabilities[i];
  }
  for (float& probability : probabilities) {
    probability /= std::max(exp_sum, 1e-6f);
  }

  int second_index = (best_index == 0) ? 1 : 0;
  for (int i = 0; i < kAudioContextClasses; ++i) {
    if (i == best_index) {
      continue;
    }
    if (probabilities[i] > probabilities[second_index]) {
      second_index = i;
    }
  }

  prediction_out->label_index = best_index;
  prediction_out->second_label_index = second_index;
  prediction_out->confidence = probabilities[best_index];
  prediction_out->second_confidence = probabilities[second_index];
  prediction_out->clip_rms = compute_rms(g_runtime.clip_buffer, kClipSamples);
  prediction_out->clip_peak = compute_peak_abs(g_runtime.clip_buffer, kClipSamples);
  prediction_out->inference_ms =
      static_cast<float>(static_cast<double>(inference_us) / 1000.0);
  return true;
}

void AudioContextEngine::run_probe_cycle() {
  ++probe_cycle_count_;
  capture_samples(g_runtime.clip_buffer, kProbeSamples);
  const float probe_rms = compute_rms(g_runtime.clip_buffer, kProbeSamples);
  const float probe_peak = compute_peak_abs(g_runtime.clip_buffer, kProbeSamples);

  if (!baseline_initialized_) {
    baseline_rms_ = probe_rms;
    baseline_peak_ = probe_peak;
    baseline_initialized_ = true;
  }

  if (kContinuousInferenceMode) {
    capture_samples(g_runtime.clip_buffer + kProbeSamples,
                    kClipSamples - kProbeSamples);
    run_model_inference(probe_rms, probe_peak);
    update_baseline(probe_rms, probe_peak, false);
    return;
  }

  if (should_label_quiet(probe_rms, probe_peak)) {
    emit_quiet_label(probe_rms, probe_peak);
    update_baseline(probe_rms, probe_peak, true);
    stable_nonquiet_skip_count_ = 0;
    have_last_nonquiet_label_ = false;
    last_label_index_ = -1;
    return;
  }

  if (!should_run_inference(probe_rms, probe_peak)) {
    emit_hold_label(probe_rms, probe_peak);
    update_baseline(probe_rms, probe_peak, false);
    ++stable_nonquiet_skip_count_;
    return;
  }

  stable_nonquiet_skip_count_ = 0;
  capture_samples(g_runtime.clip_buffer + kProbeSamples,
                  kClipSamples - kProbeSamples);
  run_model_inference(probe_rms, probe_peak);
  update_baseline(probe_rms, probe_peak, false);
}

bool AudioContextEngine::should_label_quiet(float probe_rms, float probe_peak) const {
  const float adaptive_rms = baseline_initialized_
                                 ? std::max(kAbsoluteQuietRms,
                                            baseline_rms_ * kQuietBaselineMultiplier)
                                 : kAbsoluteQuietRms;
  const float adaptive_peak = baseline_initialized_
                                  ? std::max(kAbsoluteQuietPeak,
                                             baseline_peak_ * kQuietPeakMultiplier)
                                  : kAbsoluteQuietPeak;
  return probe_rms <= adaptive_rms && probe_peak <= adaptive_peak;
}

bool AudioContextEngine::should_run_inference(float probe_rms, float probe_peak) const {
  if (!have_last_nonquiet_label_) {
    return true;
  }
  const float rms_delta = fabsf(probe_rms - baseline_rms_);
  const float peak_delta = fabsf(probe_peak - baseline_peak_);
  const bool changed =
      (rms_delta > std::max(kChangeRmsFloor, baseline_rms_ * kChangeMultiplier)) ||
      (peak_delta > std::max(kChangePeakFloor, baseline_peak_ * kChangeMultiplier));
  return changed || (stable_nonquiet_skip_count_ >= kMaxStableNonquietSkips);
}

void AudioContextEngine::update_baseline(float probe_rms, float probe_peak,
                                         bool quiet_like) {
  if (!baseline_initialized_) {
    baseline_rms_ = probe_rms;
    baseline_peak_ = probe_peak;
    baseline_initialized_ = true;
    return;
  }
  const float alpha = quiet_like ? kBaselineQuietAlpha : kBaselineActiveAlpha;
  if (quiet_like || probe_rms <= baseline_rms_ * 1.75f) {
    baseline_rms_ = ((1.0f - alpha) * baseline_rms_) + (alpha * probe_rms);
  }
  if (quiet_like || probe_peak <= baseline_peak_ * 1.75f) {
    baseline_peak_ = ((1.0f - alpha) * baseline_peak_) + (alpha * probe_peak);
  }
}

bool AudioContextEngine::extract_logmel_features_to_model_input(
    const int16_t* audio_samples) {
  for (int i = 0; i < kPadSamples; ++i) {
    g_runtime.padded_audio[i] = audio_samples[kPadSamples - i];
  }
  memcpy(g_runtime.padded_audio + kPadSamples, audio_samples,
         sizeof(int16_t) * kClipSamples);
  for (int i = 0; i < kPadSamples; ++i) {
    g_runtime.padded_audio[kPadSamples + kClipSamples + i] =
        audio_samples[kClipSamples - 2 - i];
  }

  if (g_runtime.input->type == kTfLiteFloat32) {
    float* input_data = g_runtime.input->data.f;
    for (int frame = 0; frame < kFrameCount; ++frame) {
      const int start = frame * kFrameStepSamples;
      memset(g_runtime.fft_input, 0, sizeof(float) * kFftLength);
      for (int i = 0; i < kFrameLengthSamples; ++i) {
        g_runtime.fft_input[i] =
            static_cast<float>(g_runtime.padded_audio[start + i]) * kPcmScale *
            g_runtime.hann_window[i];
      }
      tflm_signal::RfftFloatApply(g_runtime.rfft_state, g_runtime.fft_input,
                                  g_runtime.fft_output);

      for (int bin = 0; bin < kFftBins; ++bin) {
        const float real = g_runtime.fft_output[bin].real;
        const float imag = g_runtime.fft_output[bin].imag;
        g_runtime.power_spectrum[bin] = (real * real) + (imag * imag);
      }
      for (int mel = 0; mel < kNumMelBins; ++mel) {
        float mel_energy = 0.0f;
        const float* weights = g_runtime.mel_weights + (mel * kFftBins);
        for (int bin = 0; bin < kFftBins; ++bin) {
          mel_energy += g_runtime.power_spectrum[bin] * weights[bin];
        }
        float mel_db = 10.0f * log10f(std::max(mel_energy, 1e-10f));
        mel_db = std::max(kLogClipMinDb, std::min(kLogClipMaxDb, mel_db));
        input_data[(mel * kFrameCount) + frame] =
            (mel_db - kLogClipMinDb) / (kLogClipMaxDb - kLogClipMinDb);
      }
    }
    return true;
  }

  int8_t* input_data = g_runtime.input->data.int8;
  for (int frame = 0; frame < kFrameCount; ++frame) {
    const int start = frame * kFrameStepSamples;
    memset(g_runtime.fft_input, 0, sizeof(float) * kFftLength);
    for (int i = 0; i < kFrameLengthSamples; ++i) {
      g_runtime.fft_input[i] =
          static_cast<float>(g_runtime.padded_audio[start + i]) * kPcmScale *
          g_runtime.hann_window[i];
    }
    tflm_signal::RfftFloatApply(g_runtime.rfft_state, g_runtime.fft_input,
                                g_runtime.fft_output);

    for (int bin = 0; bin < kFftBins; ++bin) {
      const float real = g_runtime.fft_output[bin].real;
      const float imag = g_runtime.fft_output[bin].imag;
      g_runtime.power_spectrum[bin] = (real * real) + (imag * imag);
    }
    for (int mel = 0; mel < kNumMelBins; ++mel) {
      float mel_energy = 0.0f;
      const float* weights = g_runtime.mel_weights + (mel * kFftBins);
      for (int bin = 0; bin < kFftBins; ++bin) {
        mel_energy += g_runtime.power_spectrum[bin] * weights[bin];
      }
      float mel_db = 10.0f * log10f(std::max(mel_energy, 1e-10f));
      mel_db = std::max(kLogClipMinDb, std::min(kLogClipMaxDb, mel_db));
      const float normalized =
          (mel_db - kLogClipMinDb) / (kLogClipMaxDb - kLogClipMinDb);
      input_data[(mel * kFrameCount) + frame] =
          quantize_feature(normalized, g_runtime.input);
    }
  }
  return true;
}

void AudioContextEngine::emit_quiet_label(float probe_rms, float probe_peak) {
  Serial.printf(
      "[audio] cycle=%lu model=%s label=%s probe_rms=%.1f probe_peak=%.1f baseline_rms=%.1f baseline_peak=%.1f\n",
      static_cast<unsigned long>(probe_cycle_count_), selected_model_name(),
      kQuietLabel, probe_rms, probe_peak, baseline_rms_, baseline_peak_);
}

void AudioContextEngine::emit_hold_label(float probe_rms, float probe_peak) {
  const char* held_label = label_name(last_label_index_);
  Serial.printf(
      "[audio] cycle=%lu model=%s label=%s mode=hold probe_rms=%.1f probe_peak=%.1f baseline_rms=%.1f baseline_peak=%.1f skips=%d\n",
      static_cast<unsigned long>(probe_cycle_count_), selected_model_name(),
      held_label, probe_rms, probe_peak, baseline_rms_, baseline_peak_,
      stable_nonquiet_skip_count_);
}

void AudioContextEngine::run_model_inference(float probe_rms, float probe_peak) {
  AudioWindowPrediction prediction{};
  if (!infer_current_clip(&prediction)) {
    return;
  }

  have_last_nonquiet_label_ = true;
  last_label_index_ = prediction.label_index;

  Serial.printf(
      "[audio] cycle=%lu model=%s label=%s p=%.3f second=%s p2=%.3f probe_rms=%.1f clip_rms=%.1f probe_peak=%.1f clip_peak=%.1f infer_ms=%.2f\n",
      static_cast<unsigned long>(probe_cycle_count_), selected_model_name(),
      label_name(prediction.label_index), prediction.confidence,
      label_name(prediction.second_label_index),
      prediction.second_confidence, probe_rms, prediction.clip_rms, probe_peak,
      prediction.clip_peak, prediction.inference_ms);
}

void AudioContextEngine::mark_summary_active(bool active, uint32_t burst_sequence) {
  if (summary_mutex_ == nullptr) {
    return;
  }
  if (xSemaphoreTake(summary_mutex_, pdMS_TO_TICKS(10)) != pdTRUE) {
    return;
  }
  latest_summary_.active = active;
  latest_summary_.burst_sequence = burst_sequence;
  if (active) {
    latest_summary_.started_at_ms = millis();
  }
  xSemaphoreGive(summary_mutex_);
}

void AudioContextEngine::store_latest_summary(const AudioBurstSummary& summary) {
  if (summary_mutex_ == nullptr) {
    return;
  }
  if (xSemaphoreTake(summary_mutex_, pdMS_TO_TICKS(10)) != pdTRUE) {
    return;
  }
  latest_summary_ = summary;
  xSemaphoreGive(summary_mutex_);
}

void AudioContextEngine::log_boot_summary() const {
  Serial.printf("[audio] ready model=%s arena=%uB input=%dx%dx%d\n",
                selected_model_name(),
                g_runtime.interpreter->arena_used_bytes(),
                g_runtime.input->dims->data[1], g_runtime.input->dims->data[2],
                g_runtime.input->dims->data[3]);
}

float AudioContextEngine::compute_rms(const int16_t* audio_samples,
                                      int sample_count) const {
  double sum_sq = 0.0;
  for (int i = 0; i < sample_count; ++i) {
    const float sample = static_cast<float>(audio_samples[i]);
    sum_sq += static_cast<double>(sample * sample);
  }
  return static_cast<float>(sqrt(sum_sq / static_cast<double>(sample_count)));
}

float AudioContextEngine::compute_peak_abs(const int16_t* audio_samples,
                                           int sample_count) const {
  float peak = 0.0f;
  for (int i = 0; i < sample_count; ++i) {
    peak = std::max(peak, fabsf(static_cast<float>(audio_samples[i])));
  }
  return peak;
}
