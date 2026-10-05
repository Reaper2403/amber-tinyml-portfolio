# Recorded device observations

These are selected verbatim sections from historical technical notes. They are not new measurements. Original headings are retained.


Source: `Reaper2403/amber-watchhar-stage1/progress.md`. Original SHA-256: `ef85136fa65ed64d23580f5bfa0aba95c3fa9cf25e32dfd08bb8068b2678b8ea`.


### Firmware Bring-Up

- Built a repeatable local export pipeline from PyTorch checkpoint to:
  - Keras model
  - float32 and int8 TFLite
  - generated firmware `.h` headers
- Brought Stage 1 inference up on the ESP32-S3.
- Reduced on-device inference from about `460 ms` in float32 to about `30 ms` in int8.
- Confirmed the public-pretrained model could detect broader behaviors like
  `walk_transition`, while fine-grained office behaviors were still weak.

### Fine-Tuned Export And Device Test

- Exported the office-finetuned checkpoint through the repeatable local pipeline.
- Refreshed the generic firmware model headers with the fine-tuned model.
- Switched firmware back from capture mode to inference mode.
- Confirmed on-device typing behavior improved noticeably, with `typing_keyboard`
  winning many real typing windows at useful confidence while staying at about
  `30 ms` per inference.


Source: `Reaper2403/amber_audio_context/AUDIO_MODEL_LEARNINGS.md`. Original SHA-256: `8562bcb450b6a893918227247121c4912df92b7ddb4da219b4a755c2d860d21e`.


## Export Bug That Had To Be Fixed

- The first deployment attempt failed even though the notebook metrics looked decent.
- Raw checkpoint evaluation proved:
  - the trained stage2 checkpoint was fine
  - the deployed behavior was wrong
- The bug was in the converter's Keras reconstruction:
  - using Keras `padding=\"same\"` instead of explicit symmetric padding
  - using default Keras BatchNorm epsilon instead of `1e-5`
- The corrected reconstruction requires:
  - stem:
    - `ZeroPadding2D(((2,2),(2,2)))`
    - `Conv2D(..., padding=\"valid\")`
  - each DS block:
    - `ZeroPadding2D(((1,1),(1,1)))`
    - `DepthwiseConv2D(..., padding=\"valid\")`
  - all BatchNorm layers:
    - `epsilon=1e-5`
- After that fix:
  - single-sample Keras logits matched PyTorch to about `1e-6`
  - full device-val and device-test metrics matched the raw checkpoint
- The corrected exported graph also introduced a `PAD` op, so firmware had to register:
  - `PAD`

## Firmware Runtime Learnings

- The on-device frontend must scale raw int16 PCM by `1 / 32768.0` before the window/STFT path.
- Without that scaling, frontend saturation makes many clips look too similar.
- With the corrected export and corrected frontend scaling:
  - `stage2_finetune` produced sensible live labels on the real watch
  - `speech_present` became strong when the user actually spoke near the watch
  - quieter/non-speech windows tended toward `nonsocial_noise` or `meeting_like`
- Current prototype runtime shape on the clean-slate project:
  - mode: continuous for debugging
  - clip length: `2 s`
  - TFLM `Invoke()` time: about `84 ms`
  - actual arena use: about `89,836` bytes
- Important interpretation:
  - `infer_ms` is model invocation time only
  - end-to-end decision cadence is dominated by the `2 s` capture window, not by compute
