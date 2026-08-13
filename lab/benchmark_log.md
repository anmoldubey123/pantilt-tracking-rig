# Tracking pipeline benchmark log

Fixed rig: innomaker UVC cam, 1280x720 MJPG capture, YOLO11n (person class),
proportional control -> Pico -> PCA9685 -> MG996R pan/tilt.
Held constant across all compute targets; only the compute board / model changes.

Metric notes:
- infer_ms = isolated timer around model() call only. THE headline number.
- loop_fps = real end-to-end rate (capture + infer + control + draw).
- cap_fps = buffered read speed, NOT true camera throughput (camera ceiling ~30fps
  at 720p MJPG). Treat as diagnostic only, not a performance figure.

| Date | Compute target | Model | Runtime | infer_ms | loop_fps | clk_mhz | load_% | Notes |
|------|----------------|-------|---------|----------|----------|---------|--------|-------|
| 07/29/2026 | i7-1355U (x86 laptop) | YOLO11n | PyTorch CPU | ~44 | ~18 | ~2700 | ~62 | baseline; infer range 41-46ms |

## Transformer model note
CNN-vs-transformer axis: YOLO11n (CNN) vs DETR (transformer detector).
Backbone caveat — the transformer is NOT the identical network on every target:
- Hailo-8L stage runs DETR-ResNet18 (only DETR HEF offered in Model Zoo v2.x).
- Laptop / i.MX93 / QEMU stages run DETR-ResNet50 (official Meta checkpoint;
  no canonical R18 pretrained weights exist upstream — Hailo trained R18 in-house).
Comparison holds at CNN-vs-transformer level; R18-vs-R50 backbone-depth
difference on the Hailo stage is a documented limitation, not a hidden change.

|07/29/2026 | i7-1355U (x86 laptop) | DETR-R50 | PyTorch CPU | ~1720 | ~0.6 | ~2700 | ~82 | transformer; ~40x slower than YOLO11n on same CPU; correct detection but not real-time on CPU |

## i.MX93 Ethos-U65 Vela compile (YOLO11n int8)
Artifact path: ONNX export -> onnx2tf (-tb tf_converter) SavedModel ->
TF TFLiteConverter int8 (representative dataset) -> yolo11n_int8.tflite ->
Vela -> yolo11n_int8_vela.tflite.
Note: strict/flatbuffer_direct onnx2tf quantizer failed on YOLO11 attention
block (model.10 C2PSA); TF's own converter handled it via graceful fallback.

Vela op split: NPU 665 ops (93.8%), CPU 44 ops (6.2%) — strong NPU fit for
this CNN; the ~6% CPU fallback is the attention/misc ops. This split is the
CNN-vs-transformer headline (expect DETR to map far worse).
Compute: 3.26 GMACs/inference. Model size 2.97MB int8 (from ~10MB float32).

CONFIG ASSUMPTIONS — NOT YET NXP-CONFIRMED (verify before trusting perf):
- accelerator: ethos-u65-256 (assumed; i.MX93 may be 256 or 512 MAC)
- system-config: Ethos_U65_High_End; memory-mode: Dedicated_Sram
- Op split (93.8%) is robust to these; bandwidth/perf estimates are NOT.
- Vela summary figures are DESIGN-MODEL estimates, not measured latency.
  Real i.MX93 latency comes from running _vela.tflite on the board (pending).

## RPi5 ARM CPU baseline (YOLO11n, pre-Hailo)
Bare Pi 5 (BCM2712, Cortex-A76), Active Cooler installed, HAT+ NOT yet mounted.
Same held-constant rig: Innomaker UVC cam, 1280x720 MJPG, YOLO11n (person class),
PyTorch CPU via ultralytics. Inference-only benchmark (no gimbal/serial) — isolates
compute. Purpose: pure-ARM data point between the x86 laptop and the coming
Hailo-accelerated run on the same board.

| Date | Compute target | Model | Runtime | infer_ms | loop_fps | clk_mhz | load_% | Notes |
|------|----------------|-------|---------|----------|----------|---------|--------|-------|
| 08/04/2026 | RPi5 (ARM Cortex-A76) | YOLO11n | PyTorch CPU | ~237 | ~4.1 | — | ~51 | MEASURED; infer range 235-238ms, very stable; dets confirmed. load ~51% = PyTorch not saturating all 8 cores (default thread count) — a tuned run could go faster; logged the naive/default number for consistency with laptop baseline. cap_fps ~157 is buffered-read artifact, diagnostic only. |

Cross-target snapshot (measured infer_ms, headline): i7-1355U x86 ~44 | RPi5 ARM CPU ~237 | i.MX93 Ethos-U65 NPU ~164. Note the Pi's ARM CPU is SLOWER than the i.MX93's embedded NPU; Hailo-8L run (pending) expected to leap ahead of all three.

## RPi5 + Hailo-8L AI HAT+ (YOLO11n HEF) — accelerated
Same held-constant rig: Innomaker UVC cam, 1280x720 MJPG, YOLO11n (person class).
Inference-only benchmark (no gimbal/serial) — isolates compute, matches the CPU
baseline methodology. Model: yolov11n HEF from Hailo Model Zoo v2.19.0, compiled
for HAILO8L (640x640x3 UINT8 input; NMS + YOLOv8-style post-process baked into the
HEF, output = HAILO NMS BY CLASS). No manual decode, no quantization tuning — the
Hailo compiler handles quant + NMS on-device; detections arrive confident out of
the box (top scores ~0.85-0.90). Stack: HailoRT 4.23.0, hailo-all 5.1.1, Debian 13
(trixie). Chip confirmed HAILO8L (13 TOPS) via hailortcli.

| Date | Compute target | Model | Runtime | infer_ms | loop_fps | clk_mhz | load_% | Notes |
|------|----------------|-------|---------|----------|----------|---------|--------|-------|
| 08/05/2026 | RPi5 + Hailo-8L (13 TOPS) | YOLO11n | HailoRT 4.23.0 HEF | ~13.1 | ~29.5 | — | — | MEASURED; infer range 12.7-13.5ms, very stable; detection confident (top ~0.88). loop_fps ~29.5 is CAMERA-BOUND not compute-bound — at 13ms the model could run ~76fps but 720p MJPG capture ceiling is ~30fps; infer_ms is the honest figure here. First target where capture, not compute, is the bottleneck. Zoo datasheet quotes 157 FPS batch-1 but under PCIe Gen3 x4 + x86 host; Pi gives single PCIe lane + ARM host, so measured per-inference latency is higher than their ideal-conditions throughput — different conditions, not a discrepancy. |

Pipeline validation note: before swapping in yolo11n, validated the full HailoRT
inference path on the shipped yolov8s_h8l.hef (~17ms infer, top ~0.9) — proved HEF
load, InferVStreams API, and NMS-by-class decode before committing to the benchmark
model. yolov8s is heavier than yolo11n (28.6 vs 6.55 GOPS), hence 17ms vs 13ms.

## Cross-target summary (measured infer_ms, headline)
| Compute target | Model | infer_ms | Notes |
|----------------|-------|----------|-------|
| RPi5 ARM Cortex-A76 (CPU) | YOLO11n | ~237 | PyTorch CPU, default threads |
| i.MX93 Ethos-U65 (NPU) | YOLO11n | ~164 | int8 Vela, split-head fix |
| i7-1355U x86 (CPU) | YOLO11n | ~44 | PyTorch CPU baseline |
| RPi5 + Hailo-8L (NPU) | YOLO11n | ~13 | HEF, camera-bound loop |

Headline finding: Hailo-8L is ~3.4x faster than the x86 laptop CPU, ~13x faster than
the i.MX93 embedded NPU, and ~18x faster than the Pi's own ARM CPU — on the same
held-constant YOLO11n model and capture pipeline. Note the two embedded NPUs sit at
opposite ends: i.MX93 Ethos-U65 (~164ms) is the slowest NPU while Hailo-8L (~13ms) is
the fastest of all targets, underscoring that "NPU" spans a wide performance range.

## RPi5 + Hailo-8L — DETR-R18 (transformer detector)
CNN-vs-transformer axis, measured on the SAME accelerator as the YOLO11n row above
(same Hailo-8L, same host, same 1280x720 MJPG capture, same person target) — only
the model architecture changes. Model: detr_resnet_v1_18_bn HEF, Hailo Model Zoo
v2.19.0, compiled for HAILO8L. Input 800x800x3 UINT8 (note: larger than YOLO's 640).
Unlike the YOLO HEFs (baked-in NMS-by-class), DETR outputs RAW set-prediction heads:
conv113 (1,100,92) class logits over 100 object queries, conv116 (1,100,4) box preds.
Decode done host-side: softmax over 92 class    es (0=N/A, 1=person, 91=no-object),
sigmoid on box outputs (DETR boxes are pre-sigmoid), threshold on person prob.
No NMS needed — DETR set prediction is deduplicated by design.

| Date | Compute target | Model | Runtime | infer_ms | loop_fps | clk_mhz | load_% | Notes |
|------|----------------|-------|---------|----------|----------|---------|--------|-------|
| 08/05/2026 | RPi5 + Hailo-8L (13 TOPS) | DETR-R18 | HailoRT 4.23.0 HEF | ~71.9 | ~11.9 | — | — | MEASURED; infer range 71.9-72.1ms, tightest of any run; detection dead-confident (top=1.00). loop_fps ~12 is COMPUTE-BOUND (DETR slower than 30fps camera) — opposite regime from YOLO11n which was camera-bound. Zoo quotes 23.4 FPS batch-1 (PCIe Gen3 x4 + x86 host); Pi conditions differ. |

## CNN-vs-transformer on Hailo-8L (same-chip, same-conditions — the clean comparison)
| Model | Type | infer_ms | compute-ceiling fps | HW mAP (zoo) | input |
|-------|------|----------|--------------------|--------------|-------|
| YOLO11n | CNN | ~13 | ~76 | 37.5 | 640x640 |
| DETR-R18 | transformer | ~72 | ~14 | 31.5 | 800x800 |

Headline: on identical hardware, the transformer (DETR-R18) is ~5.5x slower than the
CNN (YOLO11n) AND lower accuracy (31.5 vs 37.5 mAP). Measured ratio (~5.5x) tracks the
zoo's batch-1 figures (157 vs 23.4 FPS ~ 6.7x); Pi/PCIe conditions compress both.
Part of DETR's disadvantage here is the lighter R18 backbone + larger 800px input.

CAVEAT — two distinct comparison axes, do not conflate:
- CNN-vs-transformer ON HAILO (above): CLEAN — both models same chip, same conditions.
- Cross-BOARD DETR (laptop/i.MX93/QEMU vs Hailo): backbone mismatch — Hailo runs
  DETR-R18 (only DETR HEF in h8l zoo), other targets run DETR-R50 (no canonical R18
  weights upstream). Documented limitation, not a hidden change.
