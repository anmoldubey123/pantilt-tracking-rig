# Physical AI Camera Tracking — Multi-Compute NPU Benchmark

A pan/tilt camera gimbal that runs real-time person detection and tracks a
target by actuating servos. Built as a benchmarking rig: the **compute target is
the swappable independent variable** — the identical tracking pipeline runs
across an x86 laptop, a Raspberry Pi 5 + Hailo-8L, and a Phytec i.MX93 +
Ethos-U65 NPU. Secondary axis: CNN (YOLO11n) vs transformer (DETR) detection.

## Architecture: held-constant rig, swappable compute

Everything except the compute board and inference model is held constant:
- **Camera**: Innomaker UVC, fixed 1280x720 MJPG capture on every target.
- **Gimbal**: 2-DOF pan/tilt, MG996R servos via PCA9685.
- **Actuation controller**: Pico 2 W (MicroPython) receives normalized
  [-1,+1] pan/tilt coordinates over USB serial and owns all servo-pulse
  specifics — this keeps the vision side completely hardware-agnostic.
- **Control**: proportional error from frame center, deadband + per-frame step
  clamp, identical constants across every compute target.

Only the compute board + model change between runs, so measured differences are
attributable to the compute target, not the pipeline.

## Layout

- `firmware/main.py` — Pico 2 W MicroPython. Boot-centering, travel-limit
  clamping, normalized AIM/CENTER/PING serial protocol, on-device servo
  slewing (gentle motion, hardware-agnostic), and a Ctrl-D soft-reboot
  handshake so a REPL-dropped Pico self-recovers on script start.
- `lab/` — x86 laptop (baseline) + the ONNX quantization pipeline:
  - `track_loop.py` — main YOLO11n tracking loop with the instrumentation
    overlay (CPU model, live clock, isolated inference latency, loop FPS) and
    structured `METRICS` log emitter for headless runs.
  - `track_loop_detr.py` — DETR-R50 transformer variant, same harness.
  - `split_concat_inputs.py` — **the graph-surgery fix** (see below).
  - `split_onnx_head.py` — the earlier, failed split attempt (kept as record).
  - `tf_quantize*.py`, `make_calib.py`, `verify_split_int8.py`,
    `compare_split_models.py`, `check_int8.py` — int8 artifact pipeline.
  - `benchmark_log.md` — all measured results + methodology + caveats.
- `phytec/` — i.MX93 + Ethos-U65. `track_loop_npu.py` (full NPU loop),
  `gst_capture_test.py` (GStreamer appsink→numpy capture, no OpenCV),
  serial via stdlib termios (no pyserial) — written for a minimal Yocto image
  with no OpenCV/pip. `npu_infer_test.py`, `decode_debug.py`,
  `serial_test.py`, `pico_probe.py`.
- `pi/` — RPi5 + Hailo-8L. `bench_cpu.py` (ARM CPU baseline),
  `bench_hailo.py` (YOLO11n HEF), `bench_detr.py`/`detr_probe.py`
  (DETR-R18, host-side set-prediction decode), `track_loop_hailo.py`
  (full tracking loop, pyserial).

## Measured results (isolated inference latency, headline metric)

| Compute target | Model | infer_ms |
|----------------|-------|----------|
| RPi5 + Hailo-8L (NPU) | YOLO11n | ~13 |
| i7-1355U x86 (CPU) | YOLO11n | ~44 |
| i.MX93 Ethos-U65 (NPU) | YOLO11n | ~164 |
| RPi5 ARM Cortex-A76 (CPU) | YOLO11n | ~237 |
| RPi5 + Hailo-8L | DETR-R18 | ~72 |

Hailo-8L is fastest overall (~3.4x the x86 laptop, ~13x the i.MX93 NPU) and the
only target where the loop is camera-bound, not compute-bound. On the same chip,
the transformer (DETR-R18, ~72ms) is ~5.5x slower than the CNN (YOLO11n, ~13ms)
and lower accuracy — the clean single-variable CNN-vs-transformer result.

## The interesting fix: a per-tensor quantization scale collision

The i.MX93 int8 model ran end-to-end but detected **nothing** — every class
score was exactly zero.

**Root cause.** YOLO's detection head emits one output tensor `(1,84,8400)`
packing box coordinates (rows 0-3, pixel-space values up to ~640) and 80 class
scores (rows 4-83, sigmoid outputs in 0-1) together. int8 quantization assigns
**one scale per tensor**, sized to the largest magnitude present. The ~640 box
values forced a coarse scale of ~2.63 — meaning the smallest representable
nonzero step was 2.63, so every class score (all < 1.0) rounded to the
zero-point and became exactly 0.0. The model was fine (float32 scored ~0.87);
the information was destroyed purely at quantization.

**Failed first attempt** (`split_onnx_head.py`). Splitting the *output* tensor
into box and score halves — but this cut *downstream* of the graph's final
Concat, where the shared scale was already baked in. Splitting an
already-ruined tensor just yields two tensors, one of which is still all zeros.

**The fix** (`split_concat_inputs.py`). Cut the graph one step earlier, at the
Concat's **inputs**: delete `model.23/Concat_3` and promote its two branch
tensors (box branch `Mul_2`, score branch `Sigmoid`) to separate graph outputs.
Now each is calibrated independently — boxes keep the coarse ~2.63 scale, scores
get a fine ~1/256 ≈ 0.0039 scale and survive.

**Intuition.** Quantization precision is per-tensor and dictated by the largest
value in that tensor. Mixing quantities of very different magnitudes in one
tensor lets the big ones set the "ruler" and erases the small ones. The fix is
to separate them so each gets a ruler sized to its own range.

(Post-fix, int8 scores are real but suppressed vs float32 — normal int8
precision loss, distinct from the shared-scale collapse — so tracking runs at a
lowered confidence threshold; ranking is preserved.)

## Artifact pipeline (i.MX93)

`yolo11n.pt` → ONNX export → onnx2tf (`-tb tf_converter`) SavedModel →
graph surgery (`split_concat_inputs.py`) → TF int8 quantize with representative
dataset → Vela compile for Ethos-U65. The default onnx2tf backend fails on
YOLO11's attention block; TF's own converter handles it via fallback.

## Environment note

Working laptop venv known-good pair: **torch 2.12.1 ↔ torchvision 0.27.1**
(each torchvision release is built against one specific torch version; a
mismatch breaks the `torchvision::nms` operator). Export-tool installs have
twice leaked newer torch into the runtime venv — keep export work in a separate
venv, and `pip install -r requirements-lock.txt` to restore.
