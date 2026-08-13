import cv2, time, sys
import numpy as np
import psutil
from ultralytics import YOLO

CAM = 0
W, H = 1280, 720
MODEL = "yolo11n.pt"
PERSON = 0
CONF = 0.4
METRICS_EVERY = 0.5
WARMUP = 5          # frames to discard before timing (JIT / allocator warmup)

model = YOLO(MODEL)

cap = cv2.VideoCapture(CAM, cv2.CAP_V4L2)
cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
cap.set(cv2.CAP_PROP_FRAME_WIDTH, W)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, H)
if not cap.isOpened():
    print("ERROR: camera failed to open on /dev/video%d" % CAM)
    sys.exit(1)

# confirm negotiated format
aw = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
ah = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
print(f"capture negotiated: {aw}x{ah}")

infer_ms_hist = []
loop_hist = []
frame_i = 0
last_metrics = time.perf_counter()

try:
    while True:
        t_loop = time.perf_counter()

        t_cap = time.perf_counter()
        ok, frame = cap.read()
        if not ok:
            continue
        cap_ms = (time.perf_counter() - t_cap) * 1000

        # isolated inference timer around ONLY the model call
        t0 = time.perf_counter()
        res = model(frame, verbose=False, conf=CONF, classes=[PERSON])
        infer_ms = (time.perf_counter() - t0) * 1000

        frame_i += 1
        if frame_i <= WARMUP:
            continue

        loop_ms = (time.perf_counter() - t_loop) * 1000
        infer_ms_hist.append(infer_ms)
        loop_hist.append(loop_ms)
        if len(infer_ms_hist) > 60: infer_ms_hist.pop(0)
        if len(loop_hist) > 60: loop_hist.pop(0)

        now = time.perf_counter()
        if now - last_metrics >= METRICS_EVERY:
            n = len(res[0].boxes) if res and res[0].boxes is not None else 0
            avg_inf = sum(infer_ms_hist) / len(infer_ms_hist)
            avg_loop = sum(loop_hist) / len(loop_hist)
            loop_fps = 1000.0 / avg_loop if avg_loop > 0 else 0
            cap_fps = 1000.0 / cap_ms if cap_ms > 0 else 0
            load = psutil.cpu_percent()
            print(f"METRICS infer_ms={avg_inf:.1f} loop_fps={loop_fps:.1f} "
                  f"cap_fps={cap_fps:.1f} load={load:.0f}% dets={n}")
            last_metrics = now
except KeyboardInterrupt:
    pass
finally:
    cap.release()
    if infer_ms_hist:
        print(f"\nSUMMARY infer_ms avg={sum(infer_ms_hist)/len(infer_ms_hist):.1f} "
              f"min={min(infer_ms_hist):.1f} max={max(infer_ms_hist):.1f}")
