import cv2, time, sys
import numpy as np
try:
    import psutil
except ImportError:
    psutil = None
from hailo_platform import (VDevice, HEF, ConfigureParams, HailoStreamInterface,
                            InferVStreams, InputVStreamParams, OutputVStreamParams,
                            FormatType)

HEF_PATH = "/home/pvaris/tracking-demo/yolov11n.hef"
CAM = 0
W, H = 1280, 720
IN_SIZE = 640
PERSON = 0
CONF = 0.4
METRICS_EVERY = 0.5
WARMUP = 5

# --- camera ---
cap = cv2.VideoCapture(CAM, cv2.CAP_V4L2)
cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
cap.set(cv2.CAP_PROP_FRAME_WIDTH, W)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, H)
assert cap.isOpened(), "camera failed to open"
print(f"capture negotiated: {int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))}x{int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))}")

# --- Hailo setup ---
vdev = VDevice()
hef = HEF(HEF_PATH)
cfg = ConfigureParams.create_from_hef(hef, interface=HailoStreamInterface.PCIe)
network_group = vdev.configure(hef, cfg)[0]
ng_params = network_group.create_params()
in_info = hef.get_input_vstream_infos()[0]
out_key = hef.get_output_vstream_infos()[0].name
in_params = InputVStreamParams.make(network_group, format_type=FormatType.UINT8)
out_params = OutputVStreamParams.make(network_group, format_type=FormatType.FLOAT32)

def best_person(result):
    # result[out_key] -> [batch][80 classes] -> (N,5) [ymin,xmin,ymax,xmax,score]
    dets = result[out_key][0][PERSON]
    if len(dets) == 0:
        return 0, 0.0
    n = int((dets[:, 4] >= CONF).sum())
    return n, float(dets[:, 4].max())

infer_hist, loop_hist = [], []
frame_i = 0
last_metrics = time.perf_counter()

try:
    with InferVStreams(network_group, in_params, out_params) as pipeline:
        with network_group.activate(ng_params):
            while True:
                t_loop = time.perf_counter()

                t_cap = time.perf_counter()
                ok, frame = cap.read()
                if not ok:
                    continue
                cap_ms = (time.perf_counter() - t_cap) * 1000

                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                resized = cv2.resize(rgb, (IN_SIZE, IN_SIZE))
                inp = np.expand_dims(resized, 0).astype(np.uint8)

                t0 = time.perf_counter()
                result = pipeline.infer({in_info.name: inp})
                infer_ms = (time.perf_counter() - t0) * 1000

                frame_i += 1
                if frame_i <= WARMUP:
                    continue

                n, top = best_person(result)
                loop_ms = (time.perf_counter() - t_loop) * 1000
                infer_hist.append(infer_ms); loop_hist.append(loop_ms)
                if len(infer_hist) > 60: infer_hist.pop(0)
                if len(loop_hist) > 60: loop_hist.pop(0)

                now = time.perf_counter()
                if now - last_metrics >= METRICS_EVERY:
                    avg_inf = sum(infer_hist)/len(infer_hist)
                    avg_loop = sum(loop_hist)/len(loop_hist)
                    loop_fps = 1000.0/avg_loop if avg_loop > 0 else 0
                    cap_fps = 1000.0/cap_ms if cap_ms > 0 else 0
                    load = psutil.cpu_percent() if psutil else -1
                    print(f"METRICS infer_ms={avg_inf:.1f} loop_fps={loop_fps:.1f} "
                          f"cap_fps={cap_fps:.1f} load={load:.0f}% dets={n} top={top:.2f}")
                    last_metrics = now
except KeyboardInterrupt:
    pass
finally:
    cap.release()
    if infer_hist:
        print(f"\nSUMMARY infer_ms avg={sum(infer_hist)/len(infer_hist):.1f} "
              f"min={min(infer_hist):.1f} max={max(infer_hist):.1f}")
