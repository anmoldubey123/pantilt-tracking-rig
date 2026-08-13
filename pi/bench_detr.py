import cv2, time
import numpy as np
try:
    import psutil
except ImportError:
    psutil = None
from hailo_platform import (VDevice, HEF, ConfigureParams, HailoStreamInterface,
                            InferVStreams, InputVStreamParams, OutputVStreamParams,
                            FormatType)

HEF_PATH = "/home/pvaris/tracking-demo/detr_resnet_v1_18_bn.hef"
CAM = 0
W, H = 1280, 720
IN_SIZE = 800
PERSON = 1
NUM_CLASSES = 92
CONF = 0.5
METRICS_EVERY = 0.5
WARMUP = 5

CLS_KEY = "detr_resnet_v1_18_bn/conv113"
BOX_KEY = "detr_resnet_v1_18_bn/conv116"

def softmax(x, axis=-1):
    x = x - np.max(x, axis=axis, keepdims=True)
    e = np.exp(x)
    return e / np.sum(e, axis=axis, keepdims=True)

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
ng = vdev.configure(hef, cfg)[0]
ng_params = ng.create_params()
in_info = hef.get_input_vstream_infos()[0]
in_params = InputVStreamParams.make(ng, format_type=FormatType.UINT8)
out_params = OutputVStreamParams.make(ng, format_type=FormatType.FLOAT32)

def decode(result):
    logits = np.array(result[CLS_KEY]).reshape(100, NUM_CLASSES)
    person_p = softmax(logits, axis=-1)[:, PERSON]
    n = int((person_p > CONF).sum())
    return n, float(person_p.max())

infer_hist, loop_hist = [], []
frame_i = 0
last_metrics = time.perf_counter()

try:
    with InferVStreams(ng, in_params, out_params) as pipeline:
        with ng.activate(ng_params):
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

                n, top = decode(result)
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
