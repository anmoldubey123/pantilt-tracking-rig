import sys, time, platform
from collections import deque
import cv2
import psutil
import serial
import torch
import torchvision.transforms as T
from PIL import Image

# --- config ---
CAM_INDEX   = 4
SERIAL_PORT = "/dev/ttyACM0"
BAUD        = 115200
DETR_CONF   = 0.7      # person-score threshold
PERSON_ID   = 1        # COCO person index in DETR's 92-class logits

GAIN        = 0.08
DEADBAND    = 0.04
MAX_STEP    = 0.10
PAN_SIGN    = -1
TILT_SIGN   = -1

METRICS_EVERY = 0.5
ROLL          = 30

# --- host info (queried once; portable across x86 + ARM boards) ---
def cpu_model():
    try:
        with open("/proc/cpuinfo") as f:
            for line in f:
                if "model name" in line:
                    return line.split(":", 1)[1].strip()
                if line.startswith("Model") and ":" in line:
                    return line.split(":", 1)[1].strip()
    except Exception:
        pass
    return platform.processor() or platform.machine() or "unknown"

HOST_CPU   = cpu_model()
CORES_PHYS = psutil.cpu_count(logical=False)
CORES_LOG  = psutil.cpu_count(logical=True)
FREQ_MAX   = psutil.cpu_freq().max if psutil.cpu_freq() else 0.0

print(f"HOST cpu='{HOST_CPU}' cores={CORES_PHYS} threads={CORES_LOG} max_mhz={FREQ_MAX:.0f}")

# --- model (DETR-R50 via torch hub, CPU) ---
model = torch.hub.load('facebookresearch/detr:main', 'detr_resnet50', pretrained=True)
model.eval()

transform = T.Compose([
    T.Resize(800),
    T.ToTensor(),
    T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

def detect_person(frame_bgr):
    # returns (x1,y1,x2,y2) in pixel coords of best person box, or None
    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    pil = Image.fromarray(rgb)
    inp = transform(pil).unsqueeze(0)
    with torch.no_grad():
        out = model(inp)
    logits = out['pred_logits'][0]          # [100, 92]
    boxes  = out['pred_boxes'][0]           # [100, 4] cx,cy,w,h normalized
    probs  = logits.softmax(-1)
    scores = probs[:, PERSON_ID]            # person score per query
    best_i = int(torch.argmax(scores))
    if float(scores[best_i]) < DETR_CONF:
        return None
    h, w = frame_bgr.shape[:2]
    cx, cy, bw, bh = boxes[best_i].tolist()
    x1 = (cx - bw / 2) * w
    y1 = (cy - bh / 2) * h
    x2 = (cx + bw / 2) * w
    y2 = (cy + bh / 2) * h
    return (x1, y1, x2, y2)

# --- setup ---
cap = cv2.VideoCapture(CAM_INDEX, cv2.CAP_V4L2)
cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
cap.set(cv2.CAP_PROP_FPS, 30)
cv2.namedWindow("track", cv2.WINDOW_NORMAL)
if not cap.isOpened():
    print("ERROR: camera not opened"); sys.exit(1)

ser = serial.Serial(SERIAL_PORT, BAUD, timeout=0.05, write_timeout=0.02)
time.sleep(1.0)
ser.reset_input_buffer()

def send(pan, tilt):
    ser.write(f"AIM {pan:.3f} {tilt:.3f}\n".encode())
    ser.reset_input_buffer()
    return ""

def clamp(v, lo=-1.0, hi=1.0):
    return max(lo, min(hi, v))

def cur_mhz():
    f = psutil.cpu_freq()
    return f.current if f else 0.0

cap_dt  = deque(maxlen=ROLL)
inf_dt  = deque(maxlen=ROLL)
loop_dt = deque(maxlen=ROLL)

def avg(d):
    return sum(d) / len(d) if d else 0.0

pan_cmd, tilt_cmd = 0.0, 0.0
send(pan_cmd, tilt_cmd)

last_metrics = time.perf_counter()

try:
    while True:
        t_loop0 = time.perf_counter()

        t0 = time.perf_counter()
        ok, frame = cap.read()
        t1 = time.perf_counter()
        if not ok:
            continue
        cap_dt.append(t1 - t0)
        h, w = frame.shape[:2]

        t0 = time.perf_counter()
        best = detect_person(frame)
        t1 = time.perf_counter()
        inf_dt.append(t1 - t0)

        if best is not None:
            x1, y1, x2, y2 = best
            cx = (x1 + x2) / 2
            cy = y1 + 0.12 * (y2 - y1)   # head-bias
            err_x = (cx - w / 2) / (w / 2)
            err_y = (cy - h / 2) / (h / 2)

            if abs(err_x) > DEADBAND:
                step = clamp(GAIN * err_x, -MAX_STEP, MAX_STEP)
                pan_cmd = clamp(pan_cmd + PAN_SIGN * step)
            if abs(err_y) > DEADBAND:
                step = clamp(GAIN * err_y, -MAX_STEP, MAX_STEP)
                tilt_cmd = clamp(tilt_cmd + TILT_SIGN * step)

            send(pan_cmd, tilt_cmd)

            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
            cv2.circle(frame, (int(cx), int(cy)), 5, (0, 0, 255), -1)
        else:
            cv2.putText(frame, "no target - holding", (10, h - 15),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), 1, cv2.LINE_AA)

        cv2.line(frame, (w // 2, 0), (w // 2, h), (200, 200, 200), 1)
        cv2.line(frame, (0, h // 2), (w, h // 2), (200, 200, 200), 1)

        inf_ms   = avg(inf_dt) * 1000.0
        cap_fps  = 1.0 / avg(cap_dt) if avg(cap_dt) else 0.0
        loop_fps = 1.0 / avg(loop_dt) if avg(loop_dt) else 0.0
        mhz      = cur_mhz()
        load     = psutil.cpu_percent()

        y = 20
        for text in [
            HOST_CPU + "  [DETR-R50]",
            f"cores {CORES_PHYS}/{CORES_LOG}  clk {mhz:.0f} MHz  load {load:.0f}%",
            f"infer {inf_ms:.1f} ms  cap {cap_fps:.1f} fps  loop {loop_fps:.1f} fps",
        ]:
            cv2.putText(frame, text, (10, y), cv2.FONT_HERSHEY_SIMPLEX,
                        0.5, (255, 0, 255), 1, cv2.LINE_AA)
            y += 22

        cv2.imshow("track", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

        t_loop1 = time.perf_counter()
        loop_dt.append(t_loop1 - t_loop0)

        now = time.perf_counter()
        if now - last_metrics >= METRICS_EVERY:
            print(f"METRICS infer_ms={inf_ms:.1f} cap_fps={cap_fps:.1f} "
                  f"loop_fps={loop_fps:.1f} clk_mhz={mhz:.0f} load_pct={load:.0f}")
            last_metrics = now
finally:
    send(0.0, 0.0)
    cap.release()
    cv2.destroyAllWindows()
    ser.close()
