import cv2, time, sys
import numpy as np
import serial
from hailo_platform import (VDevice, HEF, ConfigureParams, HailoStreamInterface,
                            InferVStreams, InputVStreamParams, OutputVStreamParams,
                            FormatType)

# --- config ---
HEF_PATH = "/home/pvaris/tracking-demo/yolov11n.hef"
CAM = 0
W, H = 1280, 720
IN_SIZE = 640
PERSON = 0
CONF = 0.4
SERIAL_PORT = "/dev/ttyACM0"
BAUD = 115200

# --- control constants (held-constant across targets) ---
PAN_SIGN, TILT_SIGN = -1, -1
GAIN = 0.08
DEADBAND = 0.04
MAX_STEP = 0.10
HEAD_BIAS = 0.12
METRICS_EVERY = 0.5

def clamp(v, lo=-1.0, hi=1.0):
    return max(lo, min(hi, v))

# --- serial with REPL-recovery handshake ---
class Pico:
    def __init__(self, port, baud):
        self.ser = serial.Serial(port, baud, timeout=0.05, write_timeout=0.02)
        time.sleep(0.2)
        self.handshake()
    def handshake(self, timeout=4.0):
        # Ctrl-D soft reboot to recover a REPL-dropped Pico, wait for READY
        self.ser.write(b"\x04")
        buf = ""
        t0 = time.time()
        while time.time() - t0 < timeout:
            chunk = self.ser.read(256)
            if chunk:
                buf += chunk.decode(errors="replace")
                if "READY" in buf:
                    print("handshake: Pico ready")
                    self.ser.reset_input_buffer()
                    return True
            time.sleep(0.05)
        print("handshake: no READY seen, proceeding anyway")
        self.ser.reset_input_buffer()
        return False
    def send(self, pan, tilt):
        self.ser.write(f"AIM {pan:.3f} {tilt:.3f}\n".encode())
        self.ser.reset_input_buffer()   # do NOT read replies (freeze fix)
    def center(self):
        self.ser.write(b"CENTER\n")
        self.ser.reset_input_buffer()

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
out_key = hef.get_output_vstream_infos()[0].name
in_params = InputVStreamParams.make(ng, format_type=FormatType.UINT8)
out_params = OutputVStreamParams.make(ng, format_type=FormatType.FLOAT32)

def best_person(result):
    # [batch][80 classes] -> (N,5) [ymin,xmin,ymax,xmax,score], normalized 0..1
    dets = result[out_key][0][PERSON]
    if len(dets) == 0:
        return None
    i = int(np.argmax(dets[:, 4]))
    if float(dets[i, 4]) < CONF:
        return None
    ymin, xmin, ymax, xmax, score = dets[i]
    cx = (xmin + xmax) / 2.0
    cy = ymin + HEAD_BIAS * (ymax - ymin)
    return float(cx), float(cy), float(score)

# --- init ---
pico = Pico(SERIAL_PORT, BAUD)
pan_cmd, tilt_cmd = 0.0, 0.0
pico.send(pan_cmd, tilt_cmd)

infer_hist, loop_hist = [], []
frame_i = 0
last_metrics = time.perf_counter()

try:
    with InferVStreams(ng, in_params, out_params) as pipeline:
        with ng.activate(ng_params):
            while True:
                t_loop = time.perf_counter()
                ok, frame = cap.read()
                if not ok:
                    continue

                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                resized = cv2.resize(rgb, (IN_SIZE, IN_SIZE))
                inp = np.expand_dims(resized, 0).astype(np.uint8)

                t0 = time.perf_counter()
                result = pipeline.infer({in_info.name: inp})
                infer_ms = (time.perf_counter() - t0) * 1000

                det = best_person(result)
                target = "none"
                if det is not None:
                    cx, cy, score = det
                    target = "person"
                    err_x = (cx - 0.5) * 2.0
                    err_y = (cy - 0.5) * 2.0
                    if abs(err_x) > DEADBAND:
                        pan_cmd = clamp(pan_cmd + clamp(PAN_SIGN * GAIN * err_x, -MAX_STEP, MAX_STEP))
                    if abs(err_y) > DEADBAND:
                        tilt_cmd = clamp(tilt_cmd + clamp(TILT_SIGN * GAIN * err_y, -MAX_STEP, MAX_STEP))
                    pico.send(pan_cmd, tilt_cmd)

                frame_i += 1
                loop_ms = (time.perf_counter() - t_loop) * 1000
                infer_hist.append(infer_ms); loop_hist.append(loop_ms)
                if len(infer_hist) > 60: infer_hist.pop(0)
                if len(loop_hist) > 60: loop_hist.pop(0)

                now = time.perf_counter()
                if now - last_metrics >= METRICS_EVERY:
                    avg_inf = sum(infer_hist)/len(infer_hist)
                    avg_loop = sum(loop_hist)/len(loop_hist)
                    loop_fps = 1000.0/avg_loop if avg_loop > 0 else 0
                    print(f"METRICS infer_ms={avg_inf:.1f} loop_fps={loop_fps:.1f} "
                          f"target={target} pan={pan_cmd:.2f} tilt={tilt_cmd:.2f}")
                    last_metrics = now
except KeyboardInterrupt:
    pass
finally:
    try:
        pico.center()
        time.sleep(1.0)   # let the slew finish before we exit
    except Exception:
        pass
    cap.release()
