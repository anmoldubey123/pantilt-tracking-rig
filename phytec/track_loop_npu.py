import os, time, termios
import numpy as np
import gi
gi.require_version('Gst', '1.0')
from gi.repository import Gst
from tflite_runtime.interpreter import Interpreter, load_delegate

MODEL    = "/root/vela_out/yolo11n_split_int8_vela.tflite"
DELEGATE = "/usr/lib/libethosu_delegate.so"
SERIAL_PORT = "/dev/ttyACM0"
W, H = 1280, 720
SIZE = 640
PERSON = 0
CONF   = 0.10

GAIN = 0.08
DEADBAND = 0.04
MAX_STEP = 0.10
PAN_SIGN = -1
TILT_SIGN = -1
METRICS_EVERY = 0.5

# --- stdlib serial (replaces pyserial) ---
class SerialPort:
    def __init__(self, dev, baud=115200):
        self.fd = os.open(dev, os.O_RDWR | os.O_NOCTTY | os.O_NONBLOCK)
        attrs = termios.tcgetattr(self.fd)
        # [iflag, oflag, cflag, lflag, ispeed, ospeed, cc]
        speed = getattr(termios, f"B{baud}")
        attrs[4] = speed          # ispeed
        attrs[5] = speed          # ospeed
        # 8N1, raw
        attrs[2] = (attrs[2] & ~termios.CSIZE) | termios.CS8
        attrs[2] &= ~termios.PARENB
        attrs[2] &= ~termios.CSTOPB
        attrs[2] |= (termios.CLOCAL | termios.CREAD)
        attrs[0] = 0              # iflag: raw input
        attrs[1] = 0              # oflag: raw output
        attrs[3] = 0              # lflag: non-canonical, no echo
        termios.tcsetattr(self.fd, termios.TCSANOW, attrs)
        termios.tcflush(self.fd, termios.TCIOFLUSH)

    def write(self, data):
        os.write(self.fd, data)

    def flush_in(self):
        termios.tcflush(self.fd, termios.TCIFLUSH)

    def handshake(self, timeout=4.0):
        # soft-reboot (Ctrl-D) to recover a REPL-dropped Pico, wait for READY
        os.write(self.fd, b"\x04")
        buf = ""
        t0 = time.time()
        while time.time() - t0 < timeout:
            try:
                c = os.read(self.fd, 256)
                if c:
                    buf += c.decode(errors="replace")
                    if "READY" in buf:
                        print("handshake: Pico ready")
                        termios.tcflush(self.fd, termios.TCIOFLUSH)
                        return True
            except BlockingIOError:
                pass
            time.sleep(0.05)
        print("handshake: no READY seen, proceeding anyway")
        termios.tcflush(self.fd, termios.TCIOFLUSH)
        return False

    def close(self):
        os.close(self.fd)

# --- model ---
delegate = load_delegate(DELEGATE)
interp = Interpreter(model_path=MODEL, experimental_delegates=[delegate])
interp.allocate_tensors()
inp = interp.get_input_details()[0]
in_scale, in_zp = inp["quantization"]
BOX_IDX = SCORE_IDX = None
for _od in interp.get_output_details():
    if _od["shape"][1] == 4:
        BOX_IDX = _od["index"]
    elif _od["shape"][1] == 80:
        SCORE_IDX = _od["index"]
assert BOX_IDX is not None and SCORE_IDX is not None, "split outputs not found"

# --- camera ---
Gst.init(None)
PIPELINE = (
    f"v4l2src device=/dev/video0 ! image/jpeg,width={W},height={H},framerate=30/1 ! "
    f"jpegdec ! videoconvert ! video/x-raw,format=RGB ! "
    f"appsink name=sink emit-signals=false max-buffers=1 drop=true sync=false"
)
pipeline = Gst.parse_launch(PIPELINE)
sink = pipeline.get_by_name("sink")
pipeline.set_state(Gst.State.PLAYING)

# --- serial ---
ser = SerialPort(SERIAL_PORT, 115200)
time.sleep(0.2)
ser.handshake()
time.sleep(1.0)
ser.flush_in()

def send(pan, tilt):
    ser.write(f"AIM {pan:.3f} {tilt:.3f}\n".encode())
    ser.flush_in()

def clamp(v, lo=-1.0, hi=1.0):
    return max(lo, min(hi, v))

def grab():
    s = sink.emit("try-pull-sample", int(1.0 * Gst.SECOND))
    if s is None:
        return None
    buf = s.get_buffer()
    ok, m = buf.map(Gst.MapFlags.READ)
    if not ok:
        return None
    frame = np.frombuffer(m.data, dtype=np.uint8).reshape((H, W, 3)).copy()
    buf.unmap(m)
    return frame

def resize_rgb(frame):
    yi = np.linspace(0, frame.shape[0]-1, SIZE).astype(np.int32)
    xi = np.linspace(0, frame.shape[1]-1, SIZE).astype(np.int32)
    return frame[yi][:, xi]

def best_person(boxes_t, scores_t):
    boxes  = boxes_t[0]          # (4, 8400)
    scores = scores_t[0][PERSON] # (8400,) person-class scores
    i = int(np.argmax(scores))
    if float(scores[i]) < CONF:
        return None
    cx, cy, bw, bh = boxes[:, i]
    return cx, cy, bw, bh

pan_cmd, tilt_cmd = 0.0, 0.0
send(pan_cmd, tilt_cmd)

inf_ms = 0.0
last_metrics = time.perf_counter()

try:
    while True:
        t_loop = time.perf_counter()
        frame = grab()
        if frame is None:
            continue
        r = resize_rgb(frame).astype(np.float32) / 255.0
        q = (r / in_scale + in_zp).round().astype(inp["dtype"])[None, ...]

        t0 = time.perf_counter()
        interp.set_tensor(inp["index"], q)
        interp.invoke()
        boxes_t  = interp.get_tensor(BOX_IDX)
        scores_t = interp.get_tensor(SCORE_IDX)
        inf_ms = (time.perf_counter() - t0) * 1000

        det = best_person(boxes_t, scores_t)
        if det is not None:
            cx, cy, bw, bh = det
            head_y = (cy - bh/2) + 0.12 * bh
            err_x = (cx - SIZE/2) / (SIZE/2)
            err_y = (head_y - SIZE/2) / (SIZE/2)
            if abs(err_x) > DEADBAND:
                step = clamp(GAIN * err_x, -MAX_STEP, MAX_STEP)
                pan_cmd = clamp(pan_cmd + PAN_SIGN * step)
            if abs(err_y) > DEADBAND:
                step = clamp(GAIN * err_y, -MAX_STEP, MAX_STEP)
                tilt_cmd = clamp(tilt_cmd + TILT_SIGN * step)
            send(pan_cmd, tilt_cmd)

        loop_fps = 1.0 / (time.perf_counter() - t_loop)
        now = time.perf_counter()
        if now - last_metrics >= METRICS_EVERY:
            tgt = "person" if det is not None else "none"
            print(f"METRICS infer_ms={inf_ms:.1f} loop_fps={loop_fps:.1f} target={tgt}")
            last_metrics = now
finally:
    send(0.0, 0.0)
    pipeline.set_state(Gst.State.NULL)
    ser.close()
