import time
import numpy as np
import gi
gi.require_version('Gst', '1.0')
from gi.repository import Gst
from tflite_runtime.interpreter import Interpreter, load_delegate

MODEL    = "/root/vela_out/yolo11n_int8_vela.tflite"
DELEGATE = "/usr/lib/libethosu_delegate.so"
W, H = 1280, 720
SIZE = 640

# --- load model with Ethos-U delegate ---
delegate = load_delegate(DELEGATE)
interp = Interpreter(model_path=MODEL, experimental_delegates=[delegate])
interp.allocate_tensors()
inp = interp.get_input_details()[0]
out = interp.get_output_details()[0]
print("input :", inp["dtype"], inp["shape"], "quant:", inp["quantization"])
print("output:", out["dtype"], out["shape"], "quant:", out["quantization"])

# --- camera via GStreamer ---
Gst.init(None)
PIPELINE = (
    f"v4l2src device=/dev/video0 ! image/jpeg,width={W},height={H},framerate=30/1 ! "
    f"jpegdec ! videoconvert ! video/x-raw,format=RGB ! "
    f"appsink name=sink emit-signals=false max-buffers=1 drop=true sync=false"
)
pipeline = Gst.parse_launch(PIPELINE)
sink = pipeline.get_by_name("sink")
pipeline.set_state(Gst.State.PLAYING)

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

def preprocess(frame):
    # center-crop-free simple resize to 640x640, quantize to int8
    import numpy as np
    small = np.array(
        __import__("PIL.Image", fromlist=["Image"]).Image.fromarray(frame).resize((SIZE, SIZE))
    ) if False else None
    return small

# resize without PIL dependency: use simple stride subsample fallback is bad;
# instead use numpy-based resize via slicing is inaccurate. Use cv-free path:
def resize_rgb(frame):
    # nearest-neighbor resize to SIZE x SIZE using numpy indexing
    yi = (np.linspace(0, frame.shape[0]-1, SIZE)).astype(np.int32)
    xi = (np.linspace(0, frame.shape[1]-1, SIZE)).astype(np.int32)
    return frame[yi][:, xi]

scale, zp = inp["quantization"]

# warmup + timed runs
lat = []
try:
    for i in range(30):
        frame = grab()
        if frame is None:
            print("no frame"); continue
        r = resize_rgb(frame).astype(np.float32) / 255.0
        q = (r / scale + zp).round().astype(inp["dtype"])[None, ...]
        t0 = time.perf_counter()
        interp.set_tensor(inp["index"], q)
        interp.invoke()
        raw = interp.get_tensor(out["index"])
        t1 = time.perf_counter()
        if i >= 5:                      # skip warmup
            lat.append((t1 - t0) * 1000)
    if lat:
        print(f"NPU inference latency: mean {np.mean(lat):.1f} ms  "
              f"min {np.min(lat):.1f}  max {np.max(lat):.1f}  (n={len(lat)})")
finally:
    pipeline.set_state(Gst.State.NULL)
