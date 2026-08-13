import time
import numpy as np
import gi
gi.require_version('Gst', '1.0')
from gi.repository import Gst

Gst.init(None)

W, H = 1280, 720
PIPELINE = (
    f"v4l2src device=/dev/video0 ! "
    f"image/jpeg,width={W},height={H},framerate=30/1 ! "
    f"jpegdec ! videoconvert ! video/x-raw,format=RGB ! "
    f"appsink name=sink emit-signals=false max-buffers=1 drop=true sync=false"
)

pipeline = Gst.parse_launch(PIPELINE)
sink = pipeline.get_by_name("sink")
pipeline.set_state(Gst.State.PLAYING)

def grab(timeout_s=1.0):
    sample = sink.emit("try-pull-sample", int(timeout_s * Gst.SECOND))
    if sample is None:
        return None
    buf = sample.get_buffer()
    ok, mapinfo = buf.map(Gst.MapFlags.READ)
    if not ok:
        return None
    frame = np.frombuffer(mapinfo.data, dtype=np.uint8).reshape((H, W, 3)).copy()
    buf.unmap(mapinfo)
    return frame

n, t0 = 0, time.time()
try:
    while n < 60:
        f = grab()
        if f is None:
            print("no frame (timeout)")
            continue
        if n == 0:
            print("first frame shape:", f.shape, "dtype:", f.dtype)
        n += 1
    dt = time.time() - t0
    print(f"grabbed {n} frames in {dt:.2f}s -> {n/dt:.1f} fps")
finally:
    pipeline.set_state(Gst.State.NULL)
