import os, glob, sys
import numpy as np
from PIL import Image

# builds calibration data for onnx2tf full-int8 quantization
# YOLO11n expects 640x640x3, float32, normalized 0..1, NCHW or NHWC per exporter
SRC = sys.argv[1] if len(sys.argv) > 1 else "calib_images"
OUT = "calib_data.npy"
N   = 20            # number of samples to use
SIZE = 640

paths = sorted(glob.glob(os.path.join(SRC, "*")))[:N]
if not paths:
    print(f"no images found in {SRC}/"); sys.exit(1)

arr = []
for p in paths:
    img = Image.open(p).convert("RGB").resize((SIZE, SIZE))
    a = np.asarray(img, dtype=np.float32) / 255.0   # 0..1
    arr.append(a)

data = np.stack(arr, axis=0)   # (N, 640, 640, 3) NHWC
np.save(OUT, data)
print(f"saved {OUT} shape={data.shape} dtype={data.dtype} from {len(paths)} images")
