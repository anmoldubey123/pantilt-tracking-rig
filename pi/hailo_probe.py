import cv2
import numpy as np
from hailo_platform import (VDevice, HEF, ConfigureParams, HailoStreamInterface,
                            InferVStreams, InputVStreamParams, OutputVStreamParams,
                            FormatType)

HEF_PATH = "/usr/share/hailo-models/yolov8s_h8l.hef"
CAM = 0
W, H = 1280, 720
IN_SIZE = 640

# --- capture one frame ---
cap = cv2.VideoCapture(CAM, cv2.CAP_V4L2)
cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
cap.set(cv2.CAP_PROP_FRAME_WIDTH, W)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, H)
assert cap.isOpened(), "camera failed to open"
for _ in range(5):            # warm up / flush
    cap.read()
ok, frame = cap.read()
cap.release()
assert ok, "frame grab failed"
print("frame:", frame.shape, frame.dtype)

# BGR->RGB, resize to model input, keep uint8 (HEF ingests uint8)
rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
resized = cv2.resize(rgb, (IN_SIZE, IN_SIZE))
inp = np.expand_dims(resized, axis=0).astype(np.uint8)   # (1,640,640,3)
print("input:", inp.shape, inp.dtype)

# --- configure Hailo ---
vdev = VDevice()
hef = HEF(HEF_PATH)
cfg = ConfigureParams.create_from_hef(hef, interface=HailoStreamInterface.PCIe)
network_group = vdev.configure(hef, cfg)[0]
ng_params = network_group.create_params()

in_info = hef.get_input_vstream_infos()[0]
out_info = hef.get_output_vstream_infos()[0]
print("input vstream :", in_info.name)
print("output vstream:", out_info.name)

in_params = InputVStreamParams.make(network_group, format_type=FormatType.UINT8)
out_params = OutputVStreamParams.make(network_group, format_type=FormatType.FLOAT32)

with InferVStreams(network_group, in_params, out_params) as pipeline:
    with network_group.activate(ng_params):
        result = pipeline.infer({in_info.name: inp})

# --- dump raw output structure ---
print("\n=== OUTPUT STRUCTURE ===")
print("result type:", type(result))
print("result keys:", list(result.keys()) if hasattr(result, "keys") else "n/a")
for k, v in result.items():
    print(f"\n[{k}] type={type(v)}")
    arr = v
    # NMS-by-class often arrives as a list (batch) of lists (classes)
    if isinstance(arr, list):
        print(f"  list len={len(arr)} (batch dim)")
        cls_list = arr[0]
        print(f"  per-batch type={type(cls_list)} len={len(cls_list)} (classes)")
        # class 0 = person
        person = cls_list[0]
        print(f"  class[0] (person) type={type(person)} shape/len={getattr(person,'shape',len(person))}")
        print(f"  class[0] contents:\n{person}")
    else:
        print(f"  ndarray shape={getattr(arr,'shape',None)} dtype={getattr(arr,'dtype',None)}")
        print(arr)
