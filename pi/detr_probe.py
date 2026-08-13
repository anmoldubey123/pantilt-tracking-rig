import cv2
import numpy as np
from hailo_platform import (VDevice, HEF, ConfigureParams, HailoStreamInterface,
                            InferVStreams, InputVStreamParams, OutputVStreamParams,
                            FormatType)

HEF_PATH = "/home/pvaris/tracking-demo/detr_resnet_v1_18_bn.hef"
CAM = 0
W, H = 1280, 720
IN_SIZE = 800
PERSON = 1          # DETR 92-class: 0=N/A, 1=person, ..., 91=no-object
NUM_CLASSES = 92

CLS_KEY = "detr_resnet_v1_18_bn/conv113"   # (1,100,92) logits
BOX_KEY = "detr_resnet_v1_18_bn/conv116"   # (1,100,4)  cx,cy,w,h norm

def softmax(x, axis=-1):
    x = x - np.max(x, axis=axis, keepdims=True)
    e = np.exp(x)
    return e / np.sum(e, axis=axis, keepdims=True)

# --- one frame ---
cap = cv2.VideoCapture(CAM, cv2.CAP_V4L2)
cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
cap.set(cv2.CAP_PROP_FRAME_WIDTH, W)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, H)
assert cap.isOpened(), "camera failed to open"
for _ in range(5):
    cap.read()
ok, frame = cap.read()
cap.release()
assert ok, "frame grab failed"
print("frame:", frame.shape)

rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
resized = cv2.resize(rgb, (IN_SIZE, IN_SIZE))
inp = np.expand_dims(resized, 0).astype(np.uint8)   # (1,800,800,3)

vdev = VDevice()
hef = HEF(HEF_PATH)
cfg = ConfigureParams.create_from_hef(hef, interface=HailoStreamInterface.PCIe)
ng = vdev.configure(hef, cfg)[0]
ng_params = ng.create_params()
in_info = hef.get_input_vstream_infos()[0]
in_params = InputVStreamParams.make(ng, format_type=FormatType.UINT8)
out_params = OutputVStreamParams.make(ng, format_type=FormatType.FLOAT32)

with InferVStreams(ng, in_params, out_params) as pipeline:
    with ng.activate(ng_params):
        result = pipeline.infer({in_info.name: inp})

logits = np.array(result[CLS_KEY]).reshape(100, NUM_CLASSES)  # (100,92)
boxes  = 1.0/(1.0+np.exp(-np.array(result[BOX_KEY]).reshape(100,4)))  # sigmoid->0..1
print("logits:", logits.shape, "boxes:", boxes.shape)

probs = softmax(logits, axis=-1)          # (100,92)
person_p = probs[:, PERSON]               # (100,)
order = np.argsort(person_p)[::-1]

print("\ntop 5 queries by person prob:")
for i in order[:5]:
    argmax_cls = int(np.argmax(probs[i]))
    cx, cy, w, h = boxes[i]
    print(f"  q{i:3d} person_p={person_p[i]:.3f}  argmax_cls={argmax_cls} "
          f"box(cx={cx:.3f} cy={cy:.3f} w={w:.3f} h={h:.3f})")

n = int((person_p > 0.5).sum())
print(f"\nqueries with person_p>0.5: {n}")
