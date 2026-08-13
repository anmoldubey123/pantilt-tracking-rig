import os, time
import cv2

OUT = os.path.expanduser("~/calib_images")
os.makedirs(OUT, exist_ok=True)
N = 20
CAM_INDEX = 4

cap = cv2.VideoCapture(CAM_INDEX, cv2.CAP_V4L2)
cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
if not cap.isOpened():
    raise SystemExit("camera not opened")

print("Capturing 20 frames over ~10s - move around / vary the scene")
n = 0
while n < N:
    ok, frame = cap.read()
    if not ok:
        continue
    path = os.path.join(OUT, f"calib_{n:02d}.jpg")
    cv2.imwrite(path, frame)
    print("saved", path)
    n += 1
    time.sleep(0.5)   # spread captures out so frames differ

cap.release()
print(f"done - {n} frames in {OUT}")
