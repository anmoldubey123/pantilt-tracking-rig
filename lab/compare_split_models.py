import numpy as np
import tensorflow as tf

F32 = "/home/lab-ubuntu/tracking-demo/saved_model_split/yolo11n_split_float32.tflite"
I8  = "/home/lab-ubuntu/tracking-demo/yolo11n_split_int8.tflite"
CALIB = "/home/lab-ubuntu/tracking-demo/calib_data.npy"

calib = np.load(CALIB)   # (20,640,640,3) float32 0..1

def load(path):
    it = tf.lite.Interpreter(model_path=path)
    it.allocate_tensors()
    return it

def top_score(interp, frame01):
    inp = interp.get_input_details()[0]
    scale, zp = inp["quantization"]
    if inp["dtype"] == np.int8 and scale != 0:
        x = np.round(frame01 / scale + zp).astype(np.int8)
    else:
        x = frame01.astype(inp["dtype"])
    interp.set_tensor(inp["index"], x)
    interp.invoke()
    for od in interp.get_output_details():
        t = interp.get_tensor(od["index"])
        if t.shape[1] == 80:               # scores branch, by shape
            return float(t[0].max())
    return None

f32 = load(F32)
i8  = load(I8)

print(f"{'frame':>5}  {'float32':>9}  {'int8':>9}")
for i in range(calib.shape[0]):
    frame = calib[i:i+1]
    s32 = top_score(f32, frame)
    s8  = top_score(i8,  frame)
    print(f"{i:>5}  {s32:>9.4f}  {s8:>9.4f}")
