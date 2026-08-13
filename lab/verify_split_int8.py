import numpy as np
import tensorflow as tf

MODEL = "/home/lab-ubuntu/tracking-demo/yolo11n_split_int8.tflite"
CALIB = "/home/lab-ubuntu/tracking-demo/calib_data.npy"

calib = np.load(CALIB)          # (N,640,640,3) float32 0..1
frame = calib[0:1]              # (1,640,640,3)

interp = tf.lite.Interpreter(model_path=MODEL)
interp.allocate_tensors()

inp = interp.get_input_details()[0]
print("input:", inp["dtype"], inp["shape"], "quant:", inp["quantization"])

# input is int8; quantize the 0..1 float frame with the model's own scale/zp
scale, zp = inp["quantization"]
if scale == 0:
    x = frame.astype(inp["dtype"])
else:
    x = np.round(frame / scale + zp).astype(inp["dtype"])
interp.set_tensor(inp["index"], x)
interp.invoke()

# pick boxes vs scores by SHAPE, not index
for od in interp.get_output_details():
    t = interp.get_tensor(od["index"])
    rows = t.shape[1]
    if rows == 4:
        print(f"\nBOXES  {t.shape}: min={t.min():.2f} max={t.max():.2f}  (expect ~0..640)")
    elif rows == 80:
        print(f"\nSCORES {t.shape}: min={t.min():.4f} max={t.max():.4f}")
        # per-anchor best class score
        best = t[0].max(axis=0)      # (8400,)
        print(f"    top score across all anchors = {best.max():.4f}")
        print(f"    anchors scoring >0.4 = {(best > 0.4).sum()}")
