import numpy as np
import tensorflow as tf

MODEL = "/home/lab-ubuntu/tracking-demo/yolo11n_int8.tflite"
CALIB = "/home/lab-ubuntu/tracking-demo/calib_data.npy"

interp = tf.lite.Interpreter(model_path=MODEL)
interp.allocate_tensors()
inp = interp.get_input_details()[0]
out = interp.get_output_details()[0]
print("input :", inp["dtype"], inp["shape"], "quant:", inp["quantization"])
print("output:", out["dtype"], out["shape"], "quant:", out["quantization"])

# feed one real calibration frame, quantized to int8 per the input spec
frame = np.load(CALIB)[0:1]                       # (1,640,640,3) float 0..1
scale, zp = inp["quantization"]
q = (frame / scale + zp).round().astype(inp["dtype"])
interp.set_tensor(inp["index"], q)
interp.invoke()
raw = interp.get_tensor(out["index"])
# dequantize output
oscale, ozp = out["quantization"]
deq = (raw.astype(np.float32) - ozp) * oscale
print("raw out dtype/shape:", raw.dtype, raw.shape)
print("dequant range:", float(deq.min()), "to", float(deq.max()))
