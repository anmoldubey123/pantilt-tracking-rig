import numpy as np
import tensorflow as tf

SAVED_MODEL = "/home/lab-ubuntu/tracking-demo/saved_model_split"
CALIB       = "/home/lab-ubuntu/tracking-demo/calib_data.npy"
OUT         = "/home/lab-ubuntu/tracking-demo/yolo11n_split_int8.tflite"

# calibration frames: (N, 640, 640, 3) float32 in 0..1
calib = np.load(CALIB)
print("calib:", calib.shape, calib.dtype)

def rep_dataset():
    for i in range(calib.shape[0]):
        yield [calib[i:i+1].astype(np.float32)]

converter = tf.lite.TFLiteConverter.from_saved_model(SAVED_MODEL)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.representative_dataset = rep_dataset
converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
converter.inference_input_type = tf.int8
converter.inference_output_type = tf.float32

tflite_model = converter.convert()
with open(OUT, "wb") as f:
    f.write(tflite_model)
print("wrote", OUT, f"({len(tflite_model)/1e6:.2f} MB)")

# --- verification: read back each output's quant scale ---
interp = tf.lite.Interpreter(model_path=OUT)
interp.allocate_tensors()
print("\n--- output details ---")
for od in interp.get_output_details():
    qp = od.get("quantization_parameters", {})
    scales = qp.get("scales")
    zps = qp.get("zero_points")
    print(f"name={od['name']}  shape={od['shape']}  dtype={od['dtype']}")
    print(f"    scales={scales}  zero_points={zps}")
print("\nNOTE: output is float32-cast, so these params may be empty.")
print("If empty, the real proof is nonzero scores at runtime on-board.")
