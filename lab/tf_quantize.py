import numpy as np
import tensorflow as tf

SAVED_MODEL = "/home/lab-ubuntu/tracking-demo/saved_model"
CALIB       = "/home/lab-ubuntu/tracking-demo/calib_data.npy"
OUT         = "/home/lab-ubuntu/tracking-demo/yolo11n_int8.tflite"

# calibration frames: (N, 640, 640, 3) float32 in 0..1
calib = np.load(CALIB)
print("calib:", calib.shape, calib.dtype)

def rep_dataset():
    for i in range(calib.shape[0]):
        # yield one sample at a time, batch dim = 1
        yield [calib[i:i+1].astype(np.float32)]

converter = tf.lite.TFLiteConverter.from_saved_model(SAVED_MODEL)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.representative_dataset = rep_dataset
# force full int8 in/out; falls back gracefully for ops it can't fully quantize
converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
converter.inference_input_type = tf.int8
converter.inference_output_type = tf.float32

tflite_model = converter.convert()
with open(OUT, "wb") as f:
    f.write(tflite_model)
print("wrote", OUT, f"({len(tflite_model)/1e6:.2f} MB)")
