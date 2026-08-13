import onnx
from onnx import helper, TensorProto

SRC = "yolo11n.onnx"
DST = "yolo11n_split.onnx"
CONCAT = "/model.23/Concat_3"
BOX_T   = "/model.23/Mul_2_output_0"      # box branch, (1,4,8400)
SCORE_T = "/model.23/Sigmoid_output_0"    # score branch, (1,80,8400)
COMBINED = "output0"

m = onnx.load(SRC)
g = m.graph

# find and remove the final concat node
concat = next((n for n in g.node if n.name == CONCAT), None)
assert concat is not None, f"{CONCAT} not found"
assert list(concat.input) == [BOX_T, SCORE_T], f"unexpected concat inputs: {list(concat.input)}"
g.node.remove(concat)
print(f"removed {CONCAT}")

# drop the old combined output
old = next((o for o in g.output if o.name == COMBINED), None)
assert old is not None, f"{COMBINED} not in graph outputs"
g.output.remove(old)

# promote the two branch tensors to graph outputs
boxes  = helper.make_tensor_value_info(BOX_T,   TensorProto.FLOAT, [1, 4,  8400])
scores = helper.make_tensor_value_info(SCORE_T, TensorProto.FLOAT, [1, 80, 8400])
g.output.extend([boxes, scores])

onnx.checker.check_model(m)
onnx.save(m, DST)
print(f"saved {DST}")
print("outputs:", [(o.name, [d.dim_value for d in o.type.tensor_type.shape.dim]) for o in g.output])
