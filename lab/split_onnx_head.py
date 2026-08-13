import onnx
from onnx import helper, TensorProto

SRC = "yolo11n.onnx"
DST = "yolo11n_split.onnx"
COMBINED = "output0"          # existing (1,84,8400) graph output
AXIS = 1
BOX_ROWS = 4
SCORE_ROWS = 80

m = onnx.load(SRC)
g = m.graph

# opset for the default (empty) domain
opset = next((o.version for o in m.opset_import if o.domain in ("", "ai.onnx")), None)
print(f"model opset = {opset}")

# sanity: confirm the combined output exists with expected shape
out = next((o for o in g.output if o.name == COMBINED), None)
assert out is not None, f"{COMBINED} not found in graph outputs"
dims = [d.dim_value for d in out.type.tensor_type.shape.dim]
print(f"{COMBINED} shape = {dims}")
assert dims == [1, BOX_ROWS + SCORE_ROWS, 8400], f"unexpected shape {dims}"

# new output value infos
boxes = helper.make_tensor_value_info("boxes",  TensorProto.FLOAT, [1, BOX_ROWS,   8400])
scores = helper.make_tensor_value_info("scores", TensorProto.FLOAT, [1, SCORE_ROWS, 8400])

# build the Split node; sizes are a node INPUT on opset>=13, an ATTRIBUTE below
if opset is not None and opset >= 13:
    split_init = helper.make_tensor("split_sizes", TensorProto.INT64, [2], [BOX_ROWS, SCORE_ROWS])
    g.initializer.append(split_init)
    split_node = helper.make_node(
        "Split", inputs=[COMBINED, "split_sizes"], outputs=["boxes", "scores"], axis=AXIS)
else:
    split_node = helper.make_node(
        "Split", inputs=[COMBINED], outputs=["boxes", "scores"],
        axis=AXIS, split=[BOX_ROWS, SCORE_ROWS])

g.node.append(split_node)

# swap graph outputs: drop the combined tensor, add the two splits
g.output.remove(out)
g.output.extend([boxes, scores])

onnx.checker.check_model(m)
onnx.save(m, DST)
print(f"saved {DST}")
print("outputs:", [(o.name, [d.dim_value for d in o.type.tensor_type.shape.dim]) for o in g.output])
