import numpy as np
import gi
gi.require_version('Gst', '1.0')
from gi.repository import Gst
from tflite_runtime.interpreter import Interpreter, load_delegate

MODEL="/root/vela_out/yolo11n_int8_vela.tflite"
DELEGATE="/usr/lib/libethosu_delegate.so"
W,H,SIZE=1280,720,640

d=load_delegate(DELEGATE)
it=Interpreter(model_path=MODEL, experimental_delegates=[d]); it.allocate_tensors()
inp=it.get_input_details()[0]; out=it.get_output_details()[0]
insc,inzp=inp["quantization"]; osc,ozp=out["quantization"]

Gst.init(None)
p=Gst.parse_launch(f"v4l2src device=/dev/video0 ! image/jpeg,width={W},height={H},framerate=30/1 ! jpegdec ! videoconvert ! video/x-raw,format=RGB ! appsink name=sink emit-signals=false max-buffers=1 drop=true sync=false")
s=p.get_by_name("sink"); p.set_state(Gst.State.PLAYING)
import time; time.sleep(0.5)

smp=s.emit("try-pull-sample", int(2e9))
buf=smp.get_buffer(); ok,m=buf.map(Gst.MapFlags.READ)
frame=np.frombuffer(m.data,dtype=np.uint8).reshape((H,W,3)).copy(); buf.unmap(m)
p.set_state(Gst.State.NULL)

yi=np.linspace(0,H-1,SIZE).astype(np.int32); xi=np.linspace(0,W-1,SIZE).astype(np.int32)
r=frame[yi][:,xi].astype(np.float32)/255.0
q=(r/insc+inzp).round().astype(inp["dtype"])[None,...]
it.set_tensor(inp["index"],q); it.invoke()
raw=it.get_tensor(out["index"])
print("raw shape", raw.shape, "dtype", raw.dtype)

deq=(raw.astype(np.float32)-ozp)*osc
deq=deq[0]                      # (84,8400)
print("deq shape", deq.shape)
# assume rows 0-3 box, rows 4-83 = 80 class scores
cls=deq[4:84,:]                 # (80,8400)
print("class-score block shape", cls.shape)
print("max class score overall:", float(cls.max()))
# best candidate by any class
best_cand=int(cls.max(axis=0).argmax())
best_cls=int(cls[:,best_cand].argmax())
print("best candidate idx", best_cand, "class idx", best_cls, "score", float(cls[best_cls,best_cand]))
print("box at best cand (rows0-3):", deq[0:4,best_cand].tolist())
# person is class 0: show its best
person=cls[0,:]
print("person(row4) max score:", float(person.max()), "at cand", int(person.argmax()))
