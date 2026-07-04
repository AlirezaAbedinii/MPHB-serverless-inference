import os, sys, onnx
from onnxconverter_common import float16
print(onnx.__version__)
SRC = "/mnt/data/gillis-open-source/partition/models/gpt2-xl.onnx"                 # your source ONNX
OUT_DIR = "/mnt/data/gillis-open-source/partition/models/"         # where to write outputs
BASE = "gpt2-xl-fp16"                # base name for outputs

os.makedirs(OUT_DIR, exist_ok=True)
dst_model = os.path.join(OUT_DIR, BASE + ".onnx")
dst_data  = os.path.join(OUT_DIR, BASE + ".data")

print("Loading:", os.path.abspath(SRC))
m = onnx.load(SRC, load_external_data=False)

print("Converting to FP16 (keep_io_types=True)…")
m16 = float16.convert_float_to_float16(m, keep_io_types=True)

print("Saving with external data:", os.path.abspath(dst_model))
# IMPORTANT: use save_model (not save) and pass BOTH the model path AND the external-data params
onnx.save_model(
    m16,
    dst_model,
    save_as_external_data=True,
    all_tensors_to_one_file=True,
    location=os.path.basename(dst_data),  # path of .data relative to the model
    size_threshold=1024,                  # force externalization (bytes)
    convert_attribute=False
)

print("Verifying files…")
print("  ONNX exists:", os.path.exists(dst_model), "size MB:", os.path.getsize(dst_model)/1e6 if os.path.exists(dst_model) else -1)
print("  DATA exists:", os.path.exists(dst_data),  "size MB:", os.path.getsize(dst_data)/1e6  if os.path.exists(dst_data)  else -1)

# Extra robust check: ensure at least one initializer points to external data
from google.protobuf.json_format import MessageToDict
mm = onnx.load(dst_model, load_external_data=False)
uses_ext = False
for t in mm.graph.initializer:
    td = MessageToDict(t)
    if "externalData" in td or td.get("dataLocation", "").upper() == "EXTERNAL":
        uses_ext = True
        break
print("uses_external_data:", uses_ext)

if not os.path.exists(dst_data):
    print("\nERROR: .data file was not created. Common causes:")
    print(" - Using old onnx version (<1.14) → upgrade and rerun.")
    print(" - OUT_DIR not writable or path invalid.")
    print(" - Called onnx.save() instead of onnx.save_model().")
    sys.exit(1)

print("\nSuccess. Use this model with Gillis:")
print("  /usr/bin/time -v python3 main.py lo -n", dst_model, "-p true")
