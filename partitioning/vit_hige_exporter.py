#!/usr/bin/env python3
import os
os.environ["HF_HOME"]="/mnt/data/huggingface"
os.environ["TRANSFORMERS_CACHE"]="/mnt/data/huggingface"
os.environ["HF_DATASETS_CACHE"]="/mnt/data/huggingface"

import torch, onnx
from transformers import ViTImageProcessor, ViTForImageClassification
from PIL import Image

MODEL="google/vit-huge-patch14-224-in21k"
OUT_DIR="/mnt/data/gillis-open-source/partitions/models"; os.makedirs(OUT_DIR, exist_ok=True)
ONNX=f"{OUT_DIR}/vit-huge-patch14-224-cls.onnx"
DATA=f"{OUT_DIR}/vit-huge-patch14-224-cls.data"

# proc = ViTImageProcessor.from_pretrained(MODEL, cache_dir=os.environ["HF_HOME"])
# model = ViTForImageClassification.from_pretrained(MODEL, cache_dir=os.environ["HF_HOME"]).eval()

# img = Image.new("RGB",(224,224),(128,128,128))
# pixel_values = proc(images=img, return_tensors="pt")["pixel_values"]

# # Export (no simplification)
# torch.onnx.export(
#     model, (pixel_values,), ONNX,
#     input_names=["input"], output_names=["logits"],
#     dynamic_axes=None, opset_version=14,
#     do_constant_folding=False, export_params=True,
#     use_external_data_format=True,  # <-- writes weights externally
# )

# # Force a single *.data file colocated with the .onnx
# m = onnx.load(ONNX)
# onnx.save_model(
#     m, ONNX,
#     save_as_external_data=True, all_tensors_to_one_file=True,
#     location=os.path.basename(DATA), size_threshold=1024, convert_attribute=False
# )

# Sanity: both files exist, .data is BIG
import os as _os
print("sizes MB:", _os.path.getsize(ONNX)/1e6, _os.path.getsize(DATA)/1e6, DATA)
