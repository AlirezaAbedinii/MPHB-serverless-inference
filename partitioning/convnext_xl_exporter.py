#!/usr/bin/env python3
import os, torch
from pathlib import Path

# keep big artifacts off /home
os.environ["HF_HOME"] = "/mnt/data/huggingface"
os.environ["TRANSFORMERS_CACHE"] = "/mnt/data/huggingface"
os.environ["TORCH_HOME"] = "/mnt/data/torch_home"

import timm
model = timm.create_model("convnext_xlarge", pretrained=True, num_classes=1000)
model.eval()

dummy = torch.randn(1, 3, 224, 224)

out_dir = Path("/mnt/data/models"); out_dir.mkdir(parents=True, exist_ok=True)
onnx_path = out_dir / "convnext_xl_fp32_bs1_224.onnx"

torch.onnx.export(
    model, dummy, onnx_path.as_posix(),
    export_params=True,
    opset_version=13,             # stable for conv models
    do_constant_folding=True,
    input_names=["input"], output_names=["logits"],
    dynamic_axes=None,            # static shapes help Gillis
    keep_initializers_as_inputs=False,
    use_external_data_format=False # force single file if <2 GB
)

print("Saved:", onnx_path, "size GB:", onnx_path.stat().st_size/(1024**3))
