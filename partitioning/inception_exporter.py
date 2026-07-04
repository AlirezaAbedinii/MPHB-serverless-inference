"""Export torchvision Inception-v3 to ONNX for the Gillis partitioner.

Only topology + tensor shapes matter to the latency predictor (it is
FLOP/shape based), so random weights are fine. We disable aux_logits and
transform_input to keep the graph a clean Conv/BN/Relu/Pool/Concat/Gemm DAG,
matching the style of the existing vgg19.onnx / WRN models.
"""
import torch
import torchvision.models as models

OUT = "partition/models/inception_v3.onnx"

model = models.inception_v3(weights=None, aux_logits=False,
                            transform_input=False, init_weights=False)
model.eval()

dummy = torch.randn(1, 3, 299, 299)  # Inception-v3 native input

torch.onnx.export(
    model, dummy, OUT,
    input_names=["input"], output_names=["output"],
    opset_version=11,
    do_constant_folding=True,
    keep_initializers_as_inputs=True,
)
print("Exported Inception-v3 to", OUT)
