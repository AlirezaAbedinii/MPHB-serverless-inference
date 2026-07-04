# # import torch
# # import torchvision.models as models

# # # Step 1: Load WRN-50-2 (PyTorch only provides 2x wider version by default)
# # model = models.wide_resnet50_2(pretrained=True)
# # model.eval()

# # # Step 2: Prepare dummy input
# # dummy_input = torch.randn(1, 3, 224, 224)

# # # Step 3: Export to ONNX
# # torch.onnx.export(
# #     model,
# #     dummy_input,
# #     "wrn-50-2.onnx",
# #     input_names=["input"],
# #     output_names=["output"],
# #     opset_version=11,
# #     do_constant_folding=True,
# #     keep_initializers_as_inputs=True  # ✅ Important!
# # )


# # print("Exported WRN-50-2 to wrn-50-2.onnx")
# import torch
# import torch.nn as nn
# from torchvision.models.resnet import Bottleneck, ResNet

# class WRN50_4(ResNet):
#     def __init__(self, num_classes=1000):
#         super().__init__(block=Bottleneck, layers=[3, 4, 6, 3])
#         width_multiplier = 4

#         # Fix: Update first conv layer to match widened channels
#         self.conv1 = nn.Conv2d(3, 64 * width_multiplier, kernel_size=7, stride=2, padding=3, bias=False)
#         self.bn1 = nn.BatchNorm2d(64 * width_multiplier)
#         self.inplanes = 64 * width_multiplier

#         # Update the residual layers to match wider channels
#         self.layer1 = self._make_layer(Bottleneck, 64 * width_multiplier, 3)
#         self.layer2 = self._make_layer(Bottleneck, 128 * width_multiplier, 4, stride=2)
#         self.layer3 = self._make_layer(Bottleneck, 256 * width_multiplier, 6, stride=2)
#         self.layer4 = self._make_layer(Bottleneck, 512 * width_multiplier, 3, stride=2)

#         # Update final FC input dimension
#         self.fc = nn.Linear(512 * width_multiplier * Bottleneck.expansion, num_classes)

# def export_wrn50_4_to_onnx(path="executed-wrn50-4.onnx"):
#     model = WRN50_4()
#     model.eval()
#     dummy_input = torch.randn(1, 3, 224, 224)
#     torch.onnx.export(
#         model,
#         dummy_input,
#         path,
#         input_names=["input"],
#         output_names=["output"],
#         opset_version=11,
#         do_constant_folding=True,
#         keep_initializers_as_inputs=True
#     )
#     print("✅ Exported WRN‑50‑4 to", path)

# # You can run export_wrn50_4_to_onnx() in your local script
# export_wrn50_4_to_onnx()

import onnx
from onnx import OperatorSetIdProto

# Load your ONNX model
model = onnx.load("net0.onnx")

# Create a new opset import
new_opset = OperatorSetIdProto()
new_opset.version = 9  # Use 9 for onnxruntime <=1.19
new_opset.domain = ""  # Default domain

# Replace by reassigning a new list
model.opset_import.extend([new_opset])

# Remove old entries (skip if none existed or only one)
if len(model.opset_import) > 1:
    del model.opset_import[0]  # Keep only the one you just added

# Save patched model
onnx.save(model, "net0_opset9.onnx")
