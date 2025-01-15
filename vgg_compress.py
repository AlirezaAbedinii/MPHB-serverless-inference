import torch
model = torch.load("/mnt/data/HarmonyBatch/AlibabaCloud/vgg19.pth")
torch.save(model, "/mnt/data/HarmonyBatch/AlibabaCloud/vgg19_compressed.pth", _use_new_zipfile_serialization=True)
