import torch
from torchvision import models

# Load the pretrained VGG19 model
model = models.vgg19(pretrained=True)
model.eval()

# Save the model
torch.save(model.state_dict(), "vgg19.pth")