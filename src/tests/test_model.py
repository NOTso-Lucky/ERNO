import torch
from model import CubeMoveModel

model = CubeMoveModel(num_classes=8)

# Fake batch (4 videos, 3 channels, 16 frames, 112x112)
dummy = torch.randn(4, 3, 16, 112, 112)
output = model(dummy)

print("Output shape:", output.shape)   # should be (4, 8)
print("Model ready!")