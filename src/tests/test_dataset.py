from dataset import CubeMoveDataset
from torch.utils.data import DataLoader
import numpy as np
# Point to your dataset folder
dataset = CubeMoveDataset(root_dir='DATASET/dataset', num_frames=16)

print("Classes found:", dataset.classes)
print("Class to index:", dataset.class_to_idx)
print("Total videos:", len(dataset))

# Load one sample
video, label = dataset[0]
print("Video tensor shape:", video.shape)   # should be (3, 16, 112, 112)
print("Label:", label, "→", dataset.classes[label])

# Test the DataLoader
loader = DataLoader(dataset, batch_size=4, shuffle=True)
batch_video, batch_label = next(iter(loader))
print("Batch shape:", batch_video.shape)    # should be (4, 3, 16, 112, 112)