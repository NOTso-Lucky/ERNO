import os
import cv2
import torch
import numpy as np
from torch.utils.data import Dataset

class CubeMoveDataset(Dataset):
    def __init__(self, root_dir, num_frames=16, augment=False):
        self.root_dir = root_dir
        self.num_frames = num_frames
        self.augment = augment

        self.classes = sorted(os.listdir(root_dir))
        self.class_to_idx = {cls: idx for idx, cls in enumerate(self.classes)}

        self.samples = []
        for cls in self.classes:
            cls_folder = os.path.join(root_dir, cls)
            for fname in os.listdir(cls_folder):
                if fname.endswith('.mp4'):
                    path = os.path.join(cls_folder, fname)
                    self.samples.append((path, self.class_to_idx[cls]))

    def _augment_frame(self, frame):
        # Brightness adjustment
        factor = np.random.uniform(0.7, 1.3)
        frame = np.clip(frame * factor, 0, 1)

        # Gaussian noise
        noise = np.random.normal(0, 0.02, frame.shape)
        frame = np.clip(frame + noise, 0, 1)

        return frame

    def _load_video(self, path):
        cap = cv2.VideoCapture(path)
        frames = []

        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = cv2.resize(frame, (112, 112))
            frames.append(frame)

        cap.release()

        total = len(frames)
        indices = [int(i * total / self.num_frames) for i in range(self.num_frames)]
        frames = [frames[i] for i in indices]

        frames = np.array(frames, dtype=np.float32) / 255.0

        if self.augment:
            frames = np.array([self._augment_frame(f) for f in frames])

        tensor = torch.tensor(frames, dtype=torch.float32)
        tensor = tensor.permute(3, 0, 1, 2)

        return tensor

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        video_tensor = self._load_video(path)
        return video_tensor, label