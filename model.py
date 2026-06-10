import torch
import torch.nn as nn

class CubeMoveModel(nn.Module):
    def __init__(self, num_classes=8):
        super(CubeMoveModel, self).__init__()

        # CNN - extracts spatial features from each frame
        self.cnn = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),       # 112 → 56

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),       # 56 → 28

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),       # 28 → 14
        )

        # LSTM - understands motion across frames
        self.lstm = nn.LSTM(
            input_size=128 * 14 * 14,
            hidden_size=256,
            num_layers=2,
            batch_first=True
        )

        # Classifier - outputs move prediction
        self.classifier = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        # x shape: (batch, C, T, H, W)
        batch, C, T, H, W = x.shape

        # Run CNN on each frame
        x = x.permute(0, 2, 1, 3, 4)        # → (batch, T, C, H, W)
        x = x.reshape(batch * T, C, H, W)    # → (batch*T, C, H, W)
        x = self.cnn(x)                       # → (batch*T, 128, 14, 14)
        x = x.reshape(batch, T, -1)           # → (batch, T, 128*14*14)

        # Run LSTM across frames
        x, _ = self.lstm(x)                   # → (batch, T, 256)
        x = x[:, -1, :]                       # take last frame → (batch, 256)

        # Classify
        x = self.classifier(x)                # → (batch, 8)
        return x