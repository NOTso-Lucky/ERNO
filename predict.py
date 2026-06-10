import cv2
import torch
import numpy as np
from model import CubeMoveModel

# --- Config ---
NUM_FRAMES = 16
NUM_CLASSES = 8
CLASSES = ['B', 'D', 'E', 'F', 'L', 'M', 'R', 'U']

# --- Load Model ---
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = CubeMoveModel(num_classes=NUM_CLASSES).to(device)
model.load_state_dict(torch.load('erno_model.pth', weights_only=True))
model.eval()
print(f"Model loaded on {device}")

def load_video(path):
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
    indices = [int(i * total / NUM_FRAMES) for i in range(NUM_FRAMES)]
    frames = [frames[i] for i in indices]

    frames = np.array(frames, dtype=np.float32) / 255.0
    tensor = torch.tensor(frames, dtype=torch.float32)
    tensor = tensor.permute(3, 0, 1, 2)          # (C, T, H, W)
    tensor = tensor.unsqueeze(0).to(device)        # (1, C, T, H, W)

    return tensor

def predict(video_path):
    tensor = load_video(video_path)

    with torch.no_grad():
        outputs = model(tensor)
        probabilities = torch.softmax(outputs, dim=1)[0]
        predicted_idx = probabilities.argmax().item()
        confidence = probabilities[predicted_idx].item() * 100

    print(f"\nVideo: {video_path}")
    print(f"Predicted Move: {CLASSES[predicted_idx]} ({confidence:.1f}% confident)")
    print("\nAll probabilities:")
    for i, cls in enumerate(CLASSES):
        bar = '█' * int(probabilities[i].item() * 20)
        print(f"  {cls}: {probabilities[i].item()*100:5.1f}% {bar}")

    return CLASSES[predicted_idx], confidence

# --- Test on a video ---
import sys
if len(sys.argv) > 1:
    predict(sys.argv[1])
else:
    print("Usage: python predict.py path/to/video.mp4")