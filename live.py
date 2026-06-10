import cv2
import torch
import numpy as np
from model import CubeMoveModel

# --- Config ---
NUM_FRAMES = 16
NUM_CLASSES = 8
CLASSES = ['B', 'D', 'E', 'F', 'L', 'M', 'R', 'U']
RECORD_SECONDS = 2
MOVEMENT_THRESHOLD = 25
MIN_MOVEMENT_AREA = 500

# --- Load Model ---
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = CubeMoveModel(num_classes=NUM_CLASSES).to(device)
model.load_state_dict(torch.load('erno_model.pth', weights_only=True))
model.eval()
print(f"Model loaded on {device}")

def frames_to_tensor(frames):
    total = len(frames)
    indices = [int(i * total / NUM_FRAMES) for i in range(NUM_FRAMES)]
    frames = [frames[i] for i in indices]

    frames = np.array(frames, dtype=np.float32) / 255.0
    tensor = torch.tensor(frames, dtype=torch.float32)
    tensor = tensor.permute(3, 0, 1, 2)           # (C, T, H, W)
    tensor = tensor.unsqueeze(0).to(device)         # (1, C, T, H, W)
    return tensor

def predict_frames(frames):
    tensor = frames_to_tensor(frames)
    with torch.no_grad():
        outputs = model(tensor)
        probabilities = torch.softmax(outputs, dim=1)[0]
        predicted_idx = probabilities.argmax().item()
        confidence = probabilities[predicted_idx].item() * 100
    return CLASSES[predicted_idx], confidence

def detect_movement(frame1, frame2):
    diff = cv2.absdiff(frame1, frame2)
    gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    _, thresh = cv2.threshold(blur, MOVEMENT_THRESHOLD, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    movement_area = sum(cv2.contourArea(c) for c in contours)
    return movement_area > MIN_MOVEMENT_AREA

# --- Main Loop ---
cap = cv2.VideoCapture(0)
fps = cap.get(cv2.CAP_PROP_FPS)
if fps == 0:
    fps = 30
record_frames_needed = int(fps * RECORD_SECONDS)

print("Watching for moves... Press Q to quit")

prev_frame = None
recording = False
recorded_frames = []

while True:
    ret, frame = cap.read()
    if not ret:
        break

    display = frame.copy()

    if prev_frame is not None:
        movement = detect_movement(prev_frame, frame)

        if movement and not recording:
            # Movement detected → start recording
            recording = True
            recorded_frames = []
            print("Move detected! Recording...")

        if recording:
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            rgb = cv2.resize(rgb, (112, 112))
            recorded_frames.append(rgb)

            # Show recording indicator
            cv2.circle(display, (30, 30), 15, (0, 0, 255), -1)

            if len(recorded_frames) >= record_frames_needed:
                # Enough frames collected → predict
                move, confidence = predict_frames(recorded_frames)
                print(f"Predicted Move: {move} ({confidence:.1f}% confident)")
                recording = False
                recorded_frames = []

    # Show status on screen
    status = "RECORDING..." if recording else "Watching..."
    cv2.putText(display, status, (10, 70),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                (0, 0, 255) if recording else (0, 255, 0), 2)

    cv2.imshow('Erno Live', display)
    prev_frame = frame.copy()

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()