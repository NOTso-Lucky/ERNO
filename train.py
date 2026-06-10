import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from dataset import CubeMoveDataset
from model import CubeMoveModel

# --- Config ---
DATASET_PATH = 'DATASET/dataset'
NUM_FRAMES = 16
BATCH_SIZE = 4
EPOCHS = 20
LEARNING_RATE = 0.0005
NUM_CLASSES = 8

# --- Dataset ---
train_dataset = CubeMoveDataset(root_dir=DATASET_PATH, num_frames=NUM_FRAMES, augment=True)
val_dataset = CubeMoveDataset(root_dir=DATASET_PATH, num_frames=NUM_FRAMES, augment=False)

train_size = int(0.8 * len(train_dataset))
val_size = len(train_dataset) - train_size

torch.manual_seed(42)
train_set, _ = random_split(train_dataset, [train_size, val_size])
_, val_set = random_split(val_dataset, [train_size, val_size])

train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_set, batch_size=BATCH_SIZE, shuffle=False)

print(f"Training samples:   {train_size}")
print(f"Validation samples: {val_size}")

# --- Model ---
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

torch.manual_seed(0)
torch.cuda.manual_seed(0)

model = CubeMoveModel(num_classes=NUM_CLASSES).to(device)

if os.path.exists('erno_best_model.pth'):
    model.load_state_dict(torch.load('erno_best_model.pth', weights_only=True))
    print("Loaded best saved model, continuing training...")
else:
    print("Starting fresh...")

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

best_val_acc = 0.0
patience = 20
epochs_no_improve = 0

# --- Training Loop ---
for epoch in range(EPOCHS):

    # Training
    model.train()
    train_loss = 0
    train_correct = 0

    for videos, labels in train_loader:
        videos, labels = videos.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(videos)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        train_correct += (outputs.argmax(1) == labels).sum().item()

    # Validation
    model.eval()
    val_loss = 0
    val_correct = 0

    with torch.no_grad():
        for videos, labels in val_loader:
            videos, labels = videos.to(device), labels.to(device)
            outputs = model(videos)
            loss = criterion(outputs, labels)

            val_loss += loss.item()
            val_correct += (outputs.argmax(1) == labels).sum().item()

    # Stats
    train_acc = train_correct / train_size * 100
    val_acc = val_correct / val_size * 100
    print(f"Epoch {epoch+1:02d}/{EPOCHS} | "
          f"Train Loss: {train_loss:.3f} | Train Acc: {train_acc:.1f}% | "
          f"Val Loss: {val_loss:.3f} | Val Acc: {val_acc:.1f}%")

    # Save best model
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        torch.save(model.state_dict(), 'erno_best_model.pth')
        print(f"  → New best model saved! Val Acc: {val_acc:.1f}%")
        epochs_no_improve = 0
    else:
        epochs_no_improve += 1
        if epochs_no_improve >= patience:
            print(f"Early stopping! No improvement for {patience} epochs.")
            break

# --- Save Final Model (best model, not last epoch) ---
import shutil
shutil.copy('erno_best_model.pth', 'erno_model.pth')
print(f"Best model saved to erno_model.pth (Val Acc: {best_val_acc:.1f}%)")