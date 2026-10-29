import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from dataset import ColorizationDataset
from unet import UNet

# Dataset
dataset = ColorizationDataset(
    r"C:\Users\APPLE\Downloads\archive\seg_train\seg_train"
)

train_loader = DataLoader(
    dataset,
    batch_size=16,
    shuffle=True
)

# Device
device = torch.device(
    "cuda" if torch.cuda.is_available()
    else "cpu"
)

print("Using Device:", device)
print("Training Started...")

# Model
model = UNet().to(device)

# Loss Functions
criterion_l1 = nn.L1Loss()
criterion_mse = nn.MSELoss()

# Optimizer
optimizer = torch.optim.Adam(
    model.parameters(),
    lr=0.0001
)

# Epochs
epochs = 50

# Best Loss
best_loss = float("inf")

# Training Loop
for epoch in range(epochs):

    model.train()

    total_loss = 0.0

    for L, ab in train_loader:

        L = L.float().to(device)
        ab = ab.float().to(device)

        pred_ab = model(L)

        loss = (
            criterion_l1(pred_ab, ab)
            + 0.5 * criterion_mse(pred_ab, ab)
        )

        optimizer.zero_grad()

        loss.backward()

        optimizer.step()

        total_loss += loss.item()

    avg_loss = total_loss / len(train_loader)

    print(
        f"Epoch [{epoch + 1}/{epochs}] "
        f"Loss: {avg_loss:.4f}"
    )

    # Save Best Model
    if avg_loss < best_loss:

        best_loss = avg_loss

        torch.save(
            model.state_dict(),
            "colorizer.pth"
        )

        print("Best Model Saved!")

print("Training Complete!")
print("Final Best Loss:", best_loss)