import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from dataset import ColorizationDataset
from generator import UNet
from discriminator import Discriminator

# Dataset
dataset = ColorizationDataset(
    r"C:\Users\APPLE\Downloads\archive\seg_train\seg_train"
)

print("Dataset Size:", len(dataset))

train_loader = DataLoader(
    dataset,
    batch_size=8,
    shuffle=True
)

# Device
device = torch.device(
    "cuda" if torch.cuda.is_available()
    else "cpu"
)

print("Using Device:", device)
print("GAN Training Started...")

# Models
generator = UNet().to(device)
discriminator = Discriminator().to(device)

# Loss Functions
adversarial_loss = nn.BCELoss()
l1_loss = nn.L1Loss()

# Optimizers
g_optimizer = torch.optim.Adam(
    generator.parameters(),
    lr=0.0002
)

d_optimizer = torch.optim.Adam(
    discriminator.parameters(),
    lr=0.0002
)

# Epochs
epochs = 10

print("Starting Epochs...")

# Training Loop
for epoch in range(epochs):

    generator.train()
    discriminator.train()

    total_g_loss = 0
    total_d_loss = 0

    for L, ab in train_loader:

        L = L.float().to(device)
        ab = ab.float().to(device)

        # Generator Forward
        fake_ab = generator(L)

        # Create fake and real images
        fake_img = torch.cat([L, fake_ab], dim=1)
        real_img = torch.cat([L, ab], dim=1)

        # Get discriminator output shape dynamically
        with torch.no_grad():
            disc_out = discriminator(real_img)

        real_labels = torch.ones_like(disc_out)
        fake_labels = torch.zeros_like(disc_out)

        # ---------------------
        # Train Generator
        # ---------------------

        g_adv = adversarial_loss(
            discriminator(fake_img),
            real_labels
        )

        g_l1 = l1_loss(
            fake_ab,
            ab
        )

        g_loss = g_adv + (100 * g_l1)

        g_optimizer.zero_grad()
        g_loss.backward()
        g_optimizer.step()

        # ---------------------
        # Train Discriminator
        # ---------------------

        real_loss = adversarial_loss(
            discriminator(real_img),
            real_labels
        )

        fake_loss = adversarial_loss(
            discriminator(fake_img.detach()),
            fake_labels
        )

        d_loss = (real_loss + fake_loss) / 2

        d_optimizer.zero_grad()
        d_loss.backward()
        d_optimizer.step()

        total_g_loss += g_loss.item()
        total_d_loss += d_loss.item()

    print(
        f"Epoch [{epoch+1}/{epochs}] "
        f"G Loss: {total_g_loss/len(train_loader):.4f} "
        f"D Loss: {total_d_loss/len(train_loader):.4f}"
    )

# Save Generator
torch.save(
    generator.state_dict(),
    "colorizer_gan.pth"
)

print("GAN Model Saved!")
print("Training Complete!")