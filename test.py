import torch
import numpy as np
from PIL import Image
from skimage.color import rgb2lab, lab2rgb

from generator import UNet

# Load model
device = torch.device("cpu")

model = UNet().to(device)

model.load_state_dict(
    torch.load(
        "colorizer_gan.pth",
        map_location=device
    )
)

model.eval()

# Load image
img = Image.open("forest_bw.jpg").convert("RGB")

# Save original size
original_size = img.size

# Resize for model
img = img.resize((256, 256))

img_np = np.array(img)

# RGB -> LAB
lab = rgb2lab(img_np).astype("float32")

L = lab[:, :, 0]

L_input = (L / 50.0) - 1.0

L_input = torch.tensor(L_input).unsqueeze(0).unsqueeze(0)

# Predict colors
with torch.no_grad():
    pred_ab = model(L_input.float())

pred_ab = pred_ab.squeeze(0)

pred_ab = pred_ab.permute(1, 2, 0).numpy()

pred_ab = pred_ab * 128

# Reconstruct LAB image
lab_output = np.zeros((256, 256, 3), dtype=np.float32)

lab_output[:, :, 0] = L
lab_output[:, :, 1:] = pred_ab

# LAB -> RGB
rgb_output = lab2rgb(lab_output)

rgb_output = (rgb_output * 255).clip(0, 255).astype(np.uint8)

# Convert to image
result = Image.fromarray(rgb_output)

# Resize back to original size
result = result.resize(original_size)

# Save result
result.save("result.jpg")

print("Saved result.jpg")