import streamlit as st
import torch
import numpy as np
from PIL import Image
from skimage.color import rgb2lab, lab2rgb

from generator import UNet


# Load GAN Model
@st.cache_resource
def load_model():

    device = torch.device("cpu")

    model = UNet().to(device)

    model.load_state_dict(
        torch.load(
            "colorizer_gan.pth",
            map_location=device
        )
    )

    model.eval()

    return model


model = load_model()

st.title("Automatic Image Colorization using GAN")

st.write(
    "Upload a grayscale image and generate a colorized image using the trained GAN model."
)

uploaded_file = st.file_uploader(
    "Upload Image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file:

    # Load Image
    img = Image.open(
        uploaded_file
    ).convert("RGB")

    original_size = img.size

    st.image(
        img,
        caption="Uploaded Image",
        use_container_width=True
    )

    # Resize
    img_resized = img.resize(
        (256, 256)
    )

    img_np = np.array(
        img_resized
    )

    # RGB → LAB
    lab = rgb2lab(
        img_np
    ).astype("float32")

    L = lab[:, :, 0]

    L_input = (
        L / 50.0
    ) - 1.0

    L_input = torch.tensor(
        L_input
    ).unsqueeze(0).unsqueeze(0)

    # Prediction
    with torch.no_grad():

        pred_ab = model(
            L_input.float()
        )

    pred_ab = pred_ab.squeeze(0)

    pred_ab = pred_ab.permute(
        1, 2, 0
    ).numpy()

    pred_ab = pred_ab * 128

    # Reconstruct LAB
    lab_output = np.zeros(
        (256, 256, 3),
        dtype=np.float32
    )

    lab_output[:, :, 0] = L

    lab_output[:, :, 1:] = pred_ab

    # LAB → RGB
    rgb_output = lab2rgb(
        lab_output
    )

    rgb_output = (
        rgb_output * 255
    ).clip(
        0,
        255
    ).astype(np.uint8)

    result = Image.fromarray(
        rgb_output
    )

    result = result.resize(
        original_size
    )

    st.image(
        result,
        caption="Colorized Image",
        use_container_width=True
    )

    # Save Output
    result.save(
        "result.jpg"
    )

    # Download
    with open(
        "result.jpg",
        "rb"
    ) as file:

        st.download_button(
            label="Download Colorized Image",
            data=file,
            file_name="colorized_image.jpg",
            mime="image/jpeg"
        )