
# Automatic Colorization of Grayscale Images Using Deep CNNs

## 📌 Overview

This project explores the automatic colorization of grayscale images using deep learning and Convolutional Neural Networks (CNNs). The system takes a grayscale image as input and generates a colorized version by predicting plausible colors for different objects and regions.

The project implements a U-Net-based image colorization model and explores a Generative Adversarial Network (GAN) approach to improve the visual quality of generated images.

## 🎯 Objectives

- Convert grayscale images into color images using deep learning.
- Implement a U-Net architecture for image colorization.
- Explore GAN-based image colorization using a generator and discriminator.
- Compare the visual outputs produced by different approaches.
- Develop an application for testing colorization on user-provided images.

## 🧠 Models and Techniques

### 1. U-Net

U-Net is an encoder-decoder neural network architecture with skip connections. It captures image features at multiple scales and uses them to generate color predictions while retaining spatial details.

### 2. Generative Adversarial Network (GAN)

The GAN-based approach consists of two components:

- **Generator:** Produces a colorized image from a grayscale input.
- **Discriminator:** Learns to distinguish generated images from real color images.

The generator and discriminator are trained adversarially to explore the generation of more realistic colorized images.

## 🛠️ Technologies Used

- Python
- PyTorch
- NumPy
- Pillow
- scikit-image
- Streamlit
- Convolutional Neural Networks (CNNs)
- U-Net
- Generative Adversarial Networks (GANs)

## 📂 Project Structure

```text
grayscale-colorization/
├── models/
│   ├── unet.py
│   ├── generator.py
│   └── discriminator.py
├── examples/
│   ├── forest_bw.jpg
│   └── result.jpg
├── app.py
├── dataset.py
├── train.py
├── train_gan_v2.py
├── test.py
├── requirements.txt
├── README.md
└── .gitignore
```

## ⚙️ Installation

1. Clone this repository:

   ```bash
   git clone https://github.com/YOUR_USERNAME/grayscale-colorization.git
   ```

2. Navigate into the project directory:

   ```bash
   cd grayscale-colorization
   ```

3. Install the dependencies:

   ```bash
   pip install -r requirements.txt
   ```

## 🚀 Usage

### Run the application

Start the Streamlit application:

```bash
python -m streamlit run app.py
```

Open the local URL displayed in the terminal and upload a grayscale image to test the colorization system.

### Train the U-Net model

```bash
python train.py
```

### Train the GAN model

Use the GAN training script you have verified works:

```bash
python train_gan_v2.py
```

**Note:** Training and application commands may need to be adjusted to match the dataset paths, model checkpoint locations, and imports used in the current code.

## 🖼️ Example Results

Sample images are provided in the `examples/` directory.

| Image | Description |
|---|---|
| `forest_bw.jpg` | Grayscale input image |
| `result.jpg` | Colorized output image |

## 📊 Evaluation

The project can be evaluated by comparing generated color images against their original color references.

Potential evaluation metrics include:

- **PSNR (Peak Signal-to-Noise Ratio):** Measures pixel-level reconstruction quality.
- **SSIM (Structural Similarity Index):** Measures similarity in image structure and appearance.

Quantitative results should be added after running the evaluation on a defined test dataset.

## 🔮 Future Improvements

- Improve color prediction and visual realism.
- Compare U-Net and GAN-based approaches quantitatively.
- Evaluate performance on diverse image categories.
- Experiment with different loss functions and network architectures.
- Improve generalization to images not seen during training.

## 👨‍💻 Author

**G. Karthik Suhaas**  


## 📄 License

This project is available for educational and research purposes. A formal license can be added to specify permissions for reuse and redistribution.

