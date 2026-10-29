from torch.utils.data import Dataset
from PIL import Image
from skimage.color import rgb2lab
import numpy as np
import torch
import os

class ColorizationDataset(Dataset):

    def __init__(self, root_dir):

        self.image_paths = []

        folders = [
            "buildings",
            "forest",
            "glacier",
            "mountain",
            "sea",
            "street"
        ]

        for folder in folders:

            folder_path = os.path.join(
                root_dir,
                folder
            )

            for image in os.listdir(folder_path):

                self.image_paths.append(
                    os.path.join(
                        folder_path,
                        image
                    )
                )

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):

        img = Image.open(
            self.image_paths[idx]
        ).convert("RGB")

        img = img.resize((128,128))

        img = np.array(img)

        lab = rgb2lab(img).astype("float32")

        L = lab[:,:,0]

        ab = lab[:,:,1:]

        L = (L / 50.0) - 1.0

        ab = ab / 128.0

        L = torch.tensor(L).unsqueeze(0)

        ab = torch.tensor(ab).permute(2,0,1)

        return L, ab