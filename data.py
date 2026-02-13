import os
import cv2
import glob
import random
import numpy as np
import torch
from torch.utils.data import Dataset


class Data(Dataset):
    def __init__(self, data_path, noise=None):
        self.data_path = data_path
        self.noise = noise

        image_files = glob.glob(os.path.join(data_path, 'image_*.png'))
        image_files.sort(key=lambda x: int(os.path.basename(x).replace('image_', '').replace('.png', '')))
        self.samples = []

        for img_path in image_files:
            number = os.path.basename(img_path).replace('image_', '').replace('.png', '')
            label_path = os.path.join(data_path, f'label_{number}.png')

            if os.path.exists(label_path):
                self.samples.append((img_path, label_path))
            else:
                print(f"Warning: no label found for image_{number}.png, skipping.")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label_path = self.samples[idx]
        rand = random.random()

        image = self.load_image(img_path, rand, image_type='image')
        label = self.load_image(label_path, rand, image_type='label')

        return torch.from_numpy(image).float(), torch.from_numpy(label).float()

    def load_image(self, filepath, rand, image_type='label'):
        image = cv2.imread(filepath, cv2.IMREAD_GRAYSCALE)

        if self.noise is not None and rand > 0.5:
            image = np.flip(image, axis=1).copy()

        image = np.expand_dims(image, axis=0).astype(np.float32) / 255.0

        if self.noise is not None and image_type == 'image':
            sigma = np.random.uniform(high=self.noise)
            noise = np.random.normal(scale=sigma, size=image.shape)
            image = np.clip(image + noise, 0, 1).astype(np.float32)

        return image


if __name__ == '__main__':
    data_path = "./dataset/data"

    dataset = Data(data_path, noise=0.025)
    print(f"Number of samples: {len(dataset)}")

    image, label = dataset[0]
    print(f"Image shape: {image.shape}")  
    print(f"Label shape: {label.shape}")  