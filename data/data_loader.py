import os
import cv2
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import torch
from torchvision import transforms
from torch.utils.data import Dataset
from data.augment import DataAugmentation

class MoNuSegDataset(Dataset):
    def __init__(self, image_folder, mask_folder, augment = False, device = "cpu"):
        self.image_folder = image_folder
        self.mask_folder = mask_folder

        self.device = device

        self.image_files = sorted(os.listdir(image_folder))
        self.mask_files = sorted(os.listdir(mask_folder))

        self.augment = augment
        self.augmentor = DataAugmentation() if augment else None

        if len(self.image_files) != len(self.mask_files):
            raise ValueError("Number of images and masks must be the same.")

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, index):
        if index >= len(self.image_files):
            raise IndexError(f"Index {index} out of range. Dataset has {len(self.image_files)} items.")

        image_path = os.path.join(self.image_folder, self.image_files[index])
        image = cv2.imread(image_path, cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError(f"Failed to load image: {image_path}")

        image = np.float32(image) / 255.0
        
        mask_path = os.path.join(self.mask_folder, self.mask_files[index])
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        if mask is None:
            raise ValueError(f"Failed to load mask: {mask_path}")

        mask = np.where(mask > 0, 1.0, 0.0).astype(np.float32)

        if self.augment:
            image, mask = self.augmentor(image, mask)

        image = torch.from_numpy(image).permute(2, 0, 1).to(self.device)
        mask = torch.from_numpy(mask).to(self.device).unsqueeze(0)

        return image, mask

class BCSSDataset(Dataset):
    def __init__(self, image_dir, mask_dir, augment=False, device="cpu"):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.image_filenames = sorted(os.listdir(image_dir))
        self.mask_filenames = sorted(os.listdir(mask_dir))
        self.device = device

        self.augment = augment 
        self.augmentor = DataAugmentation() if augment else None

    def __getitem__(self, idx):
        image_path = os.path.join(self.image_dir, self.image_filenames[idx])
        mask_path = os.path.join(self.mask_dir, self.mask_filenames[idx])

        image = cv2.imread(image_path, cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError(f"Failed to load image: {image_path}")

        image = np.float32(image) / 255.0

        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        if mask is None:
            raise ValueError(f"Failed to load mask: {mask_path}")

        if self.augment:
            image, mask = self.augmentor(image, mask)

        image = torch.from_numpy(image).permute(2, 0, 1).to(self.device)
        mask = torch.from_numpy(mask).long().to(self.device)

        return image, mask

    def __len__(self):
        return len(self.image_filenames)
