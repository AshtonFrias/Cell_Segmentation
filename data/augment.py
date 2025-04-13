import albumentations as A
import cv2
import numpy as np
import torch
import os
from torch.utils.data import Dataset

class DataAugmentation:
    def __init__(self, p=0.5):
        self.transform = A.Compose([
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.RandomRotate90(p=0.5),
            A.ElasticTransform(alpha=1, sigma=50, alpha_affine=50, p=0.3),
            A.GridDistortion(p=0.3),
            A.RandomBrightnessContrast(p=0.2),
        ])

    def __call__(self, image, mask):
        augmented = self.transform(image=image, mask=mask)
        return augmented['image'], augmented['mask']