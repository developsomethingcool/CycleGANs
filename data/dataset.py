import os
import torch
from torch.utils.data import Dataset
from PIL import Image
from torchvision import transforms
import random

class UnityToRealLife(Dataset):
    def __init__(self, unity_image_dir, real_image_dir, transform=None):
        self.unity_image_dir = unity_image_dir
        self.real_image_dir = real_image_dir
        self.transform = transform

        self.unity_images = sorted(os.listdir(self.unity_image_dir))
        self.real_images = sorted(os.listdir(self.real_image_dir))

        self.len_unity_images =  len(self.unity_images)
        self.len_real_life_images = len(self.real_images)

        self.dataset_size = max(len(self.real_images),len(self.unity_images))

    def __len__(self):
        return self.dataset_size

    def __getitem__(self, idx):
        unity_idx = idx % self.len_unity_images
        unity_image_path = os.path.join(self.unity_image_dir, self.unity_images[unity_idx])
        unity_image = Image.open(unity_image_path).convert("RGB")

        real_life_idx = idx % self.len_real_life_images
        real_image_path = os.path.join(self.real_image_dir, self.real_images[real_life_idx])
        real_image = Image.open(real_image_path).convert("RGB")

        if self.transform:
            unity_image = self.transform(unity_image)
            real_image = self.transform(real_image)

        return unity_image, real_image

