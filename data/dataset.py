import os
import torch
from torch.utils.data import Dataset
from PIL import Image
from torchvision import transforms
import random

class UnityToRealLife(Dataset):
    def __init__(self, domain_A_image_dir, domain_B_image_dir, transform=None):
        self.domain_A_image_dir = domain_A_image_dir
        self.domain_B_image_dir = domain_B_image_dir
        self.transform = transform

        self.domain_A_images = sorted(os.listdir(self.domain_A_image_dir))
        self.domain_B_images = sorted(os.listdir(self.domain_B_image_dir))

        self.len_domain_A_images =  len(self.domain_A_images)
        self.len_domnain_B_images = len(self.domain_B_images)

        self.dataset_size = max(self.len_domain_A_images, self.len_domnain_B_images)

    def __len__(self):
        return self.dataset_size

    def __getitem__(self, idx):
        domain_A_idx = idx % self.len_domain_A_images
        domain_A_image_path = os.path.join(self.domain_A_image_dir, self.domain_A_images[domain_A_idx])
        domain_A_image = Image.open(domain_A_image_path).convert("RGB")

        domain_B_idx = idx % self.len_domnain_B_images
        domain_B_image_path = os.path.join(self.domain_B_image_dir, self.domain_B_images[domain_B_idx])
        domain_B_image = Image.open(domain_B_image_path).convert("RGB")

        if self.transform:
            domain_A_image = self.transform(domain_A_image)
            domain_B_image = self.transform(domain_B_image)

        return domain_A_image, domain_B_image

