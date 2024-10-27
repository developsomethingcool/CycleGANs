import os
import torch
from torch.utils.data import Dataset
from PIL import Image
from torchvision import transforms
import random

class EdgeToRealDataset(Dataset):
    def __init__(self, edge_dir, real_image_dir, edge_transform=None, real_transform=None, augment=None):
        self.edge_dir = edge_dir
        self.real_image_dir = real_image_dir
        self.edge_transform = edge_transform
        self.real_transform = real_transform
        self.augment = augment

        self.edge_images = sorted(os.listdir(self.edge_dir))
        self.real_images = sorted(os.listdir(self.real_image_dir))

        assert len(self.edge_images) == len(self.real_images), "Number of edge images should be the same as number of real images"

        #augmentation applied to both edges andreals
        self.common_transforms = transforms.Compose([
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(degrees=10),
            transforms.RandomCrop((450, 450)),
            transforms.Resize((512, 512))
        ])

        #augmentation only real life images
        self.real_only_transforms = transforms.Compose([
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        ])

    def __len__(self):
        return len(self.real_images)

    def __getitem__(self, idx):
        edge_image_path = os.path.join(self.edge_dir, self.edge_images[idx])
        real_image_path = os.path.join(self.real_image_dir, self.real_images[idx])

        edge_image = Image.open(edge_image_path).convert("RGB")
        real_image = Image.open(real_image_path).convert("RGB")

        if self.augment:
            # Seed ensures same random transformation for edge and real
            seed = random.randint(0,10000)
            random.seed(seed)
            edge_image = self.common_transforms(edge_image)
            random.seed(seed)
            real_image = self.common_transforms(real_image)

            #apply real-image-only augmentation
            real_image = self.real_only_transforms(real_image)

        if self.edge_transform:
            edge_image = self.edge_transform(edge_image)
        if self.real_transform:
            real_image = self.real_transform(real_image)

        return edge_image, real_image
