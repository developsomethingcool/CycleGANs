import torch
from torch.utils.data import DataLoader, random_split, Subset
from .dataset import EdgeToRealDataset
from torchvision import transforms
import numpy as np

def get_dataloaders(edge_dir, real_image_dir, batch_size=16, val_split=0.2, test_split=0.1, num_workers=4, random_seed=42):
    base_transform = transforms.Compose([
        transforms.Resize((512, 512)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])

    train_dataset = EdgeToRealDataset(edge_dir=edge_dir, real_image_dir=real_image_dir, edge_transform=base_transform, real_transform=base_transform,augment=False)
    val_test_dataset = EdgeToRealDataset(edge_dir=edge_dir, real_image_dir=real_image_dir, edge_transform=base_transform, real_transform=base_transform,augment=False) 

    # Calculate split sizes for train, val, and test sets
    dataset_size = len(train_dataset)
    val_size = int(val_split * dataset_size)
    test_size = int(test_split * dataset_size)
    train_size = dataset_size - val_size - test_size

    # Randomly shuffle and split dataset indices
    indices = np.arange(dataset_size)
    
    np.random.seed(random_seed)
    np.random.shuffle(indices)

    # Create subsets using the shuffled indices
    train_indices = indices[:train_size]
    val_indices = indices[train_size:train_size + val_size]
    test_indices = indices[train_size + val_size:]

    # Create subset datasets
    train_set = Subset(train_dataset, train_indices)
    val_set = Subset(val_test_dataset, val_indices)
    test_set = Subset(val_test_dataset, test_indices)

    #Create DataLoader for each subset
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=num_workers, drop_last=True)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False, num_workers=num_workers, drop_last=True)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=num_workers, drop_last=True)

    return train_loader, val_loader, test_loader


