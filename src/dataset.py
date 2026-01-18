"""
Data loading and preprocessing for Flowers102 dataset.
"""

import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from typing import Tuple, Dict
import os


# ImageNet normalization values (required for pre-trained models)
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def get_transforms(image_size: int = 224) -> Dict[str, transforms.Compose]:
    """
    Returns train and validation transforms.
    Training includes augmentation, validation does not.
    """
    
    train_transforms = transforms.Compose([
        transforms.RandomResizedCrop(image_size, scale=(0.8, 1.0)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
    ])
    
    val_transforms = transforms.Compose([
        transforms.Resize(int(image_size * 1.14)),
        transforms.CenterCrop(image_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
    ])
    
    return {
        'train': train_transforms,
        'val': val_transforms,
        'test': val_transforms
    }


def load_flowers102(
    data_dir: str = './data',
    batch_size: int = 32,
    num_workers: int = 4,
    image_size: int = 224
) -> Tuple[DataLoader, DataLoader, DataLoader, int]:
    """
    Load Flowers102 dataset and return data loaders.
    Downloads the dataset if not present.
    """
    
    data_transforms = get_transforms(image_size)
    os.makedirs(data_dir, exist_ok=True)
    
    print("Loading Flowers102 dataset...")
    print("(First run will download ~350MB)")
    
    train_dataset = datasets.Flowers102(
        root=data_dir,
        split='train',
        transform=data_transforms['train'],
        download=True
    )
    
    val_dataset = datasets.Flowers102(
        root=data_dir,
        split='val',
        transform=data_transforms['val'],
        download=True
    )
    
    test_dataset = datasets.Flowers102(
        root=data_dir,
        split='test',
        transform=data_transforms['test'],
        download=True
    )
    
    # pin_memory only works with CUDA, not MPS
    import torch
    use_pin_memory = torch.cuda.is_available()
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=use_pin_memory
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=use_pin_memory
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=use_pin_memory
    )
    
    num_classes = 102
    
    print(f"Dataset loaded:")
    print(f"  Train: {len(train_dataset)} images")
    print(f"  Val: {len(val_dataset)} images")
    print(f"  Test: {len(test_dataset)} images")
    print(f"  Classes: {num_classes}")
    
    return train_loader, val_loader, test_loader, num_classes


def denormalize(tensor: torch.Tensor) -> torch.Tensor:
    """Reverse normalization for visualization."""
    mean = torch.tensor(IMAGENET_MEAN).view(3, 1, 1)
    std = torch.tensor(IMAGENET_STD).view(3, 1, 1)
    return tensor * std + mean
