import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import os

def get_dataset(name, batch_size=128):
    """
    Load and prepare dataset for GAN training
    
    Args:
        name: Dataset name ('cifar10' or 'celeba')
        batch_size: Batch size for training
        
    Returns:
        train_loader: DataLoader for the selected dataset
        img_shape: Shape of the images (C, H, W)
    """
    # Define transforms
    if name.lower() == 'cifar10':
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
        
        # Load CIFAR-10 dataset
        dataset = datasets.CIFAR10(
            root='./data',
            train=True,
            download=True,
            transform=transform
        )
        img_shape = (3, 32, 32)
        
    elif name.lower() == 'celeba':
        transform = transforms.Compose([
            transforms.Resize((64, 64)),
            transforms.CenterCrop(64),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
        
        # Load CelebA dataset
        dataset = datasets.ImageFolder(
            root='./data/celeba',
            transform=transform
        )
        img_shape = (3, 64, 64)
        
    else:
        raise ValueError(f"Dataset {name} not supported. Choose 'cifar10' or 'celeba'.")
    
    # Create dataloader
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )
    
    return dataloader, img_shape

def prepare_celeba():
    """
    Instructions to download and prepare CelebA dataset
    """
    print("CelebA dataset needs to be downloaded separately.")
    print("1. Download from https://mmlab.ie.cuhk.edu.hk/projects/CelebA.html")
    print("2. Extract the images to data/celeba/img_align_celeba/")
    print("3. Create a file structure: data/celeba/img_align_celeba/*.jpg")