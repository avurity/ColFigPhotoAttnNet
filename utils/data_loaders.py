import torch
import os
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

def get_data_loaders(train_path, val_path, test_path, batch_size):
    """
    Prepares DataLoaders for training, validation, and testing datasets.
    
    Args:
        train_path (str or None): Path to the training dataset. Can be None or empty.
        val_path (str or None): Path to the validation dataset. Can be None or empty.
        test_path (str or None): Path to the testing dataset. Can be None or empty.
        batch_size (int): Batch size for DataLoader.
    
    Returns:
        train_loader (DataLoader or None): DataLoader for training, or None if train_path is not provided.
        val_loader (DataLoader or None): DataLoader for validation, or None if val_path is not provided.
        test_loader (DataLoader or None): DataLoader for testing, or None if test_path is not provided.
    """
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    train_loader, val_loader, test_loader = None, None, None
    if train_path and os.path.exists(train_path):
        train_dataset = datasets.ImageFolder(train_path, transform=transform)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    else:
        print("Warning: Unable to initialize train_loader")

    if val_path and os.path.exists(val_path):
        val_dataset = datasets.ImageFolder(val_path, transform=transform)
        val_loader = DataLoader(val_dataset, batch_size=batch_size)
    else:
        print("Warning: Unable to initialize val_loader")

    if test_path and os.path.exists(test_path):
        test_dataset = datasets.ImageFolder(test_path, transform=transform)
        test_loader = DataLoader(test_dataset, batch_size=batch_size)
    else:
        print("Warning: Unable to initialize test_loader")

    return train_loader, val_loader, test_loader


