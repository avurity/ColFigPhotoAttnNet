import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

def get_data_loaders(train_path, val_path, test_path, batch_size):
    """
    Prepares DataLoaders for training, validation, and testing datasets.
    
    Args:
        train_path (str): Path to the training dataset.
        val_path (str): Path to the validation dataset.
        test_path (str): Path to the testing dataset.
        batch_size (int): Batch size for DataLoader.
    
    Returns:
        train_loader (DataLoader): DataLoader for training.
        val_loader (DataLoader): DataLoader for validation.
        test_loader (DataLoader): DataLoader for testing.
    """
    # Define transformations
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # Load datasets
    train_dataset = datasets.ImageFolder(train_path, transform=transform)
    val_dataset = datasets.ImageFolder(val_path, transform=transform)
    test_dataset = datasets.ImageFolder(test_path, transform=transform)

    # Create DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)

    return train_loader, val_loader, test_loader

