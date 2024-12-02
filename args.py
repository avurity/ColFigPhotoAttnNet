import argparse
import torch

def parse_args():
    """
    Parse command-line arguments for training and testing scripts.
    """
    parser = argparse.ArgumentParser(description="Training and Testing Integrated Model")

    # Data paths
    parser.add_argument("--train_path", type=str, required=True, help="Path to the training dataset")
    parser.add_argument("--val_path", type=str, required=True, help="Path to the validation dataset")
    parser.add_argument("--test_path", type=str, required=True, help="Path to the testing dataset")
    parser.add_argument("--model_path", type=str, default="best_model.pth", help="Path to save/load the model")

    # Training parameters
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size for training and testing")
    parser.add_argument("--epochs", type=int, default=30, help="Number of training epochs")
    parser.add_argument("--lr", type=float, default=0.0001, help="Learning rate for optimizer")
    parser.add_argument("--patience", type=int, default=5, help="Patience for early stopping")

    # Device
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="Device to run the model on")

    return parser.parse_args()



def get_test_args():
    """
    Parse command-line arguments for testing the model.
    """
    parser = argparse.ArgumentParser(description="Arguments for testing the model")

    # Data path
    parser.add_argument("--test_path", type=str, required=True, help="Path to the testing dataset")
    parser.add_argument("--model_path", type=str, required=True, help="Path to the saved model")

    # Testing parameters
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size for the DataLoader")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu",
                        help="Device to run the model on ('cuda' or 'cpu')")

    return parser.parse_args()
