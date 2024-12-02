import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

from models.integrated_model_wrapper import IntegratedModelWrapper
from models.base_model_factory import base_model_factory
from utils.data_loaders import get_data_loaders
from args import parse_args


def train_model(args):
    # Load device
    device = torch.device(args.device)

    # Initialize model
    model = IntegratedModelWrapper(base_model_factory).to(device)

    # Prepare data loaders
    train_loader, val_loader, _ = get_data_loaders(
        args.train_path, args.val_path, args.test_path, args.batch_size
    )

    # Criterion and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    # Early stopping parameters
    best_loss = float("inf")
    patience_counter = 0

    # Training loop
    for epoch in range(args.epochs):
        # Training phase
        model.train()
        running_loss = 0.0
        correct = 0

        for inputs, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs} - Training"):
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()

            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            correct += (outputs.argmax(1) == labels).sum().item()

        train_loss = running_loss / len(train_loader.dataset)
        train_accuracy = correct / len(train_loader.dataset)

        # Validation phase
        model.eval()
        val_loss = 0.0
        val_correct = 0

        with torch.no_grad():
            for val_inputs, val_labels in tqdm(val_loader, desc=f"Epoch {epoch+1}/{args.epochs} - Validation"):
                val_inputs, val_labels = val_inputs.to(device), val_labels.to(device)

                val_outputs = model(val_inputs)
                val_loss += criterion(val_outputs, val_labels).item()
                val_correct += (val_outputs.argmax(1) == val_labels).sum().item()

        val_loss /= len(val_loader.dataset)
        val_accuracy = val_correct / len(val_loader.dataset)

        # Logging results
        print(f"Epoch {epoch+1}/{args.epochs}")
        print(f"Training Loss: {train_loss:.4f}, Accuracy: {train_accuracy:.4f}")
        print(f"Validation Loss: {val_loss:.4f}, Accuracy: {val_accuracy:.4f}")

        # Early stopping
        if val_loss < best_loss:
            best_loss = val_loss
            patience_counter = 0
            torch.save(model.state_dict(), args.model_path)
            print("Validation loss improved. Model saved!")
        else:
            patience_counter += 1

        if patience_counter >= args.patience:
            print("Early stopping triggered.")
            break


if __name__ == "__main__":
    args = parse_args()
    train_model(args)
