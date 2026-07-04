"""
Training Loop for ASL Recognition BiLSTM Model
Orchestrates data loading, forward/backward passes, and model checkpointing.
"""

import os
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

# Import your custom modules
from dataset import get_dataloaders
from model import BiLSTMSignModel

def train_model(
    csv_path: str,
    output_dir: str,
    epochs: int = 50,
    batch_size: int = 32,
    learning_rate: float = 0.001
):
    """
    Main training function.
    """
    # 1. Setup Device (Use GPU if available, otherwise CPU)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🚀 Starting training on device: {device}")

    # Ensure output directory exists for saving weights
    os.makedirs(output_dir, exist_ok=True)
    best_model_path = os.path.join(output_dir, "best_bilstm_model.pth")

    # 2. Load Data
    print("\n📦 Loading DataLoaders...")
    dataloaders = get_dataloaders(metadata_csv=csv_path, batch_size=batch_size)
    train_loader = dataloaders["train"]
    val_loader = dataloaders["val"]

    # 3. Initialize Model
    # NSLT-100 has 100 classes. Feature dim is 225 from MediaPipe.
    print("🧠 Initializing BiLSTM Model...")
    model = BiLSTMSignModel(input_dim=225, hidden_dim=128, num_classes=100)
    model = model.to(device)

    # 4. Define Loss Function and Optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    best_val_accuracy = 0.0

    # 5. Training Loop
    print("\n🔥 Starting Training Loop...")
    for epoch in range(1, epochs + 1):
        print(f"\n--- Epoch {epoch}/{epochs} ---")
        
        # ==========================================
        # TRAINING PHASE
        # ==========================================
        model.train()  # Set model to training mode (enables Dropout)
        train_loss = 0.0
        train_correct = 0
        train_total = 0

        # Progress bar for training
        train_loop = tqdm(train_loader, desc="Training", leave=False)
        
        for features, labels in train_loop:
            # Move data to the active device (GPU/CPU)
            features, labels = features.to(device), labels.to(device)

            # Zero the parameter gradients
            optimizer.zero_grad()

            # Forward pass
            outputs = model(features)
            loss = criterion(outputs, labels)

            # Backward pass and optimize
            loss.backward()
            optimizer.step()

            # Calculate statistics
            train_loss += loss.item() * features.size(0)
            _, predicted = torch.max(outputs.data, 1)
            train_total += labels.size(0)
            train_correct += (predicted == labels).sum().item()

            # Update progress bar description
            train_loop.set_postfix(loss=loss.item())

        epoch_train_loss = train_loss / train_total
        epoch_train_acc = (train_correct / train_total) * 100

        # ==========================================
        # VALIDATION PHASE
        # ==========================================
        model.eval()  # Set model to evaluation mode (disables Dropout)
        val_loss = 0.0
        val_correct = 0
        val_total = 0

        # Disable gradient calculation for validation to save memory and compute
        with torch.no_grad():
            val_loop = tqdm(val_loader, desc="Validation", leave=False)
            
            for features, labels in val_loop:
                features, labels = features.to(device), labels.to(device)

                outputs = model(features)
                loss = criterion(outputs, labels)

                val_loss += loss.item() * features.size(0)
                _, predicted = torch.max(outputs.data, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()

        epoch_val_loss = val_loss / val_total
        epoch_val_acc = (val_correct / val_total) * 100

        # Print Epoch Summary
        print(f"Train Loss: {epoch_train_loss:.4f} | Train Acc: {epoch_train_acc:.2f}%")
        print(f"Val Loss:   {epoch_val_loss:.4f} | Val Acc:   {epoch_val_acc:.2f}%")

        # ==========================================
        # SAVE CHECKPOINT (Best Model)
        # ==========================================
        if epoch_val_acc > best_val_accuracy:
            print(f"⭐ Validation accuracy improved from {best_val_accuracy:.2f}% to {epoch_val_acc:.2f}%. Saving model...")
            best_val_accuracy = epoch_val_acc
            
            # Save the model's weights (state_dict)
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_accuracy': best_val_accuracy,
            }, best_model_path)

    print(f"\n🎉 Training Complete! Best Validation Accuracy: {best_val_accuracy:.2f}%")
    print(f"💾 Model saved to: {best_model_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train BiLSTM for ASL Recognition")
    
    parser.add_argument("--csv_path", type=str, required=True, help="Path to metadata_mp.csv")
    parser.add_argument("--output_dir", type=str, default="./checkpoints", help="Directory to save model weights")
    parser.add_argument("--epochs", type=int, default=50, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size for DataLoaders")
    parser.add_argument("--lr", type=float, default=0.001, help="Learning rate for Adam optimizer")
    
    args = parser.parse_args()
    
    train_model(
        csv_path=args.csv_path,
        output_dir=args.output_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr
    )