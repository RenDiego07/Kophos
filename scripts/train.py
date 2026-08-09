"""
Training Loop for ASL Recognition BiLSTM Model
Orchestrates data loading, model checkpointing, and CSV logging.
"""

import os
import sys
import argparse
import csv
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

# Parche de rutas: Obliga a Python a buscar en el mismo directorio de este script

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Ahora las importaciones funcionarán sin problema
from dataset import get_dataloaders
from model import BiLSTMSignModel

def train_model(
    csv_path: str,
    output_dir: str,
    epochs: int = 50,
    batch_size: int = 32,
    learning_rate: float = 0.001,
    num_classes: int = 34
):
    # 1. Configuración de Dispositivo
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🚀 Iniciando entrenamiento en: {device}")

    os.makedirs(output_dir, exist_ok=True)
    best_model_path = os.path.join(output_dir, "best_bilstm_model.pth")
    log_file_path = os.path.join(output_dir, "training_log.csv")

    # Inicializar archivo de log
    with open(log_file_path, mode='w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["Epoch", "Train_Loss", "Train_Acc", "Val_Loss", "Val_Acc"])

    # 2. Cargar Datos
    print("\n📦 Cargando DataLoaders...")
    dataloaders = get_dataloaders(metadata_csv=csv_path, batch_size=batch_size)
    train_loader = dataloaders["train"]
    val_loader = dataloaders["val"]

    # 3. Inicializar Modelo dinámico
    print(f"🧠 Inicializando Modelo BiLSTM para {num_classes} clases...")
    model = BiLSTMSignModel(input_dim=225, hidden_dim=128, num_classes=num_classes) # <-- Dinámico
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    best_val_accuracy = 0.0

    # 4. Bucle de Entrenamiento
    print("\n🔥 Comenzando Entrenamiento...")
    for epoch in range(1, epochs + 1):
        print(f"\n--- Epoch {epoch}/{epochs} ---")
        
        # --- FASE DE ENTRENAMIENTO ---
        model.train()
        train_loss, train_correct, train_total = 0.0, 0, 0

        train_loop = tqdm(train_loader, desc="Entrenando", leave=False)
        for features, labels in train_loop:
            features, labels = features.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(features)
            loss = criterion(outputs, labels)
            
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * features.size(0)
            _, predicted = torch.max(outputs.data, 1)
            train_total += labels.size(0)
            train_correct += (predicted == labels).sum().item()

            train_loop.set_postfix(loss=loss.item())

        epoch_train_loss = train_loss / train_total
        epoch_train_acc = (train_correct / train_total) * 100

        # --- FASE DE VALIDACIÓN ---
        model.eval()
        val_loss, val_correct, val_total = 0.0, 0, 0

        with torch.no_grad():
            val_loop = tqdm(val_loader, desc="Validando", leave=False)
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

        print(f"Train Loss: {epoch_train_loss:.4f} | Train Acc: {epoch_train_acc:.2f}%")
        print(f"Val Loss:   {epoch_val_loss:.4f} | Val Acc:   {epoch_val_acc:.2f}%")

        # --- ESCRIBIR EN EL LOG CSV ---
        with open(log_file_path, mode='a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([epoch, epoch_train_loss, epoch_train_acc, epoch_val_loss, epoch_val_acc])

        # --- GUARDAR MEJOR MODELO ---
        if epoch_val_acc > best_val_accuracy:
            print(f"⭐ Precisión mejoró de {best_val_accuracy:.2f}% a {epoch_val_acc:.2f}%. Guardando modelo...")
            best_val_accuracy = epoch_val_acc
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_accuracy': best_val_accuracy,
            }, best_model_path)

    print(f"\n🎉 ¡Entrenamiento Completo! Mejor Accuracy: {best_val_accuracy:.2f}%")
    print(f"💾 Modelo y logs guardados en: {output_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv_path", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--num_classes", type=int, default=34)
    args = parser.parse_args()
    
    train_model(
        args.csv_path, 
        args.output_dir, 
        args.epochs, 
        args.batch_size, 
        args.lr,
        args.num_classes
    )