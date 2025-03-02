import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from tqdm import tqdm
import os 
import wandb  

import time 
from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR10

from fftnet_vit import FFTNetViT
from transformer import VisionTransformer 
from fnet import FNetForCIFAR10

class EarlyStopping:
    """Early stopping to stop training when validation accuracy doesn't improve for a given patience."""
    def __init__(self, patience=5):
        """
        Args:
            patience (int): How many epochs to wait after last time validation accuracy improved.
        """
        self.patience = patience
        self.best_val_acc = 0.0
        self.counter = 0
        self.early_stop = False

    def __call__(self, val_acc):
        if val_acc > self.best_val_acc:
            self.best_val_acc = val_acc
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True


def train_model(model, train_loader, test_loader, optimizer, criterion, num_epochs, device, model_name, save_dir):
    """Train the given model and save per-epoch validation metrics to a file."""

    os.makedirs(save_dir, exist_ok=True) 

    metrics_file = f"../cifar/results/{model_name}_val_metrics_updated.txt"
    training_time_file = f"../cifar/results/training-time-{model_name}.txt"

    with open(metrics_file, "w") as f:
        f.write("Epoch,Validation Loss,Validation Accuracy\n")

    best_val_acc = 0.0  # Track best validation accuracy for saving best model
    total_training_time = 0.0  # Track total training time

    early_stopping = EarlyStopping(patience=5)


    for epoch in range(num_epochs):
        start_time = time.time()  # Track start time
        # Training phase.
        model.train()
        running_loss = 0.0
        train_correct = 0
        train_total = 0
        train_loader_tqdm = tqdm(train_loader, desc=f"{model_name} Epoch {epoch+1}/{num_epochs} - Training")
        for inputs, labels in train_loader_tqdm:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * inputs.size(0)
            _, predicted = outputs.max(1)
            train_total += labels.size(0)
            train_correct += predicted.eq(labels).sum().item()
            train_loader_tqdm.set_postfix(loss=loss.item(), acc=100.*train_correct/train_total)
        
        epoch_train_loss = running_loss / len(train_loader.dataset)
        epoch_train_acc = 100. * train_correct / train_total
        print(f"\n{model_name} Epoch [{epoch+1}/{num_epochs}] Training Loss: {epoch_train_loss:.4f} | Accuracy: {epoch_train_acc:.2f}%")
        
        # Validation phase.
        model.eval()
        test_loss = 0.0
        correct = 0
        total = 0
        test_loader_tqdm = tqdm(test_loader, desc=f"{model_name} Epoch {epoch+1}/{num_epochs} - Validation")
        with torch.no_grad():
            for inputs, labels in test_loader_tqdm:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                test_loss += loss.item() * inputs.size(0)
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()
                test_loader_tqdm.set_postfix(loss=loss.item(), acc=100.*correct/total)
        
        val_loss = test_loss / len(test_loader.dataset)
        val_acc = 100. * correct / total
        print(f"{model_name} Epoch [{epoch+1}/{num_epochs}] Validation Loss: {val_loss:.4f} | Accuracy: {val_acc:.2f}%\n")
            
        epoch_time = time.time() - start_time  # Calculate time taken
        total_training_time += epoch_time

        wandb.log({
            "train_loss": epoch_train_loss,
            "train_accuracy": epoch_train_acc,
            "val_loss": val_loss,
            "val_accuracy": val_acc,
            "epoch_time": epoch_time,
            "epoch": epoch + 1
        })

        # Save validation metrics to file
        with open(metrics_file, "a") as f:
            f.write(f"{epoch+1},{val_loss:.4f},{val_acc:.2f}\n")

        with open(training_time_file, "a") as f:
            f.write(f"{epoch+1}, {epoch_time:.2f}\n")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), os.path.join(save_dir, f"best_{model_name}.pth"))
            wandb.save(os.path.join(save_dir, f"best_{model_name}.pth"))

        # Save latest model
        torch.save(model.state_dict(), os.path.join(save_dir, f"latest_{model_name}.pth"))
        wandb.save(os.path.join(save_dir, f"latest_{model_name}.pth"))

        print(f"\n{model_name} Epoch [{epoch+1}/{num_epochs}]")
        print(f"Train Loss: {epoch_train_loss:.4f} | Train Acc: {epoch_train_acc:.2f}%")
        print(f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2f}%\n")
        print(f"Epoch training time : {epoch_time}")

        early_stopping(val_acc)
        if early_stopping.early_stop:
            print(f"Early stopping triggered at epoch {epoch+1}. No improvement in validation accuracy for {early_stopping.patience} epochs.")
            break

    with open(training_time_file, "a") as f:
        f.write(f"Total Training Time: {total_training_time:.2f} seconds\n")

def main():
    # # Hyperparameters for CIFAR10.
    num_epochs = 100
    batch_size = 128
    learning_rate = 7e-4

    root_save_dir = "../cifar/trained_weights"
    os.makedirs(root_save_dir, exist_ok=True)


    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu") # updated this to support one gpu 
 
    # # Data transforms for CIFAR10.
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    """
    Updated this to consider the data pre - downloaded instead of downloading it from stratch
    """

    cifar10_path = './data/cifar-10-batches-py'

    if os.path.exists(cifar10_path):
        # Dataset already exists, load without downloading
        print("CIFAR10 dataset already exists. Loading directly...")
        train_dataset = torchvision.datasets.CIFAR10(
            root='./data', train=True, download=False, transform=transform_train)
        test_dataset = torchvision.datasets.CIFAR10(
            root='./data', train=False, download=False, transform=transform_test)
    else:
        # Dataset doesn't exist, download it
        print("CIFAR10 dataset not found. Downloading...")
        train_dataset = torchvision.datasets.CIFAR10(
            root='./data', train=True, download=True, transform=transform_train)
        test_dataset = torchvision.datasets.CIFAR10(
            root='./data', train=False, download=True, transform=transform_test)

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False, num_workers=2)


    # # # Initialize FFTNetViT model
    fftnet_model = FFTNetViT(
        img_size=32, patch_size=4, in_chans=3, num_classes=10,
        embed_dim=192, depth=6, mlp_ratio=3.0, dropout=0.1,
        num_heads=6, adaptive_spectral=True
    ).to(device)
    
    # Initialize VisionTransformer model
    transformer_model = VisionTransformer(
        image_size=32, patch_size=4, in_channels=3, num_classes=10,
        embed_dim=192, depth=6, mlp_ratio=3.0, num_heads=6
    ).to(device)


    criterion = nn.CrossEntropyLoss()

    # Train FFTNetViT
    print("Starting training for FFTNetViT model...")

    fftnet_config = {
        "model": "FFTNetViT",
        "learning_rate": learning_rate,
        "batch_size": batch_size,
        "num_epochs": num_epochs,
        "embed_dim": 192,
        "depth": 6,
        "mlp_ratio": 3.0,
        "dropout": 0.1,
        "num_heads": 6,
        "adaptive_spectral": True
    }
    

    fftnet_save_dir = os.path.join(root_save_dir, "FFTNetViT")
    transformer_save_dir = os.path.join(root_save_dir, "VisionTransformer")
    fnet_base_save_dir = os.path.join(root_save_dir, "FNET-base")


    wandb.init(project="cifar10-models", name="FFTNetViT", config=fftnet_config)

    optimizer_fftnet = optim.Adam(fftnet_model.parameters(), lr=learning_rate)
    train_model(fftnet_model, train_loader, test_loader, optimizer_fftnet, 
                criterion, num_epochs, device, model_name="FFTNetViT", save_dir=fftnet_save_dir)
    
    wandb.finish()

    print("\nStarting training for VisionTransformer model...")

    transformer_config = {
        "model": "VisionTransformer",
        "learning_rate": learning_rate,
        "batch_size": batch_size,
        "num_epochs": num_epochs,
        "embed_dim": 192,
        "depth": 6,
        "mlp_ratio": 3.0,
        "num_heads": 6
    }


    wandb.init(project="cifar10-models", name="VisionTransformer", config=transformer_config)

    optimizer_transformer = optim.Adam(transformer_model.parameters(), lr=learning_rate)
    train_model(transformer_model, train_loader, test_loader, optimizer_transformer,
                criterion, num_epochs, device, model_name="VisionTransformer",save_dir=transformer_save_dir)
                
    wandb.finish()


    print("Starting training for FNET model...")


    fnet_config = {
        'patch_size': 4,  # Patch size for CIFAR-10 (32x32 images -> 8x8 patches)
        'embedding_size': 128,  # Embedding dimension
        'hidden_size': 128,  # Hidden size for FNet layers
        'intermediate_size': 512,  # Intermediate size for feed-forward layers
        'num_hidden_layers': 6,  # Number of FNet layers
        'num_classes': 10,  # CIFAR-10 has 10 classes
        'fourier': 'fft',  # Use FFT for Fourier Transform
        'layer_norm_eps': 1e-12,
        'dropout_rate': 0.1,
    }

    fnet_model  = FNetForCIFAR10(fnet_config)
    fnet_model.to(device)

    optimizer_fnet = torch.optim.Adam(fnet_model.parameters(), lr=1e-3)


    wandb.init(project="cifar10-models", name="fnet-base", config=fnet_config)

    train_model(fnet_model, train_loader, test_loader, optimizer_fnet,
                criterion, num_epochs, device, model_name="fnet-base",save_dir=fnet_base_save_dir)
                
    wandb.finish()


if __name__ == "__main__":
    main()



