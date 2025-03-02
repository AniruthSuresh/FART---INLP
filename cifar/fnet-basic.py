import torch
import torch.nn as nn
from torchvision import transforms
from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR10

from fnet_modules import FNetEncoder

# Define the FNet model for CIFAR-10
class FNetForCIFAR10(nn.Module):
    def __init__(self, config):
        super(FNetForCIFAR10, self).__init__()
        self.config = config

        # Patch Embedding Layer (similar to Vision Transformers)
        self.patch_embed = nn.Conv2d(
            in_channels=3,  # CIFAR-10 has 3 color channels
            out_channels=config['embedding_size'],
            kernel_size=config['patch_size'],
            stride=config['patch_size']
        )
        self.num_patches = (32 // config['patch_size']) ** 2  # CIFAR-10 images are 32x32

        # Positional Embeddings
        self.position_embeddings = nn.Parameter(
            torch.zeros(1, self.num_patches + 1, config['embedding_size'])
        )
        self.cls_token = nn.Parameter(torch.zeros(1, 1, config['embedding_size']))

        # FNet Encoder
        self.encoder = FNetEncoder(config)

        # Classification Head
        self.classifier = nn.Sequential(
            nn.LayerNorm(config['embedding_size']),
            nn.Linear(config['embedding_size'], config['num_classes'])
        )

    def forward(self, x):
        # Patch Embedding
        x = self.patch_embed(x)  # (B, embedding_size, H', W')
        x = x.flatten(2).transpose(1, 2)  # (B, num_patches, embedding_size)

        # Add CLS Token
        cls_tokens = self.cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)  # (B, num_patches + 1, embedding_size)

        # Add Positional Embeddings
        x = x + self.position_embeddings

        # FNet Encoder
        x = self.encoder(x)

        # Classification Head (use CLS token)
        cls_output = x[:, 0]
        logits = self.classifier(cls_output)
        return logits


# Configuration for CIFAR-10
config = {
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

# Create the model
model = FNetForCIFAR10(config)

# Print model summary
print(model)


# Data Preprocessing and Loading
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # Normalize to [-1, 1]
])

train_dataset = CIFAR10(root='./data', train=True, download=True, transform=transform)
test_dataset = CIFAR10(root='./data', train=False, download=True, transform=transform)

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)


# Training Loop
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

for epoch in range(10):  # Train for 10 epochs
    model.train()
    running_loss = 0.0
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    print(f"Epoch [{epoch + 1}/10], Loss: {running_loss / len(train_loader):.4f}")


# Evaluation
model.eval()
correct = 0
total = 0
with torch.no_grad():
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print(f"Test Accuracy: {100 * correct / total:.2f}%")


