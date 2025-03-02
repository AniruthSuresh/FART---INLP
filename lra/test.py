import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from lra_datasets import ImdbDataset, ListOpsDataset, Cifar10Dataset



class FNetLayer(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.hidden_size = hidden_size
        
        # Fourier transform layer
        self.fourier = nn.Linear(hidden_size, hidden_size)
        self.norm1 = nn.LayerNorm(hidden_size)
        
        # Feed-forward network
        self.ffn = nn.Sequential(
            nn.Linear(hidden_size, 4 * hidden_size),
            nn.GELU(),
            nn.Linear(4 * hidden_size, hidden_size),
            nn.Dropout(0.1)
        )
        self.norm2 = nn.LayerNorm(hidden_size)

    def forward(self, x):
        # Fourier mixing
        x_ft = torch.fft.fft(torch.fft.fft(x.float(), dim=-1), dim=-2).real
        x = self.norm1(x + x_ft)
        
        # Feedforward
        x_ffn = self.ffn(x)
        x = self.norm2(x + x_ffn)
        return x

class FNetForClassification(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.embedding = nn.Embedding(config.vocab_size, config.hidden_size)
        self.layers = nn.ModuleList([
            FNetLayer(config.hidden_size) for _ in range(config.num_layers)
        ])
        self.pooler = nn.Linear(config.hidden_size, config.hidden_size)
        self.classifier = nn.Linear(config.hidden_size, config.num_classes)
        self.dropout = nn.Dropout(config.dropout_prob)

    def forward(self, input_ids):
        embeddings = self.embedding(input_ids)
        hidden_states = embeddings
        
        for layer in self.layers:
            hidden_states = layer(hidden_states)
            hidden_states = self.dropout(hidden_states)
            
        # Mean pooling
        pooled = hidden_states.mean(dim=1)
        return self.classifier(pooled)

def train_model(config):
    # Initialize dataset and dataloader
    train_dataset = ImdbDataset(config, split='train')
    val_dataset = ImdbDataset(config, split='validation')
    
    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=config.batch_size)

    # Model setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = FNetForClassification(config).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=config.learning_rate)
    criterion = nn.CrossEntropyLoss()

    # Training loop
    for epoch in range(config.epochs):
        model.train()
        total_loss = 0
        correct = 0
        total = 0
        
        for batch in train_loader:
            inputs = batch['input_ids'].to(device)
            labels = batch['label'].to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
        
        # Validation
        model.eval()
        val_correct = 0
        val_total = 0
        with torch.no_grad():
            for batch in val_loader:
                inputs = batch['input_ids'].to(device)
                labels = batch['label'].to(device)
                
                outputs = model(inputs)
                _, predicted = torch.max(outputs.data, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()
        
        print(f"Epoch {epoch+1}/{config.epochs}")
        print(f"Train Loss: {total_loss/len(train_loader):.4f} | Acc: {100*correct/total:.2f}%")
        print(f"Val Acc: {100*val_correct/val_total:.2f}%\n")

    print("Training complete!")
    return model

def simple_tokenizer(source, max_length):
    # For text input (IMDB or ListOps), we convert characters to their ordinal values.
    # For CIFAR10, the source is a numpy array (grayscale pixels).
    if isinstance(source, str):
        tokens = [ord(c) for c in source[:max_length]]
    elif isinstance(source, np.ndarray):
        tokens = source.flatten()[:max_length].tolist()
    else:
        tokens = list(source)[:max_length]
    # Pad the sequence if needed.
    if len(tokens) < max_length:
        tokens += [0] * (max_length - len(tokens))
    # Return a tensor of shape (max_length,).
    return torch.tensor(tokens, dtype=torch.float32)

# Example config (adjust based on your actual config)
class Config:
    vocab_size = 50000
    hidden_size = 256
    num_layers = 4
    num_classes = 2
    batch_size = 64
    epochs = 10
    learning_rate = 2e-4
    dropout_prob = 0.1
    max_length = 512


config = Config()
config.tokenizer = simple_tokenizer
trained_model = train_model(config)
