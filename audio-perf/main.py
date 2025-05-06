import argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, random_split
import torchaudio
from torchaudio.datasets import SPEECHCOMMANDS
from tqdm import tqdm
import wandb
import torchaudio
torchaudio.set_audio_backend("soundfile")


# --- Fourier Layer ---
class FourierFFTLayerFlexible(nn.Module):
    def __init__(self, d_model, mode="real+complex"):
        super().__init__()
        self.mode = mode
        if mode == "real+complex":
            self.proj = nn.Linear(2 * d_model, d_model)
        else:
            self.proj = nn.Linear(d_model, d_model)
    def forward(self, x):
        fft = torch.fft.fft(x.float(), dim=-1)
        if self.mode == "real":
            features = fft.real
        elif self.mode == "complex":
            features = fft.imag
        elif self.mode == "real+complex":
            features = torch.cat([fft.real, fft.imag], dim=-1)
        else:
            raise ValueError(f"Unknown mode: {self.mode}")
        return self.proj(features).to(x.dtype)

class FNetLayerFlexible(nn.Module):
    def __init__(self, d_model, hidden_dim, mode="real+complex"):
        super().__init__()
        self.fft = FourierFFTLayerFlexible(d_model, mode=mode)
        self.norm1 = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(
            nn.Linear(d_model, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, d_model)
        )
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(0.1)
    def forward(self, x):
        x2 = self.fft(x)
        x = self.norm1(x + x2)
        x2 = self.ff(x)
        x2 = self.dropout(x2)
        x = self.norm2(x + x2)
        return x

class FNetAudioClassifier(nn.Module):
    def __init__(self, input_dim, num_classes, d_model=128, hidden_dim=256, n_layers=2, mode="real+complex"):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, d_model)
        self.layers = nn.ModuleList([FNetLayerFlexible(d_model, hidden_dim, mode=mode) for _ in range(n_layers)])
        self.classifier = nn.Linear(d_model, num_classes)
    def forward(self, x):
        x = self.input_proj(x)
        for layer in self.layers:
            x = layer(x)
        return self.classifier(x)

# --- Argument parser ---
parser = argparse.ArgumentParser()
parser.add_argument("--fourier_features", type=str, choices=["real", "complex", "real+complex"], default="real+complex")
parser.add_argument("--batch_size", type=int, default=32)
parser.add_argument("--epochs", type=int, default=5)
parser.add_argument("--sample_rate", type=int, default=16000)
parser.add_argument("--duration", type=float, default=1.0)  # seconds
args = parser.parse_args()

wandb.init(project="speechcommands-fnet", name=f"fnet-speechcommands-{args.fourier_features}")

SELECTED_CLASSES = ["yes", "no", "up", "down", "left", "right"]
import torchaudio.transforms as T

class SmallSpeechCommands(Dataset):
    def __init__(self, root, sample_rate=16000, duration=1.0, subset="training", n_mels=64):
        self.dataset = SPEECHCOMMANDS(root, download=True, subset=subset)
        self.sample_rate = sample_rate
        self.length = int(sample_rate * duration)
        self.label2idx = {c: i for i, c in enumerate(SELECTED_CLASSES)}
        self.indices = [i for i, (_, _, label, _, _) in enumerate(self.dataset) if label in SELECTED_CLASSES]
        self.mel_transform = T.MelSpectrogram(
            sample_rate=sample_rate,
            n_fft=1024,
            hop_length=256,
            n_mels=n_mels
        )
    def __len__(self):
        return len(self.indices)
    def __getitem__(self, idx):
        real_idx = self.indices[idx]
        waveform, sr, label, *_ = self.dataset[real_idx]
        waveform = waveform.mean(dim=0)  # mono
        waveform = torchaudio.functional.resample(waveform, sr, self.sample_rate)
        if waveform.size(0) > self.length:
            waveform = waveform[:self.length]
        else:
            pad = self.length - waveform.size(0)
            waveform = torch.nn.functional.pad(waveform, (0, pad))
        mel = self.mel_transform(waveform)  # (n_mels, time)
        mel = mel.clamp(min=1e-5).log()    # log-mel for stability
        return mel.flatten(), self.label2idx[label]

n_mels = 64
mel_len = int(args.sample_rate * args.duration // 256) + 1  # hop_length=256
input_dim = n_mels * mel_len



train_data = SmallSpeechCommands("./data", sample_rate=args.sample_rate, duration=args.duration, subset="training", n_mels=n_mels)
test_data = SmallSpeechCommands("./data", sample_rate=args.sample_rate, duration=args.duration, subset="testing", n_mels=n_mels)

train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True)
test_loader = DataLoader(test_data, batch_size=args.batch_size, shuffle=False)

num_classes = len(SELECTED_CLASSES)
model = FNetAudioClassifier(input_dim, num_classes, d_model=128, hidden_dim=256, n_layers=2, mode=args.fourier_features)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=2e-4)
criterion = nn.CrossEntropyLoss()

# --- Training loop ---
for epoch in range(args.epochs):
    model.train()
    total, correct, loss_sum = 0, 0, 0
    for x, y in tqdm(train_loader, desc=f"Epoch {epoch+1} [Train]"):
        x, y = x.to(device), y.to(device)
        logits = model(x)
        loss = criterion(logits, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        loss_sum += loss.item() * x.size(0)
        preds = logits.argmax(dim=1)
        correct += (preds == y).sum().item()
        total += x.size(0)
    train_acc = correct / total
    train_loss = loss_sum / total

    model.eval()
    total, correct, loss_sum = 0, 0, 0
    with torch.no_grad():
        for x, y in tqdm(test_loader, desc=f"Epoch {epoch+1} [Val]"):
            x, y = x.to(device), y.to(device)
            logits = model(x)
            loss = criterion(logits, y)
            loss_sum += loss.item() * x.size(0)
            preds = logits.argmax(dim=1)
            correct += (preds == y).sum().item()
            total += x.size(0)
    val_acc = correct / total
    val_loss = loss_sum / total

    print(f"Epoch {epoch+1}: Train Acc={train_acc:.4f}, Val Acc={val_acc:.4f}")
    wandb.log({"epoch": epoch+1, "train_acc": train_acc, "val_acc": val_acc, "train_loss": train_loss, "val_loss": val_loss})

torch.save(model.state_dict(), f"fnet_speechcommands_{args.fourier_features}.pt")