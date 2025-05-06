from transformers import BartTokenizer, BartModel
from transformers import BartForSequenceClassification
import re
import torch
import torch.utils.checkpoint
from scipy import linalg
from torch import nn

model = BartForSequenceClassification.from_pretrained("facebook/bart-base", num_labels=2)
tokenizer = BartTokenizer.from_pretrained("facebook/bart-base")

import torch
import torch.nn as nn
from torch import linalg, fft
import math
from typing import Optional, Tuple, Union, List

# Assuming necessary imports from transformers are available:
from transformers.modeling_outputs import BaseModelOutput
from transformers.models.bart.modeling_bart import (
    BartPreTrainedModel,
    BartConfig,
    BartScaledWordEmbedding,
    BartLearnedPositionalEmbedding,
)
from transformers.utils import logging # For potential warnings


from datasets import load_dataset
from transformers import BartTokenizer, BartForSequenceClassification, Trainer, TrainingArguments
from sklearn.metrics import accuracy_score
import torch
import wandb
import argparse


logger = logging.get_logger(__name__)

class FourierMMLayer(nn.Module): # dont worrry about this => we use fft and not this !

    def __init__(self, config: BartConfig):
        super().__init__()
        
        # Use parameters from BartConfig
        self.dft_mat_seq = nn.Parameter(torch.tensor(linalg.dft(config.max_position_embeddings)), requires_grad=False)
        self.dft_mat_hidden = nn.Parameter(torch.tensor(linalg.dft(config.d_model)), requires_grad=False)

    def forward(self, hidden_states):
        # Ensure correct device placement for DFT matrices
        dft_mat_hidden = self.dft_mat_hidden.to(hidden_states.device, dtype=torch.complex128)
        dft_mat_seq = self.dft_mat_seq.to(hidden_states.device, dtype=torch.complex128)

        # Match FNet paper's implementation: FFT along sequence, then along hidden dim
        # Note: Original code applied hidden first, then sequence via einsum.
        # Let's try matching the paper's order: FFT(seq), then FFT(hidden)
        # hidden_states_complex = hidden_states.type(torch.complex128)
        # seq_fft = torch.einsum("...ij,...jk->...ik", hidden_states_complex, dft_mat_seq)
        # hidden_fft = torch.einsum("...ij,...jk->...ik", seq_fft.transpose(-1,-2), dft_mat_hidden).transpose(-1,-2)
        # return hidden_fft.real.type(torch.float32)

        # --- OR Using the original einsum approach (potentially faster if dimensions match einsum intent) ---
        # This assumes the einsum correctly performs 2D DFT. Check dimensions carefully.
        # The einsum "...ij,...jk,...ni->...nk" seems unusual for a standard 2D DFT application.
        # A standard 2D DFT might look more like:
        # Step 1: DFT across hidden dim: temp = einsum("...ij,...jk->...ik", hidden_states_complex, dft_mat_hidden)
        # Step 2: DFT across seq dim: result = einsum("...ij,...jk->...ik", temp.transpose(-1,-2), dft_mat_seq).transpose(-1,-2)
        # Let's stick to the *user's provided einsum* for now, assuming it's intended.
        hidden_states_complex = hidden_states.type(torch.complex128)
        return torch.einsum(
            "...ij,...jk,...ni->...nk",
            hidden_states_complex,
            dft_mat_hidden,
            dft_mat_seq
        ).real.type(torch.float32)

class FourierFFTLayerFlexible(nn.Module):

    def __init__(self, d_model, mode="real+complex"):
        super().__init__()

        self.mode = mode

        if mode == "real+complex":
            self.proj = nn.Linear(2 * d_model, d_model)
        else:
            self.proj = nn.Linear(d_model, d_model)

    @torch.amp.autocast("cuda", enabled=False)
    @torch.amp.autocast("cpu", enabled=False)
    def forward(self, hidden_states):

        fft1 = torch.fft.fft(hidden_states.float(), dim=-2)
        fft2 = torch.fft.fft(fft1, dim=-1)

        if self.mode == "real":
            features = fft2.real
        elif self.mode == "complex":
            features = fft2.imag
        elif self.mode == "real+complex":
            features = torch.cat([fft2.real, fft2.imag], dim=-1)
        else:
            raise ValueError(f"Unknown mode: {self.mode}")
        return self.proj(features).to(hidden_states.dtype)

    
# class FNetLayer(nn.Module):
#     def __init__(self, config: BartConfig):
#         super().__init__()

#         # Add fourier_implementation to BartConfig or default to 'fft'
#         fourier_impl = getattr(config, "fourier_implementation", "fft") # Default to 'fft'

#         if fourier_impl == 'matmul':
#             self.fft = FourierMMLayer(config)
#         elif fourier_impl == 'fft':
#             self.fft = FourierFFTLayer()
#         else:
#             raise ValueError(f"Unknown fourier implementation: {fourier_impl}")

#         self.mixing_layer_norm = nn.LayerNorm(config.d_model)

#         # Use standard Bart feed-forward dimensions
#         self.feed_forward = nn.Linear(config.d_model, config.encoder_ffn_dim)

#         self.output_dense = nn.Linear(config.encoder_ffn_dim, config.d_model) # applies the transformation from ``encoder_ffn_dim`` to ``d_model`` [same emb space -> can be mapped back to word ]

#         self.output_layer_norm = nn.LayerNorm(config.d_model)

#         # Use dropout rate from BartConfig
#         self.dropout = nn.Dropout(config.dropout)
#         # Use activation function specified in BartConfig (usually GELU)
#         # Making sure activation function is consistent with BART config

#         if isinstance(config.activation_function, str):
#              self.activation = nn.GELU() # Default if string doesn't map easily, BART uses GELU
#              if config.activation_function.lower() == "relu":
#                  self.activation = nn.ReLU()
#              elif config.activation_function.lower() == "silu" or config.activation_function.lower() == "swish":
#                  self.activation = nn.SiLU()
#              # Add other activations as needed
#         else:
#              # If config.activation_function is already an nn.Module instance (less common)
#              self.activation = config.activation_function


#     def forward(self, hidden_states):
#         """
#         Follows the exact architecture of FNet as described in the architecture folder !
#         """
#         # FNet uses residual connection *before* the first layer norm
#         fft_output = self.fft(hidden_states)
#         normed_fft_output = self.mixing_layer_norm(fft_output + hidden_states) # Residual connection + Norm

#         # Feed Forward part
#         intermediate_output = self.feed_forward(normed_fft_output)
#         intermediate_output = self.activation(intermediate_output)
#         output = self.output_dense(intermediate_output)
#         output = self.dropout(output) # Apply dropout

#         # Second residual connection and layer norm
#         output = self.output_layer_norm(output + normed_fft_output) # Residual connection + Norm
#         return output

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

class FNetImageClassifier(nn.Module):
    def __init__(self, input_dim, num_classes, d_model=512, hidden_dim=1024, n_layers=4, mode="real+complex"):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, d_model)
        self.layers = nn.ModuleList([FNetLayerFlexible(d_model, hidden_dim, mode=mode) for _ in range(n_layers)])
        self.classifier = nn.Linear(d_model, num_classes)

    def forward(self, x):
        # x: (batch, input_dim)
        x = self.input_proj(x)
        for layer in self.layers:
            x = layer(x)
        return self.classifier(x)

from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from transformers import Trainer, TrainingArguments
from sklearn.metrics import accuracy_score
import wandb


# --- Argument parser ---
parser = argparse.ArgumentParser()
parser.add_argument("--fourier_features", type=str, choices=["real", "complex", "real+complex"], default="real+complex")
parser.add_argument("--batch_size", type=int, default=64)
parser.add_argument("--epochs", type=int, default=10)
args = parser.parse_args()

wandb.init(project="cifar10-fnet", name=f"fnet-cifar10-{args.fourier_features}")

# --- CIFAR-10 Dataset ---
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Lambda(lambda x: x.view(-1))  # flatten to (3072,)
])
train_data = datasets.CIFAR10(root="./data", train=True, download=True, transform=transform)
test_data = datasets.CIFAR10(root="./data", train=False, download=True, transform=transform)
train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True, num_workers=2)
test_loader = DataLoader(test_data, batch_size=args.batch_size, shuffle=False, num_workers=2)

# --- Model ---
input_dim = 3 * 32 * 32
num_classes = 10
model = FNetImageClassifier(input_dim, num_classes, d_model=512, hidden_dim=1024, n_layers=4, mode=args.fourier_features)

# --- Training setup ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=2e-4)
criterion = nn.CrossEntropyLoss()

from tqdm import tqdm

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

    # --- Validation ---
    model.eval()
    total, correct, loss_sum = 0, 0, 0
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

    
torch.save(model.state_dict(), f"fnet_cifar10_{args.fourier_features}.pt")

