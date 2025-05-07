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

logger = logging.get_logger(__name__)

class FourierMMLayer(nn.Module):
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


class FourierFFTLayer(nn.Module):
    def __init__(self):
        super().__init__()

    # Disable AMP for FFT ops if needed, following original code
    @torch.amp.autocast("cuda", enabled=False) # Specify device type if using autocast
    @torch.amp.autocast("cpu", enabled=False)
    def forward(self, hidden_states):
        # FNet applies FFT along sequence dimension, then hidden dimension
        # Ensure float32 for FFT operations as done in original code
        fft1 = torch.fft.fft(hidden_states.float(), dim=-2) # FFT along sequence length
        fft2 = torch.fft.fft(fft1, dim=-1)              # FFT along hidden dimension
        return fft2.real.to(hidden_states.dtype) # Return to original dtype


class FNetLayer(nn.Module):
    def __init__(self, config: BartConfig):
        super().__init__()
        # Add fourier_implementation to BartConfig or default to 'fft'
        fourier_impl = getattr(config, "fourier_implementation", "fft") # Default to 'fft'

        if fourier_impl == 'matmul':
            self.fft = FourierMMLayer(config)
        elif fourier_impl == 'fft':
            self.fft = FourierFFTLayer()
        else:
            raise ValueError(f"Unknown fourier implementation: {fourier_impl}")

        self.mixing_layer_norm = nn.LayerNorm(config.d_model)
        # Use standard Bart feed-forward dimensions
        self.feed_forward = nn.Linear(config.d_model, config.encoder_ffn_dim)
        self.output_dense = nn.Linear(config.encoder_ffn_dim, config.d_model)
        self.output_layer_norm = nn.LayerNorm(config.d_model)
        # Use dropout rate from BartConfig
        self.dropout = nn.Dropout(config.dropout)
        # Use activation function specified in BartConfig (usually GELU)
        # Making sure activation function is consistent with BART config
        if isinstance(config.activation_function, str):
             self.activation = nn.GELU() # Default if string doesn't map easily, BART uses GELU
             if config.activation_function.lower() == "relu":
                 self.activation = nn.ReLU()
             elif config.activation_function.lower() == "silu" or config.activation_function.lower() == "swish":
                 self.activation = nn.SiLU()
             # Add other activations as needed
        else:
             # If config.activation_function is already an nn.Module instance (less common)
             self.activation = config.activation_function


    def forward(self, hidden_states):
        # FNet uses residual connection *before* the first layer norm
        fft_output = self.fft(hidden_states)
        normed_fft_output = self.mixing_layer_norm(fft_output + hidden_states) # Residual connection + Norm

        # Feed Forward part
        intermediate_output = self.feed_forward(normed_fft_output)
        intermediate_output = self.activation(intermediate_output)
        output = self.output_dense(intermediate_output)
        output = self.dropout(output) # Apply dropout

        # Second residual connection and layer norm
        output = self.output_layer_norm(output + normed_fft_output) # Residual connection + Norm
        return output

class FNetEncoder(BartPreTrainedModel): # Inherit from BartPreTrainedModel
    """
    FNet Encoder adapted for BART, replacing self-attention with Fourier Transforms.

    Args:
        config: BartConfig
        embed_tokens (nn.Embedding): output embedding (optional)
    """

    def __init__(self, config: BartConfig, embed_tokens: Optional[nn.Embedding] = None):
        super().__init__(config) # Call parent initializer

        # --- Copy necessary attributes and setup from BartEncoder ---
        self.dropout = config.dropout
        self.layerdrop = config.encoder_layerdrop # Keep attribute even if not used in FNet loop

        embed_dim = config.d_model
        self.padding_idx = config.pad_token_id
        self.max_source_positions = config.max_position_embeddings
        # Match BART's embedding scaling
        embed_scale = math.sqrt(embed_dim) if config.scale_embedding else 1.0

        # Use BART's standard embedding layers
        if embed_tokens is not None:
             self.embed_tokens = embed_tokens
        else:
             self.embed_tokens = BartScaledWordEmbedding(
                config.vocab_size, embed_dim, self.padding_idx, embed_scale=embed_scale
             )

        self.embed_positions = BartLearnedPositionalEmbedding(
            config.max_position_embeddings,
            embed_dim,
        )
        # Initial layer norm after embeddings, as in BART
        self.layernorm_embedding = nn.LayerNorm(embed_dim)

        # --- Replace BartEncoderLayers with FNetLayers ---
        # Use config.encoder_layers to determine the number of layers
        self.layers = nn.ModuleList([FNetLayer(config) for _ in range(config.encoder_layers)])

        # --- Other BartEncoder attributes (some might not be used by FNet) ---
        self._use_flash_attention_2 = False # FNet doesn't use attention
        self._use_sdpa = False             # FNet doesn't use attention
        self.gradient_checkpointing = False # Initialize gradient checkpointing flag

        # Initialize weights and apply final processing (from BartPreTrainedModel)
        self.post_init()

    def get_input_embeddings(self):
        return self.embed_tokens

    def set_input_embeddings(self, value):
        self.embed_tokens = value

    # --- Adapt the forward method ---
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None, # Keep in signature for compatibility, but FNetLayer won't use it
        head_mask: Optional[torch.Tensor] = None,      # Keep in signature, but FNetLayer won't use it
        inputs_embeds: Optional[torch.FloatTensor] = None,
        output_attentions: Optional[bool] = None,      # FNet does not produce attentions
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, BaseModelOutput]:

        # --- Input Validation and Embedding --- (Copied and adapted from BartEncoder)
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        if output_attentions:
              logger.warning("`output_attentions=True` is not supported for FNetEncoder, returning `None` for attentions.")
              output_attentions = False # Force false as FNet has no attentions

        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # Retrieve input_ids and inputs_embeds
        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
        elif input_ids is not None:
            self.warn_if_padding_and_no_attention_mask(input_ids, attention_mask)
            input_shape = input_ids.size()
            input_ids = input_ids.view(-1, input_shape[-1])
        elif inputs_embeds is not None:
            input_shape = inputs_embeds.size()[:-1]
        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds")

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        # Get position embeddings
        if self.embed_positions is not None:
             # Original BART uses input_ids shape even if inputs_embeds provided
             # Let's stick to input_shape for consistency
             embed_pos = self.embed_positions(input_ids) # Use derived input_shape
             embed_pos = embed_pos.to(inputs_embeds.device)
        else:
             embed_pos = 0 # Or handle differently if positional embeddings are mandatory

        hidden_states = inputs_embeds + embed_pos
        hidden_states = self.layernorm_embedding(hidden_states)
        hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)

        # --- FNet Layer Processing ---
        encoder_states = () if output_hidden_states else None
        all_attentions = None # FNet does not have attentions

        # head_mask is not used by FNet layers, but we keep the check for API consistency
        if head_mask is not None:
             if head_mask.size()[0] != (len(self.layers)):
                 raise ValueError(
                     f"The head_mask should be specified for {len(self.layers)} layers, but it is for "
                     f"{head_mask.size()[0]}."
                 )

        for idx, layer_module in enumerate(self.layers):
            if output_hidden_states:
                encoder_states = encoder_states + (hidden_states,)

            # --- LayerDrop --- (Optional: Implement if desired for FNet)
            # You could add LayerDrop here similar to BartEncoder if needed,
            # although it wasn't part of the original FNet paper AFAIK.
            # For now, we skip LayerDrop for simplicity.
            # to_drop = False
            # if self.training:
            #     dropout_probability = torch.rand([])
            #     if dropout_probability < self.layerdrop:
            #         to_drop = True
            # if to_drop:
            #     continue # Skip layer

            # --- Gradient Checkpointing --- (Optional: Implement if desired)
            if self.gradient_checkpointing and self.training:
                 # Define a function for checkpointing
                 def create_custom_forward(module):
                     def custom_forward(*inputs):
                         return module(*inputs)
                     return custom_forward

                 # Apply checkpointing to the FNetLayer's forward pass
                 hidden_states = torch.utils.checkpoint.checkpoint(
                     create_custom_forward(layer_module),
                     hidden_states,
                     use_reentrant=False # Recommended for newer PyTorch versions
                 )
            else:
                # Normal forward pass
                hidden_states = layer_module(hidden_states)


        # Add final hidden state if output_hidden_states is True
        if output_hidden_states:
            encoder_states = encoder_states + (hidden_states,)

        # --- Return Value --- (Matching BaseModelOutput structure)
        if not return_dict:
             # Return tuple (last_hidden_state, all_hidden_states, all_attentions)
             # Filter None values as BartEncoder does
            return tuple(v for v in [hidden_states, encoder_states, all_attentions] if v is not None)
        else:
             return BaseModelOutput(
                last_hidden_state=hidden_states,
                hidden_states=encoder_states,
                attentions=all_attentions, # Will be None
            )
        
# config = model.config
# if not hasattr(config, 'fourier_implementation'):
#     print("Setting config.fourier_implementation = 'fft' (default)")
#     config.fourier_implementation = 'fft'        

# shared_embeddings = model.get_input_embeddings()
# fnet_encoder = FNetEncoder(config=config, embed_tokens=shared_embeddings)
# model.model.encoder = fnet_encoder
# print(f"Type of model.encoder: {type(model.model.encoder)}")   


# from datasets import load_dataset
# from transformers import BartTokenizer, BartForSequenceClassification, Trainer, TrainingArguments
# from sklearn.metrics import accuracy_score
# import torch
# import wandb

# wandb.init(project="sst2-bart-fnet-finetuning", name="bart-base-sst2")

# # Load SST-2 dataset
# dataset = load_dataset("glue", "sst2")
# # Tokenization function
# def preprocess_function(examples):
#     return tokenizer(
#         examples["sentence"],
#         truncation=True,
#         padding="max_length",
#         max_length=128
#     )

# # Tokenize the dataset
# tokenized_datasets = dataset.map(preprocess_function, batched=True)

# # Rename label column
# tokenized_datasets = tokenized_datasets.rename_column("label", "labels")

# # Format for PyTorch
# tokenized_datasets.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])

# # Split datasets
# train_dataset = tokenized_datasets["train"]
# val_dataset = tokenized_datasets["validation"]

# # Training arguments with W&B logging
# training_args = TrainingArguments(
#     output_dir="./bart_sst2_results",
#     evaluation_strategy="epoch",
#     learning_rate=2e-5,
#     per_device_train_batch_size=8,
#     per_device_eval_batch_size=8,
#     num_train_epochs=3,
#     weight_decay=0.01,
#     logging_dir="./logs_bart_sst2",
#     report_to="wandb",  # Enable wandb logging
#     logging_steps=50,
#     save_strategy="epoch",
#     load_best_model_at_end=True,
#     metric_for_best_model="accuracy"
# )

# # Accuracy metric
# def compute_metrics(p):
#     if isinstance(p.predictions, tuple):  # if predictions come with loss, etc.
#         preds = p.predictions[0]
#     else:
#         preds = p.predictions

#     if isinstance(preds, torch.Tensor):
#         preds = preds.detach().cpu().numpy()

#     preds = preds.argmax(axis=1)
#     return {"accuracy": accuracy_score(p.label_ids, preds)}


# # Trainer
# trainer = Trainer(
#     model=model,
#     args=training_args,
#     train_dataset=train_dataset,
#     eval_dataset=val_dataset,
#     compute_metrics=compute_metrics
# )

# # Train the model
# trainer.train()

# # Save the model
# trainer.save_model("./sst2_fnet_bart")

# # Evaluate on validation set
# results = trainer.evaluate()
# print(results)
