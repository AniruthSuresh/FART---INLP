import torch
from torch import nn
from typing import Optional, List, Tuple, Union
import math
from datasets import load_dataset

from transformers.modeling_outputs import BaseModelOutput
from transformers.modeling_outputs import BaseModelOutputWithPastAndCrossAttentions

from transformers.models.bart.modeling_bart import (
    BartPreTrainedModel,
    BartConfig,
    BartScaledWordEmbedding,
    BartLearnedPositionalEmbedding,
)
from transformers import BartConfig, BartTokenizer, BartForSequenceClassification, Trainer, TrainingArguments
import torch.utils.checkpoint
from sklearn.metrics import accuracy_score
import wandb

# Fourier Transform Implementations
class FourierFFTLayer(nn.Module):
    def forward(self, x):
        # Use FFT for the mixing operation
        return torch.fft.ifft(torch.fft.fft(x, dim=-1), dim=-1).real

class FourierMMLayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.d_model = config.d_model
        self.register_buffer("fourier_matrix", self._make_fourier_matrix(self.d_model))
        
        # Proper scaling for better training stability
        with torch.no_grad():
            self.fourier_matrix = self.fourier_matrix / math.sqrt(self.d_model)

    def _make_fourier_matrix(self, d):
        i = torch.arange(d).unsqueeze(0)
        j = torch.arange(d).unsqueeze(1)
        omega = torch.exp(-2j * math.pi * i * j / d)
        return omega

    def forward(self, x):
        x_freq = torch.matmul(x, self.fourier_matrix.real)  # Only using real part for simplicity
        return x_freq

class FNetLayer(nn.Module):
    def __init__(self, config: BartConfig):
        super().__init__()
        fourier_impl = getattr(config, "fourier_implementation", "fft")
        self.fft = FourierMMLayer(config) if fourier_impl == 'matmul' else FourierFFTLayer()

        # Layer for incorporating encoder information without attention
        self.encoder_projection = nn.Linear(config.d_model, config.d_model)
        self.encoder_mixing_layer_norm = nn.LayerNorm(config.d_model)
        
        self.mixing_layer_norm = nn.LayerNorm(config.d_model)
        self.feed_forward = nn.Linear(config.d_model, config.encoder_ffn_dim)
        self.output_dense = nn.Linear(config.encoder_ffn_dim, config.d_model)
        self.output_layer_norm = nn.LayerNorm(config.d_model)
        self.dropout = nn.Dropout(config.dropout)

        if isinstance(config.activation_function, str):
            act = config.activation_function.lower()
            if act == "relu":
                self.activation = nn.ReLU()
            elif act in {"silu", "swish"}:
                self.activation = nn.SiLU()
            else:
                self.activation = nn.GELU()
        else:
            self.activation = config.activation_function

    def forward(self, hidden_states, encoder_hidden_states=None, encoder_attention_mask=None):
        # Self-mixing with Fourier Transform - with proper residual connection
        residual = hidden_states
        fft_output = self.fft(hidden_states)
        normed_fft_output = self.mixing_layer_norm(fft_output + residual)
        
        # Incorporate encoder information without attention
        if encoder_hidden_states is not None:
            residual = normed_fft_output
            
            # Use encoder attention mask to properly mask padding tokens
            if encoder_attention_mask is not None:
                # Convert mask from [batch_size, seq_len] to [batch_size, seq_len, 1]
                mask_expanded = encoder_attention_mask.unsqueeze(-1).float()
                # Apply mask to encoder states
                masked_encoder = encoder_hidden_states * mask_expanded
                # Calculate sum and count of non-masked elements
                encoder_sum = torch.sum(masked_encoder, dim=1, keepdim=True)
                mask_sum = torch.sum(mask_expanded, dim=1, keepdim=True).clamp(min=1e-9)
                # Average considering only non-masked elements
                encoder_avg = encoder_sum / mask_sum
            else:
                # Simple average if no mask is provided
                encoder_avg = torch.mean(encoder_hidden_states, dim=1, keepdim=True)
                
            # Expand to match sequence length of decoder
            seq_len = hidden_states.size(1)
            encoder_proj = self.encoder_projection(encoder_avg)
            encoder_proj = encoder_proj.expand(-1, seq_len, -1)
            
            # Add encoder representation to current hidden state (instead of concatenating)
            hidden_states = normed_fft_output + encoder_proj
            hidden_states = self.encoder_mixing_layer_norm(hidden_states + residual)
        else:
            hidden_states = normed_fft_output

        # Feed forward network with residual connection
        residual = hidden_states
        intermediate_output = self.feed_forward(hidden_states)
        activated_output = self.activation(intermediate_output)
        projected_output = self.output_dense(activated_output)
        dropped_output = self.dropout(projected_output)
        output = self.output_layer_norm(dropped_output + residual)
        
        return output

class FNetDecoder(BartPreTrainedModel):
    def __init__(self, config: BartConfig, embed_tokens: Optional[nn.Embedding] = None):
        super().__init__(config)
        self.dropout = config.dropout
        self.padding_idx = config.pad_token_id
        self.max_target_positions = config.max_position_embeddings
        
        # Embedding setup with proper scaling
        embed_scale = math.sqrt(config.d_model) if config.scale_embedding else 1.0
        self.embed_scale = embed_scale
        
        if embed_tokens is not None:
            self.embed_tokens = embed_tokens
        else:
            self.embed_tokens = nn.Embedding(config.vocab_size, config.d_model, self.padding_idx)

        self.embed_positions = BartLearnedPositionalEmbedding(config.max_position_embeddings, config.d_model)
        self.layers = nn.ModuleList([FNetLayer(config) for _ in range(config.decoder_layers)])
        self.layernorm_embedding = nn.LayerNorm(config.d_model)
        
        # Add final layer norm for better stability
        self.final_layer_norm = nn.LayerNorm(config.d_model)
        
        self.gradient_checkpointing = False
        self.post_init()

    def get_input_embeddings(self):
        return self.embed_tokens

    def set_input_embeddings(self, value):
        self.embed_tokens = value

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        encoder_hidden_states: Optional[torch.FloatTensor] = None,
        encoder_attention_mask: Optional[torch.LongTensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        cross_attn_head_mask: Optional[torch.Tensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, BaseModelOutputWithPastAndCrossAttentions]:

        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both decoder_input_ids and decoder_inputs_embeds at the same time")
        elif input_ids is not None:
            input_shape = input_ids.shape
            input_ids = input_ids.view(-1, input_shape[-1])
        elif inputs_embeds is not None:
            input_shape = inputs_embeds.size()[:-1]
        else:
            raise ValueError("You have to specify either decoder_input_ids or decoder_inputs_embeds")

        # Embedding step
        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids) * self.embed_scale

        # Add positional embeddings
        positions = self.embed_positions(input_ids)
        hidden_states = inputs_embeds + positions
        hidden_states = self.layernorm_embedding(hidden_states)
        hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)

        # Apply attention mask if provided
        if attention_mask is not None:
            # Expand mask
            expanded_attn_mask = attention_mask.unsqueeze(-1).to(hidden_states.dtype)
            hidden_states = hidden_states * expanded_attn_mask

        all_hidden_states = () if output_hidden_states else None

        # Process through layers
        for idx, layer in enumerate(self.layers):
            if output_hidden_states:
                all_hidden_states += (hidden_states,)

            if self.gradient_checkpointing and self.training:
                layer_outputs = self._gradient_checkpointing_func(
                    layer.__call__,
                    hidden_states,
                    encoder_hidden_states,
                    encoder_attention_mask,
                )
            else:
                # Pass encoder outputs to each layer
                layer_outputs = layer(
                    hidden_states,
                    encoder_hidden_states=encoder_hidden_states,
                    encoder_attention_mask=encoder_attention_mask
                )
                
            hidden_states = layer_outputs

        # Apply final layer norm
        hidden_states = self.final_layer_norm(hidden_states)

        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        if not return_dict:
            return tuple(v for v in [hidden_states, all_hidden_states, None, None] if v is not None)

        return BaseModelOutputWithPastAndCrossAttentions(
            last_hidden_state=hidden_states,
            hidden_states=all_hidden_states,
            past_key_values=None,  # FNet doesn't use this
            attentions=None,       # No self-attention here
            cross_attentions=None  # Optional, also not used
        )

def init_fnet_properly(model):
    """Initialize FNet layers with appropriate scaling"""
    for name, module in model.named_modules():
        # Linear layers
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                module.bias.data.zero_()
        # LayerNorm
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

# Main training code
def main():
    # Initialize wandb
    wandb.init(project="sst2-bart-finetuning", name="bart-fnet-decoder")
    
    # Load model and tokenizer
    model = BartForSequenceClassification.from_pretrained("facebook/bart-base", num_labels=2)
    tokenizer = BartTokenizer.from_pretrained("facebook/bart-base")
    config = model.config
    
    # Replace the decoder with FNetDecoder
    model.model.decoder = FNetDecoder(config=config, embed_tokens=model.model.shared)
    
    # Properly initialize the FNet layers
    init_fnet_properly(model.model.decoder)
    
    # Load SST-2 dataset
    dataset = load_dataset("glue", "sst2")
    
    # Tokenization function
    def preprocess_function(examples):
        return tokenizer(
            examples["sentence"],
            truncation=True,
            padding="max_length",
            max_length=128
        )

    # Tokenize the dataset
    tokenized_datasets = dataset.map(preprocess_function, batched=True)
    tokenized_datasets = tokenized_datasets.rename_column("label", "labels")
    tokenized_datasets.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])

    # Split datasets
    train_dataset = tokenized_datasets["train"]
    val_dataset = tokenized_datasets["validation"]

    # Training arguments
    training_args = TrainingArguments(
        output_dir="./bart_sst2_decoder_results",
        eval_strategy="steps",
        eval_steps=200,
        learning_rate=3e-5,
        lr_scheduler_type="cosine",
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        num_train_epochs=5,
        weight_decay=0.01,
        logging_dir="./logs_bart_sst2",
        report_to="wandb",
        logging_steps=50,
        save_steps=200,
        save_strategy="no",  # <<<<<< THIS DISABLES AUTOMATIC SAVING
        load_best_model_at_end=False,  # <<<< MUST be False when not saving        metric_for_best_model="accuracy",
        warmup_ratio=0.1,
        gradient_accumulation_steps=2,  # Effective batch size of the original 32
        fp16=True,
    )

    # Accuracy metric
    def compute_metrics(p):
        if isinstance(p.predictions, tuple):
            preds = p.predictions[0]
        else:
            preds = p.predictions

        if isinstance(preds, torch.Tensor):
            preds = preds.detach().cpu().numpy()

        preds = preds.argmax(axis=1)
        return {"accuracy": accuracy_score(p.label_ids, preds)}

    # First train just the decoder with frozen encoder for stability
    print("Phase 1: Training with frozen encoder")
    for param in model.model.encoder.parameters():
        param.requires_grad = False
        
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics
    )
    trainer.train()
    
    # Now unfreeze encoder and fine-tune the whole model
    print("Phase 2: Fine-tuning the full model")
    for param in model.model.encoder.parameters():
        param.requires_grad = True
    
    # Update learning rate for fine-tuning
    training_args = TrainingArguments(
        output_dir="./bart_sst2_decoder_results_phase2",
        eval_strategy="steps",
        eval_steps=200,
        learning_rate=1e-5,  # Lower learning rate for fine-tuning
        lr_scheduler_type="cosine",
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        num_train_epochs=3,
        weight_decay=0.01,
        logging_dir="./logs_bart_sst2_phase2",
        report_to="wandb",
        logging_steps=50,
        save_strategy="no",
        save_steps=200,
        load_best_model_at_end=False,
        metric_for_best_model="accuracy",
        warmup_ratio=0.05,
        gradient_accumulation_steps=2,
        fp16=True,
    )
    
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics
    )
    trainer.train()

    # Save the final model
    trainer.save_model("./sst2_fnet_decoder_bart_final")
    
    # Evaluate on validation set
    results = trainer.evaluate()
    print("Final evaluation results:", results)
    
    # Close wandb run
    wandb.finish()

if __name__ == "__main__":
    main()
    