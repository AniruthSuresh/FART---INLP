import math
import torch
import torch.nn as nn
import torch.utils.checkpoint
from typing import Optional, Tuple, Union, List

# Assume spectre.py is in the same directory or PYTHONPATH
from spectre import (
    SPECTREBlock,
    SPECTRELayer, # Though not directly instantiated, SPECTREBlock uses it
    PrefixFFTCache # For type hinting, though not used in non-incremental encoder pass
)

from transformers import (
    BartTokenizer,
    BartForSequenceClassification,
    BartConfig,
    Trainer,
    TrainingArguments
)
from transformers.modeling_outputs import BaseModelOutput
from transformers.models.bart.modeling_bart import (
    BartPreTrainedModel,
    BartScaledWordEmbedding,
    BartLearnedPositionalEmbedding,
)
from transformers.utils import logging as hf_logging # Renamed to avoid conflict

from datasets import load_dataset
from sklearn.metrics import accuracy_score
import wandb

logger = hf_logging.get_logger(__name__)

# --- Weight Initialization Utility (adapted from SpectreLM) ---
def _init_weights_spectre(module: nn.Module) -> None:
    """Initialization that respects module type for SPECTRE components."""
    if isinstance(module, nn.Linear):
        nn.init.trunc_normal_(module.weight, std=0.02)
        if module.bias is not None:
            nn.init.zeros_(module.bias)
    elif isinstance(module, nn.LayerNorm):
        nn.init.ones_(module.weight)
        nn.init.zeros_(module.bias)
    elif isinstance(module, nn.Embedding): # Though embeddings are likely handled by BART's init
        nn.init.trunc_normal_(module.weight, std=0.02)
        if module.padding_idx is not None:
            module.weight.data[module.padding_idx].zero_()


class SpectreEncoder(BartPreTrainedModel):
    """
    BART Encoder with SPECTREBlocks replacing Self-Attention layers.
    """
    def __init__(self, config: BartConfig, embed_tokens: Optional[nn.Embedding] = None):
        super().__init__(config)

        self.dropout = config.dropout
        self.layerdrop = config.encoder_layerdrop # Kept for compatibility, not used in SPECTRE loop

        embed_dim = config.d_model
        self.padding_idx = config.pad_token_id
        self.max_source_positions = config.max_position_embeddings
        embed_scale = math.sqrt(embed_dim) if config.scale_embedding else 1.0

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
        self.layernorm_embedding = nn.LayerNorm(embed_dim)

        # SPECTRE specific config (can be added to BartConfig externally or defaulted here)
        self.memory_len_spectre = getattr(config, "memory_len_spectre", 0)
        self.share_gates_spectre = getattr(config, "share_gates_spectre", True)
        self.use_wavelet_spectre = getattr(config, "use_wavelet_spectre", False)
        # Defaults for other SPECTRELayer params if not specified, matching SPECTRELayer defaults
        self.low_rank_spectre = getattr(config, "low_rank_spectre", None)
        self.toeplitz_bandwidth_spectre = getattr(config, "toeplitz_bandwidth_spectre", 0)
        self.use_modrelu_spectre = getattr(config, "use_modrelu_spectre", True)


        self.layers = nn.ModuleList(
            [
                SPECTREBlock(
                    d_model=config.d_model,
                    n_heads=config.encoder_attention_heads, # SPECTREBlock uses n_heads for its internal SPECTRELayer
                    ffn_hidden=(config.encoder_ffn_dim // config.d_model),
                    max_seq_len=config.max_position_embeddings, # For SPECTRELayer
                    memory_len=self.memory_len_spectre,
                    # Pass other SPECTRELayer specific kwargs
                    share_gates=self.share_gates_spectre,
                    use_wavelet=self.use_wavelet_spectre,
                    low_rank=self.low_rank_spectre,
                    toeplitz_bandwidth=self.toeplitz_bandwidth_spectre,
                    use_modrelu=self.use_modrelu_spectre,
                )
                for _ in range(config.encoder_layers)
            ]
        )

        self._use_flash_attention_2 = False # SPECTRE does not use attention
        self._use_sdpa = False             # SPECTRE does not use attention
        self.gradient_checkpointing = False # Initialize

        # Initialize weights and apply final processing
        self.post_init() # Standard Hugging Face post-initialization
        
        self.layers.apply(_init_weights_spectre)


    def get_input_embeddings(self):
        return self.embed_tokens

    def set_input_embeddings(self, value):
        self.embed_tokens = value

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None, # Kept for API, SPECTRE may not use it directly
        head_mask: Optional[torch.Tensor] = None,       # Kept for API, SPECTRE does not use
        inputs_embeds: Optional[torch.FloatTensor] = None,
        output_attentions: Optional[bool] = None,       # SPECTRE does not produce attentions
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, BaseModelOutput]:

        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        if output_attentions:
            logger.warning("`output_attentions=True` is not supported for SpectreEncoder, returning `None` for attentions.")
            output_attentions = False # Force false

        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
        elif input_ids is not None:
            self.warn_if_padding_and_no_attention_mask(input_ids, attention_mask)
            input_shape = input_ids.size()
            # input_ids = input_ids.view(-1, input_shape[-1]) # This line is in BART but might be problematic if we need original shape for positions
        elif inputs_embeds is not None:
            input_shape = inputs_embeds.size()[:-1]
        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds")

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        # BART calculates position_ids from input_ids, let's ensure this logic for embed_positions
        if input_ids is not None: # Derive position_ids for BartLearnedPositionalEmbedding
            # position_ids = BartLearnedPositionalEmbedding.create_position_ids_from_input_ids(
            #     input_ids, self.padding_idx, 0 # past_key_values_length=0
            # )
            embed_pos = self.embed_positions(input_ids=input_ids, past_key_values_length=0)
        elif inputs_embeds is not None:
            # Or it can take inputs_embeds to infer shape and create position_ids
            embed_pos = self.embed_positions(inputs_embeds=inputs_embeds, past_key_values_length=past_key_values_length)
        else:
            # This case should be prevented by the initial checks on input_ids and inputs_embeds
            raise ValueError("Cannot determine positional embeddings without input_ids or inputs_embeds.")


        # embed_pos = self.embed_positions(position_ids) # Use derived input_shape for positions
        embed_pos = embed_pos.to(inputs_embeds.device)


        hidden_states = inputs_embeds + embed_pos
        hidden_states = self.layernorm_embedding(hidden_states)
        hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)

        encoder_states = () if output_hidden_states else None
        all_attentions = None # SPECTRE does not produce attentions

        # head_mask is not used by SPECTRE layers
        if head_mask is not None:
            if head_mask.size()[0] != (len(self.layers)):
                raise ValueError(
                    f"The head_mask should be specified for {len(self.layers)} layers, but it is for "
                    f"{head_mask.size()[0]}."
                )

        for idx, layer_module in enumerate(self.layers):
            if output_hidden_states:
                encoder_states = encoder_states + (hidden_states,)

            # LayerDrop (skipped for simplicity, as in FNet example)

            if self.gradient_checkpointing and self.training:
                def create_custom_forward(module):
                    def custom_forward(*inputs):
                        # SPECTREBlock.forward takes (x, cache=None, incremental_state=False)
                        # For encoder, cache is None, incremental_state is False
                        output, _ = module(*inputs, cache=None, incremental_state=False)
                        return output
                    return custom_forward

                hidden_states = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(layer_module),
                    hidden_states,
                    use_reentrant=False, # Recommended for newer PyTorch versions
                )
            else:
                # For standard encoder (non-incremental), cache is None, incremental_state is False
                hidden_states, _ = layer_module(hidden_states, cache=None, incremental_state=False)


        if output_hidden_states:
            encoder_states = encoder_states + (hidden_states,)

        if not return_dict:
            return tuple(v for v in [hidden_states, encoder_states, all_attentions] if v is not None)
        
        return BaseModelOutput(
            last_hidden_state=hidden_states,
            hidden_states=encoder_states,
            attentions=all_attentions, # Will be None
        )


def main():
    # --- Configuration ---
    MODEL_NAME = "facebook/bart-base"
    DATASET_NAME = "glue"
    DATASET_CONFIG = "sst2"
    MAX_LENGTH = 128
    TRAIN_BATCH_SIZE = 8 #16
    EVAL_BATCH_SIZE = 8 #16
    LEARNING_RATE = 2e-5
    NUM_EPOCHS = 1 # 3
    WANDB_PROJECT = "sst2-bart-fftnet-finetuning"
    WANDB_RUN_NAME = f"bart-fftnet-{DATASET_CONFIG}-epochs{NUM_EPOCHS}"
    OUTPUT_DIR = f"./{WANDB_RUN_NAME}_results"
    LOGGING_DIR = f"./{WANDB_RUN_NAME}_logs"

    # --- Initialize W&B ---
    wandb.init(project=WANDB_PROJECT, name=WANDB_RUN_NAME, config={
        "model_name": MODEL_NAME,
        "dataset": f"{DATASET_NAME}:{DATASET_CONFIG}",
        "max_length": MAX_LENGTH,
        "learning_rate": LEARNING_RATE,
        "epochs": NUM_EPOCHS,
        "train_batch_size": TRAIN_BATCH_SIZE,
        "eval_batch_size": EVAL_BATCH_SIZE,
    })

    # --- Load Tokenizer and Model ---
    tokenizer = BartTokenizer.from_pretrained(MODEL_NAME)
    model = BartForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=2) # SST-2 has 2 labels

    # --- Configure and Replace Encoder ---
    config = model.config
    # Add SPECTRE specific configurations to config if you want to override defaults
    # For example:
    # config.memory_len_spectre = 8 
    # config.use_wavelet_spectre = True

    shared_embeddings = model.model.get_input_embeddings()
    spectre_encoder = SpectreEncoder(config=config, embed_tokens=shared_embeddings)
    model.model.encoder = spectre_encoder
    
    print(f"Successfully replaced encoder. Type of model.model.encoder: {type(model.model.encoder)}")
    print(f"Number of layers in SpectreEncoder: {len(model.model.encoder.layers)}")
    if len(model.model.encoder.layers) > 0:
        print(f"Type of first layer in SpectreEncoder: {type(model.model.encoder.layers[0])}")

    # --- Load and Preprocess Dataset ---
    dataset = load_dataset(DATASET_NAME, DATASET_CONFIG)

    def preprocess_function(examples):
        return tokenizer(
            examples["sentence"],
            truncation=True,
            padding="max_length",
            max_length=MAX_LENGTH
        )

    tokenized_datasets = dataset.map(preprocess_function, batched=True)
    tokenized_datasets = tokenized_datasets.rename_column("label", "labels")
    tokenized_datasets.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])

    train_dataset = tokenized_datasets["train"]
    val_dataset = tokenized_datasets["validation"]

    # --- Training Arguments ---
    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        eval_strategy="epoch",
        learning_rate=LEARNING_RATE,
        per_device_train_batch_size=TRAIN_BATCH_SIZE,
        per_device_eval_batch_size=EVAL_BATCH_SIZE,
        num_train_epochs=NUM_EPOCHS,
        weight_decay=0.01,
        logging_dir=LOGGING_DIR,
        report_to="wandb",
        logging_steps=50,
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="accuracy",
        # fp16=torch.cuda.is_available(), # Enable if using GPU and want mixed precision
    )

    # --- Metrics ---
    def compute_metrics(p):
        if isinstance(p.predictions, tuple):
            preds = p.predictions[0]
        else:
            preds = p.predictions

        if isinstance(preds, torch.Tensor):
            preds = preds.detach().cpu().numpy()

        preds = preds.argmax(axis=1)
        return {"accuracy": accuracy_score(p.label_ids, preds)}

    # --- Trainer ---
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics,
        tokenizer=tokenizer, # Good practice to pass tokenizer to Trainer
    )

    # --- Train ---
    print("Starting training...")
    trainer.train()

    # --- Save and Evaluate ---
    trainer.save_model(f"./{WANDB_RUN_NAME}_final_model")
    print(f"Model saved to ./{WANDB_RUN_NAME}_final_model")

    results = trainer.evaluate()
    print("Evaluation results:")
    print(results)
    wandb.log({"final_eval_accuracy": results.get("eval_accuracy", 0),
               "final_eval_loss": results.get("eval_loss", 0)})

    wandb.finish()

if __name__ == "__main__":

    main()