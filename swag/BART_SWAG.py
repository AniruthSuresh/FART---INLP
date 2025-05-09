from transformers import BartTokenizer, BartModel
from transformers import BartForSequenceClassification
import re
import torch
import torch.utils.checkpoint
from scipy import linalg
from torch import nn

model = BartForSequenceClassification.from_pretrained("facebook/bart-base", num_labels=2)
tokenizer = BartTokenizer.from_pretrained("facebook/bart-base")


from transformers import BartTokenizer, BartForSequenceClassification, TrainingArguments, Trainer
from datasets import load_dataset, Dataset
import torch
import random
import accelerate
# Load SWAG dataset
raw_dataset = load_dataset("swag", "regular")

# Load tokenizer and model
model_name = "facebook/bart-base"
tokenizer = BartTokenizer.from_pretrained(model_name)
# model = BartForSequenceClassification.from_pretrained(model_name, num_labels=2)

def flatten_swag(example_batch):
    texts = []
    labels = []

    for sent1, sent2, label, endings in zip(
        example_batch["sent1"],
        example_batch["sent2"],
        example_batch["label"],
        zip(example_batch["ending0"], example_batch["ending1"], example_batch["ending2"], example_batch["ending3"])
    ):
        for i in range(4):
            input_text = f"{sent1} {sent2} {endings[i]}"
            is_correct = 1 if i == int(label) else 0
            texts.append(input_text)
            labels.append(is_correct)

    return {"text": texts, "label": labels}


train_flat = raw_dataset["train"].map(flatten_swag, batched=True, remove_columns=raw_dataset["train"].column_names)
val_flat = raw_dataset["validation"].map(flatten_swag, batched=True, remove_columns=raw_dataset["validation"].column_names)

# Tokenize
def tokenize(ex):
    encoded = tokenizer(ex["text"], truncation=True, padding="max_length", max_length=128)
    encoded["label"] = ex["label"]
    return encoded

train_dataset = Dataset.from_list(train_flat).map(tokenize)
val_dataset = Dataset.from_list(val_flat).map(tokenize)

# Set format for PyTorch
train_dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "label"])
val_dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "label"])

# TrainingArguments
training_args = TrainingArguments(
    output_dir="./bart-swag",
    eval_strategy="epoch",
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    learning_rate=2e-5,
    num_train_epochs=3,
    weight_decay=0.01,
    logging_steps=100,
    save_strategy="epoch",
    load_best_model_at_end=True,
    report_to="wandb",  
    run_name="bart-swag",  
)

import wandb
wandb.init(project="bart-swag", name="bart-swag") 
# Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
)

# Train
trainer.train()
