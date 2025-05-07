from datasets import load_dataset
from transformers import BartTokenizer, BartForSequenceClassification, Trainer, TrainingArguments
from sklearn.metrics import accuracy_score
import torch
import wandb

# Initialize Weights & Biases

# Load BART tokenizer and model
model_name = "facebook/bart-base"
tokenizer = BartTokenizer.from_pretrained(model_name)
model = BartForSequenceClassification.from_pretrained(model_name, num_labels=2)

wandb.init(project="sst2-bart-finetuning", name="bart-base-sst2")

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

# Rename label column
tokenized_datasets = tokenized_datasets.rename_column("label", "labels")

# Format for PyTorch
tokenized_datasets.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])

# Split datasets
train_dataset = tokenized_datasets["train"]
val_dataset = tokenized_datasets["validation"]

# Training arguments with W&B logging
training_args = TrainingArguments(
    output_dir="./bart_sst2_results",
    evaluation_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=3,
    weight_decay=0.01,
    logging_dir="./logs_bart_sst2",
    report_to="wandb",  # Enable wandb logging
    logging_steps=50,
    save_strategy="epoch",
    load_best_model_at_end=True,
    metric_for_best_model="accuracy"
)

# Accuracy metric
def compute_metrics(p):
    if isinstance(p.predictions, tuple):  # if predictions come with loss, etc.
        preds = p.predictions[0]
    else:
        preds = p.predictions

    if isinstance(preds, torch.Tensor):
        preds = preds.detach().cpu().numpy()

    preds = preds.argmax(axis=1)
    return {"accuracy": accuracy_score(p.label_ids, preds)}


# Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    compute_metrics=compute_metrics
)

# Train the model
trainer.train()

# Save the model
trainer.save_model("./sst2_finetuned_bart")

# Evaluate on validation set
results = trainer.evaluate()
print(results)
