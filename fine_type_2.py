from datasets import load_dataset
from transformers import BertTokenizer, BertForMultipleChoice, Trainer, TrainingArguments
import torch
from sklearn.metrics import accuracy_score

# Load the SWAG dataset
dataset = load_dataset("swag", split={'train': 'train', 'validation': 'validation'})

# Load BERT tokenizer and model
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertForMultipleChoice.from_pretrained('bert-base-uncased')

def preprocess_function(examples):
    """
    Tokenizes SWAG dataset examples for multiple-choice classification.
    """
    num_choices = 4  # SWAG has 4 choices per example
    first_sentences = []
    second_sentences = []
    
    for i in range(len(examples["sent1"])):
        context = examples["sent1"][i] + " " + examples["sent2"][i]
        choices = [examples[f"ending{j}"][i] for j in range(num_choices)]
        
        first_sentences.append([context] * num_choices)
        second_sentences.append(choices)

    # Flatten the inputs for tokenization
    first_sentences = sum(first_sentences, [])
    second_sentences = sum(second_sentences, [])

    tokenized_examples = tokenizer(
        first_sentences,
        second_sentences,
        truncation=True,
        padding="max_length",
        max_length=128,
        return_tensors="pt"
    )

    # Reshape back into (batch_size, num_choices, sequence_length)
    input_ids = tokenized_examples["input_ids"].view(-1, num_choices, 128)
    attention_mask = tokenized_examples["attention_mask"].view(-1, num_choices, 128)
    token_type_ids = tokenized_examples["token_type_ids"].view(-1, num_choices, 128)

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "token_type_ids": token_type_ids,
        "labels": torch.tensor(examples["label"])
    }

# Apply tokenization to the dataset
tokenized_datasets = dataset.map(preprocess_function, batched=True)

# Prepare train and validation datasets
train_dataset = tokenized_datasets["train"]
val_dataset = tokenized_datasets["validation"]

# Set up training arguments
training_args = TrainingArguments(
    output_dir='./results',
    evaluation_strategy="epoch",  # Replace with `eval_strategy="epoch"` for future versions
    save_strategy="no",  # Disable checkpoint saving
    learning_rate=2e-5,
    per_device_train_batch_size=8,
    num_train_epochs=3,
    logging_dir='./logs',
    # Add these to limit disk usage:
    save_total_limit=1  # Keep only the latest checkpoint
)

# Define metric computation
def compute_metrics(p):
    preds = torch.argmax(torch.tensor(p.predictions), dim=-1).numpy()
    labels = p.label_ids
    return {"accuracy": accuracy_score(labels, preds)}

# Set up the Trainer
trainer = Trainer(  
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    compute_metrics=compute_metrics
)

# Train the model
trainer.train()

# Save the model after training
trainer.save_model("./swag_finetuned_bert")

# Evaluate the model
results = trainer.evaluate()
print(results)
