from datasets import load_dataset
from transformers import BertTokenizer, BertForMultipleChoice
import torch
from sklearn.metrics import accuracy_score, f1_score
from tqdm import tqdm

# Load the SWAG test dataset (only first 100 examples)
dataset = load_dataset("swag", split="validation[:100]")  # Using first 100 as test set

def load_model(model_path="./swag_finetuned_bert"):
    """Loads the fine-tuned BERT model and tokenizer."""
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    model = BertForMultipleChoice.from_pretrained(model_path)
    model.eval()  # Set model to evaluation mode
    return tokenizer, model

def preprocess_function(examples, tokenizer):
    """Tokenizes SWAG dataset examples for multiple-choice classification."""
    num_choices = 4  # SWAG has 4 choices per example
    first_sentences = []
    second_sentences = []
    labels = []
    
    for i in range(len(examples["sent1"])):
        context = examples["sent1"][i] + " " + examples["sent2"][i]
        choices = [examples[f"ending{j}"][i] for j in range(num_choices)]
        first_sentences.append([context] * num_choices)
        second_sentences.append(choices)
        labels.append(examples["label"][i])
    
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
    
    input_ids = tokenized_examples["input_ids"].view(-1, num_choices, 128)
    attention_mask = tokenized_examples["attention_mask"].view(-1, num_choices, 128)
    token_type_ids = tokenized_examples["token_type_ids"].view(-1, num_choices, 128)
    
    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "token_type_ids": token_type_ids,
        "labels": torch.tensor(labels)
    }

def evaluate_model(model, tokenizer, dataset):
    """Evaluates the model on the test dataset and computes accuracy and F1 score."""
    processed_data = preprocess_function(dataset, tokenizer)
    preds = []
    labels = processed_data["labels"].numpy()
    
    with torch.no_grad():
        for i in tqdm(range(len(labels)), desc="Evaluating"):
            output = model(
                processed_data["input_ids"][i].unsqueeze(0),
                attention_mask=processed_data["attention_mask"][i].unsqueeze(0),
                token_type_ids=processed_data["token_type_ids"][i].unsqueeze(0)
            )
            preds.append(torch.argmax(output.logits, dim=-1).item())
    
    accuracy = accuracy_score(labels, preds)
    f1 = f1_score(labels, preds, average='weighted')
    
    print(f"Accuracy: {accuracy:.4f}")
    print(f"F1 Score: {f1:.4f}")

def predict(model, tokenizer, context, choices):
    """Predicts the best choice for a given context and choices."""
    num_choices = len(choices)
    tokenized_input = tokenizer(
        [context] * num_choices, choices,
        truncation=True, padding="max_length", max_length=128,
        return_tensors="pt"
    )
    input_ids = tokenized_input["input_ids"].unsqueeze(0)
    attention_mask = tokenized_input["attention_mask"].unsqueeze(0)
    token_type_ids = tokenized_input["token_type_ids"].unsqueeze(0)
    
    with torch.no_grad():
        outputs = model(input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
        logits = outputs.logits
        predicted_choice = torch.argmax(logits, dim=-1).item()
    
    return predicted_choice

# Load the fine-tuned model
tokenizer, model = load_model()

# Evaluate on test dataset
evaluate_model(model, tokenizer, dataset)

# Example prediction
context = "The weather was getting colder and the leaves were falling from the trees."
choices = [
    "She decided to wear a light summer dress.",
    "He put on a heavy winter coat.",
    "They went to the beach to enjoy the sun.",
    "The sun was shining brightly in the sky."
]

predicted_index = predict(model, tokenizer, context, choices)
print(f"Predicted choice: {choices[predicted_index]}")

