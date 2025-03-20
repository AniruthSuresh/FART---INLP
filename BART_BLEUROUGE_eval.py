"""
Calculates BLEU, ROUGE, BLEURT Scores
"""
# ! pip install datasets
# ! pip install rouge_score
import torch
from torch.utils.data import DataLoader
from transformers import BartTokenizer, BartForConditionalGeneration
from datasets import load_metric
import nltk
from bleurt import score
from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import Dataset, DataLoader, Subset
from transformers import BartTokenizer, BartForConditionalGeneration, AdamW
from sklearn.model_selection import train_test_split
from torch.cuda.amp import GradScaler, autocast
import json
import random
import wandb
import torch
from safetensors.torch import load_file

from torch.utils.data import Dataset, DataLoader, Subset


file_path = '../FART---INLP/masked_examples_LARGE.json'
with open(file_path, 'r') as file:
    data = json.load(file)



nltk.download('punkt')
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using {device} device")

model = BartForConditionalGeneration.from_pretrained('facebook/bart-base')


tokenizer = BartTokenizer.from_pretrained("./trained_tokenizer")
model = BartForConditionalGeneration.from_pretrained("./trained_model", local_files_only=True)

# Load the safetensors weights
state_dict = load_file("/home2/aniruth.suresh/JEDI/trained_model/model.safetensors")
model.load_state_dict(state_dict , strict=False)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

print("Fine-tuned model loaded successfully!")


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using {device} device")

model = model.to(device)
model.eval()

class DialogueDataset(Dataset):
    def __init__(self, tokenizer, inputs, targets, max_len=512):
        self.tokenizer = tokenizer
        self.inputs = inputs
        self.targets = targets
        self.max_len = max_len

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        input_text = self.inputs[idx]
        target_text = self.targets[idx]
        input_encoding = tokenizer(input_text, padding='max_length', max_length=self.max_len, truncation=True, return_tensors='pt')
        target_encoding = tokenizer(target_text, padding='max_length', max_length=self.max_len, truncation=True, return_tensors='pt')
        labels = target_encoding['input_ids'].squeeze()
        labels[labels == tokenizer.pad_token_id] = -100

        return input_encoding['input_ids'].squeeze(), labels

inputs = [item['input'] for item in data]
targets = [item['target'] for item in data]
input_train, input_val, target_train, target_val = train_test_split(inputs, targets, test_size=0.2, random_state=42)
val_dataset = DialogueDataset(tokenizer, input_val, target_val)

val_dataset = DialogueDataset(tokenizer, input_val, target_val)
val_loader = DataLoader(val_dataset, batch_size=8)

bleu_metric = load_metric('bleu')
rouge_metric = load_metric('rouge')
accuracy_metric = load_metric('accuracy')
bleurt_scorer = score.BleurtScorer("bleurt-base-128")

def get_BLEU_ROUGE_BLEURT_scores(pred_ids, labels):
    pred_str = tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
    labels_mod = labels.clone()
    labels_mod[labels_mod == -100] = tokenizer.pad_token_id
    label_str = tokenizer.batch_decode(labels_mod, skip_special_tokens=True)

    predictions_tokens = [nltk.word_tokenize(pred) for pred in pred_str]
    references_tokens = [[nltk.word_tokenize(ref)] for ref in label_str]

    bleu_metric.add_batch(predictions=predictions_tokens, references=references_tokens)
    rouge_metric.add_batch(predictions=pred_str, references=label_str)
    accuracy_metric.add_batch(predictions=pred_ids.flatten().tolist(), references=labels_mod.flatten().tolist())

    bleurt_scores = bleurt_scorer.score(references=label_str, candidates=pred_str)

    return pred_str, label_str, bleurt_scores

model.eval()
total_loss = 0
all_bleurt_scores = []
for input_ids, labels in val_loader:
    input_ids = input_ids.to(device)
    labels = labels.to(device)

    with torch.no_grad():
        outputs = model(input_ids=input_ids, labels=labels)
        logits = outputs.logits
        pred_ids = torch.argmax(logits, dim=-1)
        loss = outputs.loss
        total_loss += loss.item()

        pred_str, label_str, bleurt_scores = get_BLEU_ROUGE_BLEURT_scores(pred_ids, labels)
        all_bleurt_scores.extend(bleurt_scores)

final_bleu = bleu_metric.compute()
final_rouge = rouge_metric.compute()
final_accuracy = accuracy_metric.compute()
avg_bleurt = sum(all_bleurt_scores) / len(all_bleurt_scores)
avg_loss = total_loss / len(val_loader)

print(f"Validation Loss: {avg_loss}")
print(f"BLEU Score: {final_bleu['bleu']}")
print(f"ROUGE Score: {final_rouge}")
print(f"Accuracy: {final_accuracy['accuracy']}")
print(f"BLEURT Score: {avg_bleurt}")

# ex predictions for debugging
for i in range(100):
    print(f"Prediction: {pred_str[i]}")
    print(f"Reference: {label_str[i]}")
    print(f"BLEURT Score: {bleurt_scores[i]}")

