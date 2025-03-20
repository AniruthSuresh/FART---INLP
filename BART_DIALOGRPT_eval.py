"""
Eval for DialogRPT, BERT variant
"""
import torch
from torch.utils.data import DataLoader, Dataset
from transformers import BartTokenizer, BartForConditionalGeneration
import json
from sklearn.model_selection import train_test_split
from bert_score import score
import torch
from safetensors.torch import load_file
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# tokenizer_rpt = AutoTokenizer.from_pretrained("microsoft/DialogRPT-human-vs-machine")
# model_rpt = AutoModelForSequenceClassification.from_pretrained("microsoft/DialogRPT-human-vs-machine")

file_path = '/home2/aniruth.suresh/JEDI/masked_examples_LARGE.json'
with open(file_path, 'r') as file:
    data = json.load(file)


tokenizer_rpt = AutoTokenizer.from_pretrained("microsoft/DialogRPT-human-vs-rand")
model_rpt = AutoModelForSequenceClassification.from_pretrained("microsoft/DialogRPT-human-vs-rand")
model_rpt.to('cuda') 


tokenizer = BartTokenizer.from_pretrained("./trained_tokenizer")

model = BartForConditionalGeneration.from_pretrained("./trained_model", local_files_only=True)

state_dict = load_file("/home2/aniruth.suresh/JEDI/trained_model/model.safetensors")
model.load_state_dict(state_dict , strict=False)


def generate_responses(model, tokenizer, dataloader, device):
    model.eval()
    model.to(device)
    responses = []
    contexts = []
    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch[0].to(device)
            context = tokenizer.batch_decode(input_ids, skip_special_tokens=True)
            outputs = model.generate(input_ids, max_length=50)
            decoded_responses = tokenizer.batch_decode(outputs, skip_special_tokens=True)
            responses.extend(decoded_responses)
            contexts.extend(context)

    return contexts, responses

def drpt_eval(model_rpt, tokenizer_rpt, contexts, responses, device):
    model_rpt.eval()
    scores = []
    with torch.no_grad():
        for context, response in zip(contexts, responses):
            inputs = tokenizer_rpt.encode_plus(context, response, return_tensors="pt", max_length=512, truncation=True, padding=True)
            inputs = {k: v.to(device) for k, v in inputs.items()}  # Ensure tensor is on the correct device
            outputs = model_rpt(**inputs)
            score = torch.sigmoid(outputs.logits).squeeze().item()  # Use sigmoid if the logits are not already probabilities
            scores.append(score)
    return scores

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
val_loader = DataLoader(val_dataset, batch_size=8)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


contexts, responses = generate_responses(model, tokenizer, val_loader, device)


scores = drpt_eval(model_rpt, tokenizer_rpt, contexts, responses, device)


for i in range(min(5, len(contexts))):  # Print first 5 examples
    print(f"\nContext: {contexts[i]}")
    print(f"Generated Response: {responses[i]}")
    print(f"Expected Response: {target_val[i] if i < len(target_val) else 'N/A'}")

    print(f"DialogRPT Score: {scores[i]:.4f}")

    
print("Average DialogRPT Score:", sum(scores) / len(scores))
