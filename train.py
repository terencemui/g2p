import pandas as pd
import torch
from transformers import T5Tokenizer, T5ForConditionalGeneration, AdamW
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from jiwer import wer
from torch.utils.tensorboard import SummaryWriter
import argparse
import os

parser = argparse.ArgumentParser()
# parser.add_argument("name", type=str, help="Your name")
# parser.add_argument("-a", "--age", type=int, help="Your age", required=False)
# parser.add_argument("-v", "--verbose", action="store_true", help="Enable verbose mode")

parser.add_argument("--run_number", type=int, default=-1)
parser.add_argument("--start_epoch", type=int, default=0)
parser.add_argument("--epochs", type=int, default=10)

args = parser.parse_args()

# hyperparams
if args.run_number == -1:
    # no run number, create new run
    folders = os.listdir("logs")
    i = 1
    while f"run_{i}" in folders:
        i += 1
    run_number = i
    start_epoch = 0

else:
    run_number = args.run_number
    start_epoch = args.start_epoch

epochs = args.epochs

log_dir = f"logs/run_{run_number}"
writer = SummaryWriter(log_dir=log_dir, purge_step=start_epoch)
writer.add_text("Description", "Token-based, training on train-clean-100.csv")

batch_size = 16
lr = 5e-5

model_name = "google-t5/t5-small"
model_path = "g2p_t5_model"

# device = torch.device("mps" if torch.mps.is_available() else "cpu")
device = torch.device("cpu")

tokenizer = T5Tokenizer.from_pretrained(model_name)

# load previous model
model_path = f"weights/weights_{run_number}/epoch_{start_epoch - 1}"
try:
    model = T5ForConditionalGeneration.from_pretrained(model_path).to(device)
    print(f"Successfully loaded model: {model_path}")
except:
    print("No model found. Creating new model")
    model = T5ForConditionalGeneration.from_pretrained(model_name).to(device)
model.gradient_checkpointing_enable()

# Optimizer
optimizer = AdamW(model.parameters(), lr=lr)

writer.add_text("Hyperparameters", f"Learning Rate:\t{lr} , Batch Size:\t{batch_size}")
writer.add_scalar("Learning Rate", lr)
writer.add_scalar("Batch Size", batch_size)

print(f"Run: run{run_number}")
print(f"Start epoch: {start_epoch}")
print(f"Epochs: {epochs}")

# Function to force character-level tokenization
def format_input(text):
    return f"grapheme to phoneme: {text}"
    # return f"grapheme to phoneme: {" ".join(text)}"


# Custom dataset class
class G2PDataset(Dataset):
    def __init__(self, file_path, max_length=512):
        self.data = pd.read_csv(file_path)
        self.max_length = max_length

        print(f"Dataset: {file_path}")
        print(f"Original dataset size:\t{len(self.data)}")

        # self.data = self.data[self.data["text"].apply(lambda x: len(tokenizer(format_input(x))["input_ids"]) <= self.max_length)]
        self.data = self.data[
            self.data.apply(
                lambda x: (
                    len(tokenizer(format_input(x["text"]))["input_ids"]) <= self.max_length
                    and len(tokenizer(x["phonemes"])["input_ids"]) <= self.max_length
                ),
                axis=1
            )
        ]

        print(f"Reduced dataset size:\t{len(self.data)}\n")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        grapheme_text = self.data.iloc[idx]["text"]
        phoneme_text = self.data.iloc[idx]["phonemes"]

        # Force character-level tokenization
        formatted_input = format_input(grapheme_text)

        return formatted_input, phoneme_text

# Collate function for dynamic padding
def collate_fn(batch):
    inputs, targets = zip(*batch)

    # Tokenize with dynamic padding (longest in batch)
    input_enc = tokenizer(list(inputs), padding=True, return_tensors="pt", truncation=False)
    target_enc = tokenizer(list(targets), padding=True, return_tensors="pt", truncation=False)

    return {
        "input_ids": input_enc.input_ids,
        "attention_mask": input_enc.attention_mask,
        "labels": target_enc.input_ids,
    }

# Load dataset and dataloader
train_dataset = G2PDataset("clean_data/train-clean-100.csv")
val_dataset = G2PDataset("clean_data/test-clean.csv")

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
val_loader = DataLoader(val_dataset, batch_size=batch_size, collate_fn=collate_fn)

# Training function
def train_model(model, train_loader, val_loader, epochs, writer, verbose=True):
    model.train()

    for epoch in range(start_epoch, start_epoch + epochs):
        total_loss = 0
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{start_epoch + epochs}", leave=True)
        for batch in progress_bar:
            optimizer.zero_grad()

            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels, use_cache=False)
            loss = outputs.loss
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

            progress_bar.set_postfix(loss=loss.item())
            del input_ids, attention_mask, labels
            torch.mps.empty_cache()

        avg_train_loss = total_loss / len(train_loader)
        if verbose:
            print(f"Epoch {epoch+1}, Loss: {avg_train_loss:.4f}")

        writer.add_scalar("Training Loss", avg_train_loss, epoch)

        # Validation
        avg_val_loss, exact_match_accuracy, avg_per = validate_model(model, val_loader)
        writer.add_scalar("Validation Loss", avg_val_loss, epoch)
        writer.add_scalar("Exact Match", exact_match_accuracy, epoch)
        writer.add_scalar("Average PER", avg_per, epoch)

        # save the model
        model.save_pretrained(f"weights/weights_{run_number}/epoch_{epoch}")

        torch.mps.empty_cache()

    return


# Validation function
def validate_model(model, val_loader, verbose=True):
    model.eval()
    total_loss = 0
    total_exact = 0
    total_samples = 0
    total_per = 0

    progress_bar = tqdm(val_loader, desc="Validation", leave=True)

    with torch.no_grad():
        for batch in progress_bar:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            total_loss += outputs.loss.item()

            # calculate per
            predicted_ids = torch.argmax(outputs.logits, dim=-1)
            pred_phonemes = [tokenizer.decode(pred, skip_special_tokens=True) for pred in predicted_ids]
            true_phonemes = [tokenizer.decode(label, skip_special_tokens=True) for label in labels]

            for pred, target in zip(pred_phonemes, true_phonemes):
                if pred.strip() == target.strip():
                    total_exact += 1
                else:
                    total_per += wer(target, pred)  # WER works similarly for phonemes

            total_samples += len(batch['labels'])

            torch.mps.empty_cache()

    avg_val_loss = total_loss / len(val_loader)
    exact_match_accuracy = total_exact / total_samples
    avg_per = total_per / total_samples

    if verbose:
        print(f"Validation Loss: {avg_val_loss:.4f}")
        print(f"Validation Accuracy (Exact Match): {exact_match_accuracy * 100:.2f}%")
        print(f"Average Phoneme Error Rate (PER): {avg_per * 100:.2f}%")

    return avg_val_loss, exact_match_accuracy, avg_per

train_model(model, train_loader, val_loader, epochs, writer)

writer.close()
