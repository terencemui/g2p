import pandas as pd
import torch
from transformers import T5Tokenizer, T5ForConditionalGeneration, AutoTokenizer
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import matplotlib.pyplot as plt
from jiwer import wer
from torch.utils.tensorboard import SummaryWriter
from torch.cuda.amp import GradScaler

device = torch.device("cuda")

run_number = 0
epochs = 10
batch_size = 16
lr = 5e-5
weight_decay = 0.01

train_dataset_path = "clean_data/train-clean-100.csv"
val_dataset_path = "clean_data/test-clean.csv"

model_name = "google/byt5-small"

model = T5ForConditionalGeneration.from_pretrained(model_name).to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
tokenizer = AutoTokenizer.from_pretrained(model_name)

print("Cuda available and enabled") if torch.cuda.is_available() else None
if device.type == "cuda":
    print(f"Using CUDA with cuDNN Enabled: {torch.backends.cudnn.enabled}")

start_epoch = 0
log_dir = f"logs/run_{run_number}"
writer = SummaryWriter(log_dir=log_dir, purge_step=start_epoch)

# Custom dataset class
class G2PDataset(Dataset):
    def preprocess(self):
        self.data["text"] = "grapheme to phoneme: " + self.data["text"]
        self.data["len"] = self.data["text"].str.len() + 1
        self.data = self.data[self.data["len"] < self.max_length]

    def __init__(self, file_path, max_length=512):
        self.data = pd.read_csv(file_path)
        self.max_length = max_length

        print(f"Dataset: {file_path}")
        print(f"Original dataset size:\t{len(self.data)}")
        self.preprocess()
        print(f"Reduced dataset size:\t{len(self.data)}\n")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        grapheme_text = self.data.iloc[idx]["text"]
        phoneme_text = self.data.iloc[idx]["phonemes"]

        return grapheme_text, phoneme_text

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

train_dataset = G2PDataset(train_dataset_path)
val_dataset = G2PDataset(val_dataset_path)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
val_loader = DataLoader(val_dataset, batch_size=batch_size, collate_fn=collate_fn)

# clear gpu_cache
def clear_gpu_cache():
    if device is torch.device("cuda"):
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
        torch.backends.cudnn.benchmark = False  # Reset benchmark tuning
        torch.backends.cudnn.benchmark = True  # Re-enable after clearing
    elif device is torch.device("mps"):
        torch.mps.empty_cache()

# Training function
def train_model(model, train_loader, val_loader, epochs, writer, verbose=True):
    model.train()

    for epoch in range(start_epoch, start_epoch + epochs):
        total_loss = 0
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{start_epoch + epochs}", leave=True)
        for batch in progress_bar:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            # with torch.autocast("cuda", dtype=torch.float16):
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss

            if torch.isnan(loss) or torch.isinf(loss):
                print("Warning: NaN loss detected! Skipping batch.")
                continue

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            total_loss += loss.item()

            progress_bar.set_postfix(loss=loss.item())
            del input_ids, attention_mask, labels

        avg_train_loss = total_loss / len(train_loader)
        if verbose:
            print(f"Epoch {epoch+1}, Loss: {avg_train_loss:.4f}")

        writer.add_scalar("Training Loss", avg_train_loss, epoch)
        clear_gpu_cache()

        # Validation
        avg_val_loss, exact_match_accuracy, avg_per = validate_model(model, val_loader)
        writer.add_scalar("Validation Loss", avg_val_loss, epoch)
        writer.add_scalar("Exact Match", exact_match_accuracy, epoch)
        writer.add_scalar("Average PER", avg_per, epoch)

        # save the model
        model.save_pretrained(f"weights/weights_{run_number}")
        torch.save(optimizer.state_dict(), f"weights/weights_{run_number}/optimizer.pt")
        # model.save_pretrained(f"weights/weights_{run_number}/epoch_{epoch}")
        # torch.save(optimizer.state_dict(), f"weights/weights_{run_number}/epoch_{epoch}/optimizer.pt")

        clear_gpu_cache()

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

            with torch.autocast("cuda", dtype=torch.float16):
                outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
                loss = outputs.loss

            total_loss += loss.item()
            progress_bar.set_postfix(loss=loss.item())


            # calculate per
            predicted_ids = torch.argmax(outputs.logits, dim=-1)
            pred_phonemes = tokenizer.batch_decode(predicted_ids, skip_special_tokens=True)
            true_phonemes = tokenizer.batch_decode(labels, skip_special_tokens=True)

            for pred, target in zip(pred_phonemes, true_phonemes):
                if pred.strip() == target.strip():
                    total_exact += 1
                else:
                    total_per += wer(target, pred)  # WER works similarly for phonemes

            total_samples += len(batch['labels'])

            clear_gpu_cache()

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




