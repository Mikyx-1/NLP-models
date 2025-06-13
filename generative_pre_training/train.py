import torch
import torch.nn as nn
import torch.nn.functional as F
from model import GPT
from torch.cuda.amp import GradScaler, autocast
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer

# ============================
# Config
# ============================

MAX_SEQ_LEN = 128
BATCH_SIZE = 8
EPOCHS = 1000
LEARNING_RATE = 3e-4
DEVICE = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")

# ============================
# Tokenizer Setup
# ============================

tokeniser = AutoTokenizer.from_pretrained("gpt2")
tokeniser.pad_token = tokeniser.eos_token

# ============================
# Dataset
# ============================


class QADataset(Dataset):
    def __init__(self, filepath, tokenizer, max_length=128):
        self.examples = []
        with open(filepath, "r") as f:
            lines = f.read().strip().split("\n")
        for i in range(0, len(lines) - 1, 3):  # Q, A, ""
            q = lines[i].strip()
            a = lines[i + 1].strip()
            text = f"{q} {a}"
            tokenized = tokenizer.encode(text, truncation=True, max_length=max_length)
            self.examples.append(torch.tensor(tokenized, dtype=torch.long))

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        ids = self.examples[idx]
        return ids[:-1], ids[1:]  # input, target


def collate_fn(batch):
    inputs, targets = zip(*batch)
    inputs = pad_sequence(
        inputs, batch_first=True, padding_value=tokeniser.pad_token_id
    )
    targets = pad_sequence(targets, batch_first=True, padding_value=-100)
    return inputs, targets


# ============================
# Training Loop
# ============================


def train(model, dataloader, optimizer, device, epochs=5):
    model.train()
    scaler = GradScaler()
    loss_fn = nn.CrossEntropyLoss(ignore_index=-100)

    for epoch in range(epochs):
        for step, (inputs, targets) in enumerate(dataloader):
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            with autocast():
                logits = model(inputs)
                logits = logits.view(-1, logits.size(-1))
                targets = targets.view(-1)
                loss = loss_fn(logits, targets)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

        print(f"[Epoch {epoch+1}] Loss: {loss.item():.4f}")


# ============================
# Main
# ============================

if __name__ == "__main__":
    # Save sample tiny QA if not available
    import os

    if not os.path.exists("tiny_qa.txt"):
        with open("tiny_qa.txt", "w") as f:
            f.write(
                "Q: What is the capital of France?\nA: Paris\n\n"
                "Q: Who wrote Hamlet?\nA: William Shakespeare\n\n"
                "Q: What is 2 + 2?\nA: 4\n\n"
                "Q: Who is the president of the USA?\nA: Joe Biden\n"
            )

    dataset = QADataset("tiny_qa.txt", tokeniser, max_length=MAX_SEQ_LEN)
    dataloader = DataLoader(
        dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn
    )

    model = GPT(
        vocab_size=tokeniser.vocab_size,
        embed_dim=128,
        num_heads=4,
        num_layers=4,
        max_seq_len=MAX_SEQ_LEN,
        dropout=0.1,
    ).to(DEVICE)

    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)

    print("Starting training...")
    train(model, dataloader, optimizer, DEVICE, epochs=EPOCHS)

    torch.save(model.state_dict(), "mini_gpt_qa.pth")
    print("✅ Training complete. Model saved to 'mini_gpt_qa.pth'")
