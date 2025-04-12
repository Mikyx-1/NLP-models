import torch
import torch.nn as nn
from dataset import PreTrainingDataset
from model import BERT
from torch.utils.data import DataLoader
from tqdm import tqdm


class BERTPreTrainingHead(nn.Module):
    def __init__(self, hidden_size: int, vocab_size: int):
        super().__init__()
        self.mlm_head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.GELU(),
            nn.LayerNorm(hidden_size),
            nn.Linear(hidden_size, vocab_size),
        )
        self.nsp_head = nn.Linear(hidden_size, 2)

    def forward(self, encoded: torch.Tensor):
        # encoded shape: (batch_size, seq_len, hidden_size)
        mlm_logits = self.mlm_head(encoded)
        cls_rep = encoded[:, 0, :]  # Use CLS token for NSP
        nsp_logits = self.nsp_head(cls_rep)
        return mlm_logits, nsp_logits


class BERTForPreTraining(nn.Module):
    def __init__(self, vocab_size, max_seq_len, option="base"):
        super().__init__()
        self.bert = BERT(vocab_size, max_seq_len, option)
        hidden_size = 768 if option == "base" else 1024
        self.heads = BERTPreTrainingHead(hidden_size, vocab_size)

    def forward(self, input_ids, attention_mask, token_type_ids):
        encoded = self.bert(input_ids)
        mlm_logits, nsp_logits = self.heads(encoded)
        return mlm_logits, nsp_logits


def pretrain():
    vocab_size = 30522  # Standard for BERT-base
    max_len = 256
    device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
    model = BERTForPreTraining(vocab_size, max_len, option="base").to(device)
    dataset = PreTrainingDataset(max_length=max_len)
    dataloader = DataLoader(dataset, batch_size=16, shuffle=True, pin_memory=True)

    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5)
    mlm_loss_fn = nn.CrossEntropyLoss(ignore_index=-100)
    nsp_loss_fn = nn.CrossEntropyLoss()

    model.train()

    for epoch in range(3):
        total_loss = 0.0
        total_loss_mlm = 0.0
        total_loss_nsp = 0.0
        num_batches = 0

        for batch in tqdm(dataloader, desc=f"Epoch {epoch+1}"):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            token_type_ids = batch["token_type_ids"].to(device)
            labels = batch["labels"].to(device)
            nsp_labels = batch["next_sentence_label"].to(device)

            optimizer.zero_grad()

            mlm_logits, nsp_logits = model(input_ids, attention_mask, token_type_ids)

            loss_mlm = mlm_loss_fn(
                mlm_logits.view(-1, mlm_logits.size(-1)), labels.view(-1)
            )
            loss_nsp = nsp_loss_fn(nsp_logits, nsp_labels)

            loss = loss_mlm + loss_nsp
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            total_loss_mlm += loss_mlm.item()
            total_loss_nsp += loss_nsp.item()
            num_batches += 1

        avg_loss = total_loss / num_batches
        avg_loss_mlm = total_loss_mlm / num_batches
        avg_loss_nsp = total_loss_nsp / num_batches

        print(
            f"[Epoch {epoch+1}] Avg Loss: {avg_loss:.4f} | Avg MLM: {avg_loss_mlm:.4f} | Avg NSP: {avg_loss_nsp:.4f}"
        )


if __name__ == "__main__":
    pretrain()
