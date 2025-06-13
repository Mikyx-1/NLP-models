from typing import Optional

import torch
from torch.utils.data import DataLoader, Dataset


class QADataset(Dataset):
    def __init__(
        self, file_path: str, tokeniser: Optional[None], max_length: int = 12
    ) -> None:
        self.examples = []
        with open(file_path, "r") as f:
            lines = f.read().split("\n")

        for i in range(0, len(lines) - 1, 3):  # Q, A, ""
            q = lines[i].strip()
            a = lines[i + 1].strip()
            text = f"{q} {a}"
            tokenised = tokeniser.encode(text, truncation=True, max_length=max_length)
            self.examples.append(torch.tensor(tokenised, dtype=torch.long))

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        ids = self.examples[idx]
        return ids[:-1], ids[1:]
