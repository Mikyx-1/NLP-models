"""
Pre-training of BERT includes 2 tasks: MaskedLM and Next Sentence Prediction (NSP).

1. Masked Language Model (MLM):
    - Randomly masks some tokens in the input sequence and predicts the masked tokens.
    - The model learns to predict the masked tokens based on their context.

2. Next Sentence Prediction (NSP):
    - Given two sentences, the model predicts whether the second sentence is the next sentence in the original text.
    - This helps the model understand relationships between sentences and improves performance on tasks like question answering and natural language inference.
"""

import random

import torch
from datasets import load_dataset
from torch.utils.data import Dataset
from transformers import AutoTokenizer


class PreTrainingDataset(Dataset):
    def __init__(
        self,
        name: str = "stas/openwebtext-10k",
        split: str = "train",
        max_length: int = 512,
        mask_prob: float = 0.15,
        tokenizer_name: str = "bert-base-uncased",
    ):
        self.dataset = load_dataset(name, split=split)
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        self.max_length = max_length
        self.mask_prob = mask_prob

        # Preprocess texts into sentences
        self.sentences = []
        for item in self.dataset:
            text = item["text"]
            if text:
                sents = text.strip().split(". ")
                self.sentences.extend(
                    [s.strip() for s in sents if len(s.strip().split()) > 3]
                )

    def __len__(self):
        return len(self.sentences) - 1

    def __getitem__(self, index: int):
        is_next = random.random() < 0.5

        if is_next and index < len(self.sentences) - 1:
            # Positive sample (sentence B follows A)
            sent_a = self.sentences[index]
            sent_b = self.sentences[index + 1]
            label = 1
        else:
            # Negative sample (random sentence B)
            sent_a = self.sentences[index]
            sent_b = self.sentences[random.randint(0, len(self.sentences) - 1)]
            label = 0

        encoding = self.tokenizer(
            sent_a,
            sent_b,
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt",
        )

        # Apply random masking for MLM task
        input_ids = encoding["input_ids"].squeeze(0)
        labels = input_ids.clone()
        rand = torch.rand(input_ids.shape)
        mask_arr = (
            (rand < self.mask_prob)
            & (input_ids != self.tokenizer.cls_token_id)
            & (input_ids != self.tokenizer.sep_token_id)
            & (input_ids != self.tokenizer.pad_token_id)
        )

        # 80% [MASK], 10% random, 10% unchanged
        mask_indices = torch.where(mask_arr)[0]
        for idx in mask_indices:
            prob = random.random()
            if prob < 0.8:
                input_ids[idx] = self.tokenizer.mask_token_id
            elif prob < 0.9:
                input_ids[idx] = random.randint(0, self.tokenizer.vocab_size - 1)
            # else: leave the token unchanged

        return {
            "input_ids": input_ids,
            "token_type_ids": encoding["token_type_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "labels": labels,
            "next_sentence_label": torch.tensor(label, dtype=torch.long),
        }


if __name__ == "__main__":
    dataset = PreTrainingDataset()
    sample = dataset[0]

    print("Input IDs:", sample["input_ids"])
    print("Masked Labels:", sample["labels"])
    print("NSP Label:", sample["next_sentence_label"])
