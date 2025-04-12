"""
Bi-directional Encoder Representations from Transformers (BERT)
Pre-trained BERT model can be fine-tuned with just one additional output
layer to create SOTA models for a wide range of tasks
"""

"""
BERT base (L=12, H=768, A=12, Total params: 110M)
BERT large (L=24, H=1024, A=16, Total params: 340M)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Union


class TransformerEncoder(nn.Module):
    def __init__(self, embed_dim: int = 128, num_heads: int = 2, dropout: float = 0.1, batch_first: bool = True):
        super(TransformerEncoder, self).__init__()
        self.attention = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout, batch_first=batch_first)
        self.fc1 = nn.Linear(embed_dim, embed_dim * 4)
        self.fc2 = nn.Linear(embed_dim * 4, embed_dim)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout)
        self.activation = nn.GELU()

    def forward(self, x: torch.Tensor, mask: torch.Tensor = None):
        # x: (batch_size, seq_len, embed_dim)
        # mask: (batch_size, 1, 1, seq_len) or (batch_size, seq_len, seq_len)
        attn_output, _ = self.attention(x, x, x, attn_mask=mask)
        x = x + attn_output
        x = self.norm1(x)

        ff_output = self.fc2(self.dropout(self.activation(self.fc1(x))))
        x = x + ff_output
        x = self.norm2(x)

        return x


class InputEmbedding(nn.Module):
    def __init__(self, vocab_size: int, embed_dim: int, max_seq_len: int = 512):
        super().__init__()
        self.token_embedding = nn.Embedding(
            num_embeddings=vocab_size, embedding_dim=embed_dim
        )
        self.position_embedding = nn.Embedding(max_seq_len, embed_dim)

    def forward(self, input_ids: torch.Tensor):
        batch_size, seq_len = input_ids.size()
        token_embeds = self.token_embedding(input_ids)
        pos_ids = (
            torch.arange(seq_len, device=input_ids.device)
            .unsqueeze(0)
            .expand(batch_size, -1)
        )
        pos_embeds = self.position_embedding(pos_ids)
        return pos_embeds + token_embeds


class BERT(nn.Module):
    def __init__(self, vocab_size: int, max_seq_len: int, option: str = "base"):
        super().__init__()

        if option == "base":
            L, H, A = 12, 768, 12
            print("Base option is chosen. L = 12, H = 768, A = 12")
        elif option == "large":
            L, H, A = 24, 1024, 16
            print("Large option is chosen. L = 24, H = 1024, A = 16")
        else:
            print(f"Option is not recognised. Falling back to base!!")
            L, H, A = 12, 768, 12

        self.input_embedding = InputEmbedding(
            vocab_size=vocab_size, embed_dim=H, max_seq_len=max_seq_len
        )
        self.transformer_encoders = nn.ModuleList(
            [TransformerEncoder(embed_dim=H, num_heads=A) for _ in range(L)]
        )

    def forward(self, x: torch.Tensor):
        input_embeds = self.input_embedding(x)
        encoded = input_embeds
        for encoder in self.transformer_encoders:
            encoded = encoder(encoded)
        return encoded


def test_transformer_encoder():
    embed_dim = 128
    num_heads = 2
    dropout = 0.1

    encoder = TransformerEncoder(
        embed_dim=embed_dim, num_heads=num_heads, dropout=dropout
    )
    input_tensor = torch.randn(10, 20, embed_dim)
    mask = None

    try:
        output_tensor = encoder(x=input_tensor, mask=mask)
        print("\nTest Transformer Encoder")
        print(f"output_tensor shape: {output_tensor.shape}")
    except Exception as e:
        raise e


def test_input_embedding():
    embed_dim = 128
    max_seq_len = 32
    vocab_size = 10

    dummy = torch.randint(low=0, high=vocab_size - 1, size=(2, max_seq_len))
    input_embedding = InputEmbedding(
        vocab_size=vocab_size, embed_dim=embed_dim, max_seq_len=max_seq_len
    )

    try:
        input_embeds = input_embedding(dummy)
        print(
            f"\nTest input embedding. Embed_dim: {embed_dim}, Max_seq_len: {max_seq_len}, Vocab_size: {vocab_size}"
        )
        print(f"input_embeds shape: {input_embeds.shape}")
    except Exception as e:
        raise e


def test_bert():
    vocab_size = 100
    max_seq_len = 32
    bert_base = BERT(vocab_size=vocab_size, max_seq_len=max_seq_len, option="base")

    dummy = torch.randint(low=0, high=vocab_size - 1, size=(2, max_seq_len))
    encoded = bert_base(dummy)
    return encoded


if __name__ == "__main__":
    # test_transformer_encoder()
    # test_input_embedding()
    test_bert()
