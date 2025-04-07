'''
Implementation of Generative Pre-trained Transformer GPT version 1
Implementation date: 07 April 2025
'''

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from tokeniser import encode_text, decode_text, generate_text_beam_search, generate_text_temperature


class GPTBlock(nn.Module):
    def __init__(self, embed_dim: int, num_heads: int, dropout: float):
        super().__init__()
        
        self.ln1 = nn.LayerNorm(embed_dim)
        self.attn = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout)
        self.ln2 = nn.LayerNorm(embed_dim)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, 4 * embed_dim),
            nn.GELU(),
            nn.Linear(4 * embed_dim, embed_dim),
            nn.Dropout(dropout)
        )
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x: torch.Tensor):
        residual = x
        x = self.ln1(x)
        seq_len = x.size(0)
        attn_mask = torch.triu(torch.full((seq_len, seq_len), float("-inf"), device=x.device), diagonal=1)
        attn_output, _ = self.attn(x, x, x, attn_mask=attn_mask)
        x = residual + self.dropout(attn_output)
        
        residual = x
        x = self.ln2(x)
        x = residual + self.mlp(x)
        return x


class GPT(nn.Module):
    def __init__(self, vocab_size, embed_dim, num_heads, num_layers, max_seq_len, dropout=0.1):
        super().__init__()
        self.token_emb = nn.Embedding(vocab_size, embed_dim)
        self.pos_emb = nn.Parameter(torch.zeros(max_seq_len, embed_dim))
        self.drop = nn.Dropout(dropout)
        self.blocks = nn.ModuleList([
            GPTBlock(embed_dim, num_heads, dropout) for _ in range(num_layers)
        ])
        self.ln_f = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, vocab_size, bias=False)

    def forward(self, ids):
        # ids: (batch, seq_len)
        batch, seq_len = ids.size()
        token_embeddings = self.token_emb(ids)  # (batch, seq_len, embed_dim)
        pos_embeddings = self.pos_emb[:seq_len, :].unsqueeze(0)  # (1, seq_len, embed_dim)
        x = self.drop(token_embeddings + pos_embeddings)
        x = x.transpose(0, 1)  # (seq_len, batch, embed_dim)
        for block in self.blocks:
            x = block(x)
        x = self.ln_f(x)
        x = x.transpose(0, 1)  # (batch, seq_len, embed_dim)
        logits = self.head(x)
        return logits
    
    


if __name__ == "__main__":
    VOCAB_SIZE = 10**5
    EMBED_DIM = 128
    NUM_HEADS = 4
    NUM_LAYERS = 12
    MAX_SEQ_LEN = 1024
    DROPOUT = 0.1
    DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    gpt = GPT(vocab_size=VOCAB_SIZE, embed_dim=EMBED_DIM, num_heads=NUM_HEADS, num_layers=NUM_LAYERS, max_seq_len=MAX_SEQ_LEN, dropout=DROPOUT).to(DEVICE)


    prompt = "Hello world"
    
    print("=== Temperature Sampling ===")
    generated_text = generate_text_temperature(gpt, prompt, max_length=50, temperature=0.8)
    print("Generated Text:", generated_text)
    
    print("\n=== Beam Search ===")
    generated_text, score = generate_text_beam_search(gpt, prompt, max_length=50, beam_width=5, temperature=0.8)
    print("Generated Text:", generated_text)
    print("Beam Score:", score)
