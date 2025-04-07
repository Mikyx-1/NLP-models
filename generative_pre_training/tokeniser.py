import torch
import torch.nn.functional as F
from transformers import BertTokenizer, GPT2Tokenizer, AutoModelForCausalLM, AutoTokenizer
import torch.nn as nn

# Load a pre-trained BERT tokenizer
tokeniser = AutoTokenizer.from_pretrained("gpt2")

def encode_text(text: str):
    return tokeniser.encode(text, return_tensors="pt")

def decode_text(token_ids):
    return tokeniser.decode(token_ids, skip_special_tokens=True)


# Beam Search Inference
def generate_text_beam_search(model, prompt, max_length=50, beam_width=5, temperature=1.0):
    model.eval()
    input_ids = encode_text(prompt).to(next(model.parameters()).device)
    
    # Each beam is a tuple (sequence, cumulative log probability)
    beams = [(input_ids, 0.0)]
    
    with torch.no_grad():
        for _ in range(max_length - input_ids.size(1)):
            candidates = []
            for seq, score in beams:
                logits = model(seq)  # (1, seq_len, vocab_size)
                logits = logits[:, -1, :] / temperature  # (1, vocab_size)
                log_probs = F.log_softmax(logits, dim=-1)  # (1, vocab_size)
                top_log_probs, top_ids = log_probs.topk(beam_width, dim=-1)  # (1, beam_width)
                
                # Expand each beam candidate with the top-k tokens.
                for i in range(beam_width):
                    next_token = top_ids[0, i].unsqueeze(0).unsqueeze(0)  # shape: (1, 1)
                    new_seq = torch.cat([seq, next_token], dim=1)
                    new_score = score + top_log_probs[0, i].item()
                    candidates.append((new_seq, new_score))
            # Keep only the best beam_width candidates
            beams = sorted(candidates, key=lambda x: x[1], reverse=True)[:beam_width]
    
    best_seq, best_score = beams[0]
    return decode_text(best_seq[0].tolist()), best_score
    
# Temperature Sampling Inference
def generate_text_temperature(model, prompt, max_length=50, temperature=1.0):
    model.eval()
    # Encode prompt using the BERT tokenizer.
    input_ids = encode_text(prompt).to(next(model.parameters()).device)
    generated = input_ids  # shape: (1, seq_len)
    
    with torch.no_grad():
        for _ in range(max_length - input_ids.size(1)):
            logits = model(generated)  # (1, seq_len, vocab_size)
            # Focus only on the last token's logits, then adjust by temperature
            next_token_logits = logits[:, -1, :] / temperature
            probs = F.softmax(next_token_logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            generated = torch.cat((generated, next_token), dim=1)
            
            # Optionally, stop if an end-of-sequence token is generated.
            # if next_token.item() == tokenizer.convert_tokens_to_ids("[SEP]"):
            #     break

    return decode_text(generated[0].tolist())

if __name__ == "__main__":

    string1 = "GPT2 is a model developed by VietLH."
    encoded_text = encode_text(string1)
    print(f"Encoded text: {encoded_text}")

    # Extract the inner list (assuming batch size of 1)
    decoded_text = decode_text(encoded_text[0].tolist())
    print(f"Decoded text: {decoded_text}")
