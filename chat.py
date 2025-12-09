#!/usr/bin/env python3
"""
TinyLLM Chat - Interactive conversational AI
Main entry point for the TinyLLM project.

Usage:
    python chat.py
"""
import torch
import torch.nn.functional as F
from model import TinyLLM

# Load model
ckpt = torch.load('tiny_llm.pt', map_location='cpu', weights_only=False)
chars = ckpt['chars']
stoi = {c: i for i, c in enumerate(chars)}
itos = {i: c for c, i in stoi.items()}

model = TinyLLM(len(chars), dim=192, n_layers=5, n_heads=6, max_len=192)
model.load_state_dict(ckpt['model'])
model.eval()

def generate(prompt, max_new=150, temperature=0.7, top_k=50, top_p=0.9):
    """Generate text with top-k and nucleus (top-p) sampling."""
    ids = [stoi.get(c, 0) for c in prompt]
    
    with torch.no_grad():
        for _ in range(max_new):
            x = torch.tensor([ids[-180:]])
            logits = model(x)[0, -1] / temperature
            
            # Top-k filtering
            if top_k > 0:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[-1]] = float('-inf')
            
            # Top-p (nucleus) filtering
            if top_p < 1.0:
                sorted_logits, sorted_idx = torch.sort(logits, descending=True)
                cumsum = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                remove = cumsum > top_p
                remove[1:] = remove[:-1].clone()
                remove[0] = False
                logits[sorted_idx[remove]] = float('-inf')
            
            probs = F.softmax(logits, -1)
            nxt = torch.multinomial(probs, 1).item()
            ids.append(nxt)
            
            if '<|end|>' in ''.join(itos[i] for i in ids[-10:]):
                break
    
    return ''.join(itos[i] for i in ids)

def chat_response(user_input):
    """Get clean response from model."""
    prompt = f"<|user|>{user_input}<|bot|>"
    out = generate(prompt)
    response = out.split('<|bot|>')[-1].replace('<|end|>', '').strip()
    for token in ['<|user|>', '<|', '|>']:
        if token in response:
            response = response.split(token)[0].strip()
    return response

def main():
    print("=" * 55)
    print("  TinyLLM Chat v2.0")
    print("  2.28M params | Loss: 0.0449 | Type 'quit' to exit")
    print("=" * 55 + "\n")
    
    while True:
        try:
            user = input("You: ").strip()
            if user.lower() in ['quit', 'exit', 'q']:
                print("Goodbye!")
                break
            if not user:
                continue
            response = chat_response(user)
            print(f"Bot: {response}\n")
        except KeyboardInterrupt:
            print("\nGoodbye!")
            break

if __name__ == "__main__":
    main()
