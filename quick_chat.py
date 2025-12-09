#!/usr/bin/env python3
"""
TinyLLM Quick Demo - Non-interactive demonstration
Shows sample conversations without user input.

Usage:
    python quick_chat.py
"""
import torch
import torch.nn.functional as F
from model import TinyLLM

ckpt = torch.load('tiny_llm.pt', weights_only=False)
chars = ckpt['chars']
stoi = ckpt['stoi']
itos = ckpt['itos']

model = TinyLLM(len(chars), dim=192, n_layers=5, n_heads=6, max_len=192)
model.load_state_dict(ckpt['model'])
model.eval()

def chat(prompt, max_new=100):
    """Generate response for a prompt."""
    tokens = [stoi.get(c, 0) for c in f'<|user|>{prompt}<|bot|>']
    x = torch.tensor([tokens])
    with torch.no_grad():
        for _ in range(max_new):
            logits = model(x[:, -192:])
            probs = F.softmax(logits[:, -1] / 0.7, dim=-1)
            next_tok = torch.multinomial(probs, 1)
            x = torch.cat([x, next_tok], dim=1)
            if '<|end|>' in ''.join([itos[t.item()] for t in x[0, -6:]]):
                break
    response = ''.join([itos[t.item()] for t in x[0]])
    response = response.split('<|bot|>')[-1]
    for tag in ['<|end|>', '<|user|>', '<|bot|>']:
        response = response.split(tag)[0]
    return response.strip()

if __name__ == "__main__":
    prompts = [
        'Hello!',
        'What is AI?',
        'Explain machine learning',
        'What is programming?',
        'Tell me a joke',
        'Thank you!'
    ]
    
    print("=" * 55)
    print("  TinyLLM Quick Demo v2.0")
    print("  2.28M params | Loss: 0.0449")
    print("=" * 55 + "\n")
    
    for p in prompts:
        print(f"You: {p}")
        print(f"Bot: {chat(p)}\n")
    
    print("For interactive chat, run: python chat.py")
