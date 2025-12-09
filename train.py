#!/usr/bin/env python3
"""
TinyLLM Training Script
Train the model from scratch on dialogue.txt

Usage:
    python train.py
"""
import torch
import torch.nn.functional as F
from model import TinyLLM
import time

# Load data
text = open('dialogue.txt').read()
chars = sorted(set(text))
stoi = {c: i for i, c in enumerate(chars)}
itos = {i: c for c, i in stoi.items()}
encode = lambda s: [stoi[c] for c in s]
data = torch.tensor(encode(text), dtype=torch.long)

# Config - v2.0 architecture
BATCH = 48
SEQ_LEN = 192
EPOCHS = 12
LR = 0.002
DECAY = 0.82
device = 'cpu'

# Model
model = TinyLLM(len(chars), dim=192, n_layers=5, n_heads=6, max_len=SEQ_LEN).to(device)
opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0.01)

print("=" * 50)
print("TinyLLM Training v2.0")
print("=" * 50)
print(f"Vocabulary: {len(chars)} chars")
print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")
print(f"Dataset: {len(data):,} characters")
print(f"Config: dim=192, layers=5, heads=6, seq={SEQ_LEN}")
print(f"Training: {EPOCHS} epochs, batch={BATCH}, lr={LR}")
print("=" * 50 + "\n")

# Training
start = time.time()
for epoch in range(EPOCHS):
    model.train()
    total_loss = 0
    steps = 0
    
    # Shuffle batches
    indices = torch.randperm(len(data) - SEQ_LEN - 1)
    
    for i in range(0, len(indices) - BATCH, BATCH):
        batch_idx = indices[i:i+BATCH]
        x = torch.stack([data[j:j+SEQ_LEN] for j in batch_idx]).to(device)
        y = torch.stack([data[j+1:j+SEQ_LEN+1] for j in batch_idx]).to(device)
        
        loss = F.cross_entropy(model(x).view(-1, len(chars)), y.view(-1))
        
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        
        total_loss += loss.item()
        steps += 1
    
    avg_loss = total_loss / steps
    print(f"Epoch {epoch+1:2d}/{EPOCHS} | Loss: {avg_loss:.4f}")
    
    # Learning rate decay
    for g in opt.param_groups:
        g['lr'] *= DECAY

elapsed = time.time() - start
print(f"\nTraining completed in {elapsed:.1f}s")
print(f"Final loss: {avg_loss:.4f}")

# Save
torch.save({
    'model': model.state_dict(),
    'chars': chars,
    'stoi': stoi,
    'itos': itos
}, 'tiny_llm.pt')
print(f"Saved tiny_llm.pt ({sum(p.numel() for p in model.parameters()):,} params)")
