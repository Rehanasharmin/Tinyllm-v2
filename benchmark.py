import torch
import torch.nn as nn
import time
import os

# Setup matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
plt.switch_backend("Agg")
plt.style.use("seaborn-v0_8")
sns.set_palette("husl")
plt.rcParams["font.sans-serif"] = ["Noto Sans CJK SC", "WenQuanYi Zen Hei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

# Model definition (same as model.py)
class TinyLLM(nn.Module):
    def __init__(self, vocab_size, dim=128, n_layers=3, n_heads=4, max_len=256):
        super().__init__()
        self.tok_emb = nn.Embedding(vocab_size, dim)
        self.pos_emb = nn.Embedding(max_len, dim)
        self.layers = nn.ModuleList([
            nn.TransformerEncoderLayer(dim, n_heads, dim*4, dropout=0.1, batch_first=True)
            for _ in range(n_layers)
        ])
        self.ln = nn.LayerNorm(dim)
        self.head = nn.Linear(dim, vocab_size, bias=False)
        self.max_len = max_len
        
    def forward(self, x):
        B, T = x.shape
        pos = torch.arange(T, device=x.device)
        x = self.tok_emb(x) + self.pos_emb(pos)
        mask = torch.triu(torch.ones(T, T, device=x.device), 1).bool()
        for layer in self.layers:
            x = layer(x, src_mask=mask, is_causal=True)
        return self.head(self.ln(x))

# Load model
print("Loading model...")
ckpt = torch.load("tiny_llm.pt", map_location="cpu")
chars = ckpt["chars"]
stoi = {c: i for i, c in enumerate(chars)}
itos = {i: c for i, c in enumerate(chars)}
model = TinyLLM(len(chars), dim=192, n_layers=5, n_heads=6, max_len=192)
model.load_state_dict(ckpt["model"])
model.eval()

# Count parameters
total_params = sum(p.numel() for p in model.parameters())
print(f"Total parameters: {total_params:,}")

# Generate function
def generate(prompt, max_new=50):
    ids = [stoi.get(c, 0) for c in prompt]
    with torch.no_grad():
        for _ in range(max_new):
            x = torch.tensor([ids[-128:]])
            logits = model(x)[0, -1] / 0.7
            probs = torch.softmax(logits, -1)
            nxt = torch.multinomial(probs, 1).item()
            ids.append(nxt)
            if itos[nxt] == "|" and len(ids) > 3:
                out = "".join(itos[i] for i in ids)
                if "<|end|>" in out:
                    break
    return "".join(itos[i] for i in ids)

# Test prompts
test_prompts = [
    "<|user|>Hello!<|bot|>",
    "<|user|>What is AI?<|bot|>",
    "<|user|>Tell me a joke<|bot|>",
    "<|user|>How are you?<|bot|>",
    "<|user|>What can you do?<|bot|>",
]

# 1. Inference Speed Benchmark
print("\n=== Inference Speed Benchmark ===")
inference_times = []
tokens_generated = []

for prompt in test_prompts:
    start = time.perf_counter()
    output = generate(prompt, max_new=100)
    elapsed = time.perf_counter() - start
    num_tokens = len(output) - len(prompt)
    inference_times.append(elapsed * 1000)
    tokens_generated.append(num_tokens)
    print(f"Time: {elapsed*1000:.1f}ms | Tokens: {num_tokens}")

# 2. Throughput
print("\n=== Throughput Benchmark ===")
throughputs = []
for i, (t, tok) in enumerate(zip(inference_times, tokens_generated)):
    tps = (tok / t) * 1000 if t > 0 else 0
    throughputs.append(tps)
    print(f"Test {i+1}: {tps:.1f} tokens/sec")

# 3. Memory
print("\n=== Memory Benchmark ===")
model_size_mb = os.path.getsize("tiny_llm.pt") / (1024 * 1024)
param_memory_mb = (total_params * 4) / (1024 * 1024)
print(f"Model file: {model_size_mb:.2f} MB | Param memory: {param_memory_mb:.2f} MB")

# 4. Latency distribution
print("\n=== Latency Distribution (20 runs) ===")
latencies = []
for _ in range(20):
    start = time.perf_counter()
    _ = generate("<|user|>Hi<|bot|>", max_new=50)
    latencies.append((time.perf_counter() - start) * 1000)

avg_latency = sum(latencies) / len(latencies)
min_latency = min(latencies)
max_latency = max(latencies)
print(f"Avg: {avg_latency:.1f}ms | Min: {min_latency:.1f}ms | Max: {max_latency:.1f}ms")

# === Create Charts ===
print("\n=== Generating Charts ===")
fig, axes = plt.subplots(2, 2, figsize=(12, 10))
fig.suptitle("TinyLLM Benchmark Results (626K Parameters)", fontsize=14, fontweight='bold')

# Chart 1: Inference Time
ax1 = axes[0, 0]
ax1.bar(range(1, 6), inference_times, color='#4ECDC4', edgecolor='#2C3E50')
ax1.set_xlabel('Test Prompt')
ax1.set_ylabel('Time (ms)')
ax1.set_title('Inference Time per Prompt')
for i, v in enumerate(inference_times):
    ax1.text(i+1, v + 5, f'{v:.0f}', ha='center', fontsize=9)

# Chart 2: Throughput
ax2 = axes[0, 1]
ax2.bar(range(1, 6), throughputs, color='#FF6B6B', edgecolor='#2C3E50')
ax2.set_xlabel('Test Prompt')
ax2.set_ylabel('Tokens/Second')
ax2.set_title('Generation Throughput')
ax2.axhline(y=sum(throughputs)/len(throughputs), color='#2C3E50', linestyle='--', label=f'Avg: {sum(throughputs)/len(throughputs):.1f}')
ax2.legend()

# Chart 3: Latency Distribution
ax3 = axes[1, 0]
ax3.hist(latencies, bins=10, color='#45B7D1', edgecolor='#2C3E50', alpha=0.8)
ax3.axvline(x=avg_latency, color='#FF6B6B', linestyle='--', linewidth=2, label=f'Mean: {avg_latency:.1f}ms')
ax3.set_xlabel('Latency (ms)')
ax3.set_ylabel('Frequency')
ax3.set_title('Latency Distribution (20 runs)')
ax3.legend()

# Chart 4: Model Stats
ax4 = axes[1, 1]
stats_labels = ['Params\n(K)', 'Size\n(MB)', 'Latency\n(ms)', 'Throughput\n(tok/s)']
stats_values = [total_params/1000, model_size_mb, avg_latency, sum(throughputs)/len(throughputs)]
colors = ['#4ECDC4', '#FF6B6B', '#45B7D1', '#96CEB4']
ax4.bar(stats_labels, stats_values, color=colors, edgecolor='#2C3E50')
ax4.set_title('Model Statistics')
for i, v in enumerate(stats_values):
    ax4.text(i, v + max(stats_values)*0.02, f'{v:.1f}', ha='center', fontsize=10, fontweight='bold')

plt.tight_layout()
plt.savefig('benchmark_results.png', dpi=150, bbox_inches='tight', facecolor='white')
print("Saved: benchmark_results.png")

# Summary
print("\n" + "="*50)
print("BENCHMARK SUMMARY")
print("="*50)
print(f"Parameters:     {total_params:,}")
print(f"Model Size:     {model_size_mb:.2f} MB")
print(f"Avg Latency:    {avg_latency:.1f} ms")
print(f"Avg Throughput: {sum(throughputs)/len(throughputs):.1f} tokens/sec")
print("="*50)
