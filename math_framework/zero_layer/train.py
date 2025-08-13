"""
Training script for the Zero-Layer Transformer
"""

import torch
from torch import optim
from pathlib import Path

from model import ZeroLayerTransformer, ZLTConfig

# Data
text_path = Path("math_framework/zero_layer/data/tiny.txt")
if not text_path.exists():
    text_path.write_text("123456789" * 1000)

text = text_path.read_text()
chars = sorted(list(set(text)))
stoi = {ch:i for i,ch in enumerate(chars)}
itos = {i:ch for i,ch in enumerate(chars)}
def encode(s): return torch.tensor([stoi[c] for c in s], dtype=torch.long)
def decode(ix): return "".join(itos[i] for i in ix)

data = encode(text)

# Config / model / device
device = "mps"
block_size = 64
vocab_size = len(chars)
config = ZLTConfig(
    block_size=block_size,
    vocab_size=vocab_size,
    n_embd=128
)
model = ZeroLayerTransformer(config).to(device)

# Batching
def get_batch(
    data,
    batch_size,
    block_size,
    device
):
    ix = torch.randint(0, len(data) - block_size - 1, (batch_size,))
    x = torch.stack([data[i:i+block_size] for i in ix]).to(device)
    y = torch.stack([data[i+1:i+block_size+1] for i in ix]).to(device)
    return x, y

# Optimizer
optimizer = optim.AdamW(model.parameters(), lr=1e-3)

# Train loop
model.train()
steps = 2000
batch_size = 64
for step in range(steps):
    x, y = get_batch(
        data,
        batch_size,
        block_size,
        device
    )
    logits, loss = model(x, y)
    optimizer.zero_grad(
        set_to_none=True
    )
    loss.backward()
    optimizer.step()
    if step % 100 == 0:
        print(f"step {step} loss {loss.item():.4f}")

# Sample
model.eval()
start = "1"
ctx = encode(start).unsqueeze(0).to(device)
out = model.generate(
    ctx,
    max_new_tokens=200,
    temperature=1.0,
    top_k=None
)[0].tolist()
print(decode(out))