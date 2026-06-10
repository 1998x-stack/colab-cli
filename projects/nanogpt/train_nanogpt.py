"""
Self-contained nanoGPT training script for Colab GPU.
Downloads Tiny Shakespeare, trains a character-level GPT, generates loss plots + sample text.
"""
import os, sys, time, math, json, pickle, requests, inspect
import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ──────────────────────────────────────────────────────────────────────
# 1. GPT Model (from karpathy/nanoGPT model.py)
# ──────────────────────────────────────────────────────────────────────

class LayerNorm(nn.Module):
    def __init__(self, ndim, bias):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(ndim))
        self.bias = nn.Parameter(torch.zeros(ndim)) if bias else None

    def forward(self, input):
        return F.layer_norm(input, self.weight.shape, self.weight, self.bias, 1e-5)

class CausalSelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd, bias=config.bias)
        self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.dropout = config.dropout
        self.flash = hasattr(torch.nn.functional, "scaled_dot_product_attention")
        if not self.flash:
            self.register_buffer("bias", torch.tril(torch.ones(config.block_size, config.block_size))
                                 .view(1, 1, config.block_size, config.block_size))

    def forward(self, x):
        B, T, C = x.size()
        q, k, v = self.c_attn(x).split(self.n_embd, dim=2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        if self.flash:
            y = F.scaled_dot_product_attention(q, k, v, attn_mask=None,
                  dropout_p=self.dropout if self.training else 0, is_causal=True)
        else:
            att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
            att = att.masked_fill(self.bias[:, :, :T, :T] == 0, float("-inf"))
            att = F.softmax(att, dim=-1)
            att = self.attn_dropout(att)
            y = att @ v
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        y = self.resid_dropout(self.c_proj(y))
        return y

class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd, bias=config.bias)
        self.gelu = nn.GELU()
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd, bias=config.bias)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        x = self.dropout(x)
        return x

class Block(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.ln_1 = LayerNorm(config.n_embd, bias=config.bias)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = LayerNorm(config.n_embd, bias=config.bias)
        self.mlp = MLP(config)

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x

class GPTConfig:
    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)

class GPT(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.transformer = nn.ModuleDict(dict(
            wte=nn.Embedding(config.vocab_size, config.n_embd),
            wpe=nn.Embedding(config.block_size, config.n_embd),
            drop=nn.Dropout(config.dropout),
            h=nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
            ln_f=LayerNorm(config.n_embd, bias=config.bias),
        ))
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        self.transformer.wte.weight = self.lm_head.weight
        self.apply(self._init_weights)
        for pn, p in self.named_parameters():
            if pn.endswith("c_proj.weight"):
                torch.nn.init.normal_(p, mean=0.0, std=0.02 / math.sqrt(2 * config.n_layer))
        n_params = sum(p.numel() for p in self.parameters())
        print(f"number of parameters: {n_params/1e6:.2f}M")

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def get_num_params(self, non_embedding=True):
        n_params = sum(p.numel() for p in self.parameters())
        if non_embedding:
            n_params -= self.transformer.wpe.weight.numel()
        return n_params

    def configure_optimizers(self, weight_decay, learning_rate, betas, device_type):
        param_dict = {pn: p for pn, p in self.named_parameters()}
        param_dict = {pn: p for pn, p in param_dict.items() if p.requires_grad}
        decay_params = [p for n, p in param_dict.items() if p.dim() >= 2]
        nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2]
        optim_groups = [
            {"params": decay_params, "weight_decay": weight_decay},
            {"params": nodecay_params, "weight_decay": 0.0},
        ]
        num_decay_params = sum(p.numel() for p in decay_params)
        num_nodecay_params = sum(p.numel() for p in nodecay_params)
        print(f"num decayed parameter tensors: {len(decay_params)}, with {num_decay_params:,} parameters")
        print(f"num non-decayed parameter tensors: {len(nodecay_params)}, with {num_nodecay_params:,} parameters")
        fused_available = "fused" in inspect.signature(torch.optim.AdamW).parameters
        use_fused = fused_available and device_type == "cuda"
        extra_args = dict(fused=True) if use_fused else dict()
        optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate, betas=betas, **extra_args)
        print(f"using fused AdamW: {use_fused}")
        return optimizer

    def estimate_mfu(self, fwdbwd_per_iter, dt):
        N = self.get_num_params()
        cfg = self.config
        L, H, Q, T = cfg.n_layer, cfg.n_head, cfg.n_embd // cfg.n_head, cfg.block_size
        flops_per_token = 6 * N + 12 * L * H * Q * T
        flops_per_fwdbwd = flops_per_token * T
        flops_per_iter = flops_per_fwdbwd * fwdbwd_per_iter
        flops_achieved = flops_per_iter * (1.0 / dt)
        flops_promised = 312e12  # A100 bfloat16 peak flops
        mfu = flops_achieved / flops_promised
        return mfu

    def forward(self, idx, targets=None):
        device = idx.device
        b, t = idx.size()
        pos = torch.arange(0, t, dtype=torch.long, device=device)
        tok_emb = self.transformer.wte(idx)
        pos_emb = self.transformer.wpe(pos)
        x = self.transformer.drop(tok_emb + pos_emb)
        for block in self.transformer.h:
            x = block(x)
        x = self.transformer.ln_f(x)
        if targets is not None:
            logits = self.lm_head(x)
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1)
        else:
            logits = self.lm_head(x[:, [-1], :])
            loss = None
        return logits, loss

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None):
        for _ in range(max_new_tokens):
            idx_cond = idx if idx.size(1) <= self.config.block_size else idx[:, -self.config.block_size:]
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :] / temperature
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float("Inf")
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)
        return idx

# ──────────────────────────────────────────────────────────────────────
# 2. Data preparation
# ──────────────────────────────────────────────────────────────────────

DATA_DIR = "/content/shakespeare_char"
os.makedirs(DATA_DIR, exist_ok=True)
input_path = os.path.join(DATA_DIR, "input.txt")

if not os.path.exists(input_path):
    print("Downloading Tiny Shakespeare dataset...")
    url = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
    with open(input_path, "w") as f:
        f.write(requests.get(url).text)

with open(input_path) as f:
    data = f.read()
print(f"Dataset length: {len(data):,} characters")

chars = sorted(list(set(data)))
vocab_size = len(chars)
print(f"Vocab size: {vocab_size}")

stoi = {ch: i for i, ch in enumerate(chars)}
itos = {i: ch for i, ch in enumerate(chars)}

def encode(s):
    return [stoi[c] for c in s]

def decode(ids):
    return "".join([itos[i] for i in ids])

n = len(data)
train_ids = np.array(encode(data[: int(n * 0.9)]), dtype=np.uint16)
val_ids = np.array(encode(data[int(n * 0.9):]), dtype=np.uint16)
print(f"Train tokens: {len(train_ids):,}, Val tokens: {len(val_ids):,}")

train_ids.tofile(os.path.join(DATA_DIR, "train.bin"))
val_ids.tofile(os.path.join(DATA_DIR, "val.bin"))
with open(os.path.join(DATA_DIR, "meta.pkl"), "wb") as f:
    pickle.dump({"vocab_size": vocab_size, "itos": itos, "stoi": stoi}, f)

# ──────────────────────────────────────────────────────────────────────
# 3. Training config
# ──────────────────────────────────────────────────────────────────────

out_dir = "/content/out-nanogpt"
os.makedirs(out_dir, exist_ok=True)

eval_interval = 100
eval_iters = 80
log_interval = 10

batch_size = 64
block_size = 256
n_layer = 6
n_head = 6
n_embd = 384
dropout = 0.2
bias = False

learning_rate = 1e-3
max_iters = 500
warmup_iters = 50
lr_decay_iters = 500
min_lr = 1e-4
beta1 = 0.9
beta2 = 0.99
weight_decay = 1e-1
grad_clip = 1.0

device = "cuda" if torch.cuda.is_available() else "cpu"
dtype = "bfloat16" if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else "float16"
compile_model = True

print(f"Device: {device}, dtype: {dtype}")
print(f"max_iters={max_iters}, batch_size={batch_size}, block_size={block_size}")

# ──────────────────────────────────────────────────────────────────────
# 4. Training loop
# ──────────────────────────────────────────────────────────────────────

torch.manual_seed(1337)
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

ptdtype = {"float32": torch.float32, "bfloat16": torch.bfloat16, "float16": torch.float16}[dtype]
ctx = torch.amp.autocast(device_type="cuda", dtype=ptdtype) if device == "cuda" else torch.no_grad()

def get_batch(split):
    fname = os.path.join(DATA_DIR, f"{split}.bin")
    data_arr = np.memmap(fname, dtype=np.uint16, mode="r")
    ix = torch.randint(len(data_arr) - block_size, (batch_size,))
    x = torch.stack([torch.from_numpy((data_arr[i : i + block_size]).astype(np.int64)) for i in ix])
    y = torch.stack([torch.from_numpy((data_arr[i + 1 : i + 1 + block_size]).astype(np.int64)) for i in ix])
    if device == "cuda":
        x, y = x.pin_memory().to(device, non_blocking=True), y.pin_memory().to(device, non_blocking=True)
    else:
        x, y = x.to(device), y.to(device)
    return x, y

model_args = dict(n_layer=n_layer, n_head=n_head, n_embd=n_embd, block_size=block_size,
                  bias=bias, vocab_size=vocab_size, dropout=dropout)
gptconf = GPTConfig(**model_args)
model = GPT(gptconf)
model.to(device)

scaler = torch.amp.GradScaler("cuda", enabled=(dtype == "float16"))
optimizer = model.configure_optimizers(weight_decay, learning_rate, (beta1, beta2), "cuda" if device == "cuda" else "cpu")

if compile_model:
    print("Compiling model... (may take ~30s)")
    model = torch.compile(model)

@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    for split in ["train", "val"]:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            with ctx:
                _, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean().item()
    model.train()
    return out

def get_lr(it):
    if it < warmup_iters:
        return learning_rate * (it + 1) / (warmup_iters + 1)
    if it > lr_decay_iters:
        return min_lr
    decay_ratio = (it - warmup_iters) / (lr_decay_iters - warmup_iters)
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return min_lr + coeff * (learning_rate - min_lr)

# Training state
metrics = {"iter": [], "train_loss": [], "val_loss": [], "lr": [], "dt_ms": [], "mfu": []}
iter_num = 0
best_val_loss = 1e9
running_mfu = -1.0
X, Y = get_batch("train")
t0 = time.time()

print("\n=== Starting training ===\n")
while True:
    lr = get_lr(iter_num) if lr_decay_iters > 0 else learning_rate
    for param_group in optimizer.param_groups:
        param_group["lr"] = lr

    if iter_num % eval_interval == 0:
        losses = estimate_loss()
        dt_eval = (time.time() - t0) * 1000
        print(f"step {iter_num:5d}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}  "
              f"[{dt_eval:.0f}ms]")
        metrics["iter"].append(iter_num)
        metrics["train_loss"].append(losses["train"])
        metrics["val_loss"].append(losses["val"])
        metrics["lr"].append(lr)
        metrics["dt_ms"].append(dt_eval)
        metrics["mfu"].append(running_mfu * 100)

        if losses["val"] < best_val_loss:
            best_val_loss = losses["val"]
            checkpoint = {
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "model_args": model_args,
                "iter_num": iter_num,
                "best_val_loss": best_val_loss,
            }
            torch.save(checkpoint, os.path.join(out_dir, "ckpt.pt"))
            print(f"  -> saved checkpoint (best val loss: {best_val_loss:.4f})")
        t0 = time.time()

    for micro_step in range(1):
        with ctx:
            logits, loss = model(X, Y)
        X, Y = get_batch("train")
        scaler.scale(loss).backward()

    if grad_clip != 0.0:
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
    scaler.step(optimizer)
    scaler.update()
    optimizer.zero_grad(set_to_none=True)

    t1 = time.time()
    dt = t1 - t0
    t0 = t1
    if iter_num % log_interval == 0:
        lossf = loss.item()
        if iter_num >= 5:
            mfu = model.estimate_mfu(batch_size, dt)
            running_mfu = mfu if running_mfu == -1.0 else 0.9 * running_mfu + 0.1 * mfu
        print(f"iter {iter_num:5d}: loss {lossf:.4f}, dt {dt*1000:.2f}ms, mfu {running_mfu*100:.2f}%")

    iter_num += 1
    if iter_num > max_iters:
        break

print(f"\n=== Training complete! Best val loss: {best_val_loss:.4f} ===\n")

# Save metrics
metrics_path = os.path.join(out_dir, "metrics.json")
with open(metrics_path, "w") as f:
    json.dump(metrics, f)
print(f"Metrics saved to {metrics_path}")

# ──────────────────────────────────────────────────────────────────────
# 5. Visualizations
# ──────────────────────────────────────────────────────────────────────

print("\nGenerating plots...")

# Loss curves
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

ax = axes[0]
ax.plot(metrics["iter"], metrics["train_loss"], "b-o", markersize=4, label="Train Loss")
ax.plot(metrics["iter"], metrics["val_loss"], "r-o", markersize=4, label="Val Loss")
ax.set_xlabel("Iteration")
ax.set_ylabel("Cross-Entropy Loss")
ax.set_title("nanoGPT — Shakespeare Character-Level Training Loss")
ax.legend()
ax.grid(True, alpha=0.3)

ax = axes[1]
ax.plot(metrics["iter"], metrics["lr"], "g-s", markersize=4)
ax.set_xlabel("Iteration")
ax.set_ylabel("Learning Rate")
ax.set_title("Learning Rate Schedule (Cosine Decay + Warmup)")
ax.grid(True, alpha=0.3)

plt.tight_layout()
loss_plot_path = os.path.join(out_dir, "loss_curve.png")
fig.savefig(loss_plot_path, dpi=120)
print(f"Loss plot saved to {loss_plot_path}")

# Iteration time
fig, ax = plt.subplots(figsize=(10, 4))
ax.bar(metrics["iter"], metrics["dt_ms"], width=30, color="steelblue", edgecolor="white")
ax.set_xlabel("Iteration")
ax.set_ylabel("Time (ms)")
ax.set_title("Evaluation Time per Checkpoint")
ax.grid(True, alpha=0.3, axis="y")
plt.tight_layout()
time_plot_path = os.path.join(out_dir, "time_plot.png")
fig.savefig(time_plot_path, dpi=120)
print(f"Time plot saved to {time_plot_path}")

# ──────────────────────────────────────────────────────────────────────
# 6. Sample generation
# ──────────────────────────────────────────────────────────────────────

print("\nGenerating sample text...")
model.eval()
context = torch.tensor([encode("\n")], dtype=torch.long, device=device)
samples = {}
for temp in [0.7, 1.0, 1.2]:
    with torch.no_grad():
        with ctx:
            output = model.generate(context, max_new_tokens=500, temperature=temp, top_k=None)
    samples[f"temp_{temp}"] = decode(output[0].tolist())

samples_path = os.path.join(out_dir, "samples.json")
with open(samples_path, "w") as f:
    json.dump(samples, f)
print(f"Samples saved to {samples_path}")

# Print samples
for temp_key, text in samples.items():
    print(f"\n{'='*60}")
    print(f"  Generated sample (temperature={temp_key.split('_')[1]})")
    print(f"{'='*60}")
    print(text[:600])
    print("...")

print("\n=== All done! ===")
print(f"Outputs in {out_dir}/: ckpt.pt, metrics.json, loss_curve.png, time_plot.png, samples.json")
