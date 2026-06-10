"""T4-adapted GPT training for autoresearch.

Single-GPU, 5-minute fixed budget. Uses PyTorch SDPA (no flash-attn3).
Smaller config for T4's 16GB VRAM. Based on karpathy/autoresearch.
"""

import os
os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"
os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"

import gc, math, time
from dataclasses import dataclass, asdict

import torch
import torch.nn as nn
import torch.nn.functional as F

from prepare import MAX_SEQ_LEN, TIME_BUDGET, Tokenizer, make_dataloader, evaluate_bpb

# ── T4-friendly config ───────────────────────────────────────────────────────
ASPECT_RATIO = 32
HEAD_DIM = 64
DEPTH = 3
DEVICE_BATCH_SIZE = 64
TOTAL_BATCH_SIZE = 2**15  # 32K tokens
WINDOW_PATTERN = "L"
WARMUP_RATIO = 0.1
WARMDOWN_RATIO = 0.3
FINAL_LR_FRAC = 0.05
EMBEDDING_LR = 0.25
UNEMBEDDING_LR = 0.005
MATRIX_LR = 0.025
SCALAR_LR = 0.5
WEIGHT_DECAY = 0.12
ADAM_BETAS = (0.8, 0.95)
T4_BF16_PEAK_FLOPS = 65e12  # approximate


# ── GPT Model ────────────────────────────────────────────────────────────────
@dataclass
class GPTConfig:
    sequence_len: int = MAX_SEQ_LEN
    vocab_size: int = 2048
    n_layer: int = DEPTH
    n_head: int = 3
    n_kv_head: int = 3
    n_embd: int = 192
    window_pattern: str = WINDOW_PATTERN


def norm(x):
    return F.rms_norm(x, (x.size(-1),))


def apply_rotary_emb(x, cos, sin):
    d = x.shape[3] // 2
    x1, x2 = x[..., :d], x[..., d:]
    y1 = x1 * cos + x2 * sin
    y2 = x1 * (-sin) + x2 * cos
    return torch.cat([y1, y2], 3)


class CausalSelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.n_head = config.n_head
        self.n_kv_head = config.n_kv_head
        self.n_embd = config.n_embd
        self.head_dim = config.n_embd // config.n_head
        self.c_q = nn.Linear(config.n_embd, config.n_head * self.head_dim, bias=False)
        self.c_k = nn.Linear(config.n_embd, config.n_kv_head * self.head_dim, bias=False)
        self.c_v = nn.Linear(config.n_embd, config.n_kv_head * self.head_dim, bias=False)
        self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=False)

    def forward(self, x, cos_sin):
        B, T, C = x.size()
        q = self.c_q(x).view(B, T, self.n_head, self.head_dim)
        k = self.c_k(x).view(B, T, self.n_kv_head, self.head_dim)
        v = self.c_v(x).view(B, T, self.n_kv_head, self.head_dim)
        cos, sin = cos_sin
        q, k = apply_rotary_emb(q, cos, sin), apply_rotary_emb(k, cos, sin)
        q, k = norm(q), norm(k)

        # PyTorch SDPA (replaces flash-attn3)
        y = F.scaled_dot_product_attention(
            q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2),
            is_causal=True,
        )
        y = y.transpose(1, 2).contiguous().view(B, T, -1)
        return self.c_proj(y)


class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd, bias=False)
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd, bias=False)

    def forward(self, x):
        x = self.c_fc(x)
        x = F.relu(x).square()
        return self.c_proj(x)


class Block(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.attn = CausalSelfAttention(config)
        self.mlp = MLP(config)

    def forward(self, x, cos_sin):
        x = x + self.attn(norm(x), cos_sin)
        x = x + self.mlp(norm(x))
        return x


class GPT(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.transformer = nn.ModuleDict({
            "wte": nn.Embedding(config.vocab_size, config.n_embd),
            "h": nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
        })
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        # Precompute RoPE
        self.rotary_seq_len = config.sequence_len * 10
        head_dim = config.n_embd // config.n_head
        cos, sin = self._precompute_rotary(self.rotary_seq_len, head_dim)
        self.register_buffer("cos", cos, persistent=False)
        self.register_buffer("sin", sin, persistent=False)

    def _precompute_rotary(self, seq_len, head_dim, base=10000):
        device = self.transformer.wte.weight.device
        channel_range = torch.arange(0, head_dim, 2, dtype=torch.float32, device=device)
        inv_freq = 1.0 / (base ** (channel_range / head_dim))
        t = torch.arange(seq_len, dtype=torch.float32, device=device)
        freqs = torch.outer(t, inv_freq)
        cos, sin = freqs.cos().bfloat16(), freqs.sin().bfloat16()
        return cos[None, :, None, :], sin[None, :, None, :]

    @torch.no_grad()
    def init_weights(self):
        s = 3**0.5 * self.config.n_embd**-0.5
        torch.nn.init.normal_(self.transformer.wte.weight, mean=0.0, std=1.0)
        torch.nn.init.normal_(self.lm_head.weight, mean=0.0, std=0.001)
        for block in self.transformer.h:
            torch.nn.init.uniform_(block.attn.c_q.weight, -s, s)
            torch.nn.init.uniform_(block.attn.c_k.weight, -s, s)
            torch.nn.init.uniform_(block.attn.c_v.weight, -s, s)
            torch.nn.init.zeros_(block.attn.c_proj.weight)
            torch.nn.init.uniform_(block.mlp.c_fc.weight, -s, s)
            torch.nn.init.zeros_(block.mlp.c_proj.weight)
        self.transformer.wte.to(dtype=torch.bfloat16)

    def estimate_flops(self):
        nparams = sum(p.numel() for p in self.parameters())
        return 6 * nparams  # approximate

    def num_scaling_params(self):
        wte = sum(p.numel() for p in self.transformer.wte.parameters())
        lm_head = sum(p.numel() for p in self.lm_head.parameters())
        transformer = sum(p.numel() for p in self.transformer.h.parameters())
        return {
            "wte": wte, "lm_head": lm_head,
            "transformer_matrices": transformer,
            "total": wte + lm_head + transformer,
        }

    def setup_optimizer(self):
        model_dim = self.config.n_embd
        dmodel_lr_scale = (model_dim / 768) ** -0.5
        matrix_params = list(self.transformer.h.parameters())
        embedding_params = list(self.transformer.wte.parameters())
        lm_head_params = list(self.lm_head.parameters())

        param_groups = [
            {"kind": "adamw", "params": lm_head_params, "lr": UNEMBEDDING_LR * dmodel_lr_scale,
             "betas": ADAM_BETAS, "eps": 1e-10, "weight_decay": 0.0},
            {"kind": "adamw", "params": embedding_params, "lr": EMBEDDING_LR * dmodel_lr_scale,
             "betas": ADAM_BETAS, "eps": 1e-10, "weight_decay": 0.0},
        ]
        for shape in sorted({p.shape for p in matrix_params}):
            group_params = [p for p in matrix_params if p.shape == shape]
            param_groups.append({
                "kind": "muon", "params": group_params, "lr": MATRIX_LR,
                "momentum": 0.95, "ns_steps": 5, "beta2": 0.95,
                "weight_decay": WEIGHT_DECAY,
            })
        optimizer = MuonAdamW(param_groups)
        for group in optimizer.param_groups:
            group["initial_lr"] = group["lr"]
        return optimizer

    def forward(self, idx, targets=None, reduction="mean"):
        B, T = idx.size()
        cos_sin = (self.cos[:, :T], self.sin[:, :T])
        x = self.transformer.wte(idx)
        x = norm(x)
        for block in self.transformer.h:
            x = block(x, cos_sin)
        x = norm(x)

        softcap = 15
        logits = self.lm_head(x).float()
        logits = softcap * torch.tanh(logits / softcap)

        if targets is not None:
            return F.cross_entropy(
                logits.view(-1, logits.size(-1)), targets.view(-1),
                ignore_index=-1, reduction=reduction,
            )
        return logits


# ── MuonAdamW Optimizer ──────────────────────────────────────────────────────
polar_express_coeffs = [
    (8.156554524902461, -22.48329292557795, 15.878769915207462),
    (4.042929935166739, -2.808917465908714, 0.5000178451051316),
    (3.8916678022926607, -2.772484153217685, 0.5060648178503393),
    (3.285753657755655, -2.3681294933425376, 0.46449024233003106),
    (2.3465413258596377, -1.7097828382687081, 0.42323551169305323),
]

@torch.compile(dynamic=False, fullgraph=True)
def adamw_step(p, grad, exp_avg, exp_avg_sq, step_t, lr_t, beta1_t, beta2_t, eps_t, wd_t):
    p.mul_(1 - lr_t * wd_t)
    exp_avg.lerp_(grad, 1 - beta1_t)
    exp_avg_sq.lerp_(grad.square(), 1 - beta2_t)
    bias1 = 1 - beta1_t ** step_t
    bias2 = 1 - beta2_t ** step_t
    denom = (exp_avg_sq / bias2).sqrt() + eps_t
    p.add_(exp_avg / denom, alpha=-lr_t / bias1)

@torch.compile(dynamic=False, fullgraph=True)
def muon_step(stacked_grads, stacked_params, momentum_buffer, second_momentum_buffer,
               momentum_t, lr_t, wd_t, beta2_t, ns_steps, red_dim):
    momentum = momentum_t.to(stacked_grads.dtype)
    momentum_buffer.lerp_(stacked_grads, 1 - momentum)
    g = stacked_grads.lerp_(momentum_buffer, momentum)
    X = g.bfloat16()
    X = X / (X.norm(dim=(-2, -1), keepdim=True) * 1.02 + 1e-6)
    if g.size(-2) > g.size(-1):
        for a, b, c in polar_express_coeffs[:ns_steps]:
            A = X.mT @ X
            B = b * A + c * (A @ A)
            X = a * X + X @ B
    else:
        for a, b, c in polar_express_coeffs[:ns_steps]:
            A = X @ X.mT
            B = b * A + c * (A @ A)
            X = a * X + B @ X
    g = X
    beta2 = beta2_t.to(g.dtype)
    v_mean = g.float().square().mean(dim=red_dim, keepdim=True)
    red_dim_size = g.size(red_dim)
    v_norm_sq = v_mean.sum(dim=(-2, -1), keepdim=True) * red_dim_size
    v_norm = v_norm_sq.sqrt()
    second_momentum_buffer.lerp_(v_mean.to(dtype=second_momentum_buffer.dtype), 1 - beta2)
    step_size = second_momentum_buffer.clamp_min(1e-10).rsqrt()
    scaled_sq_sum = (v_mean * red_dim_size) * step_size.float().square()
    v_norm_new = scaled_sq_sum.sum(dim=(-2, -1), keepdim=True).sqrt()
    final_scale = step_size * (v_norm / v_norm_new.clamp_min(1e-10))
    g = g * final_scale.to(g.dtype)
    lr = lr_t.to(g.dtype)
    wd = wd_t.to(g.dtype)
    mask = (g * stacked_params) >= 0
    stacked_params.sub_(lr * g + lr * wd * stacked_params * mask)


class MuonAdamW(torch.optim.Optimizer):
    def __init__(self, param_groups):
        super().__init__(param_groups, defaults={})
        self._adamw_step_t = torch.tensor(0.0, dtype=torch.float32, device="cpu")
        self._adamw_lr_t = torch.tensor(0.0, dtype=torch.float32, device="cpu")
        self._adamw_beta1_t = torch.tensor(0.0, dtype=torch.float32, device="cpu")
        self._adamw_beta2_t = torch.tensor(0.0, dtype=torch.float32, device="cpu")
        self._adamw_eps_t = torch.tensor(0.0, dtype=torch.float32, device="cpu")
        self._adamw_wd_t = torch.tensor(0.0, dtype=torch.float32, device="cpu")
        self._muon_momentum_t = torch.tensor(0.0, dtype=torch.float32, device="cpu")
        self._muon_lr_t = torch.tensor(0.0, dtype=torch.float32, device="cpu")
        self._muon_wd_t = torch.tensor(0.0, dtype=torch.float32, device="cpu")
        self._muon_beta2_t = torch.tensor(0.0, dtype=torch.float32, device="cpu")

    def _step_adamw(self, group):
        for p in group["params"]:
            if p.grad is None:
                continue
            state = self.state[p]
            if not state:
                state["step"] = 0
                state["exp_avg"] = torch.zeros_like(p)
                state["exp_avg_sq"] = torch.zeros_like(p)
            state["step"] += 1
            self._adamw_step_t.fill_(state["step"])
            self._adamw_lr_t.fill_(group["lr"])
            self._adamw_beta1_t.fill_(group["betas"][0])
            self._adamw_beta2_t.fill_(group["betas"][1])
            self._adamw_eps_t.fill_(group["eps"])
            self._adamw_wd_t.fill_(group.get("weight_decay", 0))
            adamw_step(p, p.grad, state["exp_avg"], state["exp_avg_sq"],
                       self._adamw_step_t, self._adamw_lr_t, self._adamw_beta1_t,
                       self._adamw_beta2_t, self._adamw_eps_t, self._adamw_wd_t)

    def _step_muon(self, group):
        params = group["params"]
        if not params:
            return
        p = params[0]
        state = self.state[p]
        shape, device, dtype = p.shape, p.device, p.dtype
        num_params = len(params)
        if "momentum_buffer" not in state:
            state["momentum_buffer"] = torch.zeros(num_params, *shape, dtype=dtype, device=device)
        if "second_momentum_buffer" not in state:
            state_dims = (num_params, shape[-2], 1) if shape[-2] >= shape[-1] else (num_params, 1, shape[-1])
            state["second_momentum_buffer"] = torch.zeros(state_dims, dtype=dtype, device=device)
        red_dim = -1 if shape[-2] >= shape[-1] else -2
        stacked_grads = torch.stack([p.grad for p in params])
        stacked_params = torch.stack(params)
        self._muon_momentum_t.fill_(group["momentum"])
        self._muon_beta2_t.fill_(group.get("beta2", 0.0))
        self._muon_lr_t.fill_(group["lr"] * max(1.0, shape[-2] / shape[-1])**0.5)
        self._muon_wd_t.fill_(group.get("weight_decay", 0))
        muon_step(stacked_grads, stacked_params, state["momentum_buffer"],
                  state["second_momentum_buffer"], self._muon_momentum_t,
                  self._muon_lr_t, self._muon_wd_t, self._muon_beta2_t,
                  group["ns_steps"], red_dim)
        torch._foreach_copy_(params, list(stacked_params.unbind(0)))

    @torch.no_grad()
    def step(self):
        for group in self.param_groups:
            if group["kind"] == "adamw":
                self._step_adamw(group)
            elif group["kind"] == "muon":
                self._step_muon(group)


# ── Schedules ────────────────────────────────────────────────────────────────
def get_lr_multiplier(progress):
    if progress < WARMUP_RATIO:
        return progress / WARMUP_RATIO if WARMUP_RATIO > 0 else 1.0
    elif progress < 1.0 - WARMDOWN_RATIO:
        return 1.0
    else:
        cooldown = (1.0 - progress) / WARMDOWN_RATIO
        return cooldown * 1.0 + (1 - cooldown) * FINAL_LR_FRAC


def get_muon_momentum(step):
    frac = min(step / 300, 1)
    return (1 - frac) * 0.85 + frac * 0.95


# ── Main ─────────────────────────────────────────────────────────────────────
def main():
    t_start = time.time()
    torch.manual_seed(42)
    torch.cuda.manual_seed(42)
    torch.set_float32_matmul_precision("high")
    device = torch.device("cuda")
    autocast_ctx = torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16)

    tokenizer = Tokenizer.from_directory()
    vocab_size = tokenizer.get_vocab_size()
    print(f"Vocab size: {vocab_size:,}")

    config = GPTConfig(
        sequence_len=MAX_SEQ_LEN, vocab_size=vocab_size,
        n_layer=DEPTH, n_head=3, n_kv_head=3, n_embd=192,
        window_pattern=WINDOW_PATTERN,
    )
    print(f"Model config: {asdict(config)}")

    with torch.device("meta"):
        model = GPT(config)
    model.to_empty(device=device)
    model.init_weights()

    param_counts = model.num_scaling_params()
    print("Parameters:")
    for key, value in param_counts.items():
        print(f"  {key:24s}: {value:,}")
    num_params = param_counts["total"]
    num_flops_per_token = model.estimate_flops()

    tokens_per_fwdbwd = DEVICE_BATCH_SIZE * MAX_SEQ_LEN
    grad_accum_steps = TOTAL_BATCH_SIZE // tokens_per_fwdbwd

    optimizer = model.setup_optimizer()
    train_loader = make_dataloader(tokenizer, DEVICE_BATCH_SIZE, MAX_SEQ_LEN, "train")
    x, y, epoch = next(train_loader)

    print(f"Time budget: {TIME_BUDGET}s")
    print(f"Grad accum steps: {grad_accum_steps}")
    print(f"Tokens/fwdbwd: {tokens_per_fwdbwd:,}")

    # Warmup (excluded from timer)
    with autocast_ctx:
        loss = model(x, y)
    loss.backward()
    optimizer.zero_grad(set_to_none=True)
    gc.collect()
    gc.freeze()
    gc.disable()

    t_start_training = time.time()
    smooth_train_loss = 0
    total_training_time = 0
    step = 0
    history = []  # collect step-level metrics

    while True:
        torch.cuda.synchronize()
        t0 = time.time()

        for micro_step in range(grad_accum_steps):
            with autocast_ctx:
                loss = model(x, y)
            train_loss = loss.detach()
            (loss / grad_accum_steps).backward()
            x, y, epoch = next(train_loader)

        progress = min(total_training_time / TIME_BUDGET, 1.0)
        lrm = get_lr_multiplier(progress)
        muon_momentum = get_muon_momentum(step)
        muon_wd = WEIGHT_DECAY * (1 - progress)

        for group in optimizer.param_groups:
            group["lr"] = group["initial_lr"] * lrm
            if group["kind"] == "muon":
                group["momentum"] = muon_momentum
                group["weight_decay"] = muon_wd

        optimizer.step()
        model.zero_grad(set_to_none=True)

        train_loss_f = train_loss.item()
        if math.isnan(train_loss_f) or train_loss_f > 100:
            print("FAIL (loss exploded)")
            break

        torch.cuda.synchronize()
        dt = time.time() - t0

        if step > 5:
            total_training_time += dt

        ema_beta = 0.9
        smooth_train_loss = ema_beta * smooth_train_loss + (1 - ema_beta) * train_loss_f
        debiased = smooth_train_loss / (1 - ema_beta**(step + 1))
        tok_per_sec = int(TOTAL_BATCH_SIZE / dt) if dt > 0 else 0
        remaining = max(0, TIME_BUDGET - total_training_time)

        print(f"\rstep {step:05d} ({100*progress:.1f}%) | loss: {debiased:.6f} | "
              f"dt: {dt*1000:.0f}ms | tok/s: {tok_per_sec:,} | "
              f"remaining: {remaining:.0f}s    ", end="", flush=True)

        history.append({
            "step": step, "loss": round(debiased, 6),
            "lrm": round(lrm, 4), "tok_per_sec": tok_per_sec,
            "training_time": round(total_training_time, 1),
        })

        step += 1
        if step > 10 and total_training_time >= TIME_BUDGET:
            break

    print()

    # Final eval
    model.eval()
    with autocast_ctx:
        val_bpb = evaluate_bpb(model, tokenizer, DEVICE_BATCH_SIZE)

    total_tokens = step * TOTAL_BATCH_SIZE
    t_end = time.time()
    peak_vram = torch.cuda.max_memory_allocated() / 1024 / 1024

    print("---")
    print(f"val_bpb:          {val_bpb:.6f}")
    print(f"training_seconds: {total_training_time:.1f}")
    print(f"total_seconds:    {t_end - t_start:.1f}")
    print(f"peak_vram_mb:     {peak_vram:.1f}")
    print(f"total_tokens_M:   {total_tokens / 1e6:.1f}")
    print(f"num_steps:        {step}")
    print(f"num_params_M:     {num_params / 1e6:.1f}")

    # Sample generation
    print("\n── Sample generation ──")
    prompt = "Once upon a time"
    prompt_ids = tokenizer.encode(prompt, prepend=tokenizer.get_bos_token_id())
    x = torch.tensor([prompt_ids], device=device)
    with autocast_ctx:
        for _ in range(100):
            logits = model(x[:, -MAX_SEQ_LEN:])
            next_id = torch.multinomial(
                torch.softmax(logits[0, -1] / 0.8, dim=-1), 1,
            )
            x = torch.cat([x, next_id.unsqueeze(0)], dim=1)
    print(tokenizer.decode(x[0].tolist()))
    print()

    # Save outputs
    import json
    out_dir = "/content/autoresearch-output"
    os.makedirs(out_dir, exist_ok=True)

    torch.save({
        "model": model.state_dict(),
        "config": asdict(config),
        "val_bpb": val_bpb,
    }, os.path.join(out_dir, "model.pt"))

    with open(os.path.join(out_dir, "metrics.json"), "w") as f:
        json.dump({
            "val_bpb": round(val_bpb, 6),
            "num_params": num_params,
            "total_tokens": total_tokens,
            "num_steps": step,
            "peak_vram_mb": round(peak_vram, 1),
            "training_seconds": round(total_training_time, 1),
            "config": asdict(config),
            "history": history,
        }, f, indent=2)

    print(f"Output saved to {out_dir}/")


if __name__ == "__main__":
    main()
