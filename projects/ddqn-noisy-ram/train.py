#!/usr/bin/env python3
"""DDQN vs NoisyNet — 3 discrete-action Gymnasium environments.

Fixes from v1: LayerNorm, per-env hyperparams, Prioritized Replay for sparse envs.
CartPole-v1, MountainCar-v0, Acrobot-v1. No CNN — vector observations.
"""

import os
import sys
import json
import signal
import argparse
from datetime import datetime
from collections import deque

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import gymnasium as gym

# ── Per-environment configs ─────────────────────────────────────────────────
ENV_CONFIGS = {
    "CartPole-v1":     {"episodes": 300, "lr": 5e-3, "target_update": 50,  "warmup": 500,  "use_per": False},
    "MountainCar-v0":  {"episodes": 500, "lr": 1e-3, "target_update": 100, "warmup": 2000, "use_per": True},
    "Acrobot-v1":      {"episodes": 500, "lr": 1e-3, "target_update": 100, "warmup": 2000, "use_per": True},
}

parser = argparse.ArgumentParser()
parser.add_argument("--envs", nargs="+", default=["CartPole-v1", "MountainCar-v0", "Acrobot-v1"])
parser.add_argument("--algos", nargs="+", default=["ddqn", "noisy"])
parser.add_argument("--max_steps", type=int, default=200)
parser.add_argument("--buffer_size", type=int, default=50_000)
parser.add_argument("--batch_size", type=int, default=128)
parser.add_argument("--gamma", type=float, default=0.99)
parser.add_argument("--tau", type=float, default=0.005)
parser.add_argument("--eps_start", type=float, default=1.0)
parser.add_argument("--eps_end", type=float, default=0.02)
parser.add_argument("--eval_interval", type=int, default=25)
parser.add_argument("--eval_episodes", type=int, default=5)
parser.add_argument("--per_alpha", type=float, default=0.6)
parser.add_argument("--per_beta", type=float, default=0.4)
parser.add_argument("--per_eps", type=float, default=1e-6)
parser.add_argument("--out_dir", default="/content/ddqn-noisy-output")
parser.add_argument("--seed", type=int, default=42)
args = parser.parse_args()

os.makedirs(args.out_dir, exist_ok=True)
os.makedirs(f"{args.out_dir}/plots", exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ── Logging ─────────────────────────────────────────────────────────────────
class Tee:
    def __init__(self, path):
        self.file = open(path, "a", buffering=1)
    def write(self, s):
        sys.__stdout__.write(s)
        self.file.write(s)
        self.file.flush()
    def flush(self):
        sys.__stdout__.flush()
        self.file.flush()

log = Tee(f"{args.out_dir}/train.log")
def log_print(*a, **kw): print(*a, **kw, file=log)
log_print(f"=== DDQN vs NoisyNet v2 | device={device} | {datetime.now()} ===")

# ── Seed ────────────────────────────────────────────────────────────────────
def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

# ── Noisy Linear ────────────────────────────────────────────────────────────
class NoisyLinear(nn.Module):
    def __init__(self, in_f, out_f, sigma=0.5):
        super().__init__()
        self.in_f, self.out_f = in_f, out_f
        self.mu_w = nn.Parameter(torch.empty(out_f, in_f))
        self.mu_b = nn.Parameter(torch.empty(out_f))
        self.sigma_w = nn.Parameter(torch.empty(out_f, in_f))
        self.sigma_b = nn.Parameter(torch.empty(out_f))
        bound = 1.0 / np.sqrt(in_f)
        nn.init.uniform_(self.mu_w, -bound, bound)
        nn.init.uniform_(self.mu_b, -bound, bound)
        nn.init.constant_(self.sigma_w, sigma / np.sqrt(in_f))
        nn.init.constant_(self.sigma_b, sigma / np.sqrt(out_f))
        self.register_buffer("eps_w", torch.empty(out_f, in_f))
        self.register_buffer("eps_b", torch.empty(out_f))
        self.sample_noise()

    def _noise(self, size_in, size_out):
        e_in = torch.randn(1, size_in).sign() * torch.sqrt(torch.randn(1, size_in).abs())
        e_out = torch.randn(size_out, 1).sign() * torch.sqrt(torch.randn(size_out, 1).abs())
        return e_out @ e_in

    def sample_noise(self):
        self.eps_w.copy_(self._noise(self.in_f, self.out_f))
        self.eps_b.copy_(torch.randn(self.out_f).sign() * torch.sqrt(torch.randn(self.out_f).abs()))

    def forward(self, x):
        return F.linear(x, self.mu_w + self.sigma_w * self.eps_w,
                       self.mu_b + self.sigma_b * self.eps_b)


# ── DQN Network (with LayerNorm) ────────────────────────────────────────────
class DQN(nn.Module):
    def __init__(self, obs_dim, n_actions):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, 128),
            nn.LayerNorm(128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.LayerNorm(64),
            nn.ReLU(),
            nn.Linear(64, n_actions),
        )

    def forward(self, x):
        return self.net(x)


# ── NoisyNet Network (with LayerNorm) ───────────────────────────────────────
class NoisyDQN(nn.Module):
    def __init__(self, obs_dim, n_actions):
        super().__init__()
        self.fc1 = NoisyLinear(obs_dim, 128)
        self.ln1 = nn.LayerNorm(128)
        self.fc2 = NoisyLinear(128, 64)
        self.ln2 = nn.LayerNorm(64)
        self.out = NoisyLinear(64, n_actions)

    def forward(self, x):
        x = F.relu(self.ln1(self.fc1(x)))
        x = F.relu(self.ln2(self.fc2(x)))
        return self.out(x)

    def sample_noise(self):
        for m in self.modules():
            if isinstance(m, NoisyLinear):
                m.sample_noise()


# ── Prioritized Replay Buffer (SumTree) ─────────────────────────────────────
class SumTree:
    def __init__(self, capacity):
        self.capacity = 1
        while self.capacity < capacity:
            self.capacity *= 2
        self.tree = np.zeros(2 * self.capacity, dtype=np.float64)
        self.pos = 0

    def add(self, priority):
        idx = self.pos + self.capacity
        self.tree[idx] = priority
        self._propagate(idx, priority)
        self.pos = (self.pos + 1) % self.capacity

    def update(self, idx, priority):
        idx += self.capacity
        self.tree[idx] = priority
        self._propagate(idx, priority)

    def _propagate(self, idx, change):
        while idx > 1:
            idx //= 2
            self.tree[idx] = self.tree[2 * idx] + self.tree[2 * idx + 1]

    def sample(self, value):
        idx = 1
        while idx < self.capacity:
            left = 2 * idx
            if value <= self.tree[left]:
                idx = left
            else:
                value -= self.tree[left]
                idx = left + 1
        return idx - self.capacity

    def total(self):
        return self.tree[1]


class PrioritizedReplay:
    def __init__(self, cap, alpha=0.6, beta=0.4, eps=1e-6):
        self.buf = deque(maxlen=cap)
        self.tree = SumTree(cap)
        self.alpha, self.beta, self.eps = alpha, beta, eps
        self.max_prio = 1.0
        self.cap = cap

    def push(self, s, a, r, ns, d):
        self.buf.append((s, a, r, ns, d))
        self.tree.add(self.max_prio)  # give new transitions max priority

    def sample(self, n, step, total_steps):
        total = self.tree.total()
        if total <= 0:
            total = 1.0
        segment = total / n
        idxs, weights = [], []
        beta = min(1.0, self.beta + (1.0 - self.beta) * step / max(1, total_steps))

        buf_len = len(self.buf)
        for i in range(n):
            value = np.random.uniform(segment * i, segment * (i + 1))
            idx = self.tree.sample(value)
            # SumTree capacity (power of 2) may exceed deque maxlen; wrap
            idx = idx % buf_len
            idxs.append(idx)
            prob = self.tree.tree[idx + self.tree.capacity] / total
            weights.append((prob * buf_len) ** (-beta) if prob > 0 else 1.0)

        weights = np.array(weights, dtype=np.float32)
        weights /= weights.max()

        batch = [self.buf[i] for i in idxs]
        s, a, r, ns, d = zip(*batch)
        return (
            torch.FloatTensor(np.array(s)).to(device),
            torch.LongTensor(np.array(a)).to(device),
            torch.FloatTensor(np.array(r)).unsqueeze(1).to(device),
            torch.FloatTensor(np.array(ns)).to(device),
            torch.FloatTensor(np.array(d)).unsqueeze(1).to(device),
            idxs,
            torch.FloatTensor(weights).to(device),
        )

    def update_priorities(self, idxs, td_errors):
        for idx, td in zip(idxs, td_errors):
            prio = (abs(td) + self.eps) ** self.alpha
            self.max_prio = max(self.max_prio, prio)
            self.tree.update(idx, prio)

    def __len__(self):
        return len(self.buf)


class UniformReplay:
    def __init__(self, cap):
        self.buf = deque(maxlen=cap)

    def push(self, s, a, r, ns, d):
        self.buf.append((s, a, r, ns, d))

    def sample(self, n, step=None, total_steps=None):
        idxs = np.random.choice(len(self.buf), n, replace=False)
        batch = [self.buf[i] for i in idxs]
        s, a, r, ns, d = zip(*batch)
        return (
            torch.FloatTensor(np.array(s)).to(device),
            torch.LongTensor(np.array(a)).to(device),
            torch.FloatTensor(np.array(r)).unsqueeze(1).to(device),
            torch.FloatTensor(np.array(ns)).to(device),
            torch.FloatTensor(np.array(d)).unsqueeze(1).to(device),
            idxs,
            None,  # no weights for uniform
        )

    def update_priorities(self, idxs, td_errors):
        pass  # uniform — no priority update

    def __len__(self):
        return len(self.buf)


# ── Helpers ─────────────────────────────────────────────────────────────────
def make_buffer(use_per):
    if use_per:
        return PrioritizedReplay(args.buffer_size, args.per_alpha, args.per_beta, args.per_eps)
    return UniformReplay(args.buffer_size)


def select_action_ddqn(state, steps_done):
    eps = cfg["eps_end"] + (args.eps_start - cfg["eps_end"]) * \
          np.exp(-1.0 * steps_done / cfg["eps_decay"])
    if np.random.random() < eps:
        return np.random.randint(n_actions)
    with torch.no_grad():
        return online_net(torch.FloatTensor(state).unsqueeze(0).to(device)).argmax(dim=1).item()


def select_action_noisy(state):
    with torch.no_grad():
        return online_net(torch.FloatTensor(state).unsqueeze(0).to(device)).argmax(dim=1).item()


def compute_ddqn_target(ns, d, r):
    with torch.no_grad():
        next_actions = online_net(ns).argmax(dim=1, keepdim=True)
        next_q = target_net(ns).gather(1, next_actions)
        return r + args.gamma * next_q * (1 - d)


def compute_dqn_target(ns, d, r):
    with torch.no_grad():
        next_q, _ = target_net(ns).max(dim=1, keepdim=True)
        return r + args.gamma * next_q * (1 - d)


def soft_update(target, source):
    for tp, sp in zip(target.parameters(), source.parameters()):
        tp.data.copy_(args.tau * sp.data + (1.0 - args.tau) * tp.data)


# ── Eval ────────────────────────────────────────────────────────────────────
def evaluate():
    rewards = []
    for _ in range(args.eval_episodes):
        state, _ = eval_env.reset()
        ep_reward, done, steps = 0.0, False, 0
        while not done and steps < args.max_steps:
            if use_noisy:
                online_net.sample_noise()
            with torch.no_grad():
                a = online_net(torch.FloatTensor(state).unsqueeze(0).to(device)).argmax(dim=1).item()
            state, r, terminated, truncated, _ = eval_env.step(a)
            done = terminated or truncated
            ep_reward += r
            steps += 1
        rewards.append(ep_reward)
    return float(np.mean(rewards)), float(np.std(rewards))


# ── Plotting ────────────────────────────────────────────────────────────────
all_results = {}

def save_all():
    with open(f"{args.out_dir}/metrics.json", "w") as f:
        json.dump(all_results, f, indent=2)

def plot_all():
    n = len(all_results)
    if n == 0:
        return
    cols = min(3, n)
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(6 * cols, 5 * rows), squeeze=False)

    for idx, (name, data) in enumerate(sorted(all_results.items())):
        ax = axes[idx // cols][idx % cols]
        eps_data = data["episodes"]
        ep_nums = [e["episode"] for e in eps_data]
        rewards = [e["reward"] for e in eps_data]
        ax.plot(ep_nums, rewards, alpha=0.4, linewidth=0.5, color="#0a0a2e")
        if len(rewards) >= 10:
            w = min(10, len(rewards))
            smooth = np.convolve(rewards, np.ones(w) / w, mode="valid")
            ax.plot(ep_nums[w - 1:], smooth, linewidth=1.8, color="#00e05a")
        if data.get("evals"):
            ev_eps = [e["episode"] for e in data["evals"]]
            ev_r = [e["mean_reward"] for e in data["evals"]]
            ax.scatter(ev_eps, ev_r, c="#ff4455", s=20, zorder=5)
        algo, env = name.split("/", 1)
        ax.set_title(f"{algo.upper()} — {env}", fontsize=11, fontweight="bold")
        ax.set_xlabel("Episode")
        ax.set_ylabel("Reward")
        ax.grid(True, alpha=0.25)

    for idx in range(n, rows * cols):
        axes[idx // cols][idx % cols].set_visible(False)
    plt.tight_layout()
    plt.savefig(f"{args.out_dir}/plots/progress.png", dpi=130)
    plt.close()

    # Comparison plot
    fig, ax = plt.subplots(figsize=(12, 6))
    colors = plt.cm.tab10(np.linspace(0, 1, n))
    for idx, (name, data) in enumerate(sorted(all_results.items())):
        if data.get("evals"):
            ev_eps = [e["episode"] for e in data["evals"]]
            ev_r = [e["mean_reward"] for e in data["evals"]]
            ax.plot(ev_eps, ev_r, color=colors[idx], linewidth=1.8, label=name,
                    marker="o", markersize=3)
    ax.set_xlabel("Episode")
    ax.set_ylabel("Mean Eval Reward")
    ax.set_title("DDQN vs NoisyNet — Eval Comparison (no exploration noise)")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{args.out_dir}/plots/comparison.png", dpi=130)
    plt.close()


# ── Shutdown ────────────────────────────────────────────────────────────────
shutdown_flag = False
def on_shutdown(sig, frame):
    global shutdown_flag
    log_print(f"\n[shutdown] signal {sig}, finishing...")
    shutdown_flag = True
signal.signal(signal.SIGTERM, on_shutdown)
signal.signal(signal.SIGINT, on_shutdown)


# ── Training loop ───────────────────────────────────────────────────────────
configs = [(algo, env) for env in args.envs for algo in args.algos]
log_print(f"Configs: {[(a, e) for a, e in configs]}")

for cfg_idx, (algo, env_name) in enumerate(configs):
    if shutdown_flag:
        break

    ec = ENV_CONFIGS[env_name]
    cfg = {
        "episodes": ec["episodes"], "lr": ec["lr"], "target_update": ec["target_update"],
        "warmup": ec["warmup"], "eps_end": args.eps_end,
        "eps_decay": ec["warmup"] * 2,
    }

    log_print(f"\n{'=' * 60}")
    log_print(f"[{cfg_idx + 1}/{len(configs)}] {algo.upper()} on {env_name}  "
              f"episodes={cfg['episodes']} lr={cfg['lr']} warmup={cfg['warmup']} PER={ec['use_per']}")
    log_print(f"{'=' * 60}")

    set_seed(args.seed)
    use_noisy = (algo == "noisy")
    use_ddqn = (algo == "ddqn")

    env = gym.make(env_name, max_episode_steps=args.max_steps)
    eval_env = gym.make(env_name, max_episode_steps=args.max_steps)
    obs_dim = env.observation_space.shape[0]
    n_actions = env.action_space.n
    log_print(f"obs_dim={obs_dim} n_actions={n_actions}")

    NetClass = NoisyDQN if use_noisy else DQN
    online_net = NetClass(obs_dim, n_actions).to(device)
    target_net = NetClass(obs_dim, n_actions).to(device)
    target_net.load_state_dict(online_net.state_dict())
    optimizer = optim.Adam(online_net.parameters(), lr=cfg["lr"])

    buffer = make_buffer(ec["use_per"])
    config_key = f"{algo}/{env_name}"
    all_results[config_key] = {"episodes": [], "evals": []}
    metrics = all_results[config_key]

    step = 0
    best_eval = -float("inf")

    for episode in range(1, cfg["episodes"] + 1):
        if shutdown_flag:
            break
        if use_noisy:
            online_net.sample_noise()

        state, _ = env.reset()
        ep_reward, ep_loss, n_updates = 0.0, None, 0

        for t in range(args.max_steps):
            if use_noisy:
                action = select_action_noisy(state)
            else:
                action = select_action_ddqn(state, step)

            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            buffer.push(state, action, reward, next_state, float(done))
            state = next_state
            ep_reward += reward
            step += 1

            if len(buffer) > cfg["warmup"] and len(buffer) >= args.batch_size:
                n_updates += 1
                s, a, r, ns, d, idxs, weights = buffer.sample(args.batch_size, step, cfg["episodes"] * args.max_steps)

                if use_noisy:
                    online_net.sample_noise()
                    target_net.sample_noise()
                if use_ddqn:
                    target = compute_ddqn_target(ns, d, r)
                else:
                    target = compute_dqn_target(ns, d, r)

                q = online_net(s).gather(1, a.unsqueeze(1))
                td_errors = (target - q).detach().cpu().numpy().flatten()

                if weights is not None:
                    loss = (weights.unsqueeze(1) * F.smooth_l1_loss(q, target, reduction="none")).mean()
                else:
                    loss = F.smooth_l1_loss(q, target)

                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(online_net.parameters(), 10.0)
                optimizer.step()

                buffer.update_priorities(idxs, td_errors)
                ep_loss = float(loss.detach().cpu())

            if step % cfg["target_update"] == 0:
                soft_update(target_net, online_net)

            if done:
                break

        metrics["episodes"].append({
            "episode": episode, "reward": ep_reward, "steps": t + 1,
            "total_steps": step, "loss": ep_loss, "buffer_size": len(buffer),
        })
        log_print(f"[{algo:>5s} {env_name:<16s} ep {episode:4d}] reward={ep_reward:7.1f}  "
                  f"steps={t + 1:3d}  loss={ep_loss or float('nan'):.5f}  buf={len(buffer):6d}")

        if episode % args.eval_interval == 0:
            mean_r, std_r = evaluate()
            metrics["evals"].append({"episode": episode, "mean_reward": mean_r, "std_reward": std_r})
            log_print(f"[EVAL {algo:>5s} {env_name:<16s} ep {episode:4d}] mean={mean_r:8.1f} ± {std_r:.1f}")
            if mean_r > best_eval:
                best_eval = mean_r
                torch.save({"online": online_net.state_dict(), "episode": episode, "mean_reward": mean_r},
                           f"{args.out_dir}/best_{algo}_{env_name}.pt")
                log_print("  -> new best!")

        if episode % 100 == 0:
            save_all()
            plot_all()

    save_all()
    plot_all()
    env.close()
    eval_env.close()
    log_print(f"[DONE {algo}/{env_name}] best_eval={best_eval:.1f}")

# ── Final ───────────────────────────────────────────────────────────────────
save_all()
plot_all()
log_print(f"\n=== ALL DONE | {datetime.now()} ===")
for name, data in sorted(all_results.items()):
    evals = data.get("evals", [])
    best = max(e["mean_reward"] for e in evals) if evals else float("nan")
    final = evals[-1]["mean_reward"] if evals else float("nan")
    log_print(f"  {name:30s}  best={best:8.1f}  final={final:8.1f}")
