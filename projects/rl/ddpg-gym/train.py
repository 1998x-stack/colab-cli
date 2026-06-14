#!/usr/bin/env python3
"""DDPG on Gymnasium continuous-control envs. GPU-accelerated, plot + log + metrics output."""

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

# ── config ──────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument("--env", default="Pendulum-v1")
parser.add_argument("--episodes", type=int, default=200)
parser.add_argument("--steps_per_episode", type=int, default=200)
parser.add_argument("--buffer_size", type=int, default=100_000)
parser.add_argument("--batch_size", type=int, default=64)
parser.add_argument("--actor_lr", type=float, default=1e-3)
parser.add_argument("--critic_lr", type=float, default=1e-3)
parser.add_argument("--gamma", type=float, default=0.99)
parser.add_argument("--tau", type=float, default=0.005)
parser.add_argument("--noise_theta", type=float, default=0.15)
parser.add_argument("--noise_sigma", type=float, default=0.2)
parser.add_argument("--warmup_steps", type=int, default=1000)
parser.add_argument("--eval_interval", type=int, default=10)
parser.add_argument("--eval_episodes", type=int, default=5)
parser.add_argument("--ckpt_interval", type=int, default=50)
parser.add_argument("--out_dir", default="/content/ddpg-output")
parser.add_argument("--seed", type=int, default=42)
args = parser.parse_args()

os.makedirs(args.out_dir, exist_ok=True)
os.makedirs(f"{args.out_dir}/plots", exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Log both to file and stdout
class Tee:
    def __init__(self, path): self.file = open(path, "a", buffering=1)
    def write(self, s):
        sys.__stdout__.write(s)
        self.file.write(s)
        self.file.flush()
    def flush(self):
        sys.__stdout__.flush()
        self.file.flush()

log = Tee(f"{args.out_dir}/train.log")

def log_print(*a, **kw):
    print(*a, **kw, file=log)

log_print(f"=== DDPG {args.env} | device={device} | {datetime.now()} ===")
log_print(json.dumps(vars(args), indent=2))

# ── seed ────────────────────────────────────────────────────────────────────
def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
set_seed(args.seed)

# ── env ─────────────────────────────────────────────────────────────────────
env = gym.make(args.env, max_episode_steps=args.steps_per_episode)
eval_env = gym.make(args.env, max_episode_steps=args.steps_per_episode)

state_dim = env.observation_space.shape[0]
action_dim = env.action_space.shape[0]
max_action = float(env.action_space.high[0])

log_print(f"state_dim={state_dim} action_dim={action_dim} max_action={max_action}")

# ── networks ────────────────────────────────────────────────────────────────
def fanin_init(layer):
    fanin = layer.weight.data.size()[0]
    nn.init.uniform_(layer.weight.data, -1 / np.sqrt(fanin), 1 / np.sqrt(fanin))
    nn.init.uniform_(layer.bias.data, -1 / np.sqrt(fanin), 1 / np.sqrt(fanin))

class Actor(nn.Module):
    def __init__(self, s_dim, a_dim, max_a):
        super().__init__()
        self.fc1 = nn.Linear(s_dim, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, a_dim)
        self.max_a = max_a
        self.apply(self._init)
    def _init(self, m):
        if isinstance(m, nn.Linear): fanin_init(m)
    def forward(self, s):
        x = F.relu(self.fc1(s))
        x = F.relu(self.fc2(x))
        return torch.tanh(self.fc3(x)) * self.max_a

class Critic(nn.Module):
    def __init__(self, s_dim, a_dim):
        super().__init__()
        self.fc1 = nn.Linear(s_dim + a_dim, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, 1)
        self.apply(self._init)
    def _init(self, m):
        if isinstance(m, nn.Linear): fanin_init(m)
    def forward(self, s, a):
        x = torch.cat([s, a], dim=1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)

actor = Actor(state_dim, action_dim, max_action).to(device)
target_actor = Actor(state_dim, action_dim, max_action).to(device)
target_actor.load_state_dict(actor.state_dict())

critic = Critic(state_dim, action_dim).to(device)
target_critic = Critic(state_dim, action_dim).to(device)
target_critic.load_state_dict(critic.state_dict())

actor_opt = optim.Adam(actor.parameters(), lr=args.actor_lr)
critic_opt = optim.Adam(critic.parameters(), lr=args.critic_lr)

# ── replay buffer ───────────────────────────────────────────────────────────
class ReplayBuffer:
    def __init__(self, cap):
        self.buf = deque(maxlen=cap)
    def push(self, s, a, r, ns, d):
        self.buf.append((s, a, r, ns, d))
    def sample(self, n):
        idxs = np.random.choice(len(self.buf), n, replace=False)
        batch = [self.buf[i] for i in idxs]
        s, a, r, ns, d = zip(*batch)
        return (
            torch.FloatTensor(np.array(s)).to(device),
            torch.FloatTensor(np.array(a)).to(device),
            torch.FloatTensor(np.array(r)).unsqueeze(1).to(device),
            torch.FloatTensor(np.array(ns)).to(device),
            torch.FloatTensor(np.array(d)).unsqueeze(1).to(device),
        )
    def __len__(self): return len(self.buf)

buffer = ReplayBuffer(args.buffer_size)

# ── noise process (OU) ──────────────────────────────────────────────────────
class OUNoise:
    def __init__(self, a_dim, theta=0.15, sigma=0.2, mu=0.0):
        self.a_dim = a_dim
        self.theta, self.sigma, self.mu = theta, sigma, mu
        self.reset()
    def reset(self): self.state = np.ones(self.a_dim) * self.mu
    def sample(self):
        x = self.state
        dx = self.theta * (self.mu - x) + self.sigma * np.random.randn(self.a_dim)
        self.state = x + dx
        return self.state

noise = OUNoise(action_dim, args.noise_theta, args.noise_sigma)

# ── helpers ─────────────────────────────────────────────────────────────────
def select_action(state, add_noise=True):
    with torch.no_grad():
        s = torch.FloatTensor(state).unsqueeze(0).to(device)
        a = actor(s).cpu().numpy().flatten()
    if add_noise:
        a += noise.sample() * max_action * max(0.0, 1.0 - step / args.warmup_steps)
    return np.clip(a, -max_action, max_action)

def soft_update(target, source, tau):
    for tp, sp in zip(target.parameters(), source.parameters()):
        tp.data.copy_(tau * sp.data + (1.0 - tau) * tp.data)

# ── metrics ─────────────────────────────────────────────────────────────────
metrics = {
    "args": vars(args),
    "device": str(device),
    "episodes": [],
    "eval_episodes": [],
}

def save_metrics():
    with open(f"{args.out_dir}/metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

def plot_progress():
    eps = [e["episode"] for e in metrics["episodes"]]
    rewards = [e["reward"] for e in metrics["episodes"]]
    actor_losses = [e["actor_loss"] for e in metrics["episodes"] if e["actor_loss"] is not None]
    critic_losses = [e["critic_loss"] for e in metrics["episodes"] if e["critic_loss"] is not None]
    actor_eps = [e["episode"] for e in metrics["episodes"] if e["actor_loss"] is not None]

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Reward
    ax = axes[0, 0]
    ax.plot(eps, rewards, "b-", alpha=0.6, linewidth=0.8)
    if len(eps) >= 10:
        window = min(10, len(rewards))
        smooth = np.convolve(rewards, np.ones(window)/window, mode="valid")
        ax.plot(eps[window-1:], smooth, "b-", linewidth=1.5, label=f"MA{window}")
    ax.set_xlabel("Episode"); ax.set_ylabel("Total Reward"); ax.set_title(f"{args.env} — Episode Reward")
    ax.legend(); ax.grid(True, alpha=0.3)

    # Eval
    if metrics["eval_episodes"]:
        ev_eps = [e["episode"] for e in metrics["eval_episodes"]]
        ev_rewards = [e["mean_reward"] for e in metrics["eval_episodes"]]
        ev_stds = [e["std_reward"] for e in metrics["eval_episodes"]]
        axes[0, 1].errorbar(ev_eps, ev_rewards, yerr=ev_stds, fmt="go-", capsize=3, markersize=4)
        axes[0, 1].set_xlabel("Episode"); axes[0, 1].set_ylabel("Mean Eval Reward")
        axes[0, 1].set_title("Evaluation (no noise)"); axes[0, 1].grid(True, alpha=0.3)

    # Losses
    if actor_eps:
        axes[1, 0].plot(actor_eps, actor_losses, "tab:red", alpha=0.7, label="Actor", linewidth=0.8)
        axes[1, 0].plot(actor_eps, critic_losses, "tab:blue", alpha=0.7, label="Critic", linewidth=0.8)
        axes[1, 0].set_xlabel("Episode"); axes[1, 0].set_ylabel("Loss")
        axes[1, 0].set_title("Actor & Critic Loss"); axes[1, 0].legend(); axes[1, 0].grid(True, alpha=0.3)

    # Steps
    steps = [e.get("total_steps", 0) for e in metrics["episodes"]]
    axes[1, 1].plot(eps, steps, "purple", alpha=0.7, linewidth=0.8)
    axes[1, 1].set_xlabel("Episode"); axes[1, 1].set_ylabel("Total Env Steps")
    axes[1, 1].set_title("Training Progress"); axes[1, 1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(f"{args.out_dir}/plots/progress.png", dpi=120)
    plt.close()

# ── eval ────────────────────────────────────────────────────────────────────
def evaluate():
    rewards = []
    for _ in range(args.eval_episodes):
        state, _ = eval_env.reset()
        ep_reward, done = 0.0, False
        while not done:
            with torch.no_grad():
                s = torch.FloatTensor(state).unsqueeze(0).to(device)
                a = actor(s).cpu().numpy().flatten()
            state, r, terminated, truncated, _ = eval_env.step(a)
            done = terminated or truncated
            ep_reward += r
        rewards.append(ep_reward)
    mean_r, std_r = float(np.mean(rewards)), float(np.std(rewards))
    return mean_r, std_r

# ── graceful shutdown ───────────────────────────────────────────────────────
shutdown_flag = False

def on_shutdown(sig, frame):
    global shutdown_flag
    log_print(f"\n[shutdown] signal {sig}, finishing episode then saving...")
    shutdown_flag = True

signal.signal(signal.SIGTERM, on_shutdown)
signal.signal(signal.SIGINT, on_shutdown)

# ── training loop ───────────────────────────────────────────────────────────
step = 0
best_eval = -float("inf")

for episode in range(1, args.episodes + 1):
    if shutdown_flag:
        break

    state, _ = env.reset()
    noise.reset()
    ep_reward, ep_actor_loss, ep_critic_loss, n_updates = 0.0, None, None, 0

    for t in range(args.steps_per_episode):
        action = select_action(state, add_noise=True)
        next_state, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated

        buffer.push(state, action, reward, next_state, float(done))
        state = next_state
        ep_reward += reward
        step += 1

        if len(buffer) > args.batch_size and step % 2 == 0:
            s, a, r, ns, d = buffer.sample(args.batch_size)

            # Critic update
            with torch.no_grad():
                target_a = target_actor(ns)
                target_q = target_critic(ns, target_a)
                y = r + args.gamma * target_q * (1 - d)

            q = critic(s, a)
            critic_loss = F.mse_loss(q, y)
            critic_opt.zero_grad()
            critic_loss.backward()
            torch.nn.utils.clip_grad_norm_(critic.parameters(), 1.0)
            critic_opt.step()

            # Actor update
            pred_a = actor(s)
            actor_loss = -critic(s, pred_a).mean()
            actor_opt.zero_grad()
            actor_loss.backward()
            torch.nn.utils.clip_grad_norm_(actor.parameters(), 1.0)
            actor_opt.step()

            soft_update(target_actor, actor, args.tau)
            soft_update(target_critic, critic, args.tau)

            ep_actor_loss = float(actor_loss.detach().cpu())
            ep_critic_loss = float(critic_loss.detach().cpu())
            n_updates += 1

        if done:
            break

    # Record episode metrics
    entry = {
        "episode": episode,
        "reward": ep_reward,
        "steps": t + 1,
        "total_steps": step,
        "actor_loss": ep_actor_loss,
        "critic_loss": ep_critic_loss,
        "buffer_size": len(buffer),
    }
    metrics["episodes"].append(entry)
    log_print(f"[ep {episode:4d}] reward={ep_reward:8.2f}  steps={t+1:3d}  "
              f"a_loss={ep_actor_loss or float('nan'):.5f}  "
              f"c_loss={ep_critic_loss or float('nan'):.5f}  buf={len(buffer):6d}")

    # Eval
    if episode % args.eval_interval == 0:
        mean_r, std_r = evaluate()
        eval_entry = {"episode": episode, "mean_reward": mean_r, "std_reward": std_r}
        metrics["eval_episodes"].append(eval_entry)
        log_print(f"[eval ep {episode}] mean={mean_r:.2f} ± {std_r:.2f}")

        if mean_r > best_eval:
            best_eval = mean_r
            torch.save({"actor": actor.state_dict(), "critic": critic.state_dict(),
                        "episode": episode, "mean_reward": mean_r},
                       f"{args.out_dir}/best_model.pt")
            log_print("  -> new best! saved best_model.pt")

    # Checkpoint
    if episode % args.ckpt_interval == 0:
        torch.save({"actor": actor.state_dict(), "critic": critic.state_dict(),
                    "actor_opt": actor_opt.state_dict(), "critic_opt": critic_opt.state_dict(),
                    "episode": episode},
                   f"{args.out_dir}/ckpt_ep{episode}.pt")
        save_metrics()
        plot_progress()
        log_print("  -> ckpt + metrics + plot saved")

# ── final save ──────────────────────────────────────────────────────────────
save_metrics()
plot_progress()
torch.save({"actor": actor.state_dict(), "critic": critic.state_dict(),
            "actor_opt": actor_opt.state_dict(), "critic_opt": critic_opt.state_dict(),
            "episode": max(e["episode"] for e in metrics["episodes"])},
           f"{args.out_dir}/final_model.pt")

# Summary
summary = {
    "env": args.env,
    "episodes_completed": len(metrics["episodes"]),
    "total_steps": step,
    "best_eval_reward": best_eval,
    "final_actor_loss": metrics["episodes"][-1].get("actor_loss") if metrics["episodes"] else None,
    "final_critic_loss": metrics["episodes"][-1].get("critic_loss") if metrics["episodes"] else None,
    "n_evals": len(metrics["eval_episodes"]),
    "device": str(device),
}
with open(f"{args.out_dir}/summary.json", "w") as f:
    json.dump(summary, f, indent=2)

log_print(f"\n=== DONE | best_eval={best_eval:.2f} | total_steps={step} | {datetime.now()} ===")
log_print(f"Output: {args.out_dir}/")
log_print("  train.log  metrics.json  best_model.pt  final_model.pt  plots/progress.png")
