#!/usr/bin/env python3
"""DDPG vs TD3 head-to-head on 3 MuJoCo continuous-control envs.

Trains both algorithms sequentially on HalfCheetah-v4, Hopper-v4, Walker2d-v4.
Each (env, algo) pair gets structured outputs: CSV metrics, PNG plots, timestamped logs.
Generates per-env comparison plots and a master 3x2 dashboard.

GPU-accelerated. Auto-detects Kaggle vs Colab vs local environment.
"""

import os
import sys
import time
import subprocess
from datetime import datetime
from collections import deque

# ── Platform detection ────────────────────────────────────────────────────────
IN_KAGGLE = os.path.exists("/kaggle/working/")
IN_COLAB = os.path.exists("/content/")

if IN_KAGGLE:
    # Detect P100 GPU via subprocess BEFORE importing torch.
    # P100 (sm_60) is incompatible with pre-installed PyTorch 2.10+cu128.
    # Must reinstall with CUDA 12.6 before any `import torch`.
    r = subprocess.run(
        [sys.executable, "-c",
         "import torch; print(torch.cuda.is_available()); print(torch.cuda.get_device_capability(0))"],
        capture_output=True, text=True, timeout=30,
    )
    if r.returncode == 0:
        lines = r.stdout.strip().split("\n")
        if lines[0] == "True" and lines[1].startswith("(6,"):
            print(f"[kaggle] P100 detected {lines[1]}, reinstalling torch for CUDA 12.6...")
            subprocess.run([
                sys.executable, "-m", "pip", "install", "-q", "--force-reinstall",
                "torch", "torchvision",
                "--extra-index-url", "https://download.pytorch.org/whl/cu126"
            ], check=True, timeout=300)
            print("[kaggle] torch reinstalled for CUDA 12.6")

    # Install mujoco (not pre-installed on Kaggle)
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "gymnasium[mujoco]", "matplotlib"],
                   check=True, timeout=120)
    OUT_ROOT = "/kaggle/working/ddpg-td3-mujoco-output"
elif IN_COLAB:
    OUT_ROOT = "/content/ddpg-td3-mujoco-output"
else:
    OUT_ROOT = "./output/ddpg-td3-mujoco-output"

print(f"Platform: {'Kaggle' if IN_KAGGLE else 'Colab' if IN_COLAB else 'Local'}")
print(f"Output root: {OUT_ROOT}")

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import gymnasium as gym

# ── Config ────────────────────────────────────────────────────────────────────
ENVS = ["HalfCheetah-v4", "Hopper-v4", "Walker2d-v4"]
ALGOS = ["DDPG", "TD3"]

EPISODES_PER_ENV = {
    "HalfCheetah-v4": 400,
    "Hopper-v4": 300,
    "Walker2d-v4": 300,
}
STEPS_PER_EPISODE = 1000

# Shared hyperparams
BUFFER_SIZE = 1_000_000
BATCH_SIZE = 256
GAMMA = 0.99
TAU = 0.005

# DDPG-specific
DDPG_ACTOR_LR = 1e-3
DDPG_CRITIC_LR = 1e-3
OU_THETA = 0.15
OU_SIGMA = 0.2

# TD3-specific
TD3_ACTOR_LR = 3e-4
TD3_CRITIC_LR = 3e-4
POLICY_DELAY = 2
POLICY_NOISE = 0.2
NOISE_CLIP = 0.5
EXPLORATION_NOISE = 0.1

WARMUP_STEPS = 1000
EVAL_INTERVAL = 20
EVAL_EPISODES = 5
CKPT_INTERVAL = 100
SEED = 42

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
PRINT_EVERY = 10  # log per N episodes

# ── Utilities ─────────────────────────────────────────────────────────────────
def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def fanin_init(layer):
    fanin = layer.weight.data.size()[0]
    bound = 1.0 / np.sqrt(fanin)
    nn.init.uniform_(layer.weight.data, -bound, bound)
    nn.init.uniform_(layer.bias.data, -bound, bound)

# ── Replay Buffer ─────────────────────────────────────────────────────────────
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

# ── OU Noise (DDPG) ──────────────────────────────────────────────────────────
class OUNoise:
    def __init__(self, a_dim, theta=0.15, sigma=0.2):
        self.a_dim, self.theta, self.sigma = a_dim, theta, sigma
        self.reset()
    def reset(self): self.state = np.zeros(self.a_dim)
    def sample(self):
        self.state += self.theta * -self.state + self.sigma * np.random.randn(self.a_dim)
        return self.state.copy()

# ── DDPG Networks ─────────────────────────────────────────────────────────────
class DDPGActor(nn.Module):
    def __init__(self, s_dim, a_dim, max_a):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(s_dim, 256), nn.ReLU(),
            nn.Linear(256, 256), nn.ReLU(),
            nn.Linear(256, a_dim), nn.Tanh(),
        )
        self.max_a = max_a
        self.apply(lambda m: fanin_init(m) if isinstance(m, nn.Linear) else None)
    def forward(self, s):
        return self.net(s) * self.max_a

class DDPGCritic(nn.Module):
    def __init__(self, s_dim, a_dim):
        super().__init__()
        self.fc1 = nn.Linear(s_dim + a_dim, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, 1)
        self.apply(lambda m: fanin_init(m) if isinstance(m, nn.Linear) else None)
    def forward(self, s, a):
        x = torch.cat([s, a], dim=1)
        return self.fc3(F.relu(self.fc2(F.relu(self.fc1(x)))))

# ── TD3 Networks ──────────────────────────────────────────────────────────────
class TD3Actor(nn.Module):
    def __init__(self, s_dim, a_dim, max_a):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(s_dim, 256), nn.ReLU(),
            nn.Linear(256, 256), nn.ReLU(),
            nn.Linear(256, a_dim), nn.Tanh(),
        )
        self.max_a = max_a
        self.apply(lambda m: fanin_init(m) if isinstance(m, nn.Linear) else None)
    def forward(self, s):
        return self.net(s) * self.max_a

class TD3Critic(nn.Module):
    def __init__(self, s_dim, a_dim):
        super().__init__()
        # Q1
        self.q1 = nn.Sequential(
            nn.Linear(s_dim + a_dim, 256), nn.ReLU(),
            nn.Linear(256, 256), nn.ReLU(),
            nn.Linear(256, 1),
        )
        # Q2
        self.q2 = nn.Sequential(
            nn.Linear(s_dim + a_dim, 256), nn.ReLU(),
            nn.Linear(256, 256), nn.ReLU(),
            nn.Linear(256, 1),
        )
        self.apply(lambda m: fanin_init(m) if isinstance(m, nn.Linear) else None)
    def forward(self, s, a):
        x = torch.cat([s, a], dim=1)
        return self.q1(x), self.q2(x)
    def Q1(self, s, a):
        return self.q1(torch.cat([s, a], dim=1))

def soft_update(target, source, tau):
    for tp, sp in zip(target.parameters(), source.parameters()):
        tp.data.copy_(tau * sp.data + (1.0 - tau) * tp.data)

# ── Train one (algo, env) pair ───────────────────────────────────────────────
def train_one(algo, env_name, n_episodes, out_dir):
    log_path = f"{out_dir}/train.log"
    csv_path = f"{out_dir}/metrics.csv"
    png_path = f"{out_dir}/training_curves.png"
    os.makedirs(out_dir, exist_ok=True)

    def log(msg):
        line = f"[{datetime.now().strftime('%H:%M:%S')}] {msg}"
        print(line, flush=True)
        with open(log_path, "a") as f:
            f.write(line + "\n")

    log(f"=== {algo} on {env_name} | device={device} ===")

    set_seed(SEED)
    env = gym.make(env_name)
    eval_env = gym.make(env_name)
    s_dim = env.observation_space.shape[0]
    a_dim = env.action_space.shape[0]
    max_a = float(env.action_space.high[0])
    log(f"state_dim={s_dim} action_dim={a_dim} max_action={max_a:.3f}")

    # Init networks
    if algo == "DDPG":
        actor = DDPGActor(s_dim, a_dim, max_a).to(device)
        target_actor = DDPGActor(s_dim, a_dim, max_a).to(device)
        target_actor.load_state_dict(actor.state_dict())
        critic = DDPGCritic(s_dim, a_dim).to(device)
        target_critic = DDPGCritic(s_dim, a_dim).to(device)
        target_critic.load_state_dict(critic.state_dict())
        actor_opt = optim.Adam(actor.parameters(), lr=DDPG_ACTOR_LR)
        critic_opt = optim.Adam(critic.parameters(), lr=DDPG_CRITIC_LR)
        noise = OUNoise(a_dim, OU_THETA, OU_SIGMA)
    else:
        actor = TD3Actor(s_dim, a_dim, max_a).to(device)
        target_actor = TD3Actor(s_dim, a_dim, max_a).to(device)
        target_actor.load_state_dict(actor.state_dict())
        critic = TD3Critic(s_dim, a_dim).to(device)
        target_critic = TD3Critic(s_dim, a_dim).to(device)
        target_critic.load_state_dict(critic.state_dict())
        actor_opt = optim.Adam(actor.parameters(), lr=TD3_ACTOR_LR)
        critic_opt = optim.Adam(critic.parameters(), lr=TD3_CRITIC_LR)

    buffer = ReplayBuffer(BUFFER_SIZE)

    def select_action(state, add_noise=True):
        with torch.no_grad():
            s = torch.FloatTensor(state).unsqueeze(0).to(device)
            a = actor(s).cpu().numpy().flatten()
        if add_noise:
            if algo == "DDPG":
                a += noise.sample() * max_a * max(0.0, 1.0 - step / WARMUP_STEPS)
            else:
                a += np.random.normal(0, EXPLORATION_NOISE, size=a_dim)
        return np.clip(a, -max_a, max_a)

    # Metrics tracking
    rows = []
    header = "episode,reward,steps,avg100,actor_loss,critic_loss,q_mean,elapsed_s"
    def save_csv():
        with open(csv_path, "w") as f:
            f.write(header + "\n")
            for r in rows:
                f.write(",".join(str(v) for v in r) + "\n")

    def save_png():
        if len(rows) < 2:
            return
        eps = [r[0] for r in rows]
        rewards = [r[1] for r in rows]
        avg100s = [r[3] for r in rows]
        a_losses = [r[4] for r in rows if r[4] is not None]
        a_eps = [r[0] for r in rows if r[4] is not None]
        c_losses = [r[5] for r in rows if r[5] is not None]

        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle(f"{algo} on {env_name}", fontsize=14, fontweight="bold")

        ax = axes[0, 0]
        ax.plot(eps, rewards, "b-", alpha=0.4, linewidth=0.8, label="Raw")
        ax.plot(eps, avg100s, "b-", linewidth=1.8, label="Avg100")
        ax.set_xlabel("Episode")
        ax.set_ylabel("Reward")
        ax.set_title("Episode Reward")
        ax.legend()
        ax.grid(True, alpha=0.3)

        axes[0, 1].axis("off")
        info_text = f"Env: {env_name}\nAlgo: {algo}\nEpisodes: {len(rows)}\n"
        if avg100s:
            info_text += f"Best Avg100: {max(avg100s):.1f}\n"
            info_text += f"Last Avg100: {avg100s[-1]:.1f}"
        axes[0, 1].text(0.1, 0.7, info_text, fontsize=12, fontfamily="monospace",
                         verticalalignment="top")

        if a_eps:
            axes[1, 0].plot(a_eps, a_losses, "tab:red", alpha=0.7, label="Actor", linewidth=0.8)
            axes[1, 0].plot(a_eps, c_losses, "tab:blue", alpha=0.7, label="Critic", linewidth=0.8)
            axes[1, 0].set_xlabel("Episode")
            axes[1, 0].set_ylabel("Loss")
            axes[1, 0].set_title("Losses")
            axes[1, 0].legend()
            axes[1, 0].grid(True, alpha=0.3)

        q_means = [r[6] for r in rows if r[6] is not None]
        if q_means:
            q_eps = [r[0] for r in rows if r[6] is not None]
            axes[1, 1].plot(q_eps, q_means, "tab:green", alpha=0.7, linewidth=0.8)
            axes[1, 1].set_xlabel("Episode")
            axes[1, 1].set_ylabel("Q-mean")
            axes[1, 1].set_title("Q-value Mean")
            axes[1, 1].grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(png_path, dpi=120)
        plt.close()

    def evaluate():
        rewards = []
        for _ in range(EVAL_EPISODES):
            state, _ = eval_env.reset()
            ep_r, done = 0.0, False
            while not done:
                with torch.no_grad():
                    s = torch.FloatTensor(state).unsqueeze(0).to(device)
                    a = actor(s).cpu().numpy().flatten()
                state, r, terminated, truncated, _ = eval_env.step(a)
                done = terminated or truncated
                ep_r += r
            rewards.append(ep_r)
        return float(np.mean(rewards)), float(np.std(rewards))

    # Training loop
    step = 0
    best_eval = -float("inf")
    t_start = time.time()
    episode_rewards = deque(maxlen=100)

    for episode in range(1, n_episodes + 1):
        state, _ = env.reset()
        ep_reward, ep_a_loss, ep_c_loss, ep_q_mean = 0.0, None, None, None
        n_updates = 0

        if algo == "DDPG":
            noise.reset()

        for t in range(STEPS_PER_EPISODE):
            action = select_action(state, add_noise=True)
            ns, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            buffer.push(state, action, reward, ns, float(done))
            state = ns
            ep_reward += reward
            step += 1

            if len(buffer) >= BATCH_SIZE:
                s, a, r, ns, d = buffer.sample(BATCH_SIZE)

                if algo == "DDPG":
                    with torch.no_grad():
                        target_q = target_critic(ns, target_actor(ns))
                        y = r + GAMMA * target_q * (1 - d)
                    q = critic(s, a)
                    c_loss = F.mse_loss(q, y)
                    critic_opt.zero_grad()
                    c_loss.backward()
                    torch.nn.utils.clip_grad_norm_(critic.parameters(), 1.0)
                    critic_opt.step()

                    a_loss = -critic(s, actor(s)).mean()
                    actor_opt.zero_grad()
                    a_loss.backward()
                    torch.nn.utils.clip_grad_norm_(actor.parameters(), 1.0)
                    actor_opt.step()

                    soft_update(target_actor, actor, TAU)
                    soft_update(target_critic, critic, TAU)

                    ep_a_loss = float(a_loss.detach().cpu())
                    ep_c_loss = float(c_loss.detach().cpu())
                    ep_q_mean = float(q.detach().mean().cpu())

                else:  # TD3
                    n_updates += 1
                    with torch.no_grad():
                        noise_clip = torch.randn_like(a) * POLICY_NOISE
                        noise_clip = noise_clip.clamp(-NOISE_CLIP, NOISE_CLIP)
                        target_a = (target_actor(ns) + noise_clip).clamp(-max_a, max_a)
                        q1_t, q2_t = target_critic(ns, target_a)
                        y = r + GAMMA * torch.min(q1_t, q2_t) * (1 - d)

                    q1, q2 = critic(s, a)
                    c_loss = F.mse_loss(q1, y) + F.mse_loss(q2, y)
                    critic_opt.zero_grad()
                    c_loss.backward()
                    torch.nn.utils.clip_grad_norm_(critic.parameters(), 1.0)
                    critic_opt.step()
                    ep_c_loss = float(c_loss.detach().cpu())

                    if n_updates % POLICY_DELAY == 0:
                        a_loss = -critic.Q1(s, actor(s)).mean()
                        actor_opt.zero_grad()
                        a_loss.backward()
                        torch.nn.utils.clip_grad_norm_(actor.parameters(), 1.0)
                        actor_opt.step()
                        soft_update(target_actor, actor, TAU)
                        soft_update(target_critic, critic, TAU)
                        ep_a_loss = float(a_loss.detach().cpu())

                    ep_q_mean = float(torch.min(q1, q2).detach().mean().cpu())

            if done:
                break

        episode_rewards.append(ep_reward)
        avg100 = np.mean(episode_rewards) if episode_rewards else ep_reward
        elapsed = time.time() - t_start

        row = [episode, ep_reward, t + 1, round(avg100, 2),
               round(ep_a_loss, 6) if ep_a_loss is not None else None,
               round(ep_c_loss, 6) if ep_c_loss is not None else None,
               round(ep_q_mean, 6) if ep_q_mean is not None else None,
               round(elapsed, 1)]
        rows.append(row)

        if episode % PRINT_EVERY == 0 or episode == 1 or episode == n_episodes:
            log(f"Ep {episode:4d}/{n_episodes} | reward={ep_reward:8.2f} | avg100={avg100:8.2f} | "
                f"a_loss={ep_a_loss or float('nan'):.5f} | c_loss={ep_c_loss or float('nan'):.5f} | "
                f"q_mean={ep_q_mean or float('nan'):.4f} | elapsed={elapsed:.0f}s")

        # Eval + checkpoint
        if episode % EVAL_INTERVAL == 0:
            mean_r, std_r = evaluate()
            log(f"  EVAL ep {episode}: {mean_r:.2f} ± {std_r:.2f}  (best={best_eval:.2f})")
            if mean_r > best_eval:
                best_eval = mean_r
                torch.save({"actor": actor.state_dict(), "critic": critic.state_dict(),
                            "episode": episode, "eval_reward": mean_r},
                           f"{out_dir}/best_model.pt")

        if episode % CKPT_INTERVAL == 0:
            save_csv()
            save_png()

    # Final save
    save_csv()
    save_png()
    torch.save({"actor": actor.state_dict(), "critic": critic.state_dict(),
                "episode": n_episodes},
               f"{out_dir}/final_model.pt")

    # Summary
    import json
    summary = {
        "algo": algo, "env": env_name,
        "episodes_completed": n_episodes,
        "total_steps": step,
        "best_eval_reward": best_eval,
        "elapsed_s": round(elapsed, 1),
        "device": str(device),
    }
    with open(f"{out_dir}/summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    log(f"=== DONE {algo} {env_name} | best_eval={best_eval:.2f} | elapsed={elapsed:.0f}s ===")
    env.close()
    eval_env.close()
    return {"algo": algo, "env": env_name, "episodes": rows, "best_eval": best_eval,
            "out_dir": out_dir}

# ── Comparison plots ──────────────────────────────────────────────────────────
def plot_comparison(results, out_dir):
    """Per-env DDPG vs TD3 reward comparison + master dashboard."""
    os.makedirs(out_dir, exist_ok=True)

    # Group results by env
    by_env = {}
    for r in results:
        by_env.setdefault(r["env"], []).append(r)

    # Per-env comparison
    for env_name, pairs in by_env.items():
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        fig.suptitle(f"{env_name}: DDPG vs TD3", fontsize=14, fontweight="bold")

        colors = {"DDPG": "tab:blue", "TD3": "tab:orange"}
        for res in pairs:
            algo = res["algo"]
            eps = [r[0] for r in res["episodes"]]
            rewards = [r[1] for r in res["episodes"]]
            avg100s = [r[3] for r in res["episodes"]]
            color = colors[algo]
            axes[0].plot(eps, avg100s, color=color, linewidth=1.5, label=f"{algo} (best={res['best_eval']:.1f})")
            axes[0].plot(eps, rewards, color=color, alpha=0.15, linewidth=0.5)

        axes[0].set_xlabel("Episode")
        axes[0].set_ylabel("Reward")
        axes[0].set_title("Avg100 Reward")
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)

        # Bar chart comparison
        algos_list = [p["algo"] for p in pairs]
        bests = [p["best_eval"] for p in pairs]
        bars = axes[1].bar(algos_list, bests, color=[colors[a] for a in algos_list], width=0.4)
        for bar, val in zip(bars, bests):
            axes[1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 20,
                        f"{val:.1f}", ha="center", fontweight="bold")
        axes[1].set_title("Best Eval Reward")
        axes[1].grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(f"{out_dir}/{env_name}_comparison.png", dpi=120)
        plt.close()

    # Master dashboard: 3x2 grid
    fig, axes = plt.subplots(3, 2, figsize=(14, 15))
    fig.suptitle("DDPG vs TD3 — MuJoCo Benchmark", fontsize=16, fontweight="bold")

    for i, env_name in enumerate(ENVS):
        pairs = by_env.get(env_name, [])
        for res in pairs:
            algo = res["algo"]
            color = colors[algo]
            eps = [r[0] for r in res["episodes"]]
            avg100s = [r[3] for r in res["episodes"]]
            ax = axes[i, 0]
            ax.plot(eps, avg100s, color=color, linewidth=1.5, label=f"{algo}")
            ax.set_title(f"{env_name} — Avg100 Reward")
            ax.legend(fontsize=8)
            ax.grid(True, alpha=0.3)

            # Q-mean
            q_eps = [r[0] for r in res["episodes"] if r[6] is not None]
            q_vals = [r[6] for r in res["episodes"] if r[6] is not None]
            if q_vals:
                axes[i, 1].plot(q_eps, q_vals, color=color, linewidth=1.2, label=f"{algo}")
                axes[i, 1].set_title(f"{env_name} — Q-value Mean")
                axes[i, 1].legend(fontsize=8)
                axes[i, 1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(f"{out_dir}/master_comparison.png", dpi=150)
    plt.close()
    print(f"\n[comparison] saved to {out_dir}/")
    for env_name in ENVS:
        print(f"  {env_name}_comparison.png")
    print("  master_comparison.png")

# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    print(f"=== DDPG vs TD3 — 3 MuJoCo envs ===  device={device}  {datetime.now()}")
    for e in ENVS:
        print(f"  {e}: {EPISODES_PER_ENV[e]} episodes")
    print()

    all_results = []

    for env_name in ENVS:
        n_ep = EPISODES_PER_ENV[env_name]
        for algo in ALGOS:
            out_dir = f"{OUT_ROOT}/{env_name}/{algo}"
            print(f"\n{'='*60}")
            print(f"  {algo} on {env_name} ({n_ep} episodes)")
            print(f"  Output: {out_dir}/")
            print(f"{'='*60}\n")
            result = train_one(algo, env_name, n_ep, out_dir)
            all_results.append(result)
            torch.cuda.empty_cache()

        # Per-env comparison after both algos complete
        env_results = [r for r in all_results if r["env"] == env_name]
        plot_comparison(env_results, f"{OUT_ROOT}/comparison")

    # Master comparison
    plot_comparison(all_results, f"{OUT_ROOT}/comparison")

    # Final summary
    import json as _json
    master_summary = {
        "completed_at": str(datetime.now()),
        "device": str(device),
        "results": {f"{r['algo']}_{r['env']}": {"best_eval": r["best_eval"],
                    "episodes": len(r["episodes"])} for r in all_results},
    }
    with open(f"{OUT_ROOT}/summary.json", "w") as f:
        _json.dump(master_summary, f, indent=2)

    print(f"\n{'='*60}")
    print(f"  ALL DONE — {datetime.now()}")
    for res in all_results:
        print(f"  {res['algo']:4s} {res['env']:20s} best_eval={res['best_eval']:.2f}")
    print(f"\n  Output root: {OUT_ROOT}/")
    print(f"  Comparison:  {OUT_ROOT}/comparison/master_comparison.png")
    print(f"{'='*60}")

if __name__ == "__main__":
    main()
