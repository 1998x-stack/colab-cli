"""Tabular SARSA on CartPole-v1 with state discretization.

Generates PNGs, CSV metrics, and logs to /content/rl-sarsa-output/.
"""

import os
import csv
import time
import json
from datetime import datetime

import numpy as np
import gymnasium as gym
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ── Config ────────────────────────────────────────────────────────────────────
ENV_NAME = "CartPole-v1"
N_EPISODES = 3000
N_BINS = 12                    # per state dimension → 12^4 ≈ 20.7k discrete states
ALPHA_START = 0.5              # initial learning rate
ALPHA_END = 0.01               # final learning rate (linear decay)
GAMMA = 0.99                   # discount factor
EPSILON_START = 1.0
EPSILON_MIN = 0.01
EPSILON_DECAY = 0.9985         # per-episode multiplicative decay → min at ep ~2500
SEED = 42
PLOT_EVERY = 100               # episodes between PNG refreshes
SAVE_EVERY = 500               # episodes between Q-table checkpoints
OUT_DIR = "/content/rl-sarsa-output"

os.makedirs(f"{OUT_DIR}/logs", exist_ok=True)
os.makedirs(f"{OUT_DIR}/pngs", exist_ok=True)
os.makedirs(f"{OUT_DIR}/checkpoints", exist_ok=True)

LOG = f"{OUT_DIR}/logs/train.log"
CSV_PATH = f"{OUT_DIR}/metrics.csv"

# ── Logging ────────────────────────────────────────────────────────────────────
def log(msg):
    line = f"[{datetime.now().strftime('%H:%M:%S')}] {msg}"
    print(line)
    with open(LOG, "a") as f:
        f.write(line + "\n")

# ── Reproducibility ────────────────────────────────────────────────────────────
np.random.seed(SEED)
rng = np.random.default_rng(SEED)

# ── GPU info ───────────────────────────────────────────────────────────────────
try:
    import torch
    gpu_ok = torch.cuda.is_available()
    if gpu_ok:
        name = torch.cuda.get_device_name(0)
        vram = torch.cuda.get_device_properties(0).total_memory / 1024**3
        log(f"GPU: {name} ({vram:.1f} GB VRAM)")
    else:
        log("GPU: not available, using CPU (tabular RL is CPU-bound anyway)")
except Exception:
    log("GPU: torch not installed, using CPU (tabular RL is CPU-bound anyway)")

# ── Environment ────────────────────────────────────────────────────────────────
env = gym.make(ENV_NAME)
obs_low = env.observation_space.low
obs_high = env.observation_space.high
n_actions = env.action_space.n

# Clip unbounded dimensions for discretization
obs_low_clipped = np.array([obs_low[0], -3.0, obs_low[2], -3.0])
obs_high_clipped = np.array([obs_high[0], 3.0, obs_high[2], 3.0])

log(f"Env: {ENV_NAME}, actions={n_actions}, bins={N_BINS}")
log(f"State bounds: {obs_low_clipped} → {obs_high_clipped}")
log(f"Episodes: {N_EPISODES}, alpha={ALPHA_START}→{ALPHA_END}, gamma={GAMMA}, eps_decay={EPSILON_DECAY}")

# ── Discretization ─────────────────────────────────────────────────────────────
bins = [np.linspace(obs_low_clipped[i], obs_high_clipped[i], N_BINS + 1)[1:-1]
        for i in range(4)]

def discretize(obs):
    """Convert continuous 4D observation → discrete state index."""
    idx = 0
    for i in range(4):
        digit = np.digitize(obs[i], bins[i])
        idx = idx * N_BINS + digit
    return idx

n_states = N_BINS ** 4
log(f"Discrete states: {n_states} ({N_BINS}^4)")

# ── Q-table ────────────────────────────────────────────────────────────────────
Q = np.zeros((n_states, n_actions), dtype=np.float32)

# ── Training ───────────────────────────────────────────────────────────────────
def choose_action(state_idx, epsilon):
    if rng.random() < epsilon:
        return rng.integers(n_actions)
    q = Q[state_idx]
    return int(np.argmax(q))

episode_rewards = []
episode_lengths = []
epsilons = []
q_means = []
q_maxs = []
running_avg = []

start_time = time.time()

# Write CSV header
with open(CSV_PATH, "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["episode", "reward", "steps", "epsilon", "avg100_reward", "q_mean", "q_max",
                "elapsed_s", "td_error_mean"])

epsilon = EPSILON_START
td_errors = []

def alpha_for_episode(ep):
    """Linearly decay alpha from ALPHA_START to ALPHA_END."""
    frac = (ep - 1) / max(N_EPISODES - 1, 1)
    return ALPHA_START + (ALPHA_END - ALPHA_START) * frac

log("── Training start ──")

for ep in range(1, N_EPISODES + 1):
    obs, _ = env.reset()
    state_idx = discretize(obs)
    action = choose_action(state_idx, epsilon)
    total_reward = 0
    steps = 0
    alpha = alpha_for_episode(ep)

    while True:
        obs_next, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        state_next_idx = discretize(obs_next)

        if done:
            action_next = -1  # no next action in terminal state
            target = reward
        else:
            action_next = choose_action(state_next_idx, epsilon)
            target = reward + GAMMA * Q[state_next_idx, action_next]

        td_error = target - Q[state_idx, action]
        Q[state_idx, action] += alpha * td_error
        td_errors.append(abs(td_error))

        total_reward += reward
        steps += 1

        if done:
            break

        state_idx = state_next_idx
        action = action_next

    epsilon = max(EPSILON_MIN, epsilon * EPSILON_DECAY)

    episode_rewards.append(total_reward)
    episode_lengths.append(steps)
    epsilons.append(epsilon)
    q_means.append(float(np.mean(Q)))
    q_maxs.append(float(np.max(Q)))
    avg100 = np.mean(episode_rewards[-100:])
    running_avg.append(avg100)

    td_mean = float(np.mean(td_errors[-steps:])) if td_errors else 0.0
    elapsed = time.time() - start_time

    with open(CSV_PATH, "a", newline="") as f:
        w = csv.writer(f)
        w.writerow([ep, total_reward, steps, round(epsilon, 6), round(avg100, 2),
                    round(q_means[-1], 4), round(q_maxs[-1], 4),
                    round(elapsed, 1), round(td_mean, 6)])

    if ep % 20 == 0 or ep == 1:
        log(f"Ep {ep:5d} | reward={total_reward:4.0f} | avg100={avg100:6.1f} | "
            f"eps={epsilon:.3f} | q_mean={q_means[-1]:.3f} | elapsed={elapsed:.0f}s")

    # ── Generate PNGs ──────────────────────────────────────────────────────
    if ep % PLOT_EVERY == 0 or ep == N_EPISODES:
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle(f"SARSA on CartPole-v1 — Episode {ep}", fontsize=14, fontweight="bold")

        # Reward
        ax = axes[0, 0]
        ax.plot(episode_rewards, alpha=0.3, color="steelblue", linewidth=0.6, label="Episode")
        ax.plot(running_avg, color="darkorange", linewidth=2, label="Avg 100")
        ax.axhline(y=500, color="green", linestyle="--", alpha=0.5, label="Solved (500)")
        ax.set_xlabel("Episode")
        ax.set_ylabel("Reward")
        ax.set_title("Episode Reward")
        ax.legend()
        ax.grid(True, alpha=0.3)

        # Episode length
        ax = axes[0, 1]
        ax.plot(episode_lengths, alpha=0.5, color="steelblue", linewidth=0.5)
        ax.set_xlabel("Episode")
        ax.set_ylabel("Steps")
        ax.set_title("Episode Length")
        ax.grid(True, alpha=0.3)

        # Epsilon decay
        ax = axes[1, 0]
        ax.plot(epsilons, color="crimson", linewidth=1.5)
        ax.set_xlabel("Episode")
        ax.set_ylabel("Epsilon")
        ax.set_title(f"Exploration Decay (min={EPSILON_MIN})")
        ax.grid(True, alpha=0.3)

        # Q-value distribution
        ax = axes[1, 1]
        q_flat = Q.flatten()
        q_nonzero = q_flat[q_flat != 0]
        ax.hist(q_nonzero, bins=50, color="mediumseagreen", alpha=0.8, edgecolor="white")
        ax.axvline(x=np.mean(q_nonzero), color="red", linestyle="--", linewidth=1.5,
                   label=f"mean={np.mean(q_nonzero):.2f}")
        ax.set_xlabel("Q-value")
        ax.set_ylabel("Count")
        ax.set_title(f"Q-value Distribution ({len(q_nonzero):,} non-zero)")
        ax.legend()
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        fig.savefig(f"{OUT_DIR}/pngs/training_curves.png", dpi=120, bbox_inches="tight")
        plt.close(fig)
        log(f"  → saved PNGs (ep {ep})")

    # ── Save checkpoint ────────────────────────────────────────────────────
    if ep % SAVE_EVERY == 0:
        np.save(f"{OUT_DIR}/checkpoints/q_table_ep{ep:05d}.npy", Q)
        log(f"  → saved Q-table checkpoint (ep {ep})")

# ── Final save ─────────────────────────────────────────────────────────────────
np.save(f"{OUT_DIR}/checkpoints/q_table_final.npy", Q)

elapsed_total = time.time() - start_time
log(f"── Training complete in {elapsed_total:.0f}s ({elapsed_total/60:.1f}m) ──")
log(f"Best avg100 reward: {max(running_avg):.1f} (ep {np.argmax(running_avg)+1})")
log(f"Final avg100 reward: {running_avg[-1]:.1f}")

# ── Evaluation ─────────────────────────────────────────────────────────────────
log("── Evaluation (5 episodes, greedy) ──")
for ep in range(1, 6):
    obs, _ = env.reset()
    state_idx = discretize(obs)
    total_r = 0
    steps = 0
    while True:
        action = int(np.argmax(Q[state_idx]))
        obs, reward, terminated, truncated, _ = env.step(action)
        total_r += reward
        steps += 1
        if terminated or truncated:
            break
        state_idx = discretize(obs)
    log(f"  Eval {ep}: reward={total_r:.0f} steps={steps}")

# ── Summary JSON ───────────────────────────────────────────────────────────────
summary = {
    "env": ENV_NAME,
    "episodes": N_EPISODES,
    "n_states": n_states,
    "n_actions": int(n_actions),
    "alpha_start": ALPHA_START,
    "alpha_end": ALPHA_END,
    "gamma": GAMMA,
    "epsilon_start": EPSILON_START,
    "epsilon_end": EPSILON_MIN,
    "best_avg100_reward": float(max(running_avg)),
    "best_episode": int(np.argmax(running_avg) + 1),
    "final_avg100_reward": float(running_avg[-1]),
    "total_time_s": round(elapsed_total, 1),
}
with open(f"{OUT_DIR}/summary.json", "w") as f:
    json.dump(summary, f, indent=2)

env.close()
log("Done.")
