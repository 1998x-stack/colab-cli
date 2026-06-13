#!/usr/bin/env python3
"""PPO on Atari RAM — CleanRL-style training loop with pluggable components."""
import os
import sys
import csv
import json
import time
import argparse
from datetime import datetime
from collections import deque

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from config_loader import load_config
from networks import NETWORK_REGISTRY
from ppo_agent import PPOAgent
from env_factory import make_ram_env

parser = argparse.ArgumentParser()
parser.add_argument("--envs", nargs="+",
                    default=["ALE/Pong-ram-v5", "ALE/Asterix-ram-v5", "ALE/MsPacman-ram-v5"])
parser.add_argument("--out_dir", default="/content/ppo-atari-output")
parser.add_argument("--config_dir", default="configs")
parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
args = parser.parse_args()

os.makedirs(f"{args.out_dir}/logs", exist_ok=True)
os.makedirs(f"{args.out_dir}/pngs", exist_ok=True)
os.makedirs(f"{args.out_dir}/checkpoints", exist_ok=True)

LOG_PATH = f"{args.out_dir}/logs/train.log"
CSV_PATH = f"{args.out_dir}/metrics.csv"

device = torch.device(args.device)


# ── Logging ──────────────────────────────────────────────────────────────
def log(msg):
    line = f"[{datetime.now().strftime('%H:%M:%S')}] {msg}"
    print(line, flush=True)
    with open(LOG_PATH, "a") as f:
        f.write(line + "\n")


# ── CSV ──────────────────────────────────────────────────────────────────
csv_file = open(CSV_PATH, "w", newline="")
csv_writer = csv.writer(csv_file)
csv_writer.writerow([
    "iteration", "env", "total_steps", "mean_reward", "avg10_reward",
    "eval_reward", "pg_loss", "vf_loss", "entropy", "clip_frac",
    "lr", "elapsed_s"
])
csv_file.flush()


# ── Plotting ─────────────────────────────────────────────────────────────
def plot_curves(env_name, history, out_dir):
    """2x2 panel: reward, losses, entropy, clip fraction."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(f"PPO — {env_name}", fontsize=14, fontweight="bold")

    iters = [h["iteration"] for h in history]
    rewards = [h["mean_reward"] for h in history]
    avg10 = [h["avg10_reward"] for h in history]
    solved = history[0].get("solved_threshold", 0)

    # Reward
    ax = axes[0, 0]
    ax.plot(iters, rewards, alpha=0.35, color="steelblue", linewidth=0.6, label="Mean reward")
    ax.plot(iters, avg10, color="darkorange", linewidth=2, label="Avg10")
    if solved:
        ax.axhline(y=solved, color="green", linestyle="--", alpha=0.5,
                   label=f"Solved ({solved})")
    ax.set_xlabel("Iteration"); ax.set_ylabel("Reward")
    ax.set_title("Training Reward"); ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

    # Losses
    ax = axes[0, 1]
    ax.plot(iters, [h["pg_loss"] for h in history], color="crimson",
            linewidth=1.2, label="Policy loss")
    ax.plot(iters, [h["vf_loss"] for h in history], color="royalblue",
            linewidth=1.2, label="Value loss")
    ax.set_xlabel("Iteration"); ax.set_ylabel("Loss")
    ax.set_title("Losses"); ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

    # Entropy
    ax = axes[1, 0]
    ax.plot(iters, [h["entropy"] for h in history], color="mediumseagreen", linewidth=1.5)
    ax.set_xlabel("Iteration"); ax.set_ylabel("Entropy")
    ax.set_title("Policy Entropy"); ax.grid(True, alpha=0.3)

    # Clip fraction
    ax = axes[1, 1]
    ax.plot(iters, [h["clip_frac"] for h in history], color="darkviolet", linewidth=1.5)
    ax.axhline(y=0.1, color="gray", linestyle="--", alpha=0.5, label="clip_coef=0.1")
    ax.set_xlabel("Iteration"); ax.set_ylabel("Clip fraction")
    ax.set_title("PPO Clip Fraction"); ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

    plt.tight_layout()
    fig.savefig(f"{out_dir}/pngs/{env_name.replace('/', '_')}_curves.png",
                dpi=120, bbox_inches="tight")
    plt.close(fig)


def plot_comparison(all_env_histories, out_dir):
    """Single figure: eval reward curves for all trained envs overlaid."""
    if len(all_env_histories) < 2:
        return
    fig, ax = plt.subplots(figsize=(12, 6))
    colors = plt.cm.tab10(np.linspace(0, 1, len(all_env_histories)))
    for idx, (name, history) in enumerate(sorted(all_env_histories.items())):
        if not history:
            continue
        evals = [(h["iteration"], h["eval_reward"]) for h in history
                 if h.get("eval_reward") is not None]
        if evals:
            xs, ys = zip(*evals)
            ax.plot(xs, ys, color=colors[idx], linewidth=1.8, label=name,
                    marker="o", markersize=3)
    ax.set_xlabel("Iteration"); ax.set_ylabel("Eval Reward")
    ax.set_title("PPO Atari RAM — Evaluation Comparison")
    ax.legend(fontsize=8); ax.grid(True, alpha=0.3)
    plt.tight_layout()
    fig.savefig(f"{out_dir}/pngs/comparison.png", dpi=120, bbox_inches="tight")
    plt.close(fig)


# ── Eval ─────────────────────────────────────────────────────────────────
@torch.no_grad()
def evaluate(agent, env_id, n_episodes=5):
    import gymnasium as gym
    eval_env = gym.make(env_id, max_episode_steps=108000)
    rewards = []
    for _ in range(n_episodes):
        obs, _ = eval_env.reset()
        obs = obs.astype(np.float32) / 255.0
        ep_reward = 0.0
        done = False
        while not done:
            action, _, _ = agent.act(torch.FloatTensor(obs).unsqueeze(0))
            obs, reward, terminated, truncated, _ = eval_env.step(action.item())
            obs = obs.astype(np.float32) / 255.0
            ep_reward += reward
            done = terminated or truncated
        rewards.append(ep_reward)
    eval_env.close()
    return np.mean(rewards), np.std(rewards)


# ── Main ─────────────────────────────────────────────────────────────────
log(f"=== PPO Atari RAM | device={device} | {datetime.now()} ===")
log(f"Envs: {args.envs}")

all_histories = {}

for env_idx, env_id in enumerate(args.envs):
    log(f"\n{'=' * 60}")
    log(f"[{env_idx + 1}/{len(args.envs)}] {env_id}")
    log(f"{'=' * 60}")

    # NOTE: config filenames use hyphens: "ALE/Pong-ram-v5" → "ALE-Pong-ram-v5.json"
    cfg = load_config(env_id.replace("/", "-"), args.config_dir)
    log(f"Config: network={cfg['network']} timesteps={cfg['total_timesteps']} "
        f"n_actions={cfg['n_actions']} lr={cfg['lr']}")

    num_envs = cfg["num_envs"]
    num_steps = cfg["num_steps"]
    total_timesteps = cfg["total_timesteps"]
    n_iterations = total_timesteps // (num_envs * num_steps)

    envs = make_ram_env(env_id, num_envs=num_envs, seed=cfg["seed"])

    net = NETWORK_REGISTRY[cfg["network"]](cfg["n_actions"])
    agent = PPOAgent(
        network=net,
        lr=cfg["lr"],
        gamma=cfg["gamma"],
        gae_lambda=cfg["gae_lambda"],
        clip_coef=cfg["clip_coef"],
        ent_coef=cfg["ent_coef"],
        vf_coef=cfg["vf_coef"],
        max_grad_norm=cfg["max_grad_norm"],
        n_epochs=cfg["n_epochs"],
        n_minibatches=cfg["n_minibatches"],
        device=str(device),
    )

    history = []
    reward_buffer = deque(maxlen=10)
    best_eval = -float("inf")
    start_time = time.time()

    obs, _ = envs.reset()
    obs = torch.FloatTensor(obs)

    for iteration in range(1, n_iterations + 1):
        # ── Rollout collection ──────────────────────────────────────────
        rollouts = {
            "obs": torch.zeros(num_steps, num_envs, 128),
            "actions": torch.zeros(num_steps, num_envs, dtype=torch.long),
            "log_probs": torch.zeros(num_steps, num_envs),
            "rewards": torch.zeros(num_steps, num_envs),
            "dones": torch.zeros(num_steps, num_envs),
            "values": torch.zeros(num_steps, num_envs),
        }

        for t in range(num_steps):
            actions, log_probs, values = agent.act(obs)
            next_obs, rewards, terminateds, truncateds, _ = envs.step(actions.numpy())
            dones = np.logical_or(terminateds, truncateds)

            rollouts["obs"][t] = obs
            rollouts["actions"][t] = actions
            rollouts["log_probs"][t] = log_probs
            rollouts["rewards"][t] = torch.FloatTensor(rewards)
            rollouts["dones"][t] = torch.FloatTensor(dones)
            rollouts["values"][t] = values

            obs = torch.FloatTensor(next_obs)
            dones_t = torch.FloatTensor(dones)

        # ── Compute returns ─────────────────────────────────────────────
        next_done = torch.FloatTensor(dones)
        advantages, returns = agent.compute_returns(
            obs, next_done, rollouts["rewards"],
            rollouts["dones"], rollouts["values"],
        )
        rollouts["advantages"] = advantages
        rollouts["returns"] = returns

        # ── PPO update ──────────────────────────────────────────────────
        metrics = agent.update(rollouts)

        mean_reward = rollouts["rewards"].mean().item()
        reward_buffer.append(mean_reward)
        avg10 = np.mean(reward_buffer) if reward_buffer else mean_reward

        elapsed = time.time() - start_time
        total_steps = iteration * num_envs * num_steps

        # ── Log ─────────────────────────────────────────────────────────
        history.append({
            "iteration": iteration,
            "mean_reward": mean_reward,
            "avg10_reward": avg10,
            "eval_reward": None,
            "pg_loss": metrics["pg_loss"],
            "vf_loss": metrics["vf_loss"],
            "entropy": metrics["entropy"],
            "clip_frac": metrics["clip_frac"],
        })

        csv_writer.writerow([
            iteration, env_id, total_steps,
            round(mean_reward, 2), round(avg10, 2),
            None,
            round(metrics["pg_loss"], 6), round(metrics["vf_loss"], 6),
            round(metrics["entropy"], 4), round(metrics["clip_frac"], 4),
            cfg["lr"], round(elapsed, 1),
        ])
        csv_file.flush()

        log(f"iter {iteration:3d}/{n_iterations} | {env_id.split('/')[1]:20s} "
            f"reward={mean_reward:7.2f} avg10={avg10:7.2f} | "
            f"pg_loss={metrics['pg_loss']:.4f} vf_loss={metrics['vf_loss']:.4f} | "
            f"ent={metrics['entropy']:.2f} clip={metrics['clip_frac']:.3f} | "
            f"steps={total_steps}")

        # ── Eval ────────────────────────────────────────────────────────
        if iteration % cfg["eval_interval"] == 0:
            eval_mean, eval_std = evaluate(agent, env_id, cfg["eval_episodes"])
            history[-1]["eval_reward"] = eval_mean
            log(f"  EVAL: mean={eval_mean:.2f} +- {eval_std:.2f}")
            if eval_mean > best_eval:
                best_eval = eval_mean
                torch.save({
                    "model": agent.net.state_dict(),
                    "iteration": iteration,
                    "eval_reward": eval_mean,
                }, f"{args.out_dir}/checkpoints/{env_id.replace('/', '_')}_best.pt")
                log(f"  -> new best! saved checkpoint")

        # ── Plot ────────────────────────────────────────────────────────
        if iteration % cfg["plot_interval"] == 0:
            plot_curves(env_id.split("/")[1], history, args.out_dir)
            all_histories[env_id.split("/")[1]] = history
            plot_comparison(all_histories, args.out_dir)

    # ── End of env loop ─────────────────────────────────────────────────
    plot_curves(env_id.split("/")[1], history, args.out_dir)
    all_histories[env_id.split("/")[1]] = history
    plot_comparison(all_histories, args.out_dir)
    envs.close()
    log(f"[DONE {env_id}] best_eval={best_eval:.1f} elapsed={elapsed:.0f}s")

# ── Final ────────────────────────────────────────────────────────────────
csv_file.close()

# Summary JSON
summary = {}
for env_name, hist in all_histories.items():
    evals = [h["eval_reward"] for h in hist if h["eval_reward"] is not None]
    summary[env_name] = {
        "iterations": len(hist),
        "final_avg10_reward": hist[-1]["avg10_reward"] if hist else 0,
        "best_eval_reward": max(evals) if evals else 0,
        "final_eval_reward": evals[-1] if evals else 0,
    }

with open(f"{args.out_dir}/summary.json", "w") as f:
    json.dump(summary, f, indent=2)

log(f"\n=== ALL DONE | {datetime.now()} ===")
for env_name, s in sorted(summary.items()):
    log(f"  {env_name:30s}  best_eval={s['best_eval_reward']:8.1f}  "
        f"final_avg10={s['final_avg10_reward']:8.1f}")
