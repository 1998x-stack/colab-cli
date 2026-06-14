"""DQN on Atari Pong — CNN-based Deep Q-Network with dueling architecture.

Frame preprocessing: grayscale → resize 84×84 → stack 4 frames.
Standard DQN ingredients: experience replay, target network, epsilon-greedy.
Saves checkpoints, metrics, and evaluation video to <output_dir>/.
"""

import json
import os
import time
from collections import deque
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam

import gymnasium as gym
import ale_py
gym.register_envs(ale_py)

# ── Config ───────────────────────────────────────────────────────────────────
ENV_NAME = "ALE/Pong-v5"
OUTPUT_DIR = "/content/dqn-output"
FRAME_STACK = 4
FRAME_SIZE = 84
REPLAY_SIZE = 100_000
BATCH_SIZE = 32
LR = 1e-4
GAMMA = 0.99
TARGET_UPDATE_EVERY = 1000
EPS_START = 1.0
EPS_END = 0.01
EPS_DECAY = 50_000
MAX_EPISODES = 500
LEARN_EVERY = 4           # act every N frames
LEARN_STEPS = 1
SAVE_EVERY = 50
LOG_EVERY = 10
MAX_STEPS_PER_EP = 10_000
EVAL_EPISODES = 3

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
os.makedirs(OUTPUT_DIR, exist_ok=True)


# ── Logging ───────────────────────────────────────────────────────────────────
def log(msg):
    t = datetime.now().strftime("%H:%M:%S")
    print(f"[{t}] {msg}", flush=True)


# ── Frame preprocessing ──────────────────────────────────────────────────────
class FrameProcessor:
    """Convert (210, 160, 3) RGB → (84, 84) grayscale, normalize to [0,1]."""

    def __init__(self):
        pass

    @staticmethod
    def process(frame):
        import cv2
        # frame is (H, W, 3) uint8
        gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        resized = cv2.resize(gray, (FRAME_SIZE, FRAME_SIZE),
                             interpolation=cv2.INTER_AREA)
        return resized.astype(np.float32) / 255.0


class FrameStack:
    """Maintain a stack of the last N frames, return (N, H, W) tensor."""

    def __init__(self, n=FRAME_STACK):
        self.n = n
        self.frames = deque(maxlen=n)

    def reset(self, env):
        obs, _ = env.reset()
        frame = FrameProcessor.process(obs)
        for _ in range(self.n):
            self.frames.append(frame)
        return self.get()

    def step(self, env, action):
        obs, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        frame = FrameProcessor.process(obs)
        self.frames.append(frame)
        return self.get(), reward, done

    def get(self):
        return torch.FloatTensor(np.stack(list(self.frames))).to(DEVICE)


# ── Replay Buffer ────────────────────────────────────────────────────────────
class ReplayBuffer:
    def __init__(self, capacity):
        self.buf = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.buf.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        batch = np.random.choice(len(self.buf), batch_size, replace=False)
        states, actions, rewards, next_states, dones = [], [], [], [], []
        for i in batch:
            s, a, r, ns, d = self.buf[i]
            states.append(s)
            actions.append(a)
            rewards.append(r)
            next_states.append(ns)
            dones.append(d)
        return (
            torch.stack(states),
            torch.LongTensor(actions).to(DEVICE),
            torch.FloatTensor(rewards).unsqueeze(1).to(DEVICE),
            torch.stack(next_states),
            torch.FloatTensor(dones).unsqueeze(1).to(DEVICE),
        )

    def __len__(self):
        return len(self.buf)


# ── Dueling DQN ──────────────────────────────────────────────────────────────
class DuelingDQN(nn.Module):
    """CNN encoder → dueling streams → Q(s, a)."""

    def __init__(self, action_dim):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(FRAME_STACK, 32, 8, stride=4), nn.ReLU(),
            nn.Conv2d(32, 64, 4, stride=2), nn.ReLU(),
            nn.Conv2d(64, 64, 3, stride=1), nn.ReLU(),
        )
        # conv output size for (4, 84, 84): 64 × 7 × 7 = 3136
        self.conv_out_size = 3136

        self.value = nn.Sequential(
            nn.Linear(self.conv_out_size, 256), nn.ReLU(),
            nn.Linear(256, 1),
        )
        self.advantage = nn.Sequential(
            nn.Linear(self.conv_out_size, 256), nn.ReLU(),
            nn.Linear(256, action_dim),
        )
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            nn.init.kaiming_uniform_(m.weight, nonlinearity="relu")
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        c = self.conv(x).flatten(1)
        v = self.value(c)
        a = self.advantage(c)
        return v + a - a.mean(dim=1, keepdim=True)


# ── DQN Agent ────────────────────────────────────────────────────────────────
class DQNAgent:
    def __init__(self, action_dim):
        self.action_dim = action_dim
        self.online = DuelingDQN(action_dim).to(DEVICE)
        self.target = DuelingDQN(action_dim).to(DEVICE)
        self.target.load_state_dict(self.online.state_dict())
        self.optimizer = Adam(self.online.parameters(), lr=LR)
        self.replay = ReplayBuffer(REPLAY_SIZE)
        self.total_steps = 0
        self.epsilon = EPS_START

    def select_action(self, state, eval_mode=False):
        if eval_mode or np.random.random() > self.epsilon:
            with torch.no_grad():
                q = self.online(state.unsqueeze(0))
                return q.argmax(dim=1).item()
        return np.random.randint(self.action_dim)

    def update_epsilon(self):
        self.epsilon = max(EPS_END, EPS_START -
                           (EPS_START - EPS_END) * self.total_steps / EPS_DECAY)

    def learn(self):
        if len(self.replay) < BATCH_SIZE:
            return None

        s, a, r, ns, d = self.replay.sample(BATCH_SIZE)

        with torch.no_grad():
            q_next = self.target(ns).max(dim=1, keepdim=True).values
            q_target = r + GAMMA * (1 - d) * q_next

        q_online = self.online(s).gather(1, a.unsqueeze(1))
        loss = F.smooth_l1_loss(q_online, q_target)

        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.online.parameters(), 10)
        self.optimizer.step()

        if self.total_steps % TARGET_UPDATE_EVERY == 0:
            self.target.load_state_dict(self.online.state_dict())

        self.total_steps += 1
        self.update_epsilon()
        return loss.item()

    def save(self, path):
        torch.save({
            "online": self.online.state_dict(),
            "target": self.target.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "total_steps": self.total_steps,
            "epsilon": self.epsilon,
        }, path)

    def load(self, path):
        ckpt = torch.load(path, map_location=DEVICE)
        self.online.load_state_dict(ckpt["online"])
        self.target.load_state_dict(ckpt["target"])
        self.optimizer.load_state_dict(ckpt["optimizer"])
        self.total_steps = ckpt["total_steps"]
        self.epsilon = ckpt["epsilon"]


# ── Evaluation ───────────────────────────────────────────────────────────────
def evaluate(agent, env, episodes=EVAL_EPISODES, record_dir=None):
    agent.online.eval()
    stack = FrameStack()
    returns = []
    for ep in range(episodes):
        state = stack.reset(env)
        total_r = 0
        while True:
            action = agent.select_action(state, eval_mode=True)
            state, reward, done = stack.step(env, action)
            total_r += reward
            if done:
                break
        returns.append(total_r)
    agent.online.train()
    return np.mean(returns)


# ── Main ─────────────────────────────────────────────────────────────────────
def main():
    log(f"Starting DQN on {ENV_NAME}")
    log(f"Device: {DEVICE}  Episodes: {MAX_EPISODES}  Batch: {BATCH_SIZE}")
    log(f"Output: {OUTPUT_DIR}")

    env = gym.make(ENV_NAME, mode=0, difficulty=0)  # mode=0: easy
    action_dim = env.action_space.n
    log(f"Actions: {action_dim}")

    agent = DQNAgent(action_dim)

    # Resume from checkpoint
    checkpoints = sorted([f for f in os.listdir(OUTPUT_DIR) if f.endswith(".pt")])
    if checkpoints:
        latest = os.path.join(OUTPUT_DIR, checkpoints[-1])
        log(f"Resuming from {latest}")
        agent.load(latest)

    stack = FrameStack()
    ep_returns = deque(maxlen=100)
    best_return = -float("inf")
    metrics_log = []
    losses = []

    t0 = time.time()

    for episode in range(1, MAX_EPISODES + 1):
        state = stack.reset(env)
        episode_return = 0
        episode_loss = []

        for step in range(MAX_STEPS_PER_EP):
            # Act
            if step % LEARN_EVERY == 0:
                action = agent.select_action(state)
            else:
                action = 0  # repeat last action (frame skip)

            next_state, reward, done = stack.step(env, action)
            episode_return += reward

            # Store
            agent.replay.push(state, action, reward, next_state, float(done))
            state = next_state

            # Learn
            if step % LEARN_EVERY == 0:
                for _ in range(LEARN_STEPS):
                    loss = agent.learn()
                    if loss is not None:
                        episode_loss.append(loss)

            if done:
                break

        ep_returns.append(episode_return)
        avg_return = np.mean(ep_returns)
        avg_loss = np.mean(episode_loss) if episode_loss else 0
        losses.append(avg_loss)

        if episode_return > best_return:
            best_return = episode_return
            agent.save(os.path.join(OUTPUT_DIR, "best.pt"))

        entry = {
            "episode": episode,
            "return": round(episode_return, 1),
            "avg_return_100": round(avg_return, 1),
            "epsilon": round(agent.epsilon, 4),
            "loss": round(avg_loss, 6),
            "steps": agent.total_steps,
        }
        metrics_log.append(entry)

        if episode % LOG_EVERY == 0:
            t = datetime.now().strftime("%H:%M:%S")
            log(f"[{t}] Ep {episode:4d} | Return: {episode_return:7.1f} | "
                f"Avg100: {avg_return:7.1f} | Best: {best_return:7.1f} | "
                f"Eps: {agent.epsilon:.3f} | Loss: {avg_loss:.4f} | "
                f"Steps: {agent.total_steps:7d}")

        if episode % SAVE_EVERY == 0:
            path = os.path.join(OUTPUT_DIR, f"ep{episode:04d}.pt")
            agent.save(path)

        # Early termination on good performance
        if avg_return > 18 and episode > 100:
            log(f"Pong solved! (avg100 return {avg_return:.1f} > 18)")
            break

    train_time = time.time() - t0

    # ── Final save ──────────────────────────────────────────────────
    agent.save(os.path.join(OUTPUT_DIR, "final.pt"))

    # ── Evaluation ──────────────────────────────────────────────────
    log("\n── Evaluation ──")
    eval_env = gym.make(ENV_NAME, mode=0, difficulty=0)
    eval_return = evaluate(agent, eval_env, episodes=10)
    log(f"Eval avg return (10 eps): {eval_return:.1f}")
    eval_env.close()
    env.close()

    # ── Visualizations ──────────────────────────────────────────────
    log("Generating plots...")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    returns = [m["return"] for m in metrics_log]
    avg100 = [m["avg_return_100"] for m in metrics_log]

    axes[0].plot(returns, alpha=0.3, linewidth=0.5, color="blue")
    axes[0].plot(avg100, linewidth=2, color="blue")
    axes[0].axhline(y=18, color="green", linestyle="--", label="Solved (18)")
    axes[0].set_xlabel("Episode"); axes[0].set_ylabel("Return")
    axes[0].set_title("DQN Pong — Episode Returns"); axes[0].legend()
    axes[0].grid(True)

    axes[1].plot(losses, linewidth=0.5, color="red")
    axes[1].set_xlabel("Episode"); axes[1].set_ylabel("Loss")
    axes[1].set_title("DQN Pong — TD Loss"); axes[1].grid(True)

    epsilons = [m["epsilon"] for m in metrics_log]
    axes[2].plot(epsilons, linewidth=1.5, color="purple")
    axes[2].set_xlabel("Episode"); axes[2].set_ylabel("Epsilon")
    axes[2].set_title("DQN Pong — Epsilon Decay"); axes[2].grid(True)

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "training_curves.png"), dpi=120)
    plt.close()

    # ── Metrics ─────────────────────────────────────────────────────
    with open(os.path.join(OUTPUT_DIR, "metrics.json"), "w") as f:
        json.dump({
            "env": ENV_NAME,
            "algorithm": "DQN (dueling)",
            "best_return": round(best_return, 1),
            "eval_return_10eps": round(eval_return, 1),
            "total_episodes": len(returns),
            "total_steps": agent.total_steps,
            "train_time_seconds": round(train_time, 1),
            "device": str(DEVICE),
            "hyperparameters": {
                "batch_size": BATCH_SIZE, "lr": LR, "gamma": GAMMA,
                "replay_size": REPLAY_SIZE, "target_update": TARGET_UPDATE_EVERY,
                "eps_start": EPS_START, "eps_end": EPS_END, "eps_decay": EPS_DECAY,
                "frame_stack": FRAME_STACK, "frame_size": FRAME_SIZE,
            },
            "episode_history": metrics_log,
        }, f, indent=2)

    # Standalone summary for cron fetch
    summary = {
        "env": ENV_NAME,
        "best_return": best_return,
        "total_episodes": len(returns),
        "total_steps": agent.total_steps,
        "train_time_seconds": round(train_time, 1),
        "solved": best_return > 18,
        "device": str(DEVICE),
    }
    with open(os.path.join(OUTPUT_DIR, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    log(f"\nDone in {train_time/60:.1f}m. Best return: {best_return:.1f}")
    log(f"Output: {OUTPUT_DIR}/")


if __name__ == "__main__":
    main()
