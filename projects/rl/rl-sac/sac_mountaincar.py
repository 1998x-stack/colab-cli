"""SAC (Soft Actor-Critic) for MountainCarContinuous-v0 with automatic entropy tuning."""

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from collections import deque
import random
import os
from datetime import datetime
import json

# ── Hyperparameters ──────────────────────────────────────────────────────────
ENV_NAME = "MountainCarContinuous-v0"
SEED = 42
N_EPISODES = 500
BATCH_SIZE = 256
REPLAY_SIZE = 1_000_000
GAMMA = 0.99
TAU = 0.005
LR = 3e-4
HIDDEN_DIM = 256
START_STEPS = 10_000          # random exploration before learning
UPDATE_EVERY = 1               # update per env step
UPDATES_PER_STEP = 1
LOG_EVERY = 10                # episodes
SAVE_EVERY = 50               # episodes
CHECKPOINT_DIR = "/content/checkpoints"

os.makedirs(CHECKPOINT_DIR, exist_ok=True)

# ── Reproducibility ──────────────────────────────────────────────────────────
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.backends.cudnn.deterministic = True

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")


# ── Networks ─────────────────────────────────────────────────────────────────
def init_weights(module):
    if isinstance(module, nn.Linear):
        fanin, _ = nn.init._calculate_fan_in_and_fan_out(module.weight)
        nn.init.uniform_(module.weight, -1 / np.sqrt(fanin), 1 / np.sqrt(fanin))


class Actor(nn.Module):
    """Gaussian policy: state -> mean, log_std."""

    def __init__(self, obs_dim, act_dim, hidden_dim, action_scale):
        super().__init__()
        self.action_scale = torch.tensor(action_scale, device=DEVICE, dtype=torch.float32)
        self.fc1 = nn.Linear(obs_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.mu = nn.Linear(hidden_dim, act_dim)
        self.log_std = nn.Linear(hidden_dim, act_dim)
        # log_std bounds for numerical stability
        self.log_std_min = -20
        self.log_std_max = 2
        self.apply(init_weights)

    def forward(self, state, deterministic=False, with_logprob=True):
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        mu = self.mu(x)
        log_std = self.log_std(x)
        log_std = torch.clamp(log_std, self.log_std_min, self.log_std_max)
        std = torch.exp(log_std)

        if deterministic:
            action = torch.tanh(mu) * self.action_scale
            return action, None

        dist = torch.distributions.Normal(mu, std)
        u = dist.rsample()  # reparameterized
        action = torch.tanh(u) * self.action_scale

        if with_logprob:
            # logprob with tanh squashing correction
            log_prob = dist.log_prob(u).sum(dim=-1, keepdim=True)
            log_prob -= torch.log(1 - action.pow(2) / self.action_scale.pow(2) + 1e-6).sum(dim=-1, keepdim=True)
        else:
            log_prob = None

        return action, log_prob


class Critic(nn.Module):
    """Twin Q-networks (state, action) -> Q-value."""

    def __init__(self, obs_dim, act_dim, hidden_dim):
        super().__init__()
        # Q1
        self.fc1_1 = nn.Linear(obs_dim + act_dim, hidden_dim)
        self.fc1_2 = nn.Linear(hidden_dim, hidden_dim)
        self.q1 = nn.Linear(hidden_dim, 1)
        # Q2
        self.fc2_1 = nn.Linear(obs_dim + act_dim, hidden_dim)
        self.fc2_2 = nn.Linear(hidden_dim, hidden_dim)
        self.q2 = nn.Linear(hidden_dim, 1)
        self.apply(init_weights)

    def forward(self, state, action):
        x = torch.cat([state, action], dim=-1)
        # Q1
        q1 = F.relu(self.fc1_1(x))
        q1 = F.relu(self.fc1_2(q1))
        q1 = self.q1(q1)
        # Q2
        q2 = F.relu(self.fc2_1(x))
        q2 = F.relu(self.fc2_2(q2))
        q2 = self.q2(q2)
        return q1, q2


# ── Replay Buffer ────────────────────────────────────────────────────────────
class ReplayBuffer:
    def __init__(self, capacity):
        self.buf = deque(maxlen=capacity)

    def push(self, s, a, r, s2, d):
        self.buf.append((s, a, r, s2, d))

    def sample(self, batch_size):
        batch = random.sample(self.buf, batch_size)
        s, a, r, s2, d = zip(*batch)
        return (
            torch.FloatTensor(np.array(s)).to(DEVICE),
            torch.FloatTensor(np.array(a)).to(DEVICE),
            torch.FloatTensor(np.array(r)).unsqueeze(1).to(DEVICE),
            torch.FloatTensor(np.array(s2)).to(DEVICE),
            torch.FloatTensor(np.array(d)).unsqueeze(1).to(DEVICE),
        )

    def __len__(self):
        return len(self.buf)


# ── SAC Agent ────────────────────────────────────────────────────────────────
class SAC:
    def __init__(self, obs_dim, act_dim, action_scale):
        self.actor = Actor(obs_dim, act_dim, HIDDEN_DIM, action_scale).to(DEVICE)
        self.actor_optim = Adam(self.actor.parameters(), lr=LR)

        self.critic = Critic(obs_dim, act_dim, HIDDEN_DIM).to(DEVICE)
        self.critic_target = Critic(obs_dim, act_dim, HIDDEN_DIM).to(DEVICE)
        self.critic_target.load_state_dict(self.critic.state_dict())
        self.critic_optim = Adam(self.critic.parameters(), lr=LR)

        # Automatic entropy tuning
        self.target_entropy = -float(act_dim)
        self.log_alpha = torch.zeros(1, requires_grad=True, device=DEVICE)
        self.alpha_optim = Adam([self.log_alpha], lr=LR)

        self.replay = ReplayBuffer(REPLAY_SIZE)
        self.total_steps = 0

    def select_action(self, state, deterministic=False):
        with torch.no_grad():
            s = torch.FloatTensor(state).unsqueeze(0).to(DEVICE)
            action, _ = self.actor(s, deterministic=deterministic)
            return action.cpu().numpy()[0]

    def update(self):
        if len(self.replay) < BATCH_SIZE:
            return None

        s, a, r, s2, d = self.replay.sample(BATCH_SIZE)

        # ── Critic update ──────────────────────────────────────────────
        with torch.no_grad():
            a2, logp_a2 = self.actor(s2)
            q1_targ, q2_targ = self.critic_target(s2, a2)
            q_targ = torch.min(q1_targ, q2_targ) - self.log_alpha.exp().detach() * logp_a2
            q_target = r + GAMMA * (1 - d) * q_targ

        q1, q2 = self.critic(s, a)
        critic_loss = F.mse_loss(q1, q_target) + F.mse_loss(q2, q_target)

        self.critic_optim.zero_grad()
        critic_loss.backward()
        self.critic_optim.step()

        # ── Actor update ───────────────────────────────────────────────
        a_pi, logp_pi = self.actor(s)
        q1_pi, q2_pi = self.critic(s, a_pi)
        q_pi = torch.min(q1_pi, q2_pi)
        actor_loss = (self.log_alpha.exp().detach() * logp_pi - q_pi).mean()

        self.actor_optim.zero_grad()
        actor_loss.backward()
        self.actor_optim.step()

        # ── Alpha update ───────────────────────────────────────────────
        alpha_loss = -(self.log_alpha * (logp_pi + self.target_entropy).detach()).mean()

        self.alpha_optim.zero_grad()
        alpha_loss.backward()
        self.alpha_optim.step()

        # ── Soft target update ─────────────────────────────────────────
        with torch.no_grad():
            for p, p_targ in zip(self.critic.parameters(), self.critic_target.parameters()):
                p_targ.data.copy_(TAU * p.data + (1 - TAU) * p_targ.data)

        return {
            "critic_loss": critic_loss.item(),
            "actor_loss": actor_loss.item(),
            "alpha": self.log_alpha.exp().item(),
        }

    def save(self, path):
        torch.save({
            "actor": self.actor.state_dict(),
            "critic": self.critic.state_dict(),
            "critic_target": self.critic_target.state_dict(),
            "log_alpha": self.log_alpha,
            "actor_optim": self.actor_optim.state_dict(),
            "critic_optim": self.critic_optim.state_dict(),
            "alpha_optim": self.alpha_optim.state_dict(),
            "total_steps": self.total_steps,
        }, path)

    def load(self, path):
        ckpt = torch.load(path, map_location=DEVICE)
        self.actor.load_state_dict(ckpt["actor"])
        self.critic.load_state_dict(ckpt["critic"])
        self.critic_target.load_state_dict(ckpt["critic_target"])
        self.log_alpha = ckpt["log_alpha"].to(DEVICE)
        self.actor_optim.load_state_dict(ckpt["actor_optim"])
        self.critic_optim.load_state_dict(ckpt["critic_optim"])
        self.alpha_optim.load_state_dict(ckpt["alpha_optim"])
        self.total_steps = ckpt["total_steps"]


# ── Main Training ────────────────────────────────────────────────────────────
def main():
    print(f"[{datetime.now().strftime('%H:%M:%S')}] Starting SAC on {ENV_NAME}")
    print(f"Device: {DEVICE}, Episodes: {N_EPISODES}, Batch: {BATCH_SIZE}")

    env = gym.make(ENV_NAME)
    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.shape[0]
    action_scale = env.action_space.high[0]

    agent = SAC(obs_dim, act_dim, action_scale)

    # Resume from latest checkpoint if exists
    checkpoints = sorted([f for f in os.listdir(CHECKPOINT_DIR) if f.endswith(".pt")])
    if checkpoints:
        latest = os.path.join(CHECKPOINT_DIR, checkpoints[-1])
        print(f"Resuming from {latest}")
        agent.load(latest)

    ep_returns = deque(maxlen=100)
    best_return = -float("inf")

    state, _ = env.reset(seed=SEED)
    episode_return = 0
    episode_len = 0

    for total_steps in range(1, N_EPISODES * 1000 + 1):

        # Select action
        if total_steps < START_STEPS:
            action = env.action_space.sample()
        else:
            action = agent.select_action(state)

        # Step environment
        next_state, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        episode_return += reward
        episode_len += 1

        agent.replay.push(state, action, reward, next_state, float(done))
        state = next_state

        # Update
        if total_steps >= START_STEPS and total_steps % UPDATE_EVERY == 0:
            for _ in range(UPDATES_PER_STEP):
                agent.update()

        if done:
            ep_returns.append(episode_return)
            avg_return = np.mean(ep_returns)

            if episode_return > best_return:
                best_return = episode_return
                agent.save(os.path.join(CHECKPOINT_DIR, "best.pt"))

            if len(ep_returns) % LOG_EVERY == 0:
                alpha = agent.log_alpha.exp().item()
                t = datetime.now().strftime("%H:%M:%S")
                print(f"[{t}] Episode {len(ep_returns):4d} | "
                      f"Return: {episode_return:8.2f} | "
                      f"Avg100: {avg_return:8.2f} | "
                      f"Best: {best_return:8.2f} | "
                      f"Steps: {total_steps:7d} | "
                      f"Alpha: {alpha:.4f}")

            if len(ep_returns) % SAVE_EVERY == 0:
                path = os.path.join(CHECKPOINT_DIR, f"ep{len(ep_returns):04d}.pt")
                agent.save(path)

            if len(ep_returns) >= N_EPISODES:
                break

            state, _ = env.reset()
            episode_return = 0
            episode_len = 0

    # Final save
    agent.save(os.path.join(CHECKPOINT_DIR, "final.pt"))
    print(f"\nTraining complete. Best return: {best_return:.2f}")

    # Summary
    summary = {
        "env": ENV_NAME,
        "episodes_completed": len(ep_returns),
        "total_steps": total_steps,
        "best_return": best_return,
        "final_avg100": avg_return,
        "device": str(DEVICE),
    }
    with open("/content/sac-summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    # ── Evaluation ────────────────────────────────────────────────────
    print("\n── Evaluation ──")
    for ep in range(5):
        state, _ = env.reset()
        total_r = 0
        while True:
            action = agent.select_action(state, deterministic=True)
            state, reward, terminated, truncated, _ = env.step(action)
            total_r += reward
            if terminated or truncated:
                break
        print(f"  Eval ep {ep+1}: return = {total_r:.2f}")

    env.close()


if __name__ == "__main__":
    main()
