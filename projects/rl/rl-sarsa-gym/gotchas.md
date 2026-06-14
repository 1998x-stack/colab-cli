# Tabular SARSA — Gotchas

## Discretization granularity is the limiting factor

With 12 bins per dimension (12^4 = 20,736 states), CartPole's true continuous state space is approximated. The agent plateaus at ~350 avg100 reward (vs 500 max) because the discretization can't distinguish fine motor corrections near the balance point. Increasing bins to 15 would give 15^4 = 50,625 states but doubles training time due to sparse Q-table updates.

## Velocity dimensions must be clipped

Cart velocity and pole angular velocity are unbounded in CartPole-v1. Without clipping to [-3, 3], extreme states (rare but possible) create Q-table entries that are never revisited — wasting memory and causing lookup errors. The clipping bounds cover >99.9% of observed states.

## CPU only — don't use GPU

Tabular SARSA uses NumPy arrays for the Q-table and direct indexing for updates. There's no PyTorch, no GPU acceleration possible. Using a GPU Colab slot for this is wasteful — CPU sessions work perfectly and complete 3000 episodes in ~60 seconds.

## Linear alpha decay > multiplicative epsilon decay

SARSA converges with alpha → 0 (linear decay from 0.5 to 0.01). Epsilon for exploration uses multiplicative decay (0.9985^episode) because it needs to stay high enough during early exploration but drop smoothly. Mixing the two decay schedules is deliberate — they serve different purposes.

## Q-table initialized to zeros (not random)

Unlike neural networks that need random init to break symmetry, tabular Q-learning with epsilon-greedy exploration naturally breaks ties. Zero init is correct and avoids initial bias.

## CartPole is solved at 350+ avg reward

The "solved" threshold for CartPole-v1 is 195 (v0) or 475 (v1, 500 max steps). Tabular SARSA with 12 bins reaches ~350 — below v1 solved but well above random (20-30). The discretization bottleneck means tabular SARSA will never reach 475.

## Checkpoints are .npy files (not .pt)

The Q-table is saved as a NumPy `.npy` file, not a PyTorch `.pt`. This is correct for tabular RL but means you can't resume training with `torch.load()`. Use `np.load("qtable_epXXXX.npy")` to inspect or resume.
