# DDQN vs NoisyNet — Gotchas

## NoisyLinear noise sampling timing

Noise must be sampled at the start of each episode AND before each training step. Sampling per-action (instead of per-episode) causes the same noise for the whole episode = no exploration. Sampling only before training = stale exploration during rollout.

## SumTree capacity for Prioritized Replay

The SumTree uses a binary tree with capacity rounded up to the next power of two. When the replay buffer is smaller than the tree, indices from SumTree sampling are wrapped via modulo. This is fine in practice but can cause slight bias when buffer_size is far from a power of two.

## MountainCar needs PER (sparse reward)

MountainCar-v0 gives -1 per step until reaching the flag (reward 0). Without PER, transitions that reach the flag are drowned in the replay buffer and never sampled. PER with alpha=0.6 ensures these rare positive transitions are replayed.

## CartPole solved at 200+ avg reward

CartPole-v1 is trivially solved by both DDQN and NoisyNet. It serves as a sanity check — if CartPole doesn't converge, something is wrong with the network initialization or training loop.

## Acrobot needs more episodes than expected

Acrobot-v1 (500 episodes) often looks stuck for the first 300 episodes then rapidly improves. Don't abort early — the agent is exploring the joint space and needs enough experience to discover the swing-up policy.

## GPU not needed for RAM-based control

These environments have 2-6 dimensional observations. MLP networks (128→64) are tiny. CPU is ~90% as fast as GPU. Only use GPU if you already have a session provisioned.

## LayerNorm is critical (v2 fix)

v1 without LayerNorm had training instability — the online network's Q-values would drift and the target network couldn't catch up. LayerNorm after each hidden layer stabilized training across all three environments.
