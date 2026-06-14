# Classic Control Gymnasium Environments — Complete Reference

Data from gymnasium 1.3.0 (Colab, 2026-06-14). 5 environments, the original RL benchmarks.

## Quick Reference Table

| Env | Obs | Act | Action Type | Steps/Ep | Reward Threshold | Solved |
|-----|-----|-----|------------|----------|------------------|--------|
| CartPole-v1 | 4 | 2 | Discrete | 500 | 475.0 | 195* |
| Acrobot-v1 | 6 | 3 | Discrete | 500 | -100.0 | -100 |
| MountainCar-v0 | 2 | 3 | Discrete | 200 | -110.0 | -110 |
| MountainCarContinuous-v0 | 2 | 1 | Continuous [-1,1] | 999 | 90.0 | 90 |
| Pendulum-v1 | 3 | 1 | Continuous [-2,2] | 200 | None | ~-150† |

*CartPole "solved" in literature is avg reward ≥ 195 over 100 episodes (not the official threshold of 475).
†Pendulum has no official threshold; -150 or better over 100 episodes is considered learned.

## Environment Details

### CartPole-v1
- **Task**: Balance a pole on a moving cart by pushing left or right
- **Obs (4)**: cart position, cart velocity, pole angle, pole angular velocity
- **Act (2)**: push left (0), push right (1) — Discrete
- **Max steps**: 500 (truncated, not terminated)
- **Reward**: +1 for every step the pole stays upright
- **Termination**: pole angle > 12°, cart position > 2.4, or 500 steps reached
- **Versions**: v0 (200 steps), v1 (500 steps)
- **Notes**: The "hello world" of RL. DQN solves it in <100 episodes. Random policy gets ~22 steps. Classic benchmark for debugging.

### Acrobot-v1
- **Task**: Two-link underactuated pendulum must swing up to reach a target height
- **Obs (6)**: cos(theta1), sin(theta1), cos(theta2), sin(theta2), angular velocity 1, angular velocity 2
- **Act (3)**: positive torque, zero torque, negative torque — Discrete
- **Max steps**: 500
- **Reward**: -1 per step until goal reached (sparse negative reward)
- **Goal**: swing end-effector above target height
- **Notes**: Sparse reward makes this harder than CartPole despite small obs. DQN needs ~200 episodes. Good test for exploration.

### MountainCar-v0
- **Task**: Underpowered car must drive up a steep hill by rocking back and forth
- **Obs (2)**: car position (-1.2 to 0.6), car velocity (-0.07 to 0.07)
- **Act (3)**: accelerate left (0), no acceleration (1), accelerate right (2) — Discrete
- **Max steps**: 200
- **Reward**: -1 per step until goal reached
- **Goal**: reach position 0.5 (flag at top of right hill)
- **Solved**: avg reward ≥ -110 over 100 episodes
- **Notes**: Sparse reward + insufficient power = agent must learn to build momentum. First introduced as a challenge for SARSA. TD3/DDPG can solve it, DQN with epsilon-greedy struggles.

### MountainCarContinuous-v0
- **Task**: Same as MountainCar but with continuous throttle
- **Obs (2)**: car position (-1.2 to 0.6), car velocity (-0.07 to 0.07)
- **Act (1)**: continuous force [-1.0, 1.0] — Box
- **Max steps**: 999 (longer than discrete version)
- **Reward**: 100 - (actions² sum) for reaching goal; 0 reward for staying at bottom
- **Solved**: avg reward ≥ 90 over 100 episodes
- **Notes**: Reward function completely different from discrete version! High reward only for reaching goal quickly. DDPG/SAC solve in ~50 episodes. Good sanity test for continuous control algorithms before tackling MuJoCo.

### Pendulum-v1
- **Task**: Swing up and balance an inverted pendulum with continuous torque
- **Obs (3)**: cos(theta), sin(theta), angular velocity
- **Act (1)**: continuous torque [-2.0, 2.0] — Box
- **Max steps**: 200
- **Reward**: -(theta² + 0.1·velocity² + 0.001·action²) — negative, maximized at 0
- **Notes**: Standard continuous control baseline. Max reward is 0 (pendulum upright with zero velocity). DDPG reaches ~-150 in 20 episodes, ~-50 in 100 episodes. Best eval around -40. Catastrophic forgetting is common with DDPG — TD3 or SAC are more reliable.

## Version History

| Env | Versions | Latest |
|-----|----------|--------|
| CartPole | v0, v1 | v1 (500 steps) |
| Acrobot | v1 | v1 |
| MountainCar | v0 | v0 |
| MountainCarContinuous | v0 | v0 |
| Pendulum | v1 | v1 |

## Observation / Action Space Details

### Discrete Action Spaces
```
CartPole-v1:    Discrete(2)  → 0=left, 1=right
Acrobot-v1:     Discrete(3)  → 0=+torque, 1=zero, 2=-torque
MountainCar-v0: Discrete(3)  → 0=left, 1=neutral, 2=right
```

### Continuous Action Spaces
```
Pendulum-v1:               Box([-2.0], [2.0])    — torque
MountainCarContinuous-v0:  Box([-1.0], [1.0])    — force
```

### Observation Spaces (all unbounded Box)
```
CartPole-v1:               Box(4)   [-inf,inf] — but bounded by physics
Acrobot-v1:                Box(6)   [-inf,inf] — cos/sin in [-1,1], others bounded
MountainCar-v0:            Box(2)   [-inf,inf] — pos [-1.2,0.6], vel [-0.07,0.07]
MountainCarContinuous-v0:  Box(2)   [-inf,inf] — same
Pendulum-v1:               Box(3)   [-inf,inf] — cos/sin in [-1,1], vel bounded
```

## Training Time Estimates

All solve in <2 minutes on CPU — no GPU needed:

| Env | Episodes to Solve | Time (CPU) |
|-----|------------------|------------|
| CartPole-v1 | 50-100 | <30s |
| Acrobot-v1 | 100-300 | ~60s |
| MountainCar-v0 | 200-500 | ~60s |
| MountainCarContinuous-v0 | 30-80 | ~30s |
| Pendulum-v1 | 100-200 | ~60s |

## Algorithm Recommendations

| Env | Best | Why |
|-----|------|-----|
| CartPole | DQN | Discrete, dense reward, solved in <2 min |
| Acrobot | DQN / A2C | Discrete, sparse reward, needs exploration |
| MountainCar | DQN + ε-greedy decay | Discrete, hard exploration puzzle |
| MountainCarContinuous | SAC / TD3 | Continuous, dense reward, solved fast |
| Pendulum | TD3 / SAC | Continuous, avoid DDPG forgetting |

For CartPole/MountainCar, a simple MLP Q-network ([64,64] or [128,128]) is sufficient. For the continuous envs, a small actor-critic ([128,128] each) works.

## Minimal Test Script

```python
import gymnasium as gym

for env_id in ["CartPole-v1", "Acrobot-v1", "MountainCar-v0",
               "MountainCarContinuous-v0", "Pendulum-v1"]:
    env = gym.make(env_id)
    obs, _ = env.reset()
    print(f"{env_id:30s} obs={env.observation_space.shape} act={env.action_space}")
    env.close()
```
