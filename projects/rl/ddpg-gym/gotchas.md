# DDPG-Gym — Gotchas

Project-specific issues and lessons learned.

## OU noise warmup critical for Pendulum

Without warmup (1000 steps), the agent explores randomly and never finds the swing-up policy. The noise decay schedule (`max(0.0, 1.0 - step/warmup)`) ensures exploration transitions to exploitation smoothly. Reducing warmup below 500 steps causes training instability.

## Pendulum solved at ~-200 reward

The environment reward ranges from ~-1600 (worst) to 0 (perfect upright). A reward of -200 means the pendulum is consistently near upright. Don't expect positive rewards — the max is 0 only at perfect balance with zero velocity.

## GPU not always faster for DDPG

DDPG on Pendulum-v1 (3D state, 1D action) is CPU-bound due to small network sizes (256x256). GPU provides ~10-20% speedup mainly from batch replay sampling. For harder envs (HalfCheetah, 17D state), GPU becomes significant.

## Gradient clipping prevents value explosion

Without `clip_grad_norm_(1.0)`, the critic's value estimates can explode when the replay buffer has sparse data (early training). This manifests as sudden NaN in actor loss. The clipping is applied to both actor and critic.

## `fanin_init` matches DDPG paper

The uniform initialization `±1/sqrt(fanin)` is the original DDPG initialization (Lillicrap et al. 2016). Xavier/Glorot init produces worse results on Pendulum — the actor saturates tanh output too early.

## `metrics.json` vs `metrics.csv`

DDPG uses JSON for metrics (nested eval entries with mean±std). This is more expressive than flat CSV for RL with evaluation windows. The cron fetch.sh parses JSON directly. If you need CSV for analysis, convert with:
```python
python -c "
import json, csv
with open('metrics.json') as f: m = json.load(f)
with open('metrics.csv', 'w') as f:
    w = csv.writer(f)
    w.writerow(['episode','reward','steps','total_steps','actor_loss','critic_loss','buffer_size'])
    for e in m['episodes']:
        w.writerow([e['episode'],e['reward'],e['steps'],e['total_steps'],e.get('actor_loss'),e.get('critic_loss'),e['buffer_size']])
"
```
