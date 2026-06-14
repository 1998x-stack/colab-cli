# SAC MountainCarContinuous — Gotchas

## MountainCarContinuous is surprisingly hard for SAC

The environment gives reward proportional to `speed × position` — the agent must learn to drive back and forth to build momentum before reaching the flag. SAC typically takes 150-250 episodes before the first successful ascent. The first ~100 episodes show flat reward.

## Automatic entropy tuning (alpha) is critical

Without learnable alpha (fixed alpha=0.2), SAC either over-explores (never converges) or under-explores (never discovers the flag). With automatic tuning (target entropy = -action_dim), alpha self-adjusts: starts high (~0.5) during exploration, drops to ~0.02 once the policy stabilizes.

## 10,000 random steps before learning starts

SAC needs substantial random exploration data before the critic has enough coverage to provide meaningful Q-values. The 10K `START_STEPS` fill the replay buffer with diverse transitions. Reducing this below 5K causes the Q-function to overfit to a narrow region.

## GPU not needed for 2D continuous control

MountainCarContinuous has 2D state and 1D action. SAC networks (256x256) are tiny. CPU is ~95% as fast as GPU for this env. Use CPU to save a Colab GPU slot for pixel-based projects.

## Checkpoint resume is simple but fragile

The train script checks for the latest `.pt` file in `/content/checkpoints/` by modification time. If you upload a checkpoint with a newer timestamp (e.g., after git clone), it will resume from the wrong file.

## Log output goes to stdout only (no file logging)

Unlike other projects, SAC logs to stdout (captured by launch.py to `sac_train.log`). There's no direct file logging in the train script. The cron fetch.sh downloads `sac_train.log` from the VM, which is written by launch.py's Popen stdout redirect.

## No PNG generation

This project doesn't generate training curves. Add matplotlib-based plotting if you need visual monitoring. For SAC on MountainCarContinuous, the most informative plot is reward + alpha over episodes.
