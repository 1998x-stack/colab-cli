#!/usr/bin/env python3
"""Generate per-environment JSON configs for MuJoCo environments."""
import json
import os


def known_mujoco_envs():
    """Full list of MuJoCo v5 environments."""
    return [
        ("HalfCheetah-v5", 17, 6, "mlp-medium", 1000000),
        ("Hopper-v5", 11, 3, "mlp-medium", 1000000),
        ("Walker2d-v5", 17, 6, "mlp-medium", 1000000),
        ("Ant-v5", 105, 8, "mlp-large", 1000000),
        ("Humanoid-v5", 348, 17, "mlp-large", 2000000),
        ("HumanoidStandup-v5", 348, 17, "mlp-large", 2000000),
        ("Swimmer-v5", 8, 2, "mlp-small", 500000),
        ("Pusher-v5", 23, 7, "mlp-medium", 1000000),
        ("Reacher-v5", 10, 2, "mlp-small", 500000),
        ("InvertedPendulum-v5", 4, 1, "mlp-small", 200000),
        ("InvertedDoublePendulum-v5", 9, 1, "mlp-medium", 500000),
    ]


def main():
    config_dir = os.path.join(os.path.dirname(__file__), "configs")
    os.makedirs(config_dir, exist_ok=True)

    for env_id, obs_dim, n_actions, network, total_timesteps in known_mujoco_envs():
        config = {
            "env_id": env_id,
            "obs_dim": obs_dim,
            "n_actions": n_actions,
            "network": network,
            "total_timesteps": total_timesteps,
        }
        filepath = os.path.join(config_dir, f"{env_id}.json")
        with open(filepath, "w") as f:
            json.dump(config, f, indent=2)

    print(f"Generated {len(known_mujoco_envs())} configs in {config_dir}")


if __name__ == "__main__":
    main()
