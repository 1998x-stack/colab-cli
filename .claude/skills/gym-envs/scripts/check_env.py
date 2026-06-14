#!/usr/bin/env python3
"""Quick Gymnasium environment spec checker. Works locally or on Colab.

Usage:
  python check_env.py HalfCheetah-v5       # single env
  python check_env.py --all-mujoco         # all 11 MuJoCo envs
  python check_env.py --all-classic        # all 5 Classic Control envs
  python check_env.py --all-atari          # all 104 Atari games (summary)
  python check_env.py --atari-sample       # 10 representative Atari games
"""
import sys, argparse
import gymnasium as gym

# Known env lists (authoritative as of gymnasium 1.3.0)
MUJOCO_ENVS = [
    "HalfCheetah-v5", "Hopper-v5", "Walker2d-v5", "Ant-v5",
    "Humanoid-v5", "HumanoidStandup-v5", "Swimmer-v5", "Pusher-v5",
    "Reacher-v5", "InvertedPendulum-v5", "InvertedDoublePendulum-v5",
]
CLASSIC_ENVS = [
    "CartPole-v1", "Acrobot-v1", "MountainCar-v0",
    "MountainCarContinuous-v0", "Pendulum-v1",
]
ATARI_SAMPLE = [
    "ALE/Pong-v5", "ALE/Breakout-v5", "ALE/SpaceInvaders-v5",
    "ALE/Seaquest-v5", "ALE/Qbert-v5", "ALE/BeamRider-v5",
    "ALE/Enduro-v5", "ALE/MontezumaRevenge-v5",
    "ALE/Freeway-v5", "ALE/Asteroids-v5",
]


def check_env(env_id, obs_type=None):
    """Print full spec for one environment."""
    try:
        kwargs = {}
        if obs_type:
            kwargs["obs_type"] = obs_type
        env = gym.make(env_id, **kwargs)

        obs_space = env.observation_space
        act_space = env.action_space

        print(f"\n{'='*60}")
        print(f"  {env_id}" + (f" (obs_type={obs_type})" if obs_type else ""))
        print(f"{'='*60}")

        obs_bounded = "bounded" if hasattr(obs_space, 'is_bounded') and obs_space.is_bounded() else "unbounded"
        print(f"  Observation: {type(obs_space).__name__}{obs_space.shape}  {obs_space.dtype}  [{obs_bounded}]")

        print(f"  Action:      {type(act_space).__name__}{act_space.shape}  {act_space.dtype}", end="")
        if hasattr(act_space, 'high'):
            low, high = float(act_space.low[0]), float(act_space.high[0])
            print(f"  range=[{low:.3g}, {high:.3g}]")
        elif hasattr(act_space, 'n'):
            print(f"  n={act_space.n}")

        if env.spec:
            print(f"  Max steps:   {env.spec.max_episode_steps}")
            print(f"  Threshold:   {env.spec.reward_threshold}")

        if hasattr(env, 'get_action_meanings'):
            try:
                meanings = env.get_action_meanings()
                print(f"  Actions:     {meanings}")
            except Exception:
                pass

        env.close()
        return True
    except Exception as e:
        print(f"\n  {env_id}: ERROR — {e}")
        return False


def main():
    parser = argparse.ArgumentParser(description="Gymnasium environment spec checker")
    group = parser.add_mutually_exclusive_group()
    group.add_argument("env_id", nargs="?", help="Single environment ID to check")
    group.add_argument("--all-mujoco", action="store_true")
    group.add_argument("--all-classic", action="store_true")
    group.add_argument("--all-atari", action="store_true")
    group.add_argument("--atari-sample", action="store_true")
    args = parser.parse_args()

    if args.all_mujoco:
        envs = MUJOCO_ENVS
    elif args.all_classic:
        envs = CLASSIC_ENVS
    elif args.all_atari:
        import ale_py  # noqa
        envs = sorted([k for k in gym.envs.registry if k.startswith("ALE/")])
        # Summary mode for 104 games
        print(f"=== Atari Summary: {len(envs)} games ===\n")
        print(f"{'Game':30s} {'Acts':>5s} {'Obs Shape':>12s}")
        print("-" * 50)
        for eid in envs:
            try:
                env = gym.make(eid)
                acts = env.action_space.n
                obs_shape = str(env.observation_space.shape)
                game = eid.replace("ALE/", "").replace("-v5", "")
                print(f"{game:30s} {acts:5d} {obs_shape:>12s}")
                env.close()
            except Exception as e:
                print(f"{eid:30s} ERROR: {e}")
        return
    elif args.atari_sample:
        import ale_py  # noqa
        envs = ATARI_SAMPLE
    elif args.env_id:
        envs = [args.env_id]
    else:
        parser.print_help()
        return

    ok = 0
    for eid in envs:
        if check_env(eid):
            ok += 1
        # Also check RAM mode for Atari
        if eid.startswith("ALE/") and args.atari_sample:
            check_env(eid, obs_type="ram")

    print(f"\n{'='*60}")
    print(f"  Checked: {len(envs)}  OK: {ok}  Failed: {len(envs)-ok}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
