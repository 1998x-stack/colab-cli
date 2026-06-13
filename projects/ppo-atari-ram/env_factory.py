"""Atari RAM environment factory with vectorized envs."""
import ale_py  # registers ALE namespace in gymnasium
import numpy as np
import gymnasium as gym


def make_ram_env(env_id: str, num_envs: int = 4, seed: int = 42):
    """Create AsyncVectorEnv of Atari RAM environments.

    Observation: (128,) uint8 scaled to [0, 1].
    Action space: discrete, env-specific size.
    """

    def _make_env(rank: int):
        def _init():
            env = gym.make(env_id, max_episode_steps=108000, obs_type="ram")
            env = gym.wrappers.RecordEpisodeStatistics(env)
            env = gym.wrappers.TransformObservation(
                env, lambda obs: obs.astype(np.float32) / 255.0,
                observation_space=None,
            )
            env.reset(seed=seed + rank)
            return env
        return _init

    envs = gym.vector.AsyncVectorEnv([_make_env(i) for i in range(num_envs)])
    return envs


def get_env_info(env_id: str):
    """Return (obs_dim, n_actions) for a given env ID.

    Returns (128, n_actions) for RAM envs. Creates and disposes a temp env.
    """
    env = gym.make(env_id, obs_type="ram")
    obs_dim = env.observation_space.shape[0]
    n_actions = env.action_space.n
    env.close()
    return obs_dim, n_actions
