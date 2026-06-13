"""MuJoCo environment factory with vectorized envs."""
import numpy as np
import gymnasium as gym


def make_env(env_id: str, num_envs: int = 1, seed: int = 42):
    """Create SyncVectorEnv for MuJoCo environments."""

    def _make_env(rank: int):
        def _init():
            env = gym.make(env_id)
            env = gym.wrappers.RecordEpisodeStatistics(env)
            env = gym.wrappers.ClipAction(env)
            env = gym.wrappers.NormalizeObservation(env)
            env = gym.wrappers.TransformObservation(
                env, lambda obs: np.clip(obs, -10, 10), observation_space=None)
            env = gym.wrappers.NormalizeReward(env, gamma=0.99)
            env.reset(seed=seed + rank)
            return env
        return _init

    envs = gym.vector.SyncVectorEnv([_make_env(i) for i in range(num_envs)])
    return envs


def get_env_info(env_id: str):
    """Return (obs_dim, n_actions) for a given env ID."""
    env = gym.make(env_id)
    obs_dim = env.observation_space.shape[0]
    n_actions = env.action_space.shape[0]
    env.close()
    return obs_dim, n_actions
