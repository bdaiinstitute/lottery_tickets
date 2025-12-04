from gymnasium import Env
from gymnasium.spaces import Box
from gymnasium import ObservationWrapper
import numpy as np
import gymnasium as gym

def make_frankasim_env():
    import gym_hil
    env = gym.make("gym_hil/PandaPickCubeBase-v0", render_mode="rgb_array")
    env = ConcatRenderWrapper(env)
    env = StateConcatWrapper(env)
    return(env)

class ConcatRenderWrapper(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)

    def render(self):
        # Call the original env render
        front_view, wrist_view = self.env.render()
        # Concatenate along width
        return np.concatenate([front_view, wrist_view], axis=1)


class StateConcatWrapper(ObservationWrapper):
    def __init__(self, env: Env, agent_key: str = "agent_pos", env_key: str = "environment_state"):
        super().__init__(env)
        self.agent_key = agent_key
        self.env_key = env_key

        # We assume the base env returns something (like a dict) where
        # agent_pos and environment_pos are numpy arrays with fixed shape.
        # For example:
        agent_space = env.observation_space.spaces.get(agent_key)
        env_space = env.observation_space.spaces.get(env_key)

        assert agent_space is not None, f"No {agent_key} in observation_space"
        assert env_space is not None, f"No {env_key} in observation_space"

        assert isinstance(agent_space, Box) and isinstance(env_space, Box), \
            "Expect Box spaces for agent and environment positions"

        # New observation space: a flat vector of concatenated positions
        low = np.concatenate((agent_space.low.flatten(), env_space.low.flatten()))
        high = np.concatenate((agent_space.high.flatten(), env_space.high.flatten()))
        self.observation_space = Box(
            low=low,
            high=high,
            dtype=agent_space.dtype
        )

    def observation(self, obs):
        agent = obs[self.agent_key]
        epos = obs[self.env_key]
        # flatten and concatenate
        new_state = np.concatenate((np.asarray(agent).flatten(),
                                    np.asarray(epos).flatten()))
        return {"state": new_state}
