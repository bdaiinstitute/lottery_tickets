import time
import imageio
import gymnasium as gym
import numpy as np

import gym_hil

# Use the Franka environment
env = gym.make("gym_hil/PandaPickCubeBase-v0", render_mode="rgb_array")
print(env.reset())
print(env.render())