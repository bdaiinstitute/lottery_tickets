import argparse

import numpy as np
import robosuite as suite

from squirl_launcher.wrappers.robosuite_wrapper import RobosuiteWrapper

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--environment", type=str, default="Lift")
    parser.add_argument(
        "--robots",
        nargs="+",
        type=str,
        default="Panda",
        help="Which robot(s) to use in the env",
    )
    args = parser.parse_args()

    suite_env = suite.make(
        env_name=args.environment,
        robots=args.robots,
        has_renderer=True,
        has_offscreen_renderer=True,
        use_camera_obs=True,
    )
    env = RobosuiteWrapper(suite_env)
    print(f"Observation space: {env.observation_space}")

    obs, info = env.reset()
    done = False
    while not done:
        action = np.random.randn(env.robots[0].dof)  # sample random action
        res = env.step(action)
        obs, reward, done, truncated, info = env.step(
            action
        )  # take action in the environment
        env.render()  # render on display
