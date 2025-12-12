from lottery_tickets.franka_sim_lt.gym_utils import make_frankasim_env


def print_dict(d: dict, prefix: str = "") -> None:
    for k, v in d.items():
        if isinstance(v, dict):
            print_dict(v, f"{prefix} {k}")
        else:
            print(f"{prefix} {k}")


def main():
    env_name = "PandaPickCubeRealisticControl-v0"
    env_kwargs = {
        "max_episode_steps": 300,
        "sampling_bounds": [[0.25, -0.25], [0.55, 0.25]],
        "render_mode": "rgb_array",
    }
    env = make_frankasim_env(env_name, env_kwargs)
    obs = env.reset()[0]
    print_dict(obs)


if __name__ == "__main__":
    main()
