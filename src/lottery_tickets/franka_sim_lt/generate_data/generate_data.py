# Copyright (c) 2025 Robotics and AI Institute LLC dba RAI Institute. All rights reserved.

import copy
import io
import json
import logging
import pickle
import random
import time
from multiprocessing import Pool
from pathlib import Path

import h5py
import hydra
import numpy as np
from omegaconf import DictConfig, OmegaConf
from PIL import Image

from franka_sim.mujoco_gym_env import MujocoGymEnv
from lottery_tickets.franka_sim_lt.gym_utils import make_frankasim_env


def reach_cube(
    orig_env: MujocoGymEnv, action_mag: float, epsilon: float, noise_mag: float
) -> tuple[np.ndarray, np.ndarray, bool]:
    """Move the robot gripper towards the target cube."""
    tcp_pos = orig_env._data.sensor("2f85/pinch_pos").data
    block_pos = orig_env._data.sensor("block_pos").data

    target_pos = block_pos - tcp_pos
    norm = np.linalg.norm(target_pos)
    done = norm < epsilon

    if norm > action_mag:
        target_pos = (target_pos / norm) * action_mag
    target_pos = target_pos / orig_env._action_scale[0]
    target_pos = target_pos + np.random.normal(0, 1, 3) * noise_mag

    target_gripper = np.array([0.0]) / orig_env._action_scale[1]
    return target_pos, target_gripper, done


def grasp_cube(
    orig_env: MujocoGymEnv, planner_state, wait_steps: int
) -> tuple[np.ndarray, np.ndarray, bool]:
    """Close the gripper to grasp the cube and wait for specified steps."""
    target_pos = np.zeros(3, dtype=np.float32)
    target_gripper = np.array([1.0]) / orig_env._action_scale[1]

    planner_state["grasp_wait"] += 1
    done = planner_state["grasp_wait"] > wait_steps

    return target_pos, target_gripper, done


def lift_cube(
    orig_env: MujocoGymEnv, action_mag: float, epsilon: float, height: float, noise_mag: float
) -> tuple[np.ndarray, np.ndarray, bool]:
    """Lift the grasped cube to the specified height."""
    tcp_pos = orig_env._data.sensor("2f85/pinch_pos").data
    block_pos = orig_env._data.sensor("block_pos").data

    target_pos = block_pos - tcp_pos
    target_pos[2] = height - tcp_pos[2]
    norm = np.linalg.norm(target_pos)
    done = norm < epsilon

    if norm > action_mag:
        target_pos = (target_pos / norm) * action_mag
    target_pos = target_pos / orig_env._action_scale[0]
    target_pos = target_pos + np.random.normal(0, 1, 3) * noise_mag

    target_gripper = np.array([0.0]) / orig_env._action_scale[1]
    return target_pos, target_gripper, done


def do_nothing(orig_env: MujocoGymEnv) -> tuple[np.ndarray, np.ndarray]:
    """Keep the gripper stationary with the cube grasped."""
    target_pos = np.zeros(3, dtype=np.float32)
    target_gripper = np.array([0.0]) / orig_env._action_scale[1]
    return target_pos, target_gripper


def collect_single_demo(
    env: MujocoGymEnv, planner_cfg: DictConfig, success_threshold: float
) -> tuple[list[dict], bool]:
    """Collect a single demonstration episode."""
    orig_env = env.unwrapped

    planner_state = {
        "reach": False,
        "grasp": False,
        "lift": False,
        "grasp_wait": 0,
    }

    transitions = []
    obs, _ = env.reset()
    done_or_truncated = False

    # Sample random movement speed per episode.
    action_mag = np.random.uniform(planner_cfg.action_mag[0], planner_cfg.action_mag[1])

    while not done_or_truncated:
        if planner_state["reach"] is False:
            target_pos, target_gripper, done = reach_cube(
                orig_env=orig_env,
                action_mag=action_mag,
                epsilon=planner_cfg.reach_epsilon,
                noise_mag=planner_cfg.noise_mag,
            )
            if done:
                planner_state["reach"] = True
        elif planner_state["grasp"] is False:
            target_pos, target_gripper, done = grasp_cube(
                orig_env=orig_env,
                planner_state=planner_state,
                wait_steps=planner_cfg.grasp_wait_steps,
            )
            if done:
                planner_state["grasp"] = True
        elif planner_state["lift"] is False:
            target_pos, target_gripper, done = lift_cube(
                orig_env=orig_env,
                action_mag=action_mag,
                epsilon=planner_cfg.lift_epsilon,
                height=planner_cfg.lift_height,
                noise_mag=planner_cfg.noise_mag,
            )
            if done:
                planner_state["lift"] = True
        else:
            target_pos, target_gripper = do_nothing(orig_env)

        action = np.concatenate([target_pos, target_gripper])
        next_obs, reward, done, truncated, info = env.step(action)

        # Store transition
        transitions.append(
            {
                "observations": copy.deepcopy(obs),
                "next_observations": copy.deepcopy(next_obs),
                "actions": copy.deepcopy(action),
                "reward": reward,
                "done": done,
                "truncated": truncated,
                "info": copy.deepcopy(info) if hasattr(info, "copy") else info,
            }
        )

        obs = next_obs
        done_or_truncated = done or truncated

    success = reward > success_threshold
    return transitions, success


def worker_collect_demo(args: tuple) -> None:
    """Worker function for multiprocessing pool to collect a single demo."""
    env_name, env_kwargs, planner_cfg, success_threshold = args

    # This will ensure different random seeds even if we fork processes.
    random.seed()
    np.random.seed()

    try:
        # Each worker creates its own environment instance
        env = make_frankasim_env(env_name, env_kwargs=env_kwargs)
        transitions, success = collect_single_demo(env, planner_cfg, success_threshold)
        env.close()
        return {
            "success": True,
            "demo_success": success,
            "transitions": transitions,
            "error": None,
        }
    except Exception as e:
        logging.exception("Error in worker_collect_demo:")
        return {
            "success": False,
            "demo_success": False,
            "transitions": None,
            "error": str(e),
        }


def process_pending_results(
    pending_results: list,
    successful_demo_files: list[str],
    output_dir: Path,
    base_name: str,
    suffix: str,
    num_demos: int,
    save_as_hdf5: bool,
    wait: bool = False,
) -> tuple[list[int], list[str]]:
    """Process pending results and save successful demos.

    Args:
        pending_results: List of (attempt_num, result) tuples
        successful_demo_files: List of already saved demo files
        output_dir: Directory to save demos
        base_name: Base name for demo files
        suffix: File suffix for demos
        num_demos: Target number of demos to collect
        wait: If True, block and wait for results; if False, only process ready results

    Returns:
        Tuple of (completed_indices, updated_successful_demo_files)
    """
    completed_indices = []

    for idx, (attempt_num, result) in enumerate(pending_results):
        # Check if result is ready or wait for it
        if wait or result.ready():
            completed_indices.append(idx)
            try:
                timeout = None if wait else 0.1
                result_data = result.get(timeout=timeout)

                if result_data["success"] and result_data["demo_success"]:
                    demo_idx = len(successful_demo_files) + 1
                    if save_as_hdf5:
                        # Start counting from 0. Create a subdirectory for HDF5 files.
                        tmp = output_dir / f"episode_{demo_idx - 1}"
                        tmp.mkdir(parents=True, exist_ok=True)
                        demo_file = tmp / f"episode_{demo_idx - 1}.h5"
                        write_hdf5(demo_file, result_data["transitions"])
                    else:
                        demo_file = output_dir / f"{base_name}_{demo_idx:04d}{suffix}"
                        write_pickle(demo_file, result_data["transitions"])

                    successful_demo_files.append(str(demo_file))
                    logging.info(
                        f"Attempt {attempt_num}: SUCCESS! Saved demo {demo_idx}/{num_demos} to {demo_file}"
                    )

                    # Early exit if we've collected enough demos when waiting
                    if wait and len(successful_demo_files) >= num_demos:
                        return completed_indices, successful_demo_files

                elif result_data["success"]:
                    logging.info(
                        f"Attempt {attempt_num}: Failed (unsuccessful demo discarded)"
                    )
                else:
                    logging.info(
                        f"Attempt {attempt_num}: Error - {result_data['error']}"
                    )

            except Exception as e:
                logging.exception("Error processing result:")
                logging.info(f"Attempt {attempt_num}: Error processing result - {e}")

    return completed_indices, successful_demo_files


def write_pickle(demo_file: Path, transitions: list[dict]) -> None:
    """Save an episode of transitions as a pickle."""
    with open(demo_file, "wb") as f:
        pickle.dump(transitions, f)


def write_hdf5(demo_file: Path, transitions: list[dict]) -> None:
    """Save an episode of transitions as a VPL hdf5 dataset."""
    COLOR_KEY = "color"
    NEXT_COLOR_KEY = "next_color"
    KEY_MAPPING = {"actions": "action"}  # VPL expects 'action'.

    demos = transitions

    ignore_keys = ["observations", "next_observations", "info"]
    first_demo = demos[0]
    num_steps = len(demos)
    color_keys = list(
        sorted([k for k in first_demo["observations"].keys() if k != "state"])
    )
    num_colors_keys = len(color_keys)

    with h5py.File(demo_file, "w") as hf:
        # Create datasets.
        for key in first_demo.keys():
            if key in ignore_keys:
                continue
            if isinstance(first_demo[key], (bool, int, float)):
                hf.create_dataset(
                    KEY_MAPPING.get(key, key),
                    shape=(num_steps,),
                    dtype=type(first_demo[key]),
                )
            else:
                hf.create_dataset(
                    KEY_MAPPING.get(key, key),
                    shape=(num_steps,) + first_demo[key].shape,
                    dtype=first_demo[key].dtype,
                )

        key = "state"
        hf_key = "state"
        data = first_demo["observations"][key]
        if data.shape[0] == 1:
            data = data[0]
        hf.create_dataset(
            hf_key,
            shape=(num_steps,) + data.shape,
            dtype=data.dtype,
        )

        key = "state"
        hf_key = "next_state"
        data = first_demo["next_observations"][key]
        if data.shape[0] == 1:
            data = data[0]
        hf.create_dataset(
            hf_key,
            shape=(num_steps,) + data.shape,
            dtype=data.dtype,
        )

        dt = h5py.special_dtype(vlen=np.dtype("uint8"))
        hf.create_dataset(COLOR_KEY, shape=(num_steps, num_colors_keys), dtype=dt)
        hf.create_dataset(NEXT_COLOR_KEY, shape=(num_steps, num_colors_keys), dtype=dt)

        # Write data.
        for step, demo in enumerate(demos):
            for key in demo.keys():
                if key in ignore_keys:
                    continue
                hf[KEY_MAPPING.get(key, key)][step] = demo[key]

            # Handle state separately.
            state = demo["observations"]["state"]
            next_state = demo["next_observations"]["state"]
            if state.shape[0] == 1:
                state = state[0]
            if next_state.shape[0] == 1:
                next_state = next_state[0]
            hf["state"][step] = state
            hf["next_state"][step] = next_state

            # Handle colors separately.
            for color_i, key in enumerate(color_keys):
                color = demo["observations"][key]
                next_color = demo["next_observations"][key]
                if color.shape[0] == 1:
                    color = color[0]
                if next_color.shape[0] == 1:
                    next_color = next_color[0]
                color = compress_image_jpeg(color)
                next_color = compress_image_jpeg(next_color)
                hf[COLOR_KEY][step, color_i] = color
                hf[NEXT_COLOR_KEY][step, color_i] = next_color


def compress_image_jpeg(image: np.ndarray, quality: int = 90) -> np.ndarray:
    assert image.dtype == np.uint8, "We can only compress 8-bit images."
    im = Image.fromarray(image)
    output = io.BytesIO()
    im.save(output, format="JPEG", quality=quality)
    return np.frombuffer(output.getvalue(), dtype=np.uint8)


def decode_image(png_bytes: np.uint8) -> np.ndarray:
    """Decode JPEG bytes into a numpy array image."""
    stream = io.BytesIO(png_bytes.tobytes())
    img = Image.open(stream)
    # Ensure that image data is fully loaded.
    img.load()
    return np.array(img)


def create_metadata_json(
    output_dir: Path, successful_demo_files: list, cfg: DictConfig
) -> None:
    """Create metadata.json by reading HDF5 files and counting timesteps.

    Args:
        output_dir: Directory containing the episode subdirectories
        successful_demo_files: List of paths to HDF5 files
    """
    num_episodes = len(successful_demo_files)
    num_timesteps = []

    # Read each HDF5 file and count the timesteps
    for demo_file in successful_demo_files:
        with h5py.File(demo_file, "r") as hf:
            # Get the length of the action field
            action_length = len(hf["action"])
            num_timesteps.append(action_length)

    # Create metadata dictionary
    metadata = {
        "num_episodes": num_episodes,
        "num_timesteps": num_timesteps,
        "skill": {"name": "pick cube", "description": "pick cube"},
        "demogen_config": OmegaConf.to_container(cfg, resolve=True),
    }

    # Save metadata.json in the output directory
    metadata_path = output_dir / "metadata.json"
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)

    logging.info(f"Saved metadata to {metadata_path}")


@hydra.main(version_base="1.1", config_path="cfgs", config_name="mg_demos.yaml")
def main(cfg: DictConfig) -> None:
    """Main function to collect multiple demonstration episodes and save to file."""
    env_cfg = cfg.evaluation
    output_dir = Path(".").resolve()
    print(f"Output directory: {output_dir}")
    env_kwargs = OmegaConf.to_container(env_cfg.env_kwargs, resolve=True)

    demo_cfg = cfg.demo_collection
    planner_cfg = cfg.planner
    num_workers = demo_cfg.num_workers

    successful_demo_files = []
    attempts = 0

    logging.info(
        f"Collecting {demo_cfg.num_demos} successful demonstrations using {num_workers} workers..."
    )

    # Prepare output directory and filename components
    output_path: Path = output_dir / demo_cfg.output_file
    individual_output_dir = output_dir / "individual_demos"
    output_dir.mkdir(parents=True, exist_ok=True)
    individual_output_dir.mkdir(parents=True, exist_ok=True)
    base_name = output_path.stem if output_path.stem else "demos"
    suffix = output_path.suffix if output_path.suffix else ".pkl"

    # Use multiprocessing pool for parallel demo collection
    with Pool(processes=num_workers) as pool:
        # Keep submitting tasks until we have enough successful demos or reach max attempts
        pending_results = []

        while (
            len(successful_demo_files) < demo_cfg.num_demos
            and attempts < demo_cfg.max_attempts
        ):
            # Submit new tasks to fill up the pool
            while (
                len(pending_results) < num_workers and attempts < demo_cfg.max_attempts
            ):
                attempts += 1
                args = (
                    env_cfg.env_name,
                    env_kwargs,
                    planner_cfg,
                    demo_cfg.success_threshold,
                )
                result = pool.apply_async(worker_collect_demo, (args,))
                pending_results.append((attempts, result))

            # Process completed results
            completed_indices, successful_demo_files = process_pending_results(
                pending_results,
                successful_demo_files,
                individual_output_dir,
                base_name,
                suffix,
                demo_cfg.num_demos,
                save_as_hdf5=demo_cfg.save_as_hdf5,
                wait=False,
            )

            # Remove completed results from pending list
            for idx in reversed(completed_indices):
                pending_results.pop(idx)

            # Check if we've collected enough successful demos
            if len(successful_demo_files) >= demo_cfg.num_demos:
                break

            # Small sleep to avoid busy-waiting
            if pending_results:
                time.sleep(0.1)

        # Wait for any remaining pending results if we haven't hit the limit yet
        if len(successful_demo_files) < demo_cfg.num_demos:
            _, successful_demo_files = process_pending_results(
                pending_results,
                successful_demo_files,
                individual_output_dir,
                base_name,
                suffix,
                demo_cfg.num_demos,
                save_as_hdf5=demo_cfg.save_as_hdf5,
                wait=True,
            )

    if len(successful_demo_files) < demo_cfg.num_demos:
        logging.info(
            f"Warning: Only collected {len(successful_demo_files)} successful demos out of {demo_cfg.num_demos} requested after {attempts} attempts"
        )

    if not demo_cfg.save_as_hdf5:
        # Combine individual demo files into a single pickle at the requested output path
        combined_demos = []
        for demo_file in successful_demo_files:
            with open(demo_file, "rb") as f:
                combined_demos.append(pickle.load(f))

        combined_output_path = output_dir / output_path.name
        with open(combined_output_path, "wb") as f:
            pickle.dump(combined_demos, f)

        logging.info(
            f"Saved {len(successful_demo_files)} successful demonstrations as individual files in {output_dir}"
        )
        logging.info(f"Combined demos saved to {combined_output_path}")
    else:
        # Create metadata.json for HDF5 files.
        create_metadata_json(output_dir, successful_demo_files, cfg)
        logging.info(
            f"Saved {len(successful_demo_files)} successful demonstrations as HDF5 files in {output_dir}"
        )

    # Final summary
    logging.info(f"Total attempts: {attempts}")
    logging.info(f"Success rate: {len(successful_demo_files) / attempts * 100:.1f}%")


if __name__ == "__main__":
    main()
