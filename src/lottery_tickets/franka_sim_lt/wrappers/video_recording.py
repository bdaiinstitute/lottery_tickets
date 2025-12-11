from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, SupportsFloat

import cv2
import gymnasium as gym
import numpy as np
import torch
from gymnasium.core import ActType, ObsType


@dataclass
class VideoOptions:
    """Options for recording a video.

    Attributes:
        record_video (bool): Whether to record video or not.
        save_path (Path | None): Absolute path for the video file to be saved.
    """

    record_video: bool = False
    save_path: Path | None = None


@dataclass
class VideoOptionsGenerator:
    """Video options generator we usually use. Used for vector environments."""

    video_dir: Path
    filename_prefix: str
    record_every_n_episodes: int = 1
    """Set to 0 to disable video recording"""
    current_episode_count: int = 0

    def __post_init__(self) -> None:
        self.video_dir = Path(self.video_dir)

    def __call__(self) -> Any:
        if self.record_every_n_episodes == 0:
            return VideoOptions()
        elif self.current_episode_count % self.record_every_n_episodes == 0:
            options = VideoOptions(
                record_video=True,
                save_path=self.video_dir
                / f"{self.filename_prefix}_{self.current_episode_count}.webm",
            )
        else:
            options = VideoOptions()
        self.current_episode_count += 1
        return options


class VideoRecordingWrapper(gym.Wrapper[ObsType, ActType, ObsType, ActType]):
    """
    This wrapper records videos of the environment.
    """

    def __init__(
        self,
        env: gym.Env[ObsType, ActType],
        intervention_border_color: tuple[int, int, int] = (0, 255, 0),
        video_options_generator: Callable[[], VideoOptions] | None = None,
        fps: float = 30.0,
    ):
        """
        Args:
            env (gym.Env): The environment to wrap.
            intervention_border_color (tuple): The color of the border to indicate intervention.
            video_options_generator (Callable[[], VideoOptions] | None): A callable that generates
                video options for each episode.
                We added this because vector envs make it tricky to pass in videooptions on each reset because
                    1. VecEnv autoresets during step
                    2. VecEnv's reset method doesn't take in per-env options
            fps (float): Frames per second for the video.
        """
        super().__init__(env)
        self.fps = fps
        self.fourcc = cv2.VideoWriter_fourcc(  # type: ignore
            "V", "P", "9", "0"
        )  # WebM codec # type: ignore
        self.image_shape = None  # (H,W,3)
        self.intervention_border_color = intervention_border_color
        self.step_count = 0  # To keep track of the number of steps taken
        self.video_capture = None
        self.video_path: Path | None = None
        self.video_options_generator = video_options_generator

    def reset(
        self, *, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[ObsType, dict[str, Any]]:
        if options is None:
            options = dict()
        # Reset env
        obs, info = self.env.reset(seed=seed, options=options)
        self.step_count = 0
        # End the prior recording, if it exists.
        if self.video_capture is not None:
            self.video_capture.release()
            self.video_capture = None
            if self.video_path is not None:
                info["video"] = str(self.video_path)
                self.video_path = None
        # Get video options from options if it's present, otherwise self.video_options_generator
        # if it's present, otherwise use the default video options.
        if "video" in options.keys():
            video_options: VideoOptions = options["video"]
            if self.video_options_generator is not None:
                # Still call the video options generator, so it can increment its internal counter if it has one.
                video_options_from_generator = self.video_options_generator()
                print(
                    "Both video options and video options generator are provided. Using video options from reset."
                    f"\n If this message is annoying, make this logging.info or logging.debug instead."
                    f"video_options from options: {video_options}"
                    f"video_options from generator: {video_options_from_generator}"
                )
        else:
            if self.video_options_generator is not None:
                video_options = self.video_options_generator()
            else:
                video_options = VideoOptions()

        # Start the new recording, if requested
        if video_options.record_video:
            if video_options.save_path is None:
                raise ValueError("save path must be provided to record video.")

            image = self.get_image()
            assert len(image.shape) == 3 and image.shape[2] == 3, (
                f"Image shape: {image.shape}"
            )

            self.image_shape = image.shape
            self.video_path = video_options.save_path
            self.video_path.parent.mkdir(exist_ok=True, parents=True)
            self.video_capture = cv2.VideoWriter(
                str(self.video_path),
                self.fourcc,
                self.fps,
                (self.image_shape[1], self.image_shape[0]),
            )
            annotated_image = self.annotate_image_frame(
                image, info.get("intervened", False)
            )
            self.video_capture.write(annotated_image[:, :, ::-1])  # RGB to BGR.
        return obs, info

    def step(
        self, action: ActType
    ) -> tuple[ObsType, SupportsFloat, bool, bool, dict[str, Any]]:
        obs, reward, terminated, truncated, info = self.env.step(action)
        self.step_count += 1

        if self.video_capture is not None:
            image = self.get_image()
            annotated_image = self.annotate_image_frame(
                image, intervention=info.get("intervened", False)
            )
            self.video_capture.write(annotated_image[:, :, ::-1])  # RGB to BGR.

            # If the episode is done, stop recording
            if terminated or truncated:
                self.video_capture.release()
                self.video_capture = None
                if self.video_path is not None:
                    info["video"] = str(self.video_path)
                    self.video_path = None

        return obs, reward, terminated, truncated, info

    def get_image(self) -> np.ndarray:
        image = self.env.render()
        if isinstance(image, torch.Tensor):
            image = image.detach().cpu().numpy()

        if not isinstance(image, np.ndarray):
            raise ValueError(
                "VideoRecordingWrapper expects render to return an np.ndarray"
                " or torch.Tensor with shape [H,W,3] or [1,H,W,3]."
                f" Instead we got {type(image)}: {image}"
            )
        image = image.squeeze()
        return image

    def annotate_image_frame(
        self, image: np.ndarray, intervention: bool = False
    ) -> np.ndarray:
        """Annotate image frame to indicate step count and intervention"""
        if intervention:
            b = 5
            image[0:b, :, :] = self.intervention_border_color  # Top border
            image[-b:, :, :] = self.intervention_border_color  # Bottom border
            image[:, 0:b, :] = self.intervention_border_color  # Left border
            image[:, -b:, :] = self.intervention_border_color  # Right border

        # Add text to image
        text = f"Step: {self.step_count}"
        assert self.image_shape is not None
        image = cv2.putText(
            np.ascontiguousarray(image),
            text,
            (10, self.image_shape[1] - 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.3,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )
        return image

    def close(self) -> None:
        if self.video_capture is not None:
            self.video_capture.release()
            self.video_capture = None
        return super().close()
