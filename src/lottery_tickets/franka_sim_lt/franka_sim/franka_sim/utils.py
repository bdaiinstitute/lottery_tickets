import gymnasium as gym
import mujoco
from gymnasium.envs.mujoco.mujoco_rendering import MujocoRenderer


def create_gym_mjc_viewer_multiversion(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    height: int,
    width: int,
    camera_ids: tuple[int],
) -> MujocoRenderer | dict[int, MujocoRenderer]:
    """Create a MujocoRenderer or a dict of MujocoRenderers depending on gym version.

    gymnasium < 1.0.0: single MujocoRenderer for all cameras.
    gymnasium >= 1.0.0: dict of MujocoRenderers, one per camera.

    Args:
        model: The Mujoco model.
        data: The Mujoco data.
        height: The height of the rendered image.
        width: The width of the rendered image.
        camera_ids: The camera IDs to use for rendering.
    """
    if gym.__version__ <= "1.0.0":
        return MujocoRenderer(
            model=model,
            data=data,
        )
    else:
        return {
            camera_id: MujocoRenderer(
                model=model,
                data=data,
                height=height,
                width=width,
                camera_id=camera_id,
            )
            for camera_id in camera_ids
        }


def render_gym_mjc_viewer_multiversion(
    viewer: MujocoRenderer | list[MujocoRenderer],
    render_mode: str,
    camera_ids: tuple[int],
) -> None:
    """Render images from a MujocoRenderer or a list of MujocoRenderers depending on gym version.

    gymnasium < 1.0.0: single MujocoRenderer for all cameras.
    gymnasium >= 1.0.0: dict of MujocoRenderers, one per camera.

    Args:
        viewer: The MujocoRenderer or list of MujocoRenderers.
        render_mode: The render mode.
        camera_ids: The camera IDs to use for rendering.
    """
    if gym.__version__ <= "1.0.0":
        rendered_frames = []
        for cam_id in camera_ids:
            frame = viewer.render(render_mode=render_mode, camera_id=cam_id)
            if frame is not None:
                rendered_frames.append(frame.copy())  # copy due to flipud in render.
    else:
        rendered_frames = []
        for cam_id in camera_ids:
            frame = viewer[cam_id].render(
                render_mode=render_mode,
            )
            if frame is not None:
                rendered_frames.append(frame.copy())  # copy due to flipud in render.
    return rendered_frames


def close_gym_mjc_viewer_multiversion(
    viewer: MujocoRenderer | dict[int, MujocoRenderer],
) -> None:
    """Close a MujocoRenderer or a dict of MujocoRenderers depending on gym version.

    gymnasium < 1.0.0: single MujocoRenderer for all cameras.
    gymnasium >= 1.0.0: dict of MujocoRenderers, one per camera.

    Args:
        viewer: The MujocoRenderer or dict of MujocoRenderers.
    """
    if gym.__version__ <= "1.0.0":
        viewer.close()
    else:
        for v in viewer.values():
            v.close()
