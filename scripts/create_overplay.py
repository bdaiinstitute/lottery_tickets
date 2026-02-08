import cv2
import numpy as np
import sys
import os


def create_time_composite(
    video_path,
    output_path,
    n_frames=10,
    method="lighten",
    start_percentage=0.0,
    end_percentage=1.0,
    last_frame_weight=0.0,
):
    """
    Extracts N equally spaced frames from a video and composites them
    into a single image using various blending methods.

    Args:
        video_path: Path to the input video file
        output_path: Path for the output composite image
        n_frames: Number of frames to extract and blend
        method: Blending method to use
        start_percentage: Start fraction of video (0.0-1.0). E.g., 0.1 starts at 10%.
        end_percentage: End fraction of video (0.0-1.0). E.g., 0.9 ends at 90%.
        last_frame_weight: Weight for the last frame (0.0-1.0). E.g., 0.3 blends 70% composite + 30% last frame.

    Methods:
        - "lighten": Keep the brightest pixel at each location (best for showing motion trails)
        - "darken": Keep the darkest pixel at each location
        - "multiply": Photographic darkening - darker and richer than darken
        - "colorburn": Intense darkening with increased saturation/contrast
        - "overlay": Mix of lighten+darken - darkens darks, lightens lights (high contrast)
        - "softlight": Gentler version of overlay (balanced contrast)
        - "average": Simple pixel averaging (original method, can be noisy)
        - "median": Use median value (good noise reduction)
        - "difference": Highlight differences from first frame
    """

    # 1. Verify the file exists
    if not os.path.exists(video_path):
        print(f"Error: Video file not found at {video_path}")
        return

    # 2. Open the video file
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print("Error: Could not open video.")
        return

    # 3. Get total frame count and apply percentage
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"Total frames in video: {total_frames}")

    # Apply start and end percentages to use only a portion of the video
    start_percentage = max(0.0, min(1.0, start_percentage))  # Clamp to [0, 1]
    end_percentage = max(0.0, min(1.0, end_percentage))  # Clamp to [0, 1]
    if start_percentage >= end_percentage:
        print(
            f"Error: start_percentage ({start_percentage}) must be less than end_percentage ({end_percentage})"
        )
        return

    start_frame = int(total_frames * start_percentage)
    end_frame = int(total_frames * end_percentage)
    usable_frames = end_frame - start_frame
    usable_frames = max(1, usable_frames)  # Ensure at least 1 frame
    print(
        f"Using frames {start_percentage*100:.0f}%-{end_percentage*100:.0f}% of video (frames {start_frame}-{end_frame}, {usable_frames} frames)"
    )

    if usable_frames < n_frames:
        print(
            f"Warning: Usable frames ({usable_frames}) is less than requested ({n_frames}). Using all usable frames."
        )
        n_frames = usable_frames

    # 4. Calculate equally spaced frame indices within the usable range
    frame_indices = np.linspace(start_frame, end_frame - 1, n_frames, dtype=int)
    print(f"Extracting frames at indices: {frame_indices}")

    # 5. Get video dimensions
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # 6. Extract all frames first
    frames = []
    for idx in frame_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if ret:
            frames.append(frame)
        else:
            print(f"Warning: Could not read frame at index {idx}")

    cap.release()

    if len(frames) == 0:
        print("Error: No frames were processed.")
        return

    print(f"Successfully extracted {len(frames)} frames. Using '{method}' blending.")

    # 7. Apply the selected blending method
    if method == "lighten":
        # Keep maximum (brightest) value at each pixel - great for motion trails
        final_image = frames[0].copy()
        for frame in frames[1:]:
            final_image = np.maximum(final_image, frame)

    elif method == "darken":
        # Keep minimum (darkest) value at each pixel
        final_image = frames[0].copy()
        for frame in frames[1:]:
            final_image = np.minimum(final_image, frame)

    elif method == "multiply":
        # Multiply blend: multiplies pixel values (normalized) - creates rich, deep darks
        # Better than darken for showing motion on light backgrounds
        base = frames[0].astype(np.float32) / 255.0
        for frame in frames[1:]:
            blend = frame.astype(np.float32) / 255.0
            base = base * blend
        final_image = (base * 255).astype(np.uint8)

    elif method == "colorburn":
        # Color burn: intense darkening with increased saturation
        # Formula: 1 - (1 - base) / blend
        base = frames[0].astype(np.float32) / 255.0
        for frame in frames[1:]:
            blend = frame.astype(np.float32) / 255.0
            # Avoid division by zero
            blend = np.maximum(blend, 0.001)
            result = 1 - (1 - base) / blend
            base = np.clip(result, 0, 1)
        final_image = (base * 255).astype(np.uint8)

    elif method == "median":
        # Use median value - excellent for noise reduction
        stacked = np.stack(frames, axis=0)
        final_image = np.median(stacked, axis=0).astype(np.uint8)

    elif method == "difference":
        # Show motion as difference from first frame overlaid on base
        base_frame = frames[0].astype(np.float32)
        diff_accumulator = np.zeros_like(base_frame)
        for frame in frames[1:]:
            diff = np.abs(frame.astype(np.float32) - base_frame)
            diff_accumulator = np.maximum(diff_accumulator, diff)
        # Blend difference with base frame
        final_image = np.clip(base_frame + diff_accumulator * 0.7, 0, 255).astype(
            np.uint8
        )

    elif method == "overlay":
        # Overlay: darkens darks, lightens lights - combines both effects
        # Formula: if base < 0.5: 2*base*blend, else: 1 - 2*(1-base)*(1-blend)
        base = frames[0].astype(np.float32) / 255.0
        for frame in frames[1:]:
            blend = frame.astype(np.float32) / 255.0
            mask = base < 0.5
            result = np.where(mask, 2 * base * blend, 1 - 2 * (1 - base) * (1 - blend))
            base = result
        final_image = (np.clip(base, 0, 1) * 255).astype(np.uint8)

    elif method == "softlight":
        # Soft light: gentler version of overlay, more balanced
        # Formula: (1 - 2*blend) * base^2 + 2*blend*base
        base = frames[0].astype(np.float32) / 255.0
        for frame in frames[1:]:
            blend = frame.astype(np.float32) / 255.0
            result = (1 - 2 * blend) * (base**2) + 2 * blend * base
            base = result
        final_image = (np.clip(base, 0, 1) * 255).astype(np.uint8)

    elif method == "average":
        composite_accumulator = np.zeros((height, width, 3), dtype=np.float32)
        for frame in frames:
            composite_accumulator += frame.astype(np.float32)
        final_image = (composite_accumulator / len(frames)).astype(np.uint8)

    else:
        print(f"Unknown method '{method}', using 'lighten'")
        final_image = frames[0].copy()
        for frame in frames[1:]:
            final_image = np.maximum(final_image, frame)

    # 8. Blend with last frame if weight is specified
    if last_frame_weight > 0 and len(frames) > 0:
        last_frame_weight = max(0.0, min(1.0, last_frame_weight))  # Clamp to [0, 1]
        last_frame = frames[-1].astype(np.float32)
        final_image = (
            (1 - last_frame_weight) * final_image.astype(np.float32)
            + last_frame_weight * last_frame
        ).astype(np.uint8)
        print(f"Applied {last_frame_weight*100:.0f}% weight to last frame")

    # 9. Save the result
    cv2.imwrite(output_path, final_image)
    print(f"Success! Composite saved to: {output_path}")


if __name__ == "__main__":
    # Configuration
    INPUT_VIDEO = "2026-01-29/banana_pick/noises/837dadf6-bea6-47f4-b7b9-f94abaf5de06/eagle/camera_data_episode_8/camera/cam_front_right_camera_351322300029_color.mp4"  # Change this to your video path
    OUTPUT_IMAGE = (
        "front_img_fails/lt_eagle"  # Base output filename (without extension)
    )
    FRAMES_TO_USE = 2  # How many frames to blend (increased for better trails)
    START_PERCENTAGE = 0.0  # Start of video range (e.g., 0.1 = start at 10%)
    END_PERCENTAGE = 0.80  # End of video range (e.g., 0.9 = end at 90%)
    LAST_FRAME_WEIGHT = (
        0.55  # Weight for last frame (0.0 = no extra weight, 0.5 = 50% last frame)
    )

    # Blending method options:
    # - "lighten": Best for motion trails on darker backgrounds (recommended)
    # - "darken": Best for motion trails on lighter backgrounds
    # - "median": Good noise reduction, shows most common state
    # - "difference": Highlights motion/changes from first frame
    # - "average": Original method (can be noisy/blurry)
    BLEND_METHODS = [
        "lighten",
        "difference",
        "darken",
        "average",
        "median",
        "overlay",
        "softlight",
        "multiply",
        "colorburn",
    ]  # Generate all by default

    # Run the function for each blending method
    for method in BLEND_METHODS:
        output_path = f"{OUTPUT_IMAGE}_{method}.jpg"
        print(f"\n--- Generating {method} composite ---")
        create_time_composite(
            INPUT_VIDEO,
            output_path,
            FRAMES_TO_USE,
            method,
            START_PERCENTAGE,
            END_PERCENTAGE,
            LAST_FRAME_WEIGHT,
        )
