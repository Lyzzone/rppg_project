import numpy as np

from .. import INPUT


def green_frame(frame, roi):
    if roi:
        x, y, width, height = roi
        if x + width > frame.shape[1] or y + height > frame.shape[0]:
            raise ValueError("ROI extends outside the video frame")
        frame = frame[y:y + height, x:x + width]
    if min(frame.shape[:2]) < INPUT["patch_size"]:
        raise ValueError("The frame or ROI is smaller than 25x25")
    return frame[:, :, 1].copy()


def model_input(patch):
    expected = (INPUT["patch_size"], INPUT["patch_size"], INPUT["window_frames"])
    if patch.shape != expected:
        raise ValueError(f"Expected patch {expected}, got {patch.shape}")
    signal = patch.astype(np.float32) / 255.0
    signal -= signal.mean()
    return signal[None]


def patch_starts(size, stride):
    end = size - INPUT["patch_size"]
    return sorted({*range(0, end + 1, stride), end})
