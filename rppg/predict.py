import csv
import time
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from itertools import islice

import numpy as np
import torch

from . import INPUT, NO_PULSE, PULSE_RATES
from .checkpoint import load_model
from .config import new_run
from .data.preprocess import green_frame, model_input, patch_starts
from .data.real_video import open_video, video_fps
from .evaluate import FIELDS


@torch.inference_mode()
def predict_window(model, frames, cfg, device):
    video = np.stack(frames, axis=-1)
    height, width = video.shape[:2]
    size, stride = INPUT["patch_size"], cfg["patch_stride"]
    patches = (model_input(video[y:y + size, x:x + size])
               for y in patch_starts(height, stride) for x in patch_starts(width, stride))
    votes = np.zeros(NO_PULSE + 1, dtype=np.int64)
    while batch := list(islice(patches, cfg["batch_size"])):
        labels = model(torch.from_numpy(np.stack(batch)).to(device)).argmax(dim=1).cpu().numpy()
        votes += np.bincount(labels, minlength=len(votes))
    valid = int(votes[:-1].sum())
    bpm = float(np.dot(PULSE_RATES, votes[:-1]) / valid) if valid else None
    return bpm, valid, int(votes.sum())


def frames_from(capture, settings, fps, live):
    count = INPUT["window_frames"]
    frames, times = deque(maxlen=count), deque(maxlen=count)
    started = time.monotonic()
    frame_index = -1
    while True:
        ok, frame = capture.read()
        if not ok:
            break
        frame_index += 1
        frames.append(green_frame(frame, settings["roi"]))
        times.append(time.monotonic() - started if live else frame_index / fps)
        window_start = frame_index + 1 - count
        if window_start >= 0 and window_start % settings["frame_stride"] == 0:
            timing_ok = True
            if live:
                span = times[-1] - times[0]
                timing_ok = (span > 0 and abs((count - 1) / span - fps) <= 1.5
                             and max(np.diff(times)) <= 2.5 / fps)
            yield window_start, tuple(frames), times[0], times[-1] + 1 / fps, timing_ok
    if frame_index + 1 < count:
        raise ValueError(f"Source must provide at least {count} decodable frames")


def predict(cfg, config_path, checkpoint, device):
    import cv2

    settings = cfg["predict"]
    live = settings["source"] == "camera"
    if not live and not settings["video_path"]:
        raise ValueError("Set predict.video_path for video prediction")
    if live and settings["camera_index"] < 0:
        raise ValueError("predict.camera_index must be nonnegative")
    source = settings["camera_index"] if live else settings["video_path"]
    model, _ = load_model(checkpoint, device)
    capture = open_video(source)
    try:
        if live:
            fps = INPUT["fps"]
            capture.set(cv2.CAP_PROP_FPS, fps)
        else:
            fps = video_fps(capture)
        video_id = f"camera:{source}" if live else source
        run = new_run(cfg, config_path, "predict", checkpoint)
        windows = frames_from(capture, settings, fps, live)
        with (run / "predictions.csv").open("w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=FIELDS)
            writer.writeheader()

            def emit(window, result):
                start_frame, _, start_s, end_s, timing_ok = window
                bpm, valid, total = result
                status = "invalid_timing" if not timing_ok else "no_pulse" if bpm is None else "ok"
                writer.writerow({"sample_id": f"window-{start_frame}", "video_id": video_id,
                                 "start_s": start_s, "end_s": end_s, "prediction_bpm": bpm,
                                 "status": status, "valid_patches": valid, "total_patches": total})
                stream.flush()
                if live:
                    value = f"{bpm:.1f} bpm" if bpm is not None else status
                    print(f"{end_s:.1f}s {value}", flush=True)

            if live:
                print(f"Saving predictions to {run / 'predictions.csv'}", flush=True)
                print(f"Collecting {INPUT['window_frames']} frames. Press Ctrl+C to stop.", flush=True)

                def infer_and_emit(window):
                    emit(window, predict_window(model, window[1], settings, device))

                pending = None
                with ThreadPoolExecutor(max_workers=1) as pool:
                    try:
                        for window in windows:
                            if pending is not None:
                                if not pending.done():
                                    continue
                                pending.result()
                                pending = None
                            if window[-1]:
                                pending = pool.submit(infer_and_emit, window)
                            else:
                                emit(window, (None, 0, 0))
                    except KeyboardInterrupt:
                        pass
                    if pending is not None:
                        pending.result()
            else:
                for window in windows:
                    emit(window, predict_window(model, window[1], settings, device))
    finally:
        capture.release()
    return run
