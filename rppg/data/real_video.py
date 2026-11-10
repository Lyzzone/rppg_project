import csv
from pathlib import Path

import numpy as np
from torch.utils.data import Dataset

from .. import INPUT, NO_PULSE, PULSE_RATES
from .preprocess import green_frame, model_input


MANIFEST_FIELDS = (
    "subject_id", "session_id", "video_path", "start_frame", "roi_x", "roi_y",
    "split", "label_kind", "reference_bpm",
)


def open_video(source):
    import cv2

    capture = cv2.VideoCapture(source)
    if not capture.isOpened():
        capture.release()
        raise ValueError(f"Cannot open video source: {source}")
    return capture


def video_fps(capture):
    import cv2

    fps = capture.get(cv2.CAP_PROP_FPS)
    if not abs(fps - INPUT["fps"]) <= 0.1:
        raise ValueError(f"Expected a 30 fps video, got {fps:.3f} fps")
    return fps


class VideoWindowDataset(Dataset):
    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        import cv2

        sample_id, subject, path, start, x, y, bpm, target = self.rows[index]
        count, size = INPUT["window_frames"], INPUT["patch_size"]
        capture = open_video(str(path))
        try:
            fps = video_fps(capture)
            if start and not capture.set(cv2.CAP_PROP_POS_FRAMES, start):
                raise ValueError(f"Cannot seek to frame {start} in {path}")
            frames = []
            for _ in range(count):
                ok, frame = capture.read()
                if not ok:
                    raise ValueError(f"{path}: fewer than {count} decodable frames from frame {start}")
                frames.append(green_frame(frame, (x, y, size, size)))
        finally:
            capture.release()
        return {
            "input": model_input(np.stack(frames, axis=-1)), "target": target,
            "reference_bpm": bpm, "sample_id": sample_id, "video_id": str(path),
            "subject_id": subject, "start_s": start / fps, "end_s": (start + count) / fps,
        }


class RealVideoDataset(VideoWindowDataset):
    def __init__(self, manifest, split):
        manifest = Path(manifest)
        self.rows = []
        subject_splits, video_splits, seen_windows = {}, {}, set()
        with manifest.open(encoding="utf-8-sig", newline="") as stream:
            reader = csv.DictReader(stream)
            missing = set(MANIFEST_FIELDS) - set(reader.fieldnames or [])
            if missing:
                raise ValueError(f"{manifest}: missing columns {', '.join(sorted(missing))}")
            for line, row in enumerate(reader, start=2):
                try:
                    subject, session, video_name, partition, kind = (
                        row[key].strip() for key in ("subject_id", "session_id", "video_path", "split", "label_kind")
                    )
                    if not subject or not session or not video_name or partition not in {"train", "val", "test"}:
                        raise ValueError("subject_id, session_id, video_path and a valid split are required")
                    path = (manifest.parent / video_name).resolve()
                    if subject_splits.setdefault(subject, partition) != partition or video_splits.setdefault(path, partition) != partition:
                        raise ValueError("A subject or video occurs in more than one split")
                    start, x, y = (int(row[key]) for key in ("start_frame", "roi_x", "roi_y"))
                    if min(start, x, y) < 0:
                        raise ValueError("Frame index and ROI coordinates must be nonnegative")
                    key = (path, start)
                    if key in seen_windows:
                        raise ValueError("Duplicate video window; use one patch per window")
                    seen_windows.add(key)
                    if kind == "pulse":
                        bpm = float(row["reference_bpm"])
                        if not PULSE_RATES[0] <= bpm <= PULSE_RATES[-1]:
                            raise ValueError("reference_bpm must be in [55, 237.5]")
                        target = int(np.argmin(np.abs(np.asarray(PULSE_RATES) - bpm)))
                    elif kind == "no_pulse" and not row["reference_bpm"].strip():
                        bpm, target = float("nan"), NO_PULSE
                    else:
                        raise ValueError("Use label_kind=pulse with BPM or no_pulse with blank BPM")
                    if partition == split:
                        if not path.is_file():
                            raise ValueError(f"Video does not exist: {path}")
                        self.rows.append((f"real-{line}", subject, path, start, x, y, bpm, target))
                except (AttributeError, TypeError, ValueError) as error:
                    raise ValueError(f"{manifest}:{line}: {error}") from error
        if not self.rows:
            raise ValueError(f"{manifest}: add labeled rows for split={split}")

