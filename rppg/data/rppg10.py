from collections import Counter
from pathlib import Path

import numpy as np

from .. import INPUT, PULSE_RATES
from .real_video import VideoWindowDataset, open_video, video_fps


def read_r_peak_times(ecg_path, sample_rate):
    """Read one ECG recording and detect R peaks once for all test windows."""
    from scipy.signal import butter, find_peaks, sosfiltfilt

    ecg = np.load(ecg_path, allow_pickle=False)
    if ecg.ndim != 1 or ecg.size < sample_rate * 2:
        raise ValueError("ECG must be a 1-D waveform of at least 2 seconds")
    if not np.isfinite(ecg).all():
        raise ValueError("ECG contains non-finite values")

    # Emphasize the QRS complex without shifting peak times.
    sos = butter(3, [5, 20], btype="bandpass", fs=sample_rate, output="sos")
    filtered = sosfiltfilt(sos, ecg)
    low, high = np.percentile(filtered, [0.5, 99.5])
    if -low > high:
        filtered = -filtered

    # The resting-data preset matches the Subject_1 pilot. At 0.4 s minimum
    # spacing, this detector needs separate validation for rates above 150 BPM.
    peaks, _ = find_peaks(filtered, distance=round(0.4 * sample_rate),
                          prominence=0.55 * np.percentile(filtered, 99.5))
    if len(peaks) < 2:
        raise ValueError("ECG does not contain enough detected R peaks")
    return peaks / sample_rate


def window_reference_bpm(r_peaks_s, start_s, end_s):
    """Average 60/RR over intervals, weighted by overlap with the window."""
    rr_s = np.diff(r_peaks_s)
    overlap_s = np.maximum(0.0, np.minimum(r_peaks_s[1:], end_s)
                          - np.maximum(r_peaks_s[:-1], start_s))
    # Require the detected intervals to cover the entire window.
    if not np.isclose(overlap_s.sum(), end_s - start_s, rtol=0.0, atol=1e-6):
        return None, "insufficient_ecg_coverage"
    used = overlap_s > 0
    if np.any((rr_s[used] < 0.4 - 1e-6) | (rr_s[used] > 1.5 + 1e-6)):
        return None, "invalid_rr_interval"

    bpm = float(np.average(60.0 / rr_s[used], weights=overlap_s[used]))
    if not PULSE_RATES[0] <= bpm <= PULSE_RATES[-1]:
        return None, "outside_model_range"
    return bpm, "ok"


class RPPG10Dataset(VideoWindowDataset):
    def __init__(self, data):
        import cv2

        video = Path(data["video_path"])
        capture = open_video(str(video))
        try:
            fps = video_fps(capture)
            frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
            width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        finally:
            capture.release()

        count, size = INPUT["window_frames"], INPUT["patch_size"]
        x, y = data["roi_x"], data["roi_y"]
        if x + size > width or y + size > height:
            raise ValueError(f"rppg10 ROI extends outside the {width}x{height} video")
        if data["stop_frame"] > frame_count:
            raise ValueError(f"rppg10 stop_frame exceeds the video's {frame_count} frames")

        r_peaks_s = read_r_peak_times(data["ecg_path"], data["ecg_sample_rate"])
        starts = range(data["start_frame"], data["stop_frame"] - count + 1, data["stride_frames"])
        offset_s = data["ecg_offset_s"]
        rates = np.asarray(PULSE_RATES)
        skipped = Counter()
        self.rows = []
        for start in starts:
            # Positive offset means a later ECG time for the same video frame.
            bpm, status = window_reference_bpm(r_peaks_s, start / fps + offset_s,
                                                (start + count) / fps + offset_s)
            if bpm is None:
                skipped[status] += 1
                continue
            target = int(np.argmin(np.abs(rates - bpm)))
            sample_id = f"{data['subject_id']}-{data['session_id']}-frame-{start}"
            self.rows.append((sample_id, data["subject_id"], video, start, x, y, bpm, target))

        self.reference_windows = {
            "candidates": len(starts), "labeled": len(self.rows),
            "skipped": sum(skipped.values()), "skipped_by_reason": dict(skipped),
        }
        if not self.rows:
            raise ValueError(f"rppg10 has no valid reference windows: {self.reference_windows}")
