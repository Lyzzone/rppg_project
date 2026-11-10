import json
from math import isfinite
from datetime import datetime
from pathlib import Path
from tempfile import mkdtemp

try:
    import tomllib
except ModuleNotFoundError:
    import tomli as tomllib

from . import INPUT


DEFAULTS = {
    "runtime": {"device": "cpu", "num_threads": 2, "seed": 42},
    "model": {"name": "convnet3d"},
    "input": INPUT,
    "train": {"epochs": 2, "batch_size": 1, "learning_rate": 1e-4,
              "init_checkpoint": "", "freeze_backbone": False, "test_after_train": False},
    "predict": {"source": "video", "video_path": "", "camera_index": 0,
                "roi": [], "frame_stride": 30, "patch_stride": 25,
                "batch_size": 1},
    "output": {"root": "../runs", "experiment": "rppg"},
}


def with_defaults(values, defaults, section):
    if not isinstance(values, dict):
        raise ValueError(f"[{section}] must be a TOML table")
    unknown = values.keys() - defaults.keys()
    if unknown:
        raise ValueError(f"Unknown options in [{section}]: {', '.join(sorted(unknown))}")
    for key, value in values.items():
        expected = type(defaults[key])
        if type(value) is not expected and not (expected is float and type(value) is int):
            raise ValueError(f"{section}.{key} must be {expected.__name__}")
    return defaults | values


def rppg10_options(values, config_dir):
    data = with_defaults(values, {
        "source": "rppg10", "subject_id": "", "session_id": "",
        "video_path": "", "ecg_path": "", "roi_x": 0, "roi_y": 0,
        "start_frame": 0, "stop_frame": 0, "stride_frames": INPUT["window_frames"],
        "ecg_sample_rate": 1000, "ecg_offset_s": 0.0,
    }, "data.test")
    for key in ("subject_id", "session_id", "video_path", "ecg_path"):
        if not data[key].strip():
            raise ValueError(f"data.test.{key} is required for rppg10")
    video = (config_dir / data["video_path"]).resolve()
    ecg = (config_dir / data["ecg_path"]).resolve()
    subject = data["subject_id"]
    video_stems = {f"{subject}_{region}" for region in ("Forehead", "Cheek1", "Cheek2")}
    if video.suffix.lower() != ".avi" or video.stem.rstrip("_") not in video_stems:
        raise ValueError("rppg10 video_path must name the selected subject's Forehead, Cheek1 or Cheek2 AVI")
    if ecg.name != f"{subject}_ECG.npy":
        raise ValueError("rppg10 ecg_path must name the selected subject's ECG file")
    if min(data["roi_x"], data["roi_y"], data["start_frame"]) < 0:
        raise ValueError("rppg10 ROI coordinates and start_frame must be nonnegative")
    if data["stride_frames"] < 1 or data["stop_frame"] - data["start_frame"] < INPUT["window_frames"]:
        raise ValueError("rppg10 needs a positive stride and a range covering at least one full window")
    if data["ecg_sample_rate"] <= 40 or not isfinite(data["ecg_offset_s"]):
        raise ValueError("rppg10 needs ecg_sample_rate > 40 Hz and a finite ecg_offset_s")
    data.update(video_path=str(video), ecg_path=str(ecg))
    return data


def load_config(path):
    path = Path(path).resolve()
    with path.open("rb") as stream:
        raw = tomllib.load(stream)
    unknown = raw.keys() - DEFAULTS.keys() - {"data"}
    if unknown:
        raise ValueError(f"Unknown config sections: {', '.join(sorted(unknown))}")
    cfg = {name: with_defaults(raw.get(name, {}), defaults, name)
           for name, defaults in DEFAULTS.items()}
    if cfg["input"] != INPUT or cfg["model"]["name"] != "convnet3d":
        raise ValueError(f"convnet3d requires the fixed input contract {INPUT}")
    runtime, train, predict, output = (cfg[key] for key in ("runtime", "train", "predict", "output"))
    if runtime["num_threads"] < 1 or runtime["seed"] < 0:
        raise ValueError("num_threads must be positive and seed must be nonnegative")
    if min(train["epochs"], train["batch_size"]) < 1 or not 0 < train["learning_rate"] < float("inf"):
        raise ValueError("epochs, batch_size and learning_rate must be positive and finite")
    if predict["source"] not in {"video", "camera"}:
        raise ValueError("predict.source must be video or camera")
    if min(predict["frame_stride"], predict["batch_size"]) < 1 or not 1 <= predict["patch_stride"] <= INPUT["patch_size"]:
        raise ValueError("Invalid prediction strides or batch size")
    roi = predict["roi"]
    if roi and (len(roi) != 4 or any(type(value) is not int for value in roi)
                or min(roi[:2]) < 0 or min(roi[2:]) < INPUT["patch_size"]):
        raise ValueError("predict.roi must be [x, y, width, height] and contain a 25x25 patch")
    experiment = output["experiment"]
    if not experiment or any(not (char.isalnum() or char in "-_") for char in experiment):
        raise ValueError("output.experiment must contain letters, numbers, hyphens or underscores")

    sections = raw.get("data", {})
    if not isinstance(sections, dict) or sections.keys() - {"train", "val", "test"}:
        raise ValueError("Use [data.train], [data.val] and [data.test] for data splits")
    cfg["data"] = {}
    synthetic_seeds = set()
    for split, values in sections.items():
        source = values.get("source") if isinstance(values, dict) else None
        section = f"data.{split}"
        if source == "synthetic":
            data = with_defaults(values, {"source": source, "samples_per_class": 1,
                                         "seed": -1, "refresh_each_epoch": False}, section)
            if data["seed"] < 0 or data["samples_per_class"] < 1:
                raise ValueError(f"[{section}] needs a nonnegative seed and positive samples_per_class")
            if data["seed"] in synthetic_seeds or (split != "train" and data["refresh_each_epoch"]):
                raise ValueError("Synthetic splits need distinct seeds; only train may refresh each epoch")
            synthetic_seeds.add(data["seed"])
        elif source == "real":
            data = with_defaults(values, {"source": source, "manifest": ""}, section)
            if not data["manifest"]:
                raise ValueError(f"[{section}] needs a manifest path")
            data["manifest"] = str((path.parent / data["manifest"]).resolve())
        elif source == "rppg10" and split == "test":
            data = rppg10_options(values, path.parent)
        else:
            raise ValueError(f"[{section}].source must be synthetic or real; rppg10 is available in [data.test]")
        cfg["data"][split] = data
    if len({data["manifest"] for data in cfg["data"].values() if data["source"] == "real"}) > 1:
        raise ValueError("Use the same manifest for all real splits so groups can be checked")
    for settings, key in ((train, "init_checkpoint"), (predict, "video_path"), (output, "root")):
        if settings[key]:
            settings[key] = str((path.parent / settings[key]).resolve())
    if not output["root"]:
        raise ValueError("output.root must be a directory path")
    return cfg


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")


def new_run(cfg, config_path, action, checkpoint=None, *, root=None):
    output = cfg["output"]
    if root is None:
        root = Path(output["root"]) / output["experiment"]
    root.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    path = Path(mkdtemp(prefix=f"{stamp}-{action}-", dir=root))
    write_json(path / "config.json", {"action": action, "source_config": str(config_path),
                                      "checkpoint": str(checkpoint) if checkpoint else None,
                                      "settings": cfg})
    return path
