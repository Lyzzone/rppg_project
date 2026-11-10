import csv
import math
from collections import defaultdict
from pathlib import Path
from statistics import mean

import torch
from torch.nn import functional as F

from . import NO_PULSE, PULSE_RATES
from .checkpoint import load_model
from .config import new_run, write_json
from .data.build import make_loader


FIELDS = ("sample_id", "video_id", "subject_id", "start_s", "end_s", "reference_bpm",
          "prediction_bpm", "error_bpm", "status", "valid_patches", "total_patches")


def summarize(rows):
    errors = [row["error_bpm"] for row in rows if row["error_bpm"] is not None]
    pulse_rows = [row for row in rows if row["target_class"] != NO_PULSE]
    noise_count = len(rows) - len(pulse_rows)
    return {
        "samples": len(rows),
        "pulse_targets": len(pulse_rows),
        "pulse_predictions_for_pulse_targets": len(errors),
        "accuracy": mean(row["target_class"] == row["prediction_class"] for row in rows),
        "valid_output_ratio": mean(row["prediction_bpm"] is not None for row in rows),
        "pulse_coverage": len(errors) / len(pulse_rows) if pulse_rows else None,
        "noise_recall": (sum(row["target_class"] == row["prediction_class"] == NO_PULSE for row in rows)
                         / noise_count if noise_count else None),
        "pulse_mae_bpm": mean(abs(error) for error in errors) if errors else None,
        "pulse_rmse_bpm": math.sqrt(mean(error ** 2 for error in errors)) if errors else None,
        "pulse_bias_bpm": mean(errors) if errors else None,
    }


def write_csv(path, fields, rows):
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


@torch.inference_mode()
def evaluate(cfg, config_path, checkpoint, device, *, output_dir=None):
    model, metadata = load_model(checkpoint, device)
    if "test" not in cfg["data"]:
        raise ValueError("Missing [data.test] in config")
    data = cfg["data"]["test"]
    if data["source"] == "synthetic" and data["seed"] in (metadata.get("train_seed"), metadata.get("val_seed")):
        raise ValueError("Test seed overlaps with training or validation seed in the checkpoint")
    loader = make_loader(cfg, "test")
    total_loss = 0.0
    rows = []
    by_video = defaultdict(list)
    for batch in loader:
        logits = model(batch["input"].to(device))
        loss = F.cross_entropy(logits, batch["target"].to(device), reduction="sum").item()
        if not math.isfinite(loss):
            raise ValueError("Non-finite evaluation loss; check the input data and checkpoint")
        total_loss += loss
        for index, prediction in enumerate(logits.argmax(dim=1).cpu().tolist()):
            bpm = PULSE_RATES[prediction] if prediction != NO_PULSE else None
            reference = batch["reference_bpm"][index].item()
            video = batch["video_id"][index]
            row = {
                "sample_id": batch["sample_id"][index], "video_id": video,
                "subject_id": batch["subject_id"][index],
                "start_s": batch["start_s"][index].item() if video else None,
                "end_s": batch["end_s"][index].item() if video else None,
                "reference_bpm": reference if math.isfinite(reference) else None,
                "prediction_bpm": bpm,
                "error_bpm": bpm - reference if bpm is not None and math.isfinite(reference) else None,
                "status": "ok" if bpm is not None else "no_pulse",
                "valid_patches": int(bpm is not None), "total_patches": 1,
                "target_class": batch["target"][index].item(), "prediction_class": prediction,
            }
            rows.append(row)
            if video:
                by_video[video].append(row)
    metrics = summarize(rows)
    metrics.update(test_set=data["source"], loss=total_loss / len(rows), videos=len(by_video),
                   seed=data.get("seed"), checkpoint_epoch=metadata.get("epoch"))
    if data["source"] == "rppg10":
        windows = loader.dataset.reference_windows
        metrics["reference_windows"] = windows
        print(f"reference_windows={windows}")
    if output_dir is None:
        root = None
        if data["source"] == "rppg10":
            training_run = Path(checkpoint).resolve().parent
            if (training_run / "training.csv").is_file():
                root = training_run / "evaluate" / cfg["output"]["experiment"]
        run = new_run(cfg, config_path, "evaluate", checkpoint, root=root)
    else:
        run = output_dir
        run.mkdir()
    write_json(run / "metrics.json", metrics)
    write_csv(run / "predictions.csv", (*FIELDS, "target_class", "prediction_class"), rows)
    if by_video:
        groups = [{"video_id": video, **summarize(video_rows)} for video, video_rows in by_video.items()]
        write_csv(run / "per_video_metrics.csv", groups[0].keys(), groups)
    print(f"samples={metrics['samples']} MAE={metrics['pulse_mae_bpm']} coverage={metrics['pulse_coverage']}")
    return run
