import csv
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator


def plot_training_loss(training_csv, test_metrics=None):
    with training_csv.open(encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    epochs = [int(row["epoch"]) for row in rows]
    train_loss = [float(row["train_loss"]) for row in rows]
    val_loss = [float(row["val_loss"]) for row in rows]

    fig, ax = plt.subplots(figsize=(7.2, 4.2), layout="constrained")
    ax.plot(epochs, train_loss, "o-", label="Training loss")
    ax.plot(epochs, val_loss, "s-", label="Validation loss")
    if test_metrics is not None:
        epoch, loss = test_metrics["checkpoint_epoch"], test_metrics["loss"]
        ax.scatter(epoch, loss, marker="D", color="tab:red", zorder=4,
                   label=f"Final test loss: {loss:.4f} (best epoch {epoch})")
    ax.set(xlabel="Epoch", ylabel="Mean cross-entropy", title="Training and evaluation loss")
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax.grid(axis="y", alpha=0.25)
    ax.legend()
    path = training_csv.with_name("loss_curve.png")
    try:
        fig.savefig(path, dpi=150)
    finally:
        plt.close(fig)
    return path


def visualize(predictions, video_id=None):
    predictions = Path(predictions)
    with predictions.open(encoding="utf-8", newline="") as stream:
        reader = csv.DictReader(stream)
        required = {"video_id", "start_s", "prediction_bpm", "reference_bpm"}
        if required - set(reader.fieldnames or []):
            raise ValueError("Use a predictions.csv produced by evaluate or predict")
        rows = list(reader)
    if not rows:
        raise ValueError("No prediction records to plot")

    videos = defaultdict(list)
    for row in rows:
        if row["video_id"] and row["start_s"]:
            videos[row["video_id"]].append(row)
    if video_id is not None:
        if video_id not in videos:
            raise ValueError(f"Video not found in predictions: {video_id}")
        rows = videos[video_id]
    output = predictions.parent / "plots"
    files = []

    def save(fig, name):
        output.mkdir(exist_ok=True)
        path = output / name
        fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        files.append(path)

    if videos:
        selected = video_id if video_id is not None else next(iter(videos))
        sequence = sorted(videos[selected], key=lambda row: float(row["start_s"]))
        seconds = [float(row["start_s"]) for row in sequence]
        bpm = [float(row["prediction_bpm"] or "nan") for row in sequence]
        fig, ax = plt.subplots(figsize=(8, 3.5))
        ax.plot(seconds, bpm, ".-", label="Prediction")
        if any(row["reference_bpm"] for row in sequence):
            reference = [float(row["reference_bpm"] or "nan") for row in sequence]
            ax.plot(seconds, reference, ".-", label="Reference")
        ax.set(xlabel="Window start (s)", ylabel="Pulse rate (BPM)", title=Path(selected).name)
        ax.legend()
        save(fig, "bpm_timeline.png")

    pairs = [(float(row["reference_bpm"]), float(row["prediction_bpm"]))
             for row in rows if row["reference_bpm"] and row["prediction_bpm"]]
    if pairs:
        reference, prediction = zip(*pairs)
        low, high = min(min(reference), min(prediction)) - 1, max(max(reference), max(prediction)) + 1
        fig, ax = plt.subplots(figsize=(5, 5))
        ax.scatter(reference, prediction, s=14, alpha=0.6)
        ax.plot([low, high], [low, high], "--", color="0.4")
        ax.set(xlabel="Reference (BPM)", ylabel="Prediction (BPM)", xlim=(low, high), ylim=(low, high))
        ax.set_aspect("equal", adjustable="box")
        save(fig, "bpm_scatter.png")

        fig, ax = plt.subplots(figsize=(6, 3.5))
        ax.hist([pred - ref for ref, pred in pairs], bins=20)
        ax.axvline(0, linestyle="--", color="0.4")
        ax.set(xlabel="Prediction − reference (BPM)", ylabel="Windows / samples")
        save(fig, "error_histogram.png")

    if not files:
        raise ValueError("No timed windows or valid reference/prediction pairs to plot")
    return files
