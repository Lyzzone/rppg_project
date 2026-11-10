from pathlib import Path
from tempfile import NamedTemporaryFile

import torch

from . import INPUT, PULSE_RATES
from .model import ConvNet3D


def load_model(path, device):
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    if "model_state_dict" in checkpoint:
        if tuple(checkpoint["heart_rates_bpm"]) != PULSE_RATES:
            raise ValueError("Checkpoint pulse rate classes differ from the current model")
        if checkpoint.get("input", INPUT) != INPUT:
            raise ValueError("Checkpoint input contract differs from the current model")
        state = checkpoint["model_state_dict"]
        metadata = checkpoint
    else:
        state = checkpoint
        metadata = {}
    model = ConvNet3D()
    model.load_state_dict(state)
    return model.to(device).eval(), metadata


def save_model(path, model, epoch, val_loss, data):
    seeds = {f"{split}_seed": data[split]["seed"] for split in ("train", "val")
             if data[split]["source"] == "synthetic"}
    with NamedTemporaryFile(dir=path.parent, prefix=".checkpoint-", delete=False) as stream:
        temp = Path(stream.name)
        try:
            torch.save({
                "model_state_dict": model.state_dict(),
                "heart_rates_bpm": list(PULSE_RATES),
                "input": INPUT,
                "epoch": epoch,
                "val_loss": val_loss,
                **seeds,
            }, stream)
        except BaseException:
            stream.close()
            temp.unlink(missing_ok=True)
            raise
    try:
        temp.replace(path)
    finally:
        temp.unlink(missing_ok=True)
