import csv
import math

import torch
from torch.nn import functional as F

from .checkpoint import load_model, save_model
from .config import new_run
from .data.build import make_loader
from .model import ConvNet3D


def run_epoch(model, loader, device, optimizer=None, freeze_backbone=False):
    training = optimizer is not None
    model.train(training)
    if training and freeze_backbone:
        model.layer1.eval()
    total_loss, correct, count = 0.0, 0, 0
    with torch.set_grad_enabled(training):
        for batch in loader:
            inputs, labels = batch["input"].to(device), batch["target"].to(device)
            if training:
                optimizer.zero_grad(set_to_none=True)
            logits = model(inputs)
            loss = F.cross_entropy(logits, labels)
            if not math.isfinite(loss.item()):
                raise ValueError("Non-finite loss; check the input data and learning rate")
            if training:
                loss.backward()
                optimizer.step()
            count += labels.numel()
            correct += (logits.argmax(dim=1) == labels).sum().item()
            total_loss += loss.item() * labels.numel()
    return total_loss / count, correct / count


def train(cfg, config_path, device):
    settings = cfg["train"]
    checkpoint = settings["init_checkpoint"]
    freeze = settings["freeze_backbone"]
    if freeze and not checkpoint:
        raise ValueError("freeze_backbone requires init_checkpoint")
    if checkpoint:
        model, _ = load_model(checkpoint, device)
    else:
        model = ConvNet3D().to(device)
    if freeze:
        model.layer1.requires_grad_(False)
    optimizer = torch.optim.Adam((p for p in model.parameters() if p.requires_grad),
                                 lr=settings["learning_rate"])
    train_loader, val_loader = make_loader(cfg, "train"), make_loader(cfg, "val")
    refresh = cfg["data"]["train"].get("refresh_each_epoch", False)
    run = new_run(cfg, config_path, "train", checkpoint)
    best_loss = float("inf")
    with (run / "training.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(("epoch", "train_loss", "train_accuracy", "val_loss", "val_accuracy"))
        for epoch in range(1, settings["epochs"] + 1):
            if refresh and epoch > 1:
                del train_loader
                train_loader = make_loader(cfg, "train", refresh_index=epoch - 1)
            loss, accuracy = run_epoch(model, train_loader, device, optimizer, freeze)
            val_loss, val_accuracy = run_epoch(model, val_loader, device)
            if val_loss < best_loss:
                best_loss = val_loss
                save_model(run / "best.pt", model, epoch, val_loss, cfg["data"])
            writer.writerow((epoch, loss, accuracy, val_loss, val_accuracy))
            stream.flush()
            print(f"epoch={epoch} train_loss={loss:.4f} val_loss={val_loss:.4f} val_accuracy={val_accuracy:.2%}", flush=True)
    return run
