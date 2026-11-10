import torch
from torch.utils.data import DataLoader, Dataset

from .. import NO_PULSE, PULSE_RATES
from .real_video import RealVideoDataset
from .rppg10 import RPPG10Dataset
from .synthetic import DataGen


class SyntheticDataset(Dataset):
    def __init__(self, data, refresh_index):
        self.inputs, self.labels = DataGen(
            no_videos_by_class=data["samples_per_class"], seed=[data["seed"], refresh_index]
        ).gen_signal()

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        target = int(self.labels[index])
        return {
            "input": self.inputs[index], "target": target,
            "reference_bpm": PULSE_RATES[target] if target < NO_PULSE else float("nan"),
            "sample_id": f"synthetic-{index}", "video_id": "", "subject_id": "",
            "start_s": float("nan"), "end_s": float("nan"),
        }


def make_loader(cfg, split, refresh_index=0):
    if split not in cfg["data"]:
        raise ValueError(f"Missing [data.{split}] in config")
    data = cfg["data"][split]
    if data["source"] == "synthetic":
        dataset = SyntheticDataset(data, refresh_index)
    elif data["source"] == "rppg10":
        dataset = RPPG10Dataset(data)
    else:
        dataset = RealVideoDataset(data["manifest"], split)
    return DataLoader(dataset, batch_size=cfg["train"]["batch_size"], shuffle=split == "train",
                      generator=torch.Generator().manual_seed(cfg["runtime"]["seed"] + refresh_index))
