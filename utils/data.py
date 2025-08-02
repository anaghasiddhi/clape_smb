import os
import torch
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
import pickle
import pytorch_lightning as pl
import time

# ------------------- Dataset Class -------------------
class LigandData(Dataset):
    def __init__(self, data_dict_path):
        self.features = []
        self.labels = []
        self.seen_pids = set()

        rank = int(os.environ.get("RANK", 0))  # Get current DDP rank
        
        if os.path.isfile(data_dict_path):
            print(f"[Rank {rank}] Loading .pkl file: {data_dict_path}")
            with open(data_dict_path, "rb") as f:
                data_dict = pickle.load(f)
            for pid, (feat, label) in data_dict.items():
                self.seen_pids.add(pid)
                self.features.append(feat)
                self.labels.append(label)
            print(f"[Rank {rank}] Loaded {len(self.features)} entries from file.")

        elif os.path.isdir(data_dict_path):
            print(f"[Rank {rank}] Scanning directory: {data_dict_path}")
            for fname in sorted(os.listdir(data_dict_path)):
                if not fname.endswith(".pkl"):
                    continue
                if f"rank{rank}" not in fname:
                    continue
                path = os.path.join(data_dict_path, fname)
                print(f"[Rank {rank}] Loading {path}...")
                try:
                    with open(path, "rb") as f:
                        data_dict = pickle.load(f)
                    new_items = 0
                    for pid, (feat, label) in data_dict.items():
                        if pid in self.seen_pids:
                            continue
                        self.seen_pids.add(pid)
                        self.features.append(feat)
                        self.labels.append(label)
                        new_items += 1
                    print(f"[Rank {rank}] Loaded {new_items} unique entries.")
                except Exception as e:
                    print(f"[Rank {rank}] Failed to load {path}: {e}")
        else:
            raise ValueError(f"Invalid path provided: {data_dict_path}")


    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        feature = self.features[idx].clone().detach().float()
        label = self.labels[idx].clone().detach().long()
        return feature, label

# ------------------- Collate Function -------------------
class BatchCollate:
    def __call__(self, data):
        features = pad_sequence([t[0] for t in data], batch_first=True)
        labels = pad_sequence([t[1] for t in data], batch_first=True)
        return features, labels

# ------------------- Lightning DataModule -------------------
class ProteinLigandData(pl.LightningDataModule):
    def __init__(self, batch_size, train_data_root, val_data_root, workers=1, pin_memory=False):
        super().__init__()
        self.batch_size = batch_size
        self.workers = workers
        self.pin_memory = pin_memory

        self.collate_fn = BatchCollate()
        self.train_data = LigandData(train_data_root)
        self.val_data = LigandData(val_data_root)

    def train_dataloader(self):
        return DataLoader(
            self.train_data,
            batch_size=self.batch_size,
            collate_fn=self.collate_fn,
            num_workers=self.workers,
            pin_memory=self.pin_memory,
            shuffle=True,
            persistent_workers=False,
            prefetch_factor=1
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_data,
            batch_size=self.batch_size,
            collate_fn=self.collate_fn,
            num_workers=self.workers,
            pin_memory=self.pin_memory
        )
