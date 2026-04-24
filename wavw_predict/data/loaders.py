"""
DataLoader factory functions.
"""
from __future__ import annotations

from torch.utils.data import DataLoader, Dataset


def make_loader(
    dataset: Dataset,
    batch_size: int,
    shuffle: bool = True,
    num_workers: int = 4,
    pin_memory: bool = True,
    drop_last: bool = False,
) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=drop_last,
    )


def make_train_val_loaders(
    train_dataset: Dataset,
    val_dataset: Dataset,
    batch_size: int,
    num_workers: int = 4,
) -> tuple[DataLoader, DataLoader]:
    train_loader = make_loader(train_dataset, batch_size, shuffle=True,
                               num_workers=num_workers, drop_last=True)
    val_loader = make_loader(val_dataset, batch_size, shuffle=False,
                             num_workers=num_workers, drop_last=False)
    return train_loader, val_loader
