"""Data loader factory used by train.py / test.py."""

from typing import Tuple

from torch.utils.data import DataLoader

from .datasets import build_dataset
from .transforms import My_Test_Transforms, My_Train_Transforms


def _worker_init_fn_factory(seed: int):
    import random

    import numpy as np

    def _init(worker_id):
        s = seed + worker_id
        random.seed(s)
        np.random.seed(s)

    return _init


def get_loaders(
    dataset: str,
    root: str,
    batch_size: int = 16,
    num_workers: int = 8,
    image_size: int = 256,
    seed: int = 1337,
    include_test: bool = False,
) -> Tuple[DataLoader, DataLoader]:
    """Build train and val DataLoaders. If ``include_test=True`` returns
    ``(train, val, test)``; otherwise ``(train, val)``.

    Splits resolved per dataset:
      - Preprocessed ``{train,val,test}_images/`` layouts are used as-is.
      - Otherwise a deterministic 70/30 split keeps train as 70% and uses
        the held-out 30% for both val and test (matching the KMUNet
        protocol on small datasets where no separate val split exists).
    """
    train_tf = My_Train_Transforms(image_size=image_size)
    test_tf = My_Test_Transforms(image_size=image_size)

    train_ds = build_dataset(dataset, root, split="train", transform=train_tf, seed=seed)
    val_ds = build_dataset(dataset, root, split="val", transform=test_tf, seed=seed)

    init_fn = _worker_init_fn_factory(seed)

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=True, drop_last=False,
        worker_init_fn=init_fn,
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True,
    )
    if not include_test:
        return train_loader, val_loader

    test_ds = build_dataset(dataset, root, split="test", transform=test_tf, seed=seed)
    test_loader = DataLoader(
        test_ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True,
    )
    return train_loader, val_loader, test_loader


def get_test_loader(
    dataset: str,
    root: str,
    batch_size: int = 16,
    num_workers: int = 8,
    image_size: int = 256,
    seed: int = 1337,
    split: str = "test",
) -> DataLoader:
    test_tf = My_Test_Transforms(image_size=image_size)
    ds = build_dataset(dataset, root, split=split, transform=test_tf, seed=seed)
    return DataLoader(
        ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True,
    )
