"""Generic training utilities shared across train / test scripts."""

import logging
import os
import random
import sys

import numpy as np
import torch
import torch.backends.cudnn as cudnn


class AverageMeter:
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0.0
        self.sum = 0.0
        self.count = 0
        self.avg = 0.0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / max(self.count, 1)


def set_seed(seed: int, deterministic: bool = True) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if deterministic:
        cudnn.benchmark = False
        cudnn.deterministic = True
    else:
        cudnn.benchmark = True
        cudnn.deterministic = False


def worker_init_fn(worker_id: int, seed: int = 1337) -> None:
    random.seed(seed + worker_id)
    np.random.seed(seed + worker_id)


def setup_logger(log_dir: str, name: str = "wbkmnet") -> logging.Logger:
    os.makedirs(log_dir, exist_ok=True)
    log_path = os.path.join(log_dir, "log.txt")

    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    fmt = logging.Formatter("[%(asctime)s.%(msecs)03d] %(message)s", datefmt="%H:%M:%S")
    fh = logging.FileHandler(log_path)
    fh.setFormatter(fmt)
    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(fmt)
    logger.addHandler(fh)
    logger.addHandler(sh)
    return logger


def count_parameters(model: torch.nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
