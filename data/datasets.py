"""Dataset classes for the KMUNet evaluation suite.

Two directory layouts are supported because public versions of the three
target datasets ship in different shapes:

1. **Preprocessed split layout** (preferred — matches KMUNet's loader):
       <root>/train_images/  <root>/train_labels/
       <root>/val_images/    <root>/val_labels/
       <root>/test_images/   <root>/test_labels/

2. **Single-folder layout** (raw releases):
       <root>/images/        <root>/masks/    (or labels/, gt/)
   In this case the dataset wrapper deterministically splits 70/30 into
   train and test (and reuses the test split for val) using ``--seed``.

Image file extensions: ``.jpg``, ``.png``, ``.jpeg``, ``.bmp``, ``.tif``.

For BUSI the original release stores benign/malignant/normal in three
sub-folders with one or more ``*_mask*.png`` per case. ``BUSIDataset``
flattens these and pairs each image with its (logical-or-merged) mask.
"""

import glob
import os
import random
from typing import List, Optional, Tuple

import numpy as np
from PIL import Image
from torch.utils.data import Dataset


_IMG_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff")


def _list_images(folder: str) -> List[str]:
    files = []
    for ext in _IMG_EXTS:
        files.extend(glob.glob(os.path.join(folder, f"*{ext}")))
        files.extend(glob.glob(os.path.join(folder, f"*{ext.upper()}")))
    return sorted(files)


def _deterministic_split(files: List[str], split: str, ratio: float = 0.7,
                         seed: int = 1337) -> List[str]:
    rng = random.Random(seed)
    idx = list(range(len(files)))
    rng.shuffle(idx)
    n_train = int(round(len(files) * ratio))
    if split == "train":
        keep = set(idx[:n_train])
    else:  # val or test reuse the held-out tail
        keep = set(idx[n_train:])
    return [files[i] for i in sorted(keep)]


class _SplitFolderDataset(Dataset):
    """Layout 1: ``<root>/{train,val,test}_{images,labels}/``."""

    def __init__(self, root: str, split: str, transform):
        self.root = root
        self.split = split
        self.transform = transform
        img_dir = os.path.join(root, f"{split}_images")
        lbl_dir = os.path.join(root, f"{split}_labels")
        if not os.path.isdir(img_dir) or not os.path.isdir(lbl_dir):
            raise FileNotFoundError(img_dir + " or " + lbl_dir)
        names = [os.path.basename(p) for p in _list_images(img_dir)]
        # keep only items whose label also exists
        self.samples = []
        for name in names:
            stem = os.path.splitext(name)[0]
            lbl = self._find_label(lbl_dir, stem)
            if lbl is not None:
                self.samples.append((os.path.join(img_dir, name), lbl))

    @staticmethod
    def _find_label(lbl_dir: str, stem: str) -> Optional[str]:
        for ext in _IMG_EXTS:
            for cand in (stem + ext, stem + "_mask" + ext, stem + "_segmentation" + ext):
                p = os.path.join(lbl_dir, cand)
                if os.path.isfile(p):
                    return p
        return None

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        img_path, lbl_path = self.samples[idx]
        image = Image.open(img_path).convert("RGB")
        label = Image.open(lbl_path).convert("L")
        sample = {"image": image, "label": label,
                  "filename": os.path.basename(img_path)}
        if self.transform is not None:
            sample = self.transform(sample)
        return sample


class _FlatFolderDataset(Dataset):
    """Layout 2: a single ``images/`` + ``masks/`` (or ``labels/`` / ``gt/``)
    folder, split deterministically inside __init__.
    """

    def __init__(self, root: str, split: str, transform,
                 split_ratio: float = 0.7, seed: int = 1337):
        self.root = root
        self.split = split
        self.transform = transform

        img_dir = None
        for cand in ("images", "image", "Images"):
            if os.path.isdir(os.path.join(root, cand)):
                img_dir = os.path.join(root, cand)
                break
        lbl_dir = None
        for cand in ("masks", "mask", "labels", "label", "gt", "GT", "Masks"):
            if os.path.isdir(os.path.join(root, cand)):
                lbl_dir = os.path.join(root, cand)
                break
        if img_dir is None or lbl_dir is None:
            raise FileNotFoundError(
                f"Could not locate images/ + masks/ folders under {root}"
            )

        all_images = _list_images(img_dir)
        kept = _deterministic_split(all_images, split, ratio=split_ratio, seed=seed)

        self.samples: List[Tuple[str, str]] = []
        for p in kept:
            stem = os.path.splitext(os.path.basename(p))[0]
            lbl = _SplitFolderDataset._find_label(lbl_dir, stem)
            if lbl is not None:
                self.samples.append((p, lbl))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        img_path, lbl_path = self.samples[idx]
        image = Image.open(img_path).convert("RGB")
        label = Image.open(lbl_path).convert("L")
        sample = {"image": image, "label": label,
                  "filename": os.path.basename(img_path)}
        if self.transform is not None:
            sample = self.transform(sample)
        return sample


class CVCClinicDB(Dataset):
    """CVC-ClinicDB colonoscopy polyp dataset (~612 images).

    Accepts either preprocessed split layout or the official release where
    images live under ``Original/`` and ground-truth under ``Ground Truth/``.
    """

    def __init__(self, root: str, split: str, transform=None,
                 split_ratio: float = 0.7, seed: int = 1337):
        if os.path.isdir(os.path.join(root, f"{split}_images")):
            self._inner = _SplitFolderDataset(root, split, transform)
            return
        # Fall back to "Original" / "Ground Truth" subfolders if present.
        if os.path.isdir(os.path.join(root, "Original")):
            wrapper_root = os.path.join(root, "_cvc_alias")
            os.makedirs(wrapper_root, exist_ok=True)
            self._setup_alias(root, "Original", "Ground Truth", wrapper_root)
            self._inner = _FlatFolderDataset(
                wrapper_root, split, transform, split_ratio=split_ratio, seed=seed
            )
            return
        self._inner = _FlatFolderDataset(
            root, split, transform, split_ratio=split_ratio, seed=seed
        )

    @staticmethod
    def _setup_alias(root: str, img_sub: str, lbl_sub: str, alias_root: str):
        # Create symlinks "images" / "masks" inside alias_root so the flat
        # loader picks them up; skip if already linked.
        for tgt_name, src_sub in (("images", img_sub), ("masks", lbl_sub)):
            link = os.path.join(alias_root, tgt_name)
            target = os.path.abspath(os.path.join(root, src_sub))
            if os.path.islink(link) or os.path.exists(link):
                continue
            try:
                os.symlink(target, link, target_is_directory=True)
            except OSError:
                pass  # fall back will raise downstream if both layouts fail

    def __len__(self):
        return len(self._inner)

    def __getitem__(self, idx):
        return self._inner[idx]


class GlasDataset(Dataset):
    """Glas gland-segmentation challenge dataset (~165 images)."""

    def __init__(self, root: str, split: str, transform=None,
                 split_ratio: float = 0.7, seed: int = 1337):
        if os.path.isdir(os.path.join(root, f"{split}_images")):
            self._inner = _SplitFolderDataset(root, split, transform)
        else:
            self._inner = _FlatFolderDataset(
                root, split, transform, split_ratio=split_ratio, seed=seed
            )

    def __len__(self):
        return len(self._inner)

    def __getitem__(self, idx):
        return self._inner[idx]


class BUSIDataset(Dataset):
    """BUSI breast ultrasound (~210 malignant images by default).

    The official release stores cases under ``benign/``, ``malignant/``,
    ``normal/``. The KMUNet paper used only the malignant set (~210 images).
    Pass ``categories`` to override (e.g. ``("benign", "malignant")``).
    For each image ``X.png`` the release may contain one or more
    ``X_mask*.png``; multiple masks are merged via logical OR.
    """

    def __init__(self, root: str, split: str, transform=None,
                 categories: Tuple[str, ...] = ("malignant",),
                 split_ratio: float = 0.7, seed: int = 1337):
        if os.path.isdir(os.path.join(root, f"{split}_images")):
            self._inner = _SplitFolderDataset(root, split, transform)
            self._merged = False
            return

        self._merged = True
        self.transform = transform
        self.samples: List[Tuple[str, List[str]]] = []
        for cat in categories:
            cat_dir = os.path.join(root, cat)
            if not os.path.isdir(cat_dir):
                continue
            for img_path in _list_images(cat_dir):
                name = os.path.basename(img_path)
                stem, ext = os.path.splitext(name)
                if "_mask" in stem:
                    continue  # skip masks themselves
                mask_paths = sorted(
                    glob.glob(os.path.join(cat_dir, f"{stem}_mask*{ext}"))
                    + glob.glob(os.path.join(cat_dir, f"{stem}_mask*.png"))
                )
                if mask_paths:
                    self.samples.append((img_path, mask_paths))

        # Deterministic train/test split.
        rng = random.Random(seed)
        rng.shuffle(self.samples)
        n_train = int(round(len(self.samples) * split_ratio))
        if split == "train":
            self.samples = self.samples[:n_train]
        else:
            self.samples = self.samples[n_train:]

    def __len__(self):
        if self._merged:
            return len(self.samples)
        return len(self._inner)

    def __getitem__(self, idx):
        if not self._merged:
            return self._inner[idx]

        img_path, mask_paths = self.samples[idx]
        image = Image.open(img_path).convert("RGB")
        # OR-merge all mask files
        merged = None
        for mp in mask_paths:
            arr = np.array(Image.open(mp).convert("L"))
            arr = (arr > 127).astype(np.uint8) * 255
            merged = arr if merged is None else np.maximum(merged, arr)
        if merged is None:
            merged = np.zeros((image.height, image.width), dtype=np.uint8)
        label = Image.fromarray(merged)
        sample = {"image": image, "label": label,
                  "filename": os.path.basename(img_path)}
        if self.transform is not None:
            sample = self.transform(sample)
        return sample


DATASET_REGISTRY = {
    "cvc_clinicdb": CVCClinicDB,
    "cvc": CVCClinicDB,
    "glas": GlasDataset,
    "busi": BUSIDataset,
}


def build_dataset(name: str, root: str, split: str, transform=None,
                  seed: int = 1337):
    """Factory: pick a dataset class by name (case-insensitive)."""
    key = name.lower().strip()
    if key not in DATASET_REGISTRY:
        raise ValueError(
            f"Unknown dataset '{name}'. Available: {sorted(DATASET_REGISTRY)}"
        )
    cls = DATASET_REGISTRY[key]
    return cls(root=root, split=split, transform=transform, seed=seed)
