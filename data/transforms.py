"""Train / test image-mask transforms.

The augmentation policy mirrors KMUNet's ``My_Train_Transforms``:
random horizontal / vertical flip + arbitrary rotation, then a resize
to ``(image_size, image_size)``. Test transforms only resize.

Sample contract throughout the project:
    {"image": Tensor[3, H, W] float in [0, 1],
     "label": Tensor[H, W]    long / float in {0, 1},
     "filename": str}
"""

import random
from typing import Dict

import numpy as np
import torch
import torchvision.transforms.functional as TF
from PIL import Image


class _BaseTransform:
    def __init__(self, image_size: int = 256):
        self.image_size = image_size

    @staticmethod
    def _to_pil(x):
        if isinstance(x, Image.Image):
            return x
        if isinstance(x, np.ndarray):
            return Image.fromarray(x)
        raise TypeError(f"Unsupported image type {type(x)}")

    def _resize_pair(self, image: Image.Image, label: Image.Image):
        image = image.resize((self.image_size, self.image_size), Image.BILINEAR)
        label = label.resize((self.image_size, self.image_size), Image.NEAREST)
        return image, label

    @staticmethod
    def _finalise(image: Image.Image, label: Image.Image):
        # image -> float32 (3, H, W) in [0, 1]
        img_arr = np.array(image, dtype=np.float32) / 255.0
        if img_arr.ndim == 2:
            img_arr = np.stack([img_arr] * 3, axis=-1)
        img_t = torch.from_numpy(img_arr.transpose(2, 0, 1)).contiguous()

        # label -> long (H, W) in {0, 1}
        lbl_arr = np.array(label)
        if lbl_arr.ndim == 3:
            lbl_arr = lbl_arr[..., 0]
        lbl_arr = (lbl_arr > 127).astype(np.int64)
        lbl_t = torch.from_numpy(lbl_arr).contiguous()
        return img_t, lbl_t


class My_Train_Transforms(_BaseTransform):
    """Random flip + rotation + resize, then tensor conversion."""

    def __init__(self, image_size: int = 256, p_hflip: float = 0.5,
                 p_vflip: float = 0.5, max_rotation: int = 360):
        super().__init__(image_size=image_size)
        self.p_hflip = p_hflip
        self.p_vflip = p_vflip
        self.max_rotation = max_rotation

    def __call__(self, sample: Dict) -> Dict:
        image = self._to_pil(sample["image"])
        label = self._to_pil(sample["label"])

        if random.random() < self.p_hflip:
            image = TF.hflip(image)
            label = TF.hflip(label)
        if random.random() < self.p_vflip:
            image = TF.vflip(image)
            label = TF.vflip(label)

        angle = random.randint(0, self.max_rotation)
        image = TF.rotate(image, angle, interpolation=TF.InterpolationMode.BILINEAR)
        label = TF.rotate(label, angle, interpolation=TF.InterpolationMode.NEAREST)

        image, label = self._resize_pair(image, label)
        img_t, lbl_t = self._finalise(image, label)
        return {"image": img_t, "label": lbl_t,
                "filename": sample.get("filename", "")}


class My_Test_Transforms(_BaseTransform):
    """Deterministic: only resize to ``image_size`` + tensor conversion."""

    def __call__(self, sample: Dict) -> Dict:
        image = self._to_pil(sample["image"])
        label = self._to_pil(sample["label"])
        image, label = self._resize_pair(image, label)
        img_t, lbl_t = self._finalise(image, label)
        return {"image": img_t, "label": lbl_t,
                "filename": sample.get("filename", "")}
