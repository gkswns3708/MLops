import io
import numpy as np
from PIL import Image

import torch
import albumentations as A
import albumentations.pytorch
from albumentations.pytorch import ToTensorV2


def transform_image(image_bytes: bytes) -> torch.Tensor:
    transform = A.Compose([A.Resize(256, 256), ToTensorV2()])
    # TODO Numpy로 Load하면 더 빠르지 않을까?
    image = Image.open(io.BytesIO(image_bytes))
    image = image.convert("RGB")
    image = np.array(image, dtype=np.float32)
    return transform(image=image)["image"].unsqueeze(0)
