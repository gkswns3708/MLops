# Transform
import albumentations as A
from albumentations.pytorch import ToTensorV2
from albumentations.pytorch.transforms import ToTensor

MEAN_IMAGENET = [0.485, 0.456, 0.406]
STD_IMAGENET = [0.229, 0.224, 0.225]


def transforms_select(method, default, mean=MEAN_IMAGENET, std=STD_IMAGENET):
    lib = {
        "train_Filp_transform": A.Compose(
            [
                A.OneOf(
                    [
                        A.HorizontalFlip(p=1.0),
                        A.VerticalFlip(p=1.0),
                        A.RandomRotate90(p=1.0),
                    ],
                    p=0.75,
                ),
                A.RandomBrightness(p=0.5, limit=(-0.2, 0.25)),
                A.RandomContrast(p=0.5, limit=(-0.25, 0.25)),
                ToTensorV2(p=1.0),  # dtype float32, transpose 한번에
            ]
        ),
        "train_deault_transform": A.Compose([ToTensorV2(p=1.0)]),
    }
    return lib[method], lib[default]