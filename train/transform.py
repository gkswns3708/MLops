# Transform
import albumentations as A
from albumentations.pytorch import ToTensorV2
from albumentations.pytorch.transforms import ToTensor

train_albumentation =  A.Compose([
                  A.OneOf([
                           A.HorizontalFlip(p=1.0),
                           A.VerticalFlip(p=1.0),
                           A.RandomRotate90(p=1.0)
                           ],p=0.75),
                  A.RandomBrightness(p=0.5, limit=(-0.2, 0.25)),
                  A.RandomContrast(p=0.5, limit=(-0.25, 0.25)),
                  ToTensorV2(p=1.0)# dtype float32, transpose 한번에 
                  ])
test_albumentation =  A.Compose([
                      ToTensorV2(p=1.0)# dtype float32, transpose 한번에 
                      ])  
