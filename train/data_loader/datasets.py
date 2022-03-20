import cv2
import json
import torch
import numpy
import torch.nn as nn
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split
import os
import pandas as pd


class CustomDataset(Dataset):
    def __init__(
        self,
        path,
        label,
        label_encoder=None,
        label_decoder=None,
        transform=None,
        mode="train",
    ):
        self.path = path
        self.label = label
        self.transform = transform
        self.mode = mode

    def __len__(self):
        return len(self.path)

    def __getitem__(self, idx):
        data_path = self.path[idx]
        data_name = data_path.split("/")[-1]

        # 전처리
        image = cv2.imread(f"{data_path}/{data_name}.jpg")
        image = self.transform(image=image)["image"]  # alubmentation transform

        if self.mode == "train":
            data_json = json.load(open(f"{data_path}/{data_name}.json", "r"))
            crop = data_json["annotations"]["crop"]
            disease = data_json["annotations"]["disease"]
            risk = data_json["annotations"]["risk"]
            label = f"{crop}_{disease}_{risk}"
            return {
                "image": torch.tensor(image / 255, dtype=torch.float32),
                "label": torch.tensor(self.label_encoder[label], dtype=torch.long),
            }
        else:
            return {"image": image}

class BaseDataset(Dataset):
    def __init__(self, dataset_df, transform=None):
        super().__init__()
        assert transform is not None, "Set the transform on train set"
        self.transform = transform
        self.dataset_df = dataset_df

    def __len__(self):
        return len(self.dataset_df)

    def __getitem__(self, idx):
        # TODO : Image Path를 csv 파일에 추가하는게 좋은 선택지 같음.
        self.dataset_df['image_path']
        image = cv2.imread(self.dataset_df['image_path'].iloc[idx])
        # TODO : TOTensorV2를 쓰면 numpy 채널 축도 변하고 ToTensor의 기능을 포함함.
        # image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = self.transform(image=image)['image'].float()
        return image, torch.tensor(self.dataset_df['label'].iloc[idx])
    
    def get_labels(self):
        return self.dataset_df['label'].iloc[:]

class CropDataset():
    def __init__(self, transform=None, default_transform=None, config=None):
        super().__init__()
        # TODO : 다양한 assert문 utils로 추가해놓기 (params : variable name)
        assert transform is not None, "Set the transform on train set"
        assert default_transform is not None, "Set the default transform"

        self.transform = transform
        self.default_transform = default_transform
        self.dir_path = os.path.dirname(config['Prepared_Data_Path'])
        self.df_path = os.path.join(self.dir_path, 'train.csv')
        self.img_dir_path = config['Prepared_Data_Path']

        self.dataset_df = pd.read_csv(os.path.join(config['Prepared_Csv_Path'], 'train.csv'))
        # TODO : Class Imbalance 문제를 해결하기 위해 각 class weight를 지정하는 코드 Cross_Entropy를 사용할 때 사용할 듯
        # self.class_weight = self._get_class_weight() # 학습 고도화를 위한 각 class Imbalance 문제를 위한 코드


    def split_validation(self, valid_split_ratio):
        # TODO : Stratify의 처리가 제대로되는지 확인하기
        df_train, df_val = train_test_split(self.dataset_df, test_size=valid_split_ratio, random_state=42, stratify=self.dataset_df['label'].to_numpy())
        train_dataset = BaseDataset(df_train, transform=self.transform)
        val_dataset = BaseDataset(df_val, transform=self.default_transform)
        return train_dataset, val_dataset