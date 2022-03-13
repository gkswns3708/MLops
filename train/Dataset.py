import cv2
import json
import torch
import numpy
import torch.nn as nn
from torch.utils.data import Dataset

class CustomDataset(Dataset):
    def __init__(self, path, config,transform=None, mode='train'):
        self.path = path
        self.transform = transform
        self.mode = mode
        self.label_encoder = config['Annotation_info']['label_encoder']
        self.label_decoder = config['Annotation_info']['label_decoder']
    
    def __len__(self):
        return len(self.path)

    def __getitem__(self, idx):
        data_path = self.path[idx]
        data_name = data_path.split('/')[-1]

        # 전처리
        image = cv2.imread(f"{data_path}/{data_name}.jpg")
        image = self.transform(image=image)['image'] # alubmentation transform

        if self.mode == 'train':
            data_json = json.load(open(f"{data_path}/{data_name}.json", 'r'))
            crop = data_json['annotations']['crop']
            disease = data_json['annotations']['disease']
            risk = data_json['annotations']['risk']
            label = f'{crop}_{disease}_{risk}'
            return {
                'image' : torch.tensor(image/255, dtype=torch.float32),
                'label' : torch.tensor(self.label_encoder[label], dtype= torch.long) 
                # 이거 꼭 tensor형태로 return 해야하나?
            }
        else:
            return {
                'image' : image
            }