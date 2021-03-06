import os
from .datasets import CustomDataset, CropDataset
from .transforms import transforms_select
from torch.utils.data import DataLoader
from torchsampler import ImbalancedDatasetSampler  # 이거 git 다운 받아야 함.


class CropDataLoader:
    def __init__(
        self,
        data_dir,
        batch_size,
        shuffle=True,
        valid_split_ratio=0.1,
        num_workers=1,
        sampler=None,
        transform=None,
        default_transform=None,
        submit=False,
        is_main=None,
        config=None,
    ):
        # TODO : 이거 eval보단 test로 Path를 json에 지정하는게 좋을 듯.
        # TODO : inference를 위한 도구(data_loader)도 필요한 지 고민하기.
        # TODO : is_main 의미 파악하기
        self.data_dir = data_dir  # config에서 Prepared_Data까지만 허용 됨.
        self.shuffle = shuffle
        self.train_dir = os.path.join(self.data_dir, "train")
        self.test_dir = os.path.join(self.data_dir, "eval")  
        self.label_encoder = config["Annotation_info"]["label_encoder"]

        assert transform is not None, "Set the transform on train set"
        assert default_transform is not None, "Set the default_transform on valid/test set"

        self.transform = transform
        self.default_transform = default_transform
        # if you use base_filp, use default_base_flip

        print(f"Current transforms : {self.transform}")
        print(f"num_workers : {num_workers}")

        self.base_dataset = CropDataset(
            transform=self.transform,
            default_transform=self.default_transform,
            config=config,
        )

        assert 1 >= valid_split_ratio >= 0, "Set the valid_split_ratio correctly(1>= p >=0)"
        assert config is not None, "Set the configure file on config.json"
        self.train_dataset, self.valid_dataset = self.base_dataset.split_validation(valid_split_ratio)

        # TODO : Oversampling, normal Sampling 구현
        # TODO : assert 제대로 배우고 sampler 값을 assert로 구현하기
        # TODO : train/valid Shuffle 어떻게 해야하는지 고민하기.
        if sampler == "over":
            print("Sampler is OverSampling Mode")
            self.shuffle = False
            self.train_dataloader = DataLoader(
                self.train_dataset,
                batch_size,
                shuffle=self.shuffle,
                num_workers=num_workers,
                sampler=ImbalancedDatasetSampler(self.train_dataset),
                pin_memory=True,
            )
        elif sampler == "normal":
            print("No sampling method")
            self.train_dataloader = DataLoader(
                self.train_dataset,
                batch_size,
                shuffle=self.shuffle,
                numworkers=num_workers,
                pin_memory=True,
            )
        self.valid_dataloader = DataLoader(
            self.valid_dataset,
            batch_size,
            shuffle=self.shuffle,
            num_workers=num_workers,
            pin_memory=True,
        )
        # TODO : inference를 위한 Dataloader도 필요한가 고민하기 -> 아마 필요 없을 거 같긴함
        # 그냥 model(image)하면 나오는 값으로 학습하면 되지 않을까 싶음.
        # if submit:

    def split_validation(self):
        return self.train_dataloader, self.valid_dataloader
