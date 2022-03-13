import os
from .datasets import CustomDataset
from .transforms import transforms_select
from torch.utils.data import DataLoader
from torchsampler import ImbalancedDatasetSampler #이거 git 다운 받아야 함.

class CropDataLoader():
    def __init__(self, data_dir, batch_size, shuffle=True,
                validation_split=0.1, num_workers=1, sampler=None,
                transform=None, default_transform=None, submit=False, is_main=None,
                config=None):
        self.data_dir = data_dir
        self.shuffle = shuffle
        self.train_dir = os.path.join(self.data_dir, 'train')
        self.eval_dir = os.path.join(self.data_dir, 'eval') # train과 eval이 이런식으로 존재하는 듯 함.
        self.label_encoder = config[]

        assert transform is not None, "Set the transform on train set"
        assert default_transform is not None, "Set the default_transform on valid/test set"

        self.transform = transform
        self.default_transform = default_transform
        print(f"Current transforms : {self.transform}")
        print(f'num_workers : {num_workers}')

        # TODO : 현재는 어떤가
        # Custom_Dataset은 현재 pathlist를 주면 해당 path_list에서 json과 image를 가져왔음
        # 현재는 csv파일에 path가 있어서 거기서 가지고 오는 듯 하다. 우리도 그렇게 구현하면 좋을 거 같다.
        # 즉 dataframe을 넘기면 될 거 같다.
        # 그럼 먼저 Dataframe을 만드는 코드가 필요하다.
        self.train_dataset = CustomDataset(self.data_dir, config, transform=self.transform, mode='train')
        self.valid_dataset = CustomDataset(self.data_dir, config, transform=self.transform, mode='train')



