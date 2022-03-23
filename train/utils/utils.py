import os
import cv2
import json
import torch
import pandas as pd
import os.path as osp
# TODO : pathlib 이친구 뭐하는 친구인지 알아내기
from pathlib import Path
from glob import glob
from collections import OrderedDict

# 필요한 directory를 만들어주는 Utils
def make_necessary_dir(Prepared_Data_Path: str, Model_Save_Path: str):
    """필요한 directory를 만들어주는 함수.

    Args:
        Prepared_Data_Path (str)): 전처리된 Data들이 저장될 Path
        Model_Save_Path (str): Training된 Model의 weight가 저장될 Path
    """
    os.makedirs(Prepared_Data_Path, exist_ok=True)
    os.makedirs(Model_Save_Path, exist_ok=True)


def get_image_json_path(Raw_Data_Path: str) -> list:
    """label을 얻기 위한 json_path 얻는 함수

    Args:
        Raw_Data_Path (str): Raw Data들이 있는 Path

    Returns:
        list : Raw Data들의 label이 있는 list
    """
    raw_data = sorted(glob(osp.join(Raw_Data_Path, "*")))
    raw_image_json = list()
    for raw_data_path in raw_data:
        Image_name = raw_data_path.split("/")[-1]
        Image = osp.join(raw_data_path, Image_name + ".jpg")
        Json = osp.join(raw_data_path, Image_name + ".json")
        raw_image_json.append((Image, Json))

    return raw_image_json


def make_label_encoder_decoder(config: dict) -> None:
    """config_train.json에 들어가는 Annotation_info를 만들어주는 함수.

    Args:
        config (dict): Config dict
    """
    path_list = sorted(glob(osp.join(config["Prepared_Data_Path"], "*/*.json")))
    with open("config.json") as f:
        config = json.load(f)
    crop_dict = config["Annotation_info"]["crop_dict"]
    disease_dict = config["Annotation_info"]["disease_dict"]
    risk_dict = config["Annotation_info"]["risk_dict"]
    area_dict = config["Annotation_info"]["area_dict"]
    grow_dict = config["Annotation_info"]["grow_dict"]
    label_set = set()
    all_label_set = set()
    for idx, path in enumerate(path_list):
        now_json = json.load(open(path, "r"))
        crop = crop_dict[str(now_json["annotations"]["crop"])]
        disease = disease_dict[str(now_json["annotations"]["disease"])]
        risk = risk_dict[str(now_json["annotations"]["risk"])]
        area = area_dict[str(now_json["annotations"]["area"])]
        grow = grow_dict[str(now_json["annotations"]["grow"])]
        label_set.add(crop + "_" + disease + "_" + risk)
        all_label_set.add(crop + "_" + disease + "_" + risk + "_" + area + "_" + grow)
    label_encoder = {key: idx for idx, key in enumerate(sorted(label_set))}
    label_decoder = {val: key for key, val in label_encoder.items()}
    all_label_encoder = {key: idx for idx, key in enumerate(sorted(all_label_set))}
    all_label_decoder = {val: key for key, val in all_label_encoder.items()}
    config["Annotation_info"]["label_encoder"] = label_encoder
    config["Annotation_info"]["label_decoder"] = label_decoder
    config["Annotation_info"]["all_label_encoder"] = all_label_encoder
    config["Annotation_info"]["all_label_decoder"] = all_label_decoder
    with open("config.json", "w") as f:
        f.write(json.dumps(config, ensure_ascii=False))


def get_Resized_Image_Dataset(config: dict) -> None:
    # 5762장 Resized 실행시 1분 50분 정도 Local 환경에서 소요 됨.
    Raw_Data_Path = config["Raw_Data_Path"]
    Raw_Data_Path_list = sorted(glob(osp.join(Raw_Data_Path, "*")))
    for now_dir_path in Raw_Data_Path_list:
        data_num = now_dir_path.split("/")[-1]
        now_image_path = osp.join(now_dir_path, f"{data_num}.jpg")
        now_json_path = osp.join(now_dir_path, f"{data_num}.json")
        name = str(len(glob(osp.join(config["Prepared_Data_Path"], "*"))) + 1)
        img = cv2.imread(now_image_path)
        img = cv2.resize(
            img, dsize=(config["Width"], config["Height"]), interpolation=cv2.INTER_AREA
        )
        now_json = json.load(open(now_json_path, "r"))

        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        Data_save_path = osp.join(config["Prepared_Data_Path"], name)

        # Save File
        if osp.isdir(Data_save_path) == False:
            os.makedirs(Data_save_path, exist_ok=True)

        cv2.imwrite(osp.join(Data_save_path, name + ".jpg"), img)
        with open(osp.join(Data_save_path, name + ".json"), "w") as f:
            json.dump(now_json, f, indent=4)



# TODO : 이 친구는 Build할 때만 필요했던 친구(config 제작할 때) 이런 경우에는 git에서 삭제하나?
def make_label_encoder_decoder(config):
    save_path = config['Prepared_Data_Path']
    path_list = sorted(glob(osp.join(save_path, "*/*.json")))
    label_set = set()
    all_label_set = set()
    for idx, path in enumerate(path_list):
        now_json = json.load(open(path, "r"))
        crop = crop_dict[now_json["annotations"]["crop"]]
        disease = disease_dict[now_json["annotations"]["disease"]]
        risk = risk_dict[now_json["annotations"]["risk"]]
        area = area_dict[now_json["annotations"]["area"]]
        grow = grow_dict[now_json["annotations"]["grow"]]
        label_set.add(crop + "_" + disease + "_" + risk)
        all_label_set.add(crop + "_" + disease + "_" + risk + "_" + area + "_" + grow)
    label_encoder = {key: idx for idx, key in enumerate(sorted(label_set))}
    label_decoder = {val: key for key, val in label_encoder.items()}
    all_label_encoder = {key: idx for idx, key in enumerate(sorted(all_label_set))}
    all_label_decoder = {val: key for key, val in all_label_encoder.items()}
    return label_encoder, label_decoder, all_label_encoder, all_label_decoder


def Set_Dataset_CSV(config):
    """train할 때 사용하는 Dataset의 정보를 담고 있는 CSV 생성 함수

    Args:
        config (dict): csv 파일을 만드는데 사용되는 config를 포함하고 있는 config dict
    """
    data = {
        "name": [],
        "path": [],
        "image_path" : [],
        "Described_label": [],  # 0
        "Described_delabel": [],  # '고추_고추탄저병-1_중기'
        "Described_all_label": [],  # 0,
        "Described_all_delabel": [],  # '고추_고추탄저병-1_중기_열매_착화/과실기',
        "label": [],  # 0,
        "delabel": [],  # '1_00_0',
        "all_label": [],  # 0,
        "all_delabel": [],  #'1_00_0_3_11',
    }
    path = sorted(glob(osp.join(config["Prepared_Data_Path"], "*")))
    for now_path in path:
        if now_path in ".json":
            continue
        name = now_path.split("/")[-1]
        image_path = osp.join(now_path, name + ".jpg")
        json_path = osp.join(now_path, name + ".json")
        now_json = json.load(open(json_path, "r"))
        # TODO : 여기 다 뜯어 고치고 싶네 어떻게 안되나...?
        crop = str(now_json["annotations"]["crop"])
        disease = now_json["annotations"]["disease"]
        risk = str(now_json["annotations"]["risk"])
        area = str(now_json["annotations"]["area"])
        grow = str(now_json["annotations"]["grow"])

        human_label_encoder = config["Annotation_info"]["human_label_encoder"]
        human_all_label_encoder = config["Annotation_info"]["human_all_label_encoder"]
        label_encoder = config["Annotation_info"]["label_encoder"]
        all_label_encoder = config["Annotation_info"]["all_label_encoder"]

        crop_dict = config["Annotation_info"]["crop_dict"]
        disease_dict = config["Annotation_info"]["disease_dict"]
        risk_dict = config["Annotation_info"]["risk_dict"]
        grow_dict = config["Annotation_info"]["grow_dict"]
        area_dict = config["Annotation_info"]["area_dict"]

        Described_delabel = (
            f"{crop_dict[crop]}_{disease_dict[disease]}_{risk_dict[risk]}"
        )
        Described_label = human_label_encoder[Described_delabel]
        Described_all_delabel = f"{crop_dict[crop]}_{disease_dict[disease]}_{risk_dict[risk]}_{area_dict[area]}_{grow_dict[grow]}"
        Described_all_label = human_all_label_encoder[Described_all_delabel]

        delabel = f"{crop}_{disease}_{risk}"
        label = label_encoder[delabel]
        all_delabel = f"{crop}_{disease}_{risk}_{area}_{grow}"
        all_label = all_label_encoder[all_delabel]
        
        variable_names = [
            "name",
            "path",
            "image_path",
            "Described_label",
            "Described_delabel",
            "Described_all_label",
            "Described_all_delabel",
            "label",
            "delabel",
            "all_label",
            "all_delabel",
        ]
        variables = [
            name,
            now_path,
            image_path,
            Described_label,
            Described_delabel,
            Described_all_label,
            Described_all_delabel,
            label,
            delabel,
            all_label,
            all_delabel,
        ]
        for variable_name, variable in zip(variable_names, variables):
            data[f"{variable_name}"].append(variable)

    df = pd.DataFrame(data)
    df.to_csv(osp.join(config["Prepared_Csv_Path"], "train.csv"))

def prepare_device(n_gpu_use):
    """
    setup GPU device if available. get gpu device indices which are used for DataParallel
    """
    n_gpu = torch.cuda.device_count()
    if n_gpu_use > 0 and n_gpu == 0:
        print("Warning: There\'s no GPU available on this machine,"
              "training will be performed on CPU.")
        n_gpu_use = 0
    if n_gpu_use > n_gpu:
        print(f"Warning: The number of GPU\'s configured to use is {n_gpu_use}, but only {n_gpu} are "
              "available on this machine.")
        n_gpu_use = n_gpu
    device = torch.device('cuda:0' if n_gpu_use > 0 else 'cpu')
    list_ids = list(range(n_gpu_use))
    return device, list_ids

def ensure_dir(dirname):
    dirname = Path(dirname)
    if not dirname.is_dir():
        dirname.mkdir(parents=True, exist_ok=False)

def read_json(fname):
    fname = Path(fname)
    with fname.open('rt') as handle:
        return json.load(handle, object_hook=OrderedDict)

def write_json(content, fname):
    fname = Path(fname)
    with fname.open('wt') as handle:
        json.dump(content, handle, indent=4, sort_keys=False)

def inf_loop(data_loader):
    ''' wrapper function for endless data loader. '''
    for loader in repeat(data_loader):
        yield from loader

class MetricTracker:
    def __init__(self, *keys, writer=None):
        # self.writer = writer
        self._data = pd.DataFrame(index=keys, columns=['total', 'counts', 'average'])
        self.reset()

    def reset(self):
        for col in self._data.columns:
            self._data[col].values[:] = 0

    def update(self, key, value, n=1):
        # if self.writer is not None:
        #     self.writer.add_scalar(key, value)
        self._data.total[key] += value * n
        self._data.counts[key] += n
        self._data.average[key] = self._data.total[key] / self._data.counts[key]

    def avg(self, key):
        return self._data.average[key]

    def result(self):
        return dict(self._data.average)
