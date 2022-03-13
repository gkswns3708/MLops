import json
import os
import os.path as osp

from glob import glob

# 필요한 directory를 만들어주는 Utils
def make_necessary_dir(Prepared_Data_Path: str, Model_Save_Path: str):
    """_summary_

    Args:
        Prepared_Data_Path (str)): 전처리된 Data들이 저장될 Path
        Model_Save_Path (str): Training된 Model의 weight가 저장될 Path
    """
    os.makedirs(Prepared_Data_Path, exist_ok=True)
    os.makedirs(Model_Save_Path, exist_ok=True)


def get_image_json_path(Raw_Data_Path: str) -> list:
    """_summary_

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

def make_label_encoder_decoder(config : dict) -> None:
    """config_train.json에 들어가는 Annotation_info를 만들어주는 함수.

    Args:
        config (dict): Config dict
    """
    path_list = sorted(glob(osp.join(config['Prepared_Data_Path'], "*/*.json")))
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
        now_crop = crop_dict[str(now_json["annotations"]["crop"])]
        now_disease = disease_dict[str(now_json["annotations"]["disease"])]
        now_risk = risk_dict[str(now_json["annotations"]["risk"])]
        now_area = area_dict[str(now_json["annotations"]["area"])]
        now_grow = grow_dict[str(now_json["annotations"]["grow"])]
        label_set.add(now_crop + "_" + now_disease + "_" + now_risk)
        all_label_set.add(
            now_crop
            + "_"
            + now_disease
            + "_"
            + now_risk
            + "_"
            + now_area
            + "_"
            + now_grow
        )
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


# 5000장 44초 정도 걸림 (512, 384) -> (256,256)
def get_Resized_Image_Dataset(raw_image_json: list) -> None:
    for (now_image, now_json) in raw_image_json:
        name = str(len(glob(osp.join(save_path, "*"))) + 1)
        img = cv2.imread(now_image)
        now_json = json.load(open(now_json, "r"))
        img = cv2.resize(img, dsize=(width, height), interpolation=cv2.INTER_AREA)

        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        tmp_save_path = osp.join(save_path, name)
        # Save File
        if osp.isdir(tmp_save_path) == False:
            os.makedirs(tmp_save_path, exist_ok=True)

        cv2.imwrite(osp.join(tmp_save_path, name + ".jpg"), img)
        with open(osp.join(tmp_save_path, name + ".json"), "w") as f:
            json.dump(now_json, f, indent=4)


def make_label_encoder_decoder(save_path):
    path_list = sorted(glob(osp.join(save_path, "*/*.json")))
    label_set = set()
    all_label_set = set()
    for idx, path in enumerate(path_list):
        now_json = json.load(open(path, "r"))
        now_crop = crop_dict[now_json["annotations"]["crop"]]
        now_disease = disease_dict[now_json["annotations"]["disease"]]
        now_risk = risk_dict[now_json["annotations"]["risk"]]
        now_area = area_dict[now_json["annotations"]["area"]]
        now_grow = grow_dict[now_json["annotations"]["grow"]]
        label_set.add(now_crop + "_" + now_disease + "_" + now_risk)
        all_label_set.add(
            now_crop
            + "_"
            + now_disease
            + "_"
            + now_risk
            + "_"
            + now_area
            + "_"
            + now_grow
        )
    label_encoder = {key: idx for idx, key in enumerate(sorted(label_set))}
    label_decoder = {val: key for key, val in label_encoder.items()}
    all_label_encoder = {key: idx for idx, key in enumerate(sorted(all_label_set))}
    all_label_decoder = {val: key for key, val in all_label_encoder.items()}
    return label_encoder, label_decoder, all_label_encoder, all_label_decoder


def Set_Dataset_CSV(path):
    # dict crop_dict들이 필요함.
    # dict를 이용해 csv를 만들 수 있음.
    data = {
        'name' : [],
        'path' : [],
        'Described_label' : [], # 0
        'Described_delabel' : [], # '고추_고추탄저병-1_중기'
        'Described_all_label' : [], # 0,
        'Described_all_delabel' : [], # '고추_고추탄저병-1_중기_열매_착화/과실기',

        'label' : [], # 0,
        'delabel' : [], # '1_00_0',
        'all_label' : [], # 0,
        'all_delabel' : [], #'1_00_0_3_11',
    }
    path = sorted(glob(osp.join(path, '*')))
    for now_path in path:
        now_name = now_path.split('/')[-1]
        now_image_path = osp.join(now_path,now_name+'.jpg')
        now_json_path = osp.join(now_path,now_name+'.json')
        now_json = json.load(open(now_json_path,'r'))
        now_crop = now_json['annotations']['crop']
        now_disease = now_json['annotations']['disease']
        now_risk = now_json['annotations']['risk']
        now_area = now_json['annotations']['area']
        now_grow = now_json['annotations']['grow']

        Described_delabel = f'{crop_dict[now_crop]}_{disease_dict[now_disease]}_{risk_dict[now_risk]}'
        Described_label = label_encoder[Described_delabel]
        Described_all_delabel = f'{crop_dict[now_crop]}_{disease_dict[now_disease]}_{risk_dict[now_risk]}_{area_dict[now_area]}_{grow_dict[now_grow]}'
        Described_all_label = all_label_encoder[Described_all_delabel]

        delabel = f'{now_crop}_{now_disease}_{now_risk}'
        label = dataset_label_encoder[delabel]
        all_delabel = f'{now_crop}_{now_disease}_{now_risk}_{now_area}_{now_grow}'
        all_label = dataset_all_label_encoder[all_delabel]

        data['name'].append(now_name)
        data['path'].append(now_path)

        data['Described_label'].append(Described_label)
        data['Described_delabel'].append(Described_delabel)
        data['Described_all_label'].append(Described_all_label)
        data['Described_all_delabel'].append(Described_all_delabel)

        data['label'].append(label)
        data['delabel'].append(delabel)
        data['all_label'].append(all_label)
        data['all_delabel'].append(all_delabel)

    df = pd.DataFrame(data)
    df.to_csv(osp.join('/content','train.csv'))