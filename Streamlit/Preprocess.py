import json
from glob import glob
import os.path as osp


def make_label_encoder_decoder(save_path):
    path_list = sorted(glob(osp.join(save_path, "*/*.json")))
    with open("./Streamlit/config.json") as f:
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
