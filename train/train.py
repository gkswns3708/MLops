import numpy as np
import pandas as pd
import cv2
import random
import os
import json

from tqdm import tqdm
from glob import glob

import utils

# temp
from pprint import pprint
from collections import Counter
from parse_config import config_parser

import data_loader.data_loaders as Custom_loader
import data_loader.transforms as Custom_transforms

import torch
import torch.nn as nn
from torchvision import models
from torch.utils.data import Dataset
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedKFold  # 구현 안할 듯 함. Inference 속도 Issue

SEED = 123
torch.manual_seed(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(SEED)


def main(config):
    # config_parser는 말 그대로 dict type이 맞음 []로 접근할 수 있음.
    def _main(config):
        # config = json.load(open('./config_train.json', 'r'))
        if config["is_Preprocessing"]:
            utils.raw_image_json = utils.get_image_json_path(config["Raw_Data_Path"])
            utils.get_Resized_Image_Dataset(config)
            utils.make_label_encoder_decoder(config)
            utils.Set_Dataset_CSV(config["Prepared_Data_Path"])

        # Training Process with Config Parser
        train_transform, default_transform = config.init_ftn(
            "transform_name", Custom_transforms
        )()
        # TODO config.json에 trainsform_name란을 만들면 된다(type과 args를 추가)
        Dataloader = config.init_obj(
            "Dataset",
            Custom_loader,
            train_transform=train_transform,
            default_transform=default_transform,
        )


# TODO :  Preprocess 절차를 생략할지 말지 정하는 config 와 argparser가 있으면 좋을 듯 하다.
if __name__ == "__main__":
    args = argparse.ArgumentParser(description="PyTorch Template")
    args.add_argument(
        "-c",
        "--config",
        default=None,
        type=str,
        help="config file path (default: None)",
    )
    args.add_argument(
        "-r",
        "--resume",
        default=None,
        type=str,
        help="path to latest checkpoint (default: None)",
    )
    args.add_argument(
        "-d",
        "--device",
        default=None,
        type=str,
        help="indices of GPUs to enable (default: all)",
    )

    # custom cli options to modify configuration from default values given in json file.
    CustomArgs = collections.namedtuple("CustomArgs", "flags type target")
    options = [
        CustomArgs(["--lr", "--learning_rate"], type=float, target="optimizer;args;lr"),
        CustomArgs(
            ["--bs", "--batch_size"], type=int, target="data_loader;args;batch_size"
        ),
    ]
    config = ConfigParser.from_args(args, options)

    # wandb.login()
    main(config)
