import torch
import streamlit as st
from model import CNN
from utils import transform_image
import yaml
from typing import Tuple
import json


@st.cache
def load_model() -> CNN:
    """This is return Pre-trained Model

    Returns:
        CNN: Pre-trained Model
    """
    # with open("config.yaml") as f:
    #     config = yaml.load(f, Loader=yaml.FullLoader)
    with open("./Streamlit/config.json") as f:
        config = json.load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = CNN(class_n=25, rate=0.).to(device)
    model.load_state_dict(torch.load(config["model_path"], map_location=device))

    return model


def get_prediction(model: CNN, image_bytes: bytes) -> Tuple[torch.Tensor, torch.Tensor]:
    transformed_Image = transform_image(image_bytes=image_bytes).cuda()
    logits = model.forward(transformed_Image)
    prediction = logits.argmax(1)
    return prediction
