import streamlit as st
import io
import os
import yaml
import json
from PIL import Image
from predict import load_model, get_prediction
from confirm_button_hack import cache_on_button_press
from Preprocess import make_label_encoder_decoder

st.set_page_config(layout="wide")
root_password = "chj"


def main():
    st.title("Crop_Disease Classification Model")

    # with open("config.yaml") as f:
    #     config = yaml.load(f, Loader=yaml.FullLoader)
    with open("config.json") as f:
        config = json.load(f)

    model = load_model()
    model.eval()

    make_label_encoder_decoder(config["Prepared_Data_Path"])

    uploaded_file = st.file_uploader("Choose an image", type=["jpg", "jpeg", "png"])

    if uploaded_file:
        image_bytes = uploaded_file.getvalue()
        image = Image.open(io.BytesIO(image_bytes))

        st.image(image, caption="Uploaded Image")
        st.write("Classifying...")
        label = get_prediction(model, image_bytes).item()
        label = config["Annotation_info"]["label_decoder"][str(label)]

        st.write(f"label is {label}")


@cache_on_button_press("Authenticate")
def authenticate(password) -> bool:
    print(type(password))
    return password == root_password


password = st.text_input("password", type="password")

if authenticate(password):
    st.success("You are authenticated!")
    main()
else:
    st.error("The password is invalid.")
