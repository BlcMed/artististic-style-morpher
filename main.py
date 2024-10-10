import streamlit as st
from src.image_generation import reconstruct_image_from_content_style
from src.image_operations import read_image, image_to_tensor, tensor_to_image
from src.model_utils import load_model
from config import config
from src.data_manager import get_files

model_config = config.get("model")
paths_config = config.get("paths")
streamlit_config = config.get("streamlit")
st.title(streamlit_config["title"])
st.write(streamlit_config["description"])

if "model" not in st.session_state:
    vgg_model, transform = load_model()
    st.session_state["model"] = vgg_model
    st.session_state["transform"] = transform

default_style_images = get_files(paths_config["style_references"])
col1, col2 = st.columns(2)

with col1:
    content_image_path = st.file_uploader(
        "Upload an image as content reference", type=["jpg"]
    )
    if content_image_path:
        content_image = read_image(content_image_path)
        st.image(content_image, use_column_width=True)

with col2:
    default = streamlit_config["default_style_messsage"]
    upload = streamlit_config["upload_style_message"]
    style_option = st.radio("Choose an option", [default, upload])
    if style_option == upload:
        style_image_path = st.file_uploader(
            "Upload an image as style reference", type=["jpg"]
        )
        if style_image_path:
            style_image = read_image(style_image_path)
            st.image(style_image, use_column_width=True)
    else:
        selected_style_image = st.selectbox(
            "Select a default style image", list(default_style_images.keys())
        )
        style_image_path = default_style_images[selected_style_image]
        style_image = read_image(style_image_path)
        st.image(style_image, use_column_width=True)


if st.button(streamlit_config["transfer_style_button"]):
    if content_image and style_image:
        model = st.session_state.model
        transform = st.session_state.transform

        content_image_t = image_to_tensor(content_image, transform=transform)
        style_image_t = image_to_tensor(style_image, transform=transform)
        with st.spinner("Generating the image ..."):
            new_image_t = reconstruct_image_from_content_style(
                content_image_t=content_image_t,
                style_image_t=style_image_t,
                content_layer=model_config["content_layer"],
                style_layers=model_config["style_layers"],
                model=model,
                transform=transform,
                generated_image_resolution=model_config["generated_image_resolution"],
                content_weight=model_config["content_weight"],
                num_iteration=model_config["num_iteration"],
                learning_rate=model_config["learning_rate"],
            )
            new_image = tensor_to_image(new_image_t)

        st.image(
            new_image,
            caption=streamlit_config["result_image_caption"],
            use_column_width=True,
        )
    else:
        st.warning("Please upload both a content image and a style image.")
