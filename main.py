import streamlit as st
from src.image_generation import reconstruct_image_from_content_style
from src.image_operations import read_image, image_to_tensor, tensor_to_image
from src.model_utils import load_model

st.title("Articstic Style Morpher")

if "model" not in st.session_state:
    vgg_model, transform = load_model()
    st.session_state["model"] = vgg_model
    st.session_state["transform"] = transform

col1, col2 = st.columns(2)

with col1:
    content_image_path = st.file_uploader(
        "Upload an image as content reference", type=["jpg"]
    )
    if content_image_path:
        content_image = read_image(content_image_path)
        st.image(content_image, use_column_width=True)

with col2:
    style_image_path = st.file_uploader(
        "Upload an image as style reference", type=["jpg"]
    )
    if style_image_path:
        style_image = read_image(style_image_path)
        st.image(style_image, use_column_width=True)

if st.button("transfer style", help="Apply the style to the content image"):
    if content_image and style_image:
        model = st.session_state.model
        transform = st.session_state.transform

        content_image_t = image_to_tensor(content_image, transform=transform)
        style_image_t = image_to_tensor(style_image, transform=transform)
        with st.spinner("Generating the image ..."):
            new_image_t = reconstruct_image_from_content_style(
                content_image_t=content_image_t,
                style_image_t=style_image_t,
                content_layer=0,
                style_layers=list(range(4)),
                model=model,
                transform=transform,
                content_weight=0.97,
                num_iteration=300,
                learning_rate=0.01,
            )
            new_image = tensor_to_image(new_image_t)

        st.image(new_image, caption="Result Image", use_column_width=True)
    else:
        st.warning("Please upload both a content image and a style image.")
