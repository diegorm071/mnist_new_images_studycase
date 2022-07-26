import streamlit as st
import numpy as np
import cv2 as cv
import os
from matplotlib import pyplot as plt
from image_processing import *
import tensorflow as tf
import wandb
import config as config
from tensorflow import keras


@st.cache(suppress_st_warning=True)
def load_model():
    os.system(f"wandb login --relogin {config.WandbAcess().WANDB_URL}")
    best_model = wandb.restore(
        "model-best.h5", run_path="diego25rm/mnist_deep_classification/5u5r18i0"
    )

    return best_model.name


def use_model():
    img = cv.imread("app/digitos.jpg")
    model = keras.models.load_model(st.session_state["best_model"])
    img_gray = cv.cvtColor(img.copy(), cv.COLOR_BGR2GRAY)

    st.session_state.original_image = img_gray

    st.session_state.moment_image = img_gray.copy()
    st.session_state.stage = "boxes_contours"
    if st.session_state.stage == "initial":
        img = cv.imread("digitos.jpg")
        st.session_state.original_image = img
        img_gray = cv.cvtColor(img.copy(), cv.COLOR_BGR2GRAY)
        st.session_state.moment_image = img_gray.copy()
        st.image(img_gray)

    if st.session_state.stage == "blur":
        mask_size = st.sidebar.slider(
            label="Gaussian mask size", min_value=5, max_value=71
        )
        st.session_state.moment_image = blur_image(
            st.session_state.moment_image, mask_size=mask_size
        )
        st.image(st.session_state.moment_image)
    if st.session_state.stage == "canny":
        multiplier = st.sidebar.slider(
            label="Canny multiplier for threshold", min_value=2, max_value=7
        )
        st.session_state.moment_image = edges_canny(
            st.session_state.moment_image,
            lower_threshold=100,
            multiplier=multiplier,
            aperture_size=3,
        )
        st.image(st.session_state.moment_image)

    if st.session_state.stage == "dilate":
        kernel_size = st.sidebar.slider(
            label="Kernel size for dilate", min_value=5, max_value=8
        )
        interation = st.sidebar.slider(
            label="max iterations for dilate function", min_value=1, max_value=8
        )
        st.session_state.moment_image = dilate_image(
            st.session_state.moment_image,
            kernel_size=kernel_size,
            iterations=interation,
        )
        st.image(st.session_state.moment_image)

    if st.session_state.stage == "boxes_contours":
        mask_size = st.sidebar.slider(
            label="Gaussian mask size", min_value=5, max_value=71
        )
        st.session_state.moment_image = blur_image(
            st.session_state.moment_image, mask_size=mask_size
        )
        multiplier = st.sidebar.slider(
            label="Canny multiplier for threshold", min_value=2, max_value=7
        )
        aperture = st.sidebar.slider(
            label="Canny aperture size", min_value=3, max_value=7, step=2
        )
        st.session_state.moment_image = edges_canny(
            st.session_state.moment_image,
            lower_threshold=100,
            multiplier=multiplier,
            aperture_size=aperture,
        )
        kernel_size = st.sidebar.slider(
            label="Kernel size for dilate", min_value=5, max_value=8
        )
        interation = st.sidebar.slider(
            label="max iterations for dilate function", min_value=1, max_value=5
        )
        st.session_state.moment_image = dilate_image(
            st.session_state.moment_image,
            kernel_size=kernel_size,
            iterations=interation,
        )
        contours_poly, boundRect, contours = find_contours_and_boxes(
            st.session_state.moment_image
        )
        st.session_state.bound_rect = boundRect
        boxes_img = draw_bounding_boxes_contours(
            st.session_state.original_image.copy(), contours_poly, boundRect, contours
        )
        st.image(boxes_img)
        button = st.button(label="extract roi")
        button_predict = st.button(label="predict")
        if button:
            rois = extract_roi_black_white(
                st.session_state.original_image.copy(), st.session_state.bound_rect
            )

            resize = resize_rois(rois)
            for roi in resize:
                st.image(roi)
        if button_predict:
            rois = extract_roi_black_white(
                st.session_state.original_image.copy(), st.session_state.bound_rect
            )
            resize = resize_rois(rois)
            #        try:
            resize2 = list(np.float_(resize) / 255)
            print(resize2.__format__)
            imgs = np.array([np.reshape(np.array(i), (28, 28)) for i in resize2])
            predctions = model.predict(imgs)
            for i in range(len(resize)):
                st.image(resize[i])
                st.write(np.argmax(predctions[i]))


model = None
st.session_state["best_model"] = load_model()
use_model()
