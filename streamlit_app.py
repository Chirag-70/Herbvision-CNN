import json
from pathlib import Path

import numpy as np
import streamlit as st
import tensorflow as tf
from PIL import Image


# ============================================================
# CONFIGURATION
# ============================================================

BASE_DIR = Path(__file__).resolve().parent

MODEL_PATH = BASE_DIR / "herb_classifier.keras"
CLASS_NAMES_PATH = BASE_DIR / "class_names.json"

IMG_SIZE = (224, 224)


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="HerbVision AI",
    page_icon="🌿",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ============================================================
# LOAD MODEL
# ============================================================

@st.cache_resource
def load_model():

    if not MODEL_PATH.exists():
        return None

    return tf.keras.models.load_model(
        MODEL_PATH,
        compile=False
    )


# ============================================================
# LOAD CLASS NAMES
# ============================================================

@st.cache_data
def load_class_names():

    if not CLASS_NAMES_PATH.exists():
        return []

    with open(
        CLASS_NAMES_PATH,
        "r",
        encoding="utf-8"
    ) as file:

        return json.load(file)


# ============================================================
# IMAGE PREPROCESSING
# ============================================================

def preprocess_image(image):

    image = image.convert("RGB")

    image = image.resize(
        IMG_SIZE
    )

    image_array = np.asarray(
        image,
        dtype=np.float32
    )

    image_array = np.expand_dims(
        image_array,
        axis=0
    )

    return image_array


# ============================================================
# PREDICTION
# ============================================================

def predict_image(
    model,
    image
):

    processed_image = preprocess_image(
        image
    )

    predictions = model.predict(
        processed_image,
        verbose=0
    )[0]

    top_indices = np.argsort(
        predictions
    )[::-1][:5]

    return predictions, top_indices


# ============================================================
# LOAD RESOURCES
# ============================================================

model = load_model()

class_names = load_class_names()


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.title("🌿 HerbVision AI")

    st.caption(
        "Medicinal Plant Classifier"
    )

    st.divider()

    st.subheader("📊 Model Information")

    if class_names:

        st.metric(
            "Herb Classes",
            len(class_names)
        )

    st.write(
        "**Model:** Custom CNN"
    )

    st.write(
        "**Input:** 224 × 224 RGB"
    )

    st.write(
        "**Output:** Herb classification"
    )

    st.write(
        "**Format:** `.keras`"
    )

    st.divider()

    st.subheader("💡 Image Tips")

    st.write(
        """
        • Use a clear leaf image

        • Keep the leaf visible

        • Use good lighting

        • Avoid excessive blur

        • Avoid very dark images

        • Keep the plant centered
        """
    )

    st.divider()

    st.caption(
        "CNN based medicinal plant recognition"
    )


# ============================================================
# MAIN HEADER
# ============================================================

st.title("🌿 HerbVision AI")

st.write(
    "CNN based medicinal plant image classification"
)

st.divider()


# ============================================================
# MODEL CHECK
# ============================================================

if model is None:

    st.error(
        "❌ Model file 'herb_classifier.keras' was not found."
    )

    st.info(
        "Please place herb_classifier.keras in the same folder as streamlit_app.py."
    )

    st.stop()


# ============================================================
# CLASS NAMES CHECK
# ============================================================

if not class_names:

    st.error(
        "❌ class_names.json was not found."
    )

    st.info(
        "Please place class_names.json in the same folder as streamlit_app.py."
    )

    st.stop()


# ============================================================
# UPLOAD SECTION
# ============================================================

with st.container(border=True):

    st.subheader(
        "📷 Upload a Medicinal Plant Image"
    )

    st.write(
        "Upload a clear image of a leaf or medicinal plant. "
        "The trained CNN model will identify the most likely herb."
    )

    uploaded_file = st.file_uploader(
        "Choose an image",
        type=[
            "jpg",
            "jpeg",
            "png",
            "webp"
        ],
        help="Upload JPG, JPEG, PNG or WEBP image."
    )


# ============================================================
# PREDICTION
# ============================================================

if uploaded_file is not None:

    try:

        image = Image.open(
            uploaded_file
        ).convert("RGB")

    except Exception:

        st.error(
            "❌ Unable to read this image."
        )

        st.stop()


    st.write("")


    # ========================================================
    # IMAGE + ANALYSIS COLUMNS
    # ========================================================

    left, right = st.columns(
        [1, 1],
        gap="large"
    )


    # ========================================================
    # LEFT COLUMN
    # ========================================================

    with left:

        with st.container(border=True):

            st.subheader(
                "🖼️ Uploaded Image"
            )

            st.image(
                image,
                use_container_width=True
            )

            st.caption(
                f"File: {uploaded_file.name}"
            )


    # ========================================================
    # RIGHT COLUMN
    # ========================================================

    with right:

        with st.container(border=True):

            st.subheader(
                "🔍 AI Analysis"
            )

            st.write(
                "Click the button below to identify the medicinal plant."
            )

            predict_button = st.button(
                "🌿 Identify Herb",
                type="primary",
                use_container_width=True
            )


            if predict_button:

                with st.spinner(
                    "Analyzing plant image..."
                ):

                    predictions, top_indices = predict_image(
                        model,
                        image
                    )


                # =================================================
                # BEST PREDICTION
                # =================================================

                best_index = top_indices[0]

                predicted_class = class_names[
                    best_index
                ]

                confidence = (
                    float(predictions[best_index])
                    * 100
                )


                st.success(
                    "Prediction completed successfully."
                )


                st.metric(
                    label="🌿 Predicted Herb",
                    value=predicted_class,
                    delta=f"{confidence:.2f}% confidence"
                )


                st.progress(
                    float(predictions[best_index])
                )


                # =================================================
                # CONFIDENCE MESSAGE
                # =================================================

                if confidence >= 90:

                    st.success(
                        "🟢 High-confidence prediction"
                    )

                elif confidence >= 70:

                    st.info(
                        "🔵 Moderate-confidence prediction"
                    )

                elif confidence >= 50:

                    st.warning(
                        "🟡 Low-confidence prediction"
                    )

                else:

                    st.error(
                        "🔴 Very low-confidence prediction. "
                        "Try a clearer image."
                    )


    # ========================================================
    # TOP 5 RESULTS
    # ========================================================

    st.write("")

    with st.container(border=True):

        st.subheader(
            "📊 Top 5 Predictions"
        )

        for rank, index in enumerate(
            top_indices,
            start=1
        ):

            herb_name = class_names[
                index
            ]

            probability = float(
                predictions[index]
            )

            score = probability * 100


            col1, col2 = st.columns(
                [4, 1]
            )


            with col1:

                st.write(
                    f"**{rank}. {herb_name}**"
                )


            with col2:

                st.write(
                    f"**{score:.2f}%**"
                )


            st.progress(
                probability
            )


# ============================================================
# INITIAL STATE
# ============================================================

else:

    st.write("")

    with st.container(border=True):

        st.subheader(
            "🌱 Ready to Identify Your Herb"
        )

        st.write(
            "Upload a medicinal plant image above "
            "to start AI-based classification."
        )


# ============================================================
# FOOTER
# ============================================================

st.divider()

st.caption(
    "🌿 HerbVision AI • CNN-based Medicinal Plant Classification"
)