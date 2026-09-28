import os
from pathlib import Path

import numpy as np
import streamlit as st
import tensorflow as tf
from PIL import Image


# ==============================
# SETTINGS
# ==============================

APP_DIR = Path(__file__).resolve().parent

DATA_DIR = APP_DIR / "Snake"
MODEL_PATH = APP_DIR / "snake_model.keras"

IMAGE_SIZE = (128, 128)
BATCH_SIZE = 32
EPOCHS = 10

CLASS_NAMES = [
    "Non Venomous",
    "Venomous"
]


# ==============================
# STREAMLIT PAGE
# ==============================

st.set_page_config(
    page_title="Snake AI Classifier",
    page_icon="🐍",
    layout="centered"
)

st.title("🐍 Snake AI Classifier")

st.write(
    "Upload a snake image to classify it as "
    "**Venomous** or **Non Venomous**."
)


# ==============================
# CHECK DATASET
# ==============================

non_venomous_dir = DATA_DIR / "Non Venomous"
venomous_dir = DATA_DIR / "Venomous"


if not DATA_DIR.exists():

    st.error(
        "❌ Snake dataset folder not found."
    )

    st.info(
        "Create a folder named 'Snake' beside app.py "
        "with 'Non Venomous' and 'Venomous' folders."
    )

    st.stop()


if not non_venomous_dir.exists() or not venomous_dir.exists():

    st.error(
        "❌ Dataset folders are incomplete."
    )

    st.write("Required structure:")

    st.code(
        """
Snake/
├── Non Venomous/
└── Venomous/
        """
    )

    st.stop()


# ==============================
# TRAIN MODEL
# ==============================

def train_model():

    st.info(
        "🧠 First run: Training Snake AI model..."
    )

    progress = st.progress(0)

    train_ds = tf.keras.utils.image_dataset_from_directory(
        DATA_DIR,
        labels="inferred",
        label_mode="categorical",
        class_names=CLASS_NAMES,
        image_size=IMAGE_SIZE,
        batch_size=BATCH_SIZE,
        validation_split=0.20,
        subset="training",
        seed=42
    )

    val_ds = tf.keras.utils.image_dataset_from_directory(
        DATA_DIR,
        labels="inferred",
        label_mode="categorical",
        class_names=CLASS_NAMES,
        image_size=IMAGE_SIZE,
        batch_size=BATCH_SIZE,
        validation_split=0.20,
        subset="validation",
        seed=42
    )

    train_ds = train_ds.prefetch(
        tf.data.AUTOTUNE
    )

    val_ds = val_ds.prefetch(
        tf.data.AUTOTUNE
    )


    model = tf.keras.Sequential([

        tf.keras.layers.Input(
            shape=(128, 128, 3)
        ),

        tf.keras.layers.Rescaling(
            1.0 / 255
        ),

        tf.keras.layers.RandomFlip(
            "horizontal"
        ),

        tf.keras.layers.RandomRotation(
            0.1
        ),

        tf.keras.layers.RandomZoom(
            0.1
        ),

        tf.keras.layers.Conv2D(
            32,
            (3, 3),
            activation="relu"
        ),

        tf.keras.layers.MaxPooling2D(),

        tf.keras.layers.Conv2D(
            64,
            (3, 3),
            activation="relu"
        ),

        tf.keras.layers.MaxPooling2D(),

        tf.keras.layers.Conv2D(
            128,
            (3, 3),
            activation="relu"
        ),

        tf.keras.layers.MaxPooling2D(),

        tf.keras.layers.Flatten(),

        tf.keras.layers.Dense(
            128,
            activation="relu"
        ),

        tf.keras.layers.Dropout(
            0.5
        ),

        tf.keras.layers.Dense(
            2,
            activation="softmax"
        )
    ])


    model.compile(
        optimizer="adam",
        loss="categorical_crossentropy",
        metrics=["accuracy"]
    )


    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=EPOCHS
    )


    model.save(
        MODEL_PATH
    )

    progress.progress(100)

    return model


# ==============================
# LOAD OR TRAIN MODEL
# ==============================

@st.cache_resource
def get_model():

    if MODEL_PATH.exists():

        try:
            return tf.keras.models.load_model(
                MODEL_PATH
            )

        except Exception:

            st.warning(
                "Existing model could not be loaded. "
                "Training a new model..."
            )

    return train_model()


model = get_model()


# ==============================
# UPLOAD IMAGE
# ==============================

uploaded_file = st.file_uploader(
    "Upload Snake Image",
    type=[
        "jpg",
        "jpeg",
        "png",
        "webp",
        "bmp"
    ]
)


# ==============================
# PREDICTION
# ==============================

if uploaded_file is not None:

    image = Image.open(
        uploaded_file
    ).convert("RGB")

    st.image(
        image,
        caption="Uploaded Snake Image",
        use_container_width=True
    )


    resized_image = image.resize(
        IMAGE_SIZE
    )


    image_array = np.asarray(
        resized_image,
        dtype=np.float32
    )


    image_array = np.expand_dims(
        image_array,
        axis=0
    )


    with st.spinner(
        "🔍 Analyzing snake..."
    ):

        prediction = model.predict(
            image_array,
            verbose=0
        )[0]


    predicted_index = int(
        np.argmax(prediction)
    )

    confidence = float(
        prediction[predicted_index]
    )

    predicted_class = CLASS_NAMES[
        predicted_index
    ]


    st.divider()

    st.subheader("Result")


    # ==============================
    # RESULT
    # ==============================

    if confidence < 0.70:

        st.warning(
            "⚠️ UNCERTAIN"
        )

        st.write(
            f"Confidence: **{confidence:.1%}**"
        )

        st.info(
            "The AI is not confident enough "
            "about this image."
        )


    elif predicted_class == "Venomous":

        st.error(
            "🔴 VENOMOUS"
        )

        st.metric(
            "Confidence",
            f"{confidence:.1%}"
        )

        st.warning(
            "⚠️ Do not touch, approach, or handle "
            "the snake. This AI prediction should "
            "not be used as a safety guarantee."
        )


    else:

        st.success(
            "🟢 NON VENOMOUS"
        )

        st.metric(
            "Confidence",
            f"{confidence:.1%}"
        )

        st.warning(
            "⚠️ Do not handle the snake based only "
            "on this AI prediction."
        )


    # ==============================
    # PROBABILITIES
    # ==============================

    st.divider()

    st.subheader(
        "Prediction Probabilities"
    )


    st.write(
        f"**Non Venomous:** "
        f"{prediction[0]:.1%}"
    )

    st.progress(
        float(prediction[0])
    )


    st.write(
        f"**Venomous:** "
        f"{prediction[1]:.1%}"
    )

    st.progress(
        float(prediction[1])
    )


st.divider()

st.caption(
    "Snake AI • CNN Image Classification"
)

st.caption(
    "AI predictions may be incorrect. "
    "Never rely on this system alone for snake identification or safety."
)