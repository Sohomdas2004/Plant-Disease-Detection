import streamlit as st
import numpy as np
from PIL import Image
from pathlib import Path
import tensorflow as tf

# Keep this file in the same folder as model.keras.
MODEL_PATH = Path(__file__).parent / "model.keras"

# IMPORTANT:
# These labels must match the order used when the model was trained.
# If you trained with ImageDataGenerator.flow_from_directory(), check:
# print(train_generator.class_indices)
# For alphabetically named folders Healthy, Powdery, Rust, this order is:
CLASS_NAMES = ["Healthy", "Powdery", "Rust"]

IMAGE_SIZE = (225, 225)  # Read from the uploaded model's input shape.

st.set_page_config(
    page_title="Leaf Disease Detection",
    page_icon="🌿",
    layout="centered",
)

@st.cache_resource
def load_model():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Model file not found at {MODEL_PATH}. Put model.keras beside app.py."
        )
    return tf.keras.models.load_model(MODEL_PATH, compile=False)

def preprocess_image(image: Image.Image) -> np.ndarray:
    """Resize and prepare the image for the model."""
    image = image.convert("RGB").resize(IMAGE_SIZE)
    image_array = np.asarray(image, dtype=np.float32)

    # This model's expected input is float32 image pixels.
    # Do not divide by 255 unless you also used /255 rescaling during training.
    image_array = np.expand_dims(image_array, axis=0)
    return image_array

st.title("🌿 Leaf Disease Detection")
st.write("Upload a leaf image in JPEG/JPG or PNG format to classify it as **Healthy**, **Powdery**, or **Rust**.")

uploaded_file = st.file_uploader(
    "Choose a leaf image",
    type=["jpg", "jpeg", "png"],
    help="For best results, use a clear image with the leaf in focus.",
)

if uploaded_file is not None:
    try:
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded leaf image", use_container_width=True)

        if st.button("Predict disease", type="primary"):
            with st.spinner("Analyzing the leaf image..."):
                model = load_model()
                input_array = preprocess_image(image)
                predictions = np.asarray(model.predict(input_array, verbose=0))[0]

            if len(predictions) != len(CLASS_NAMES):
                st.error(
                    f"The model returned {len(predictions)} outputs, but "
                    f"{len(CLASS_NAMES)} class labels are configured. "
                    "Update CLASS_NAMES in app.py to match your model's training classes."
                )
            else:
                predicted_index = int(np.argmax(predictions))
                predicted_class = CLASS_NAMES[predicted_index]
                confidence = float(predictions[predicted_index]) * 100

                st.subheader("Prediction")
                if predicted_class.lower() == "healthy":
                    st.success(f"🌱 {predicted_class}")
                else:
                    st.warning(f"🍃 {predicted_class}")

                st.metric("Confidence", f"{confidence:.2f}%")
                st.write("Class probabilities")
                probability_map = {
                    name: float(score) for name, score in zip(CLASS_NAMES, predictions)
                }
                for name, score in probability_map.items():
                    st.write(f"**{name}:** {score * 100:.2f}%")
                    st.progress(float(np.clip(score, 0.0, 1.0)))

                st.caption(
                    "This is a model prediction, not a professional plant-health diagnosis. "
                    "Accuracy depends on the training data and image quality."
                )
    except Exception as exc:
        st.error(f"Could not process the image or load the model: {exc}")
else:
    st.info("Upload a JPEG/JPG image to get started.")