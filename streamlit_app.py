from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn
import streamlit as st
from PIL import Image, ImageOps

from preprocessing import FeatureConfig, extract_processed_features, resize_rgb


APP_DIR = Path(__file__).resolve().parent
MODEL_PATH = APP_DIR / "models" / "grabcut_cielab_kmeans_svm.joblib"

st.set_page_config(
    page_title="Banana Ripeness Classifier",
    page_icon="🍌",
    layout="wide",
)


@st.cache_resource
def load_model_bundle(path: Path):
    return joblib.load(path)


def config_from_bundle(bundle: dict) -> FeatureConfig:
    return FeatureConfig(
        image_size=tuple(bundle.get("image_size", (96, 96))),
        k_candidates=tuple(bundle.get("k_candidates", (2, 3, 4))),
        pixel_sample=int(bundle.get("pixel_sample", 1200)),
        hist_bins=int(bundle.get("hist_bins", 12)),
        grabcut_iterations=int(bundle.get("grabcut_iterations", 5)),
        grabcut_margin_ratio=float(bundle.get("grabcut_margin_ratio", 0.02)),
        seed=int(bundle.get("seed", 42)),
    )


def model_classes(model) -> np.ndarray:
    if hasattr(model, "classes_"):
        return np.asarray(model.classes_)
    if hasattr(model, "named_steps") and "svc" in model.named_steps:
        return np.asarray(model.named_steps["svc"].classes_)
    return np.asarray([])


def expected_feature_count(model):
    if hasattr(model, "n_features_in_"):
        return int(model.n_features_in_)
    if hasattr(model, "named_steps"):
        for step in model.named_steps.values():
            if hasattr(step, "n_features_in_"):
                return int(step.n_features_in_)
    return None


st.title("🍌 Banana Ripeness Classifier")
st.caption("GrabCut foreground extraction + adaptive CIELAB K-means + SVM")

if not MODEL_PATH.exists():
    st.error("The trained SVM model file is missing.")
    st.code(
        "models/grabcut_cielab_kmeans_svm.joblib",
        language=None,
    )
    st.write(
        "Download this file from the Colab training output, place it at the "
        "path shown above, commit it to GitHub, and redeploy the app."
    )
    st.stop()

try:
    bundle = load_model_bundle(MODEL_PATH)
except Exception as error:
    st.error("The model could not be loaded.")
    st.exception(error)
    st.info(
        "A common cause is a different scikit-learn version between Colab "
        "and Streamlit. Pin the Colab version in requirements.txt."
    )
    st.stop()

if not isinstance(bundle, dict) or "model" not in bundle:
    st.error("The joblib file does not contain the expected model bundle.")
    st.stop()

model = bundle["model"]
config = config_from_bundle(bundle)
classes = model_classes(model)
trained_sklearn_version = bundle.get("sklearn_version")

if (
    trained_sklearn_version
    and trained_sklearn_version != sklearn.__version__
):
    st.warning(
        "This model was trained with scikit-learn "
        f"{trained_sklearn_version}, but the app is running "
        f"{sklearn.__version__}. Pin the training version in "
        "requirements.txt if loading or prediction fails."
    )

with st.sidebar:
    st.header("Model information")
    st.write("Input size:", f"{config.image_size[0]}×{config.image_size[1]}")
    st.write("Candidate K values:", ", ".join(map(str, config.k_candidates)))
    if len(classes):
        st.write("Classes:", ", ".join(str(value) for value in classes))
    st.divider()
    st.caption(
        "This classifier is an experimental model and should not be used as "
        "the sole basis for food-safety decisions."
    )

uploaded_file = st.file_uploader(
    "Upload one banana image",
    type=["jpg", "jpeg", "png", "webp", "bmp"],
)

if uploaded_file is None:
    st.info("Upload an image to begin.")
    st.stop()

try:
    pil_image = ImageOps.exif_transpose(Image.open(uploaded_file)).convert("RGB")
    original_rgb = np.asarray(pil_image, dtype=np.uint8)
    resized_rgb = resize_rgb(original_rgb, config)
except Exception as error:
    st.error("The uploaded file could not be read as an image.")
    st.exception(error)
    st.stop()

with st.spinner("Extracting the foreground and segmenting CIELAB colors..."):
    try:
        feature_vector, preview = extract_processed_features(resized_rgb, config)
    except Exception as error:
        st.error("Image preprocessing failed.")
        st.exception(error)
        st.stop()

expected_count = expected_feature_count(model)
if expected_count is not None and len(feature_vector) != expected_count:
    st.error(
        "The app generated "
        f"{len(feature_vector)} features, but the saved SVM expects "
        f"{expected_count}. The app and model were produced from different "
        "preprocessing configurations."
    )
    st.stop()

original_column, foreground_column, segmentation_column = st.columns(3)
with original_column:
    st.image(resized_rgb, caption="Resized RGB", width=300)
with foreground_column:
    st.image(
        preview["foreground"],
        caption=(
            "GrabCut foreground "
            f"({preview['foreground_fraction']:.0%} of pixels kept)"
        ),
        width=300,
    )
with segmentation_column:
    st.image(
        preview["segmented"],
        caption=f"Adaptive CIELAB K-means (K={preview['chosen_k']})",
        width=300,
    )

if preview["foreground_fraction"] < 0.10:
    st.warning(
        "GrabCut retained less than 10% of the image. Check the foreground "
        "preview carefully because the prediction may be unreliable."
    )

if st.button("Predict ripeness", type="primary", use_container_width=True):
    try:
        prediction = model.predict(feature_vector.reshape(1, -1))[0]
    except Exception as error:
        st.error("SVM prediction failed.")
        st.exception(error)
        st.stop()

    st.success(f"Predicted ripeness: **{str(prediction).replace('_', ' ').title()}**")

    if hasattr(model, "decision_function") and len(classes):
        decision = np.asarray(
            model.decision_function(feature_vector.reshape(1, -1))
        ).reshape(-1)
        if len(decision) == len(classes):
            st.subheader("Relative SVM decision scores")
            score_data = {
                str(label).replace("_", " ").title(): float(score)
                for label, score in zip(classes, decision)
            }
            score_frame = pd.DataFrame.from_dict(
                score_data,
                orient="index",
                columns=["decision score"],
            )
            st.bar_chart(score_frame)
            st.caption(
                "These are relative SVM decision scores, not calibrated "
                "probabilities."
            )
