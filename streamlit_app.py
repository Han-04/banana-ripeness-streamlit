from io import BytesIO
from pathlib import Path

import cv2
import joblib
import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image, ImageOps
from scipy.stats import skew
from skimage.feature import graycomatrix, graycoprops, local_binary_pattern


RANDOM_STATE = 42
DEFAULT_MODEL_PATH = Path(__file__).with_name("banana_ripeness_color_svm.joblib")
DISPLAY_NAMES = {
    "unripe": "Unripe",
    "ripe": "Ripe",
    "overripe": "Overripe",
    "rotten": "Rotten",
}


def resize_with_padding(rgb, side):
    h, w = rgb.shape[:2]
    scale = side / max(h, w)
    nw, nh = max(1, round(w * scale)), max(1, round(h * scale))
    interpolation = cv2.INTER_AREA if scale < 1 else cv2.INTER_CUBIC
    resized = cv2.resize(rgb, (nw, nh), interpolation=interpolation)
    border = np.concatenate(
        [resized[0], resized[-1], resized[:, 0], resized[:, -1]], axis=0
    )
    pad_color = tuple(np.median(border, axis=0).astype(np.uint8).tolist())
    top, bottom = (side - nh) // 2, side - nh - (side - nh) // 2
    left, right = (side - nw) // 2, side - nw - (side - nw) // 2
    return cv2.copyMakeBorder(
        resized, top, bottom, left, right, cv2.BORDER_CONSTANT, value=pad_color
    )


def preprocess(rgb, side, config):
    rgb = resize_with_padding(rgb, side)
    if config.get("bilateral_enabled", True):
        rgb = cv2.bilateralFilter(
            rgb,
            d=int(config.get("bilateral_d", 5)),
            sigmaColor=float(config.get("bilateral_sigma_color", 25)),
            sigmaSpace=float(config.get("bilateral_sigma_space", 25)),
        )
    lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB)
    lightness, a_channel, b_channel = cv2.split(lab)
    tile = tuple(config.get("clahe_tile_grid", (8, 8)))
    lightness = cv2.createCLAHE(
        clipLimit=float(config.get("clahe_clip_limit", 1.5)), tileGridSize=tile
    ).apply(lightness)
    return cv2.cvtColor(
        cv2.merge([lightness, a_channel, b_channel]), cv2.COLOR_LAB2RGB
    )


def fill_holes(mask):
    inverted = cv2.bitwise_not(mask)
    flood = inverted.copy()
    padded = np.zeros((mask.shape[0] + 2, mask.shape[1] + 2), np.uint8)
    cv2.floodFill(flood, padded, (0, 0), 0)
    return cv2.bitwise_or(mask, flood)


def clean_mask(mask, max_components=6, reject_border=True):
    h, w = mask.shape
    kernel_size = max(3, int(round(min(h, w) * 0.02)) | 1)
    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (kernel_size, kernel_size)
    )
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    count, labels, stats, centroids = cv2.connectedComponentsWithStats(mask, 8)
    if count <= 1:
        return mask

    center_y, center_x = h / 2, w / 2
    candidates, interior = [], []
    for component_id in range(1, count):
        area = stats[component_id, cv2.CC_STAT_AREA]
        distance = np.hypot(
            (centroids[component_id][0] - center_x) / w,
            (centroids[component_id][1] - center_y) / h,
        )
        if area < 0.003 * h * w:
            continue
        score = area * np.exp(-2 * distance)
        candidates.append((score, component_id))
        component = labels == component_id
        touches_border = (
            component[0].any()
            or component[-1].any()
            or component[:, 0].any()
            or component[:, -1].any()
        )
        if not touches_border:
            interior.append((score, component_id))

    pool = interior if reject_border and interior else candidates
    keep = [component_id for _, component_id in sorted(pool, reverse=True)[:max_components]]
    if not keep:
        keep = [1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))]
    result = np.isin(labels, keep).astype(np.uint8) * 255
    return fill_holes(result)


def banana_color_seeds(rgb):
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB)
    hue, saturation, value = cv2.split(hsv)
    lightness = lab[..., 0]
    green = (hue >= 25) & (hue <= 95) & (saturation >= 45) & (value >= 30)
    yellow = (hue >= 14) & (hue <= 39) & (saturation >= 65) & (value >= 65)
    brown = (hue <= 24) & (saturation >= 55) & (value >= 20) & (value <= 190)
    dark_peel = (value <= 85) & (lightness <= 100)

    green_fraction, yellow_fraction = green.mean(), yellow.mean()
    if green_fraction >= 0.035 and green_fraction >= 0.75 * yellow_fraction:
        primary = green | yellow
    elif yellow_fraction >= 0.035:
        primary = yellow | green
    else:
        primary = brown | dark_peel

    radius = max(7, int(round(min(rgb.shape[:2]) * 0.055)) | 1)
    nearby = cv2.dilate(
        primary.astype(np.uint8) * 255,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (radius, radius)),
        iterations=2,
    ) > 0
    if green_fraction >= 0.035 or yellow_fraction >= 0.035:
        seed_boolean = primary | ((brown | dark_peel) & nearby)
    else:
        seed_boolean = primary
    seed = seed_boolean.astype(np.uint8) * 255
    kernel_size = max(3, int(round(min(rgb.shape[:2]) * 0.012)) | 1)
    return cv2.morphologyEx(
        seed,
        cv2.MORPH_OPEN,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size)),
    )


def lab_kmeans_raw_mask(rgb):
    h, w = rgb.shape[:2]
    segmentation_scale = min(1.0, 160 / max(h, w))
    sw, sh = max(32, round(w * segmentation_scale)), max(32, round(h * segmentation_scale))
    small = cv2.resize(rgb, (sw, sh), interpolation=cv2.INTER_AREA)
    color_seed = banana_color_seeds(small) > 0
    lab = cv2.cvtColor(small, cv2.COLOR_RGB2LAB).astype(np.float32)
    yy, xx = np.mgrid[0:sh, 0:sw].astype(np.float32)
    samples = np.column_stack(
        [
            lab[..., 0].ravel() / 255,
            (lab[..., 1].ravel() - 128) / 128,
            (lab[..., 2].ravel() - 128) / 128,
            0.18 * xx.ravel() / sw,
            0.18 * yy.ravel() / sh,
        ]
    ).astype(np.float32)
    criteria = (
        cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER,
        30,
        0.01,
    )
    cv2.setRNGSeed(RANDOM_STATE)
    _, cluster_labels, _ = cv2.kmeans(
        samples, 5, None, criteria, 3, cv2.KMEANS_PP_CENTERS
    )
    cluster_labels = cluster_labels.reshape(sh, sw)

    ring = max(2, round(min(sh, sw) * 0.08))
    border = np.zeros((sh, sw), bool)
    border[:ring] = border[-ring:] = True
    border[:, :ring] = border[:, -ring:] = True
    center = (
        ((xx - sw / 2) / (0.42 * sw)) ** 2
        + ((yy - sh / 2) / (0.42 * sh)) ** 2
    ) <= 1
    scores, border_shares = [], []
    for cluster_id in range(5):
        cluster = cluster_labels == cluster_id
        seed_overlap = color_seed[cluster].mean() if cluster.any() else 0
        border_share = (cluster_labels[border] == cluster_id).mean()
        center_share = (cluster_labels[center] == cluster_id).mean()
        scores.append(2.4 * seed_overlap + 0.35 * center_share - 1.25 * border_share)
        border_shares.append(border_share)
    ranked = np.argsort(scores)[::-1]
    selected = [
        cluster_id
        for cluster_id in ranked
        if scores[cluster_id] > 0.22 and border_shares[cluster_id] < 0.30
    ][:4]
    raw = (
        np.isin(cluster_labels, selected) if selected else color_seed
    ).astype(np.uint8) * 255
    support = cv2.dilate(
        color_seed.astype(np.uint8) * 255, np.ones((9, 9), np.uint8), iterations=2
    )
    raw = cv2.bitwise_and(raw, support)
    return cv2.resize(raw, (w, h), interpolation=cv2.INTER_NEAREST)


def clean_and_validate_mask(mask):
    cleaned = clean_mask(mask, max_components=6, reject_border=True)
    ratio = (cleaned > 0).mean()
    if not 0.02 <= ratio <= 0.90:
        h, w = cleaned.shape
        cleaned[:] = 0
        cleaned[int(0.06 * h) : int(0.94 * h), int(0.06 * w) : int(0.94 * w)] = 255
    return cleaned


def refine_mask_grabcut(rgb, initial):
    h, w = initial.shape
    color_seed = banana_color_seeds(rgb)
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    bright_neutral = (hsv[..., 1] < 55) & (hsv[..., 2] > 115)
    grabcut_mask = np.full((h, w), cv2.GC_PR_BGD, np.uint8)
    grabcut_mask[initial > 0] = cv2.GC_PR_FGD
    grabcut_mask[bright_neutral] = cv2.GC_BGD
    ring = max(2, round(min(h, w) * 0.02))
    grabcut_mask[:ring] = grabcut_mask[-ring:] = cv2.GC_BGD
    grabcut_mask[:, :ring] = grabcut_mask[:, -ring:] = cv2.GC_BGD
    eroded = cv2.erode(
        cv2.bitwise_and(initial, color_seed), np.ones((5, 5), np.uint8), iterations=1
    )
    if (eroded > 0).sum() > 20:
        grabcut_mask[eroded > 0] = cv2.GC_FGD
    try:
        background = np.zeros((1, 65), np.float64)
        foreground = np.zeros((1, 65), np.float64)
        cv2.grabCut(
            cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR),
            grabcut_mask,
            None,
            background,
            foreground,
            2,
            cv2.GC_INIT_WITH_MASK,
        )
        refined = np.isin(
            grabcut_mask, [cv2.GC_FGD, cv2.GC_PR_FGD]
        ).astype(np.uint8) * 255
        refined = clean_mask(refined, max_components=6, reject_border=True)
        if 0.02 <= (refined > 0).mean() <= 0.90:
            return refined
    except cv2.error:
        pass
    return initial


def segment_banana(rgb):
    raw = lab_kmeans_raw_mask(rgb)
    cleaned = clean_and_validate_mask(raw)
    return refine_mask_grabcut(rgb, cleaned)


def normalized_hist(channel, pixels, bins, value_range):
    histogram, _ = np.histogram(channel[pixels], bins=bins, range=value_range)
    histogram = histogram.astype(np.float64)
    return histogram / max(histogram.sum(), 1)


def extract_features(image, mask):
    foreground = mask > 0
    if foreground.sum() < 50:
        foreground[:] = True
    hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    features, names = [], []

    histogram_specs = [
        ("H", hsv[..., 0], 18, (0, 180)),
        ("S", hsv[..., 1], 12, (0, 256)),
        ("V", hsv[..., 2], 12, (0, 256)),
        ("Lab_a", lab[..., 1], 16, (0, 256)),
        ("Lab_b", lab[..., 2], 16, (0, 256)),
    ]
    for prefix, channel, bins, value_range in histogram_specs:
        values = normalized_hist(channel, foreground, bins, value_range)
        features.extend(values)
        names.extend([f"{prefix}_hist_{index}" for index in range(bins)])

    for space_name, array in [("HSV", hsv), ("LAB", lab)]:
        for channel_index in range(3):
            values = array[..., channel_index][foreground].astype(float)
            channel_skew = float(skew(values, bias=False)) if len(values) > 2 else 0.0
            features.extend([values.mean(), values.std(), np.nan_to_num(channel_skew)])
            names.extend(
                [
                    f"{space_name}{channel_index}_mean",
                    f"{space_name}{channel_index}_std",
                    f"{space_name}{channel_index}_skew",
                ]
            )

    hue, saturation, value = hsv[..., 0], hsv[..., 1], hsv[..., 2]
    valid_pixels = max(foreground.sum(), 1)
    cue_masks = {
        "green_ratio": foreground & (hue >= 35) & (hue <= 90) & (saturation > 45) & (value > 35),
        "yellow_ratio": foreground & (hue >= 18) & (hue < 35) & (saturation > 45) & (value > 65),
        "brown_ratio": foreground & (hue >= 3) & (hue < 20) & (saturation > 45) & (value >= 25) & (value < 200),
        "dark_ratio": foreground & (value < 70),
        "low_saturation_ratio": foreground & (saturation < 45),
    }
    features.extend([cue.sum() / valid_pixels for cue in cue_masks.values()])
    names.extend(cue_masks.keys())

    points, radius = 24, 3
    lbp = local_binary_pattern(gray, points, radius, method="uniform")
    lbp_histogram = normalized_hist(lbp, foreground, points + 2, (0, points + 2))
    features.extend(lbp_histogram)
    names.extend([f"LBP_{index}" for index in range(points + 2)])

    ys, xs = np.where(foreground)
    crop = gray[ys.min() : ys.max() + 1, xs.min() : xs.max() + 1].copy()
    crop_mask = foreground[ys.min() : ys.max() + 1, xs.min() : xs.max() + 1]
    crop[~crop_mask] = int(np.median(gray[foreground]))
    quantized = np.minimum(crop // 16, 15).astype(np.uint8)
    glcm = graycomatrix(
        quantized,
        distances=[1, 3],
        angles=[0, np.pi / 4, np.pi / 2, 3 * np.pi / 4],
        levels=16,
        symmetric=True,
        normed=True,
    )
    for property_name in ["contrast", "dissimilarity", "homogeneity", "energy", "correlation"]:
        values = graycoprops(glcm, property_name)
        features.extend([values.mean(), values.std()])
        names.extend([f"GLCM_{property_name}_mean", f"GLCM_{property_name}_std"])

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contour = max(contours, key=cv2.contourArea) if contours else None
    h, w = mask.shape
    if contour is None:
        shape_features = [1, 0, 1, 1, 1, 0]
    else:
        area = cv2.contourArea(contour)
        perimeter = cv2.arcLength(contour, True)
        _, _, box_width, box_height = cv2.boundingRect(contour)
        hull_area = cv2.contourArea(cv2.convexHull(contour))
        circularity = 4 * np.pi * area / max(perimeter**2, 1)
        shape_features = [
            foreground.mean(),
            perimeter / (2 * (h + w)),
            area / max(hull_area, 1),
            area / max(box_width * box_height, 1),
            box_width / max(box_height, 1),
            circularity,
        ]
    features.extend(shape_features)
    names.extend(
        ["mask_area_ratio", "perimeter_norm", "solidity", "extent", "bbox_aspect", "circularity"]
    )
    return np.asarray(features, dtype=np.float32), names


@st.cache_resource
def load_local_bundle(model_path):
    return joblib.load(model_path)


@st.cache_resource
def load_uploaded_bundle(model_bytes):
    return joblib.load(BytesIO(model_bytes))


def load_rgb(uploaded_image):
    with Image.open(uploaded_image) as image:
        return np.asarray(ImageOps.exif_transpose(image).convert("RGB"))


def validate_bundle(bundle):
    required = {"model", "target_side", "feature_names"}
    missing = required.difference(bundle)
    if missing:
        raise ValueError("Model bundle is missing: " + ", ".join(sorted(missing)))
    if not hasattr(bundle["model"], "predict_proba"):
        raise ValueError("The SVM was not trained with probability=True.")


st.set_page_config(
    page_title="Banana Ripeness Classifier", page_icon="🍌", layout="wide"
)
st.title("🍌 Banana Ripeness Classification")
st.caption("Adaptive LAB color segmentation + handcrafted features + SVM")

with st.sidebar:
    st.header("Model")
    uploaded_model = st.file_uploader(
        "Upload the trained model bundle",
        type=["joblib"],
        help="Only load a .joblib file that you created or trust.",
    )
    st.info(
        "If banana_ripeness_color_svm.joblib is beside this app, it is loaded automatically."
    )

try:
    if uploaded_model is not None:
        bundle = load_uploaded_bundle(uploaded_model.getvalue())
        model_source = "Uploaded model bundle"
    elif DEFAULT_MODEL_PATH.exists():
        bundle = load_local_bundle(str(DEFAULT_MODEL_PATH))
        model_source = DEFAULT_MODEL_PATH.name
    else:
        bundle = None
        model_source = None
except Exception as error:
    bundle = None
    st.error(f"The model bundle could not be loaded: {error}")

if bundle is not None:
    try:
        validate_bundle(bundle)
        st.sidebar.success(f"Loaded: {model_source}")
        if "test_metrics" in bundle:
            metrics = bundle["test_metrics"]
            st.sidebar.caption(
                "Saved test accuracy: "
                + f"{float(metrics.get('accuracy', float('nan'))):.1%}"
            )
    except Exception as error:
        st.error(str(error))
        bundle = None

uploaded_image = st.file_uploader(
    "Upload a banana image", type=["jpg", "jpeg", "png", "webp", "bmp"]
)

if uploaded_image is None:
    st.write("Upload an image to begin.")
elif bundle is None:
    st.warning(
        "Train and download banana_ripeness_color_svm.joblib from the Colab notebook, "
        "then upload it in the sidebar or place it beside this application."
    )
else:
    try:
        with st.spinner("Segmenting the banana and calculating ripeness…"):
            original = load_rgb(uploaded_image)
            side = int(bundle["target_side"])
            config = bundle.get(
                "preprocessing_config",
                {
                    "bilateral_enabled": True,
                    "bilateral_d": 5,
                    "bilateral_sigma_color": 25,
                    "bilateral_sigma_space": 25,
                    "clahe_clip_limit": 1.5,
                    "clahe_tile_grid": (8, 8),
                },
            )
            processed = preprocess(original, side, config)
            mask = segment_banana(processed)
            segmented = cv2.bitwise_and(processed, processed, mask=mask)
            feature_vector, feature_names = extract_features(processed, mask)

            expected_names = list(bundle["feature_names"])
            if feature_names != expected_names:
                raise ValueError(
                    "Feature definitions do not match the trained model. Use the app supplied with this notebook."
                )

            model = bundle["model"]
            vector_2d = feature_vector.reshape(1, -1)
            prediction = str(model.predict(vector_2d)[0])
            probabilities = model.predict_proba(vector_2d)[0]
            probability_series = pd.Series(
                probabilities, index=[str(label) for label in model.classes_], name="Probability"
            ).sort_values(ascending=False)
            confidence = float(probability_series.loc[prediction])

        prediction_name = DISPLAY_NAMES.get(prediction, prediction.title())
        metric_left, metric_right, metric_area = st.columns(3)
        metric_left.metric("Predicted ripeness", prediction_name)
        metric_right.metric("Model confidence", f"{confidence:.2%}")
        metric_area.metric("Segmented area", f"{(mask > 0).mean():.1%}")

        if confidence < 0.60:
            st.warning(
                "Low-confidence prediction. Try an image with one clearly visible banana, "
                "good lighting and less background clutter."
            )

        image_columns = st.columns(3)
        image_columns[0].image(original, caption="Uploaded image", use_container_width=True)
        image_columns[1].image(mask, caption="LAB color mask", clamp=True, use_container_width=True)
        image_columns[2].image(segmented, caption="Segmented banana ROI", use_container_width=True)

        st.subheader("Class probabilities")
        probability_frame = (
            probability_series.rename(index=lambda label: DISPLAY_NAMES.get(label, label.title()))
            .mul(100)
            .rename("Confidence (%)")
            .to_frame()
        )
        st.bar_chart(probability_frame)
        st.dataframe(
            probability_frame.style.format("{:.2f}%"), use_container_width=True
        )
        st.caption(
            "Confidence is the SVC probability for the predicted class. It expresses model "
            "certainty and does not guarantee that the prediction is correct."
        )
    except Exception as error:
        st.error(f"Prediction failed: {error}")
