"""Inference preprocessing for the GrabCut + adaptive CIELAB SVM.

This module intentionally mirrors the Colab training notebook. Do not change
feature order or preprocessing constants without retraining the SVM.
"""

from dataclasses import dataclass

import cv2
import numpy as np
from sklearn.cluster import MiniBatchKMeans
from sklearn.metrics import calinski_harabasz_score


@dataclass(frozen=True)
class FeatureConfig:
    image_size: tuple[int, int] = (96, 96)
    k_candidates: tuple[int, ...] = (2, 3, 4)
    pixel_sample: int = 1200
    hist_bins: int = 12
    grabcut_iterations: int = 5
    grabcut_margin_ratio: float = 0.02
    seed: int = 42


def resize_rgb(rgb: np.ndarray, config: FeatureConfig) -> np.ndarray:
    """Resize an RGB uint8 image exactly as in the training notebook."""
    if rgb.ndim != 3 or rgb.shape[2] != 3:
        raise ValueError("Expected an RGB image with shape (height, width, 3).")
    if rgb.dtype != np.uint8:
        rgb = np.clip(rgb, 0, 255).astype(np.uint8)

    interpolation = (
        cv2.INTER_AREA
        if rgb.shape[1] > config.image_size[0]
        or rgb.shape[0] > config.image_size[1]
        else cv2.INTER_CUBIC
    )
    return cv2.resize(rgb, config.image_size, interpolation=interpolation)


def grabcut_foreground_mask(
    rgb: np.ndarray,
    config: FeatureConfig,
) -> np.ndarray:
    """Return GrabCut foreground and probable-foreground pixels."""
    height, width = rgb.shape[:2]
    margin = max(
        1,
        int(round(min(height, width) * config.grabcut_margin_ratio)),
    )
    rectangle_width = width - 2 * margin
    rectangle_height = height - 2 * margin
    if rectangle_width <= 1 or rectangle_height <= 1:
        return np.ones((height, width), dtype=bool)

    rectangle = (margin, margin, rectangle_width, rectangle_height)
    grabcut_labels = np.zeros((height, width), dtype=np.uint8)
    background_model = np.zeros((1, 65), dtype=np.float64)
    foreground_model = np.zeros((1, 65), dtype=np.float64)
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)

    try:
        cv2.grabCut(
            bgr,
            grabcut_labels,
            rectangle,
            background_model,
            foreground_model,
            config.grabcut_iterations,
            cv2.GC_INIT_WITH_RECT,
        )
    except cv2.error:
        return np.ones((height, width), dtype=bool)

    foreground = np.isin(
        grabcut_labels,
        (cv2.GC_FGD, cv2.GC_PR_FGD),
    )
    if float(foreground.mean()) < 0.03:
        return np.ones((height, width), dtype=bool)
    return foreground


def _adaptive_lab_kmeans(
    lab: np.ndarray,
    foreground_mask: np.ndarray,
    config: FeatureConfig,
):
    all_pixels = lab.reshape(-1, 3).astype(np.float32)
    flat_foreground = foreground_mask.ravel()
    pixels = all_pixels[flat_foreground]
    if len(pixels) < max(config.k_candidates) * 2:
        flat_foreground = np.ones(len(all_pixels), dtype=bool)
        foreground_mask = flat_foreground.reshape(lab.shape[:2])
        pixels = all_pixels

    lab_scale = np.array([100.0, 128.0, 128.0], dtype=np.float32)
    scaled_pixels = pixels / lab_scale

    kmeans_seed = config.seed + 10_000
    rng = np.random.default_rng(kmeans_seed)
    sample_n = min(config.pixel_sample, len(scaled_pixels))
    sample_index = rng.choice(
        len(scaled_pixels),
        size=sample_n,
        replace=False,
    )
    sample = scaled_pixels[sample_index]

    best = None
    for k in config.k_candidates:
        model = MiniBatchKMeans(
            n_clusters=k,
            random_state=kmeans_seed,
            n_init=3,
            batch_size=512,
            max_iter=100,
        )
        sample_labels = model.fit_predict(sample)
        if len(np.unique(sample_labels)) < 2:
            score = -np.inf
        else:
            score = calinski_harabasz_score(sample, sample_labels)
        if best is None or score > best[0]:
            best = (score, model, k)

    _, best_model, best_k = best
    foreground_labels = best_model.predict(scaled_pixels)
    centers_lab = best_model.cluster_centers_ * lab_scale

    label_image = np.full(lab.shape[:2], -1, dtype=np.int32)
    label_image.ravel()[flat_foreground] = foreground_labels

    # Removed background is white in the visualization but excluded from all
    # numerical features below.
    quantized_lab = np.empty_like(lab, dtype=np.float32)
    quantized_lab[...] = np.array([100.0, 0.0, 0.0], dtype=np.float32)
    quantized_lab.reshape(-1, 3)[flat_foreground] = centers_lab[
        foreground_labels
    ]

    proportions = np.bincount(
        foreground_labels,
        minlength=best_k,
    ).astype(np.float32)
    proportions /= proportions.sum()

    cluster_stds = np.vstack(
        [
            pixels[foreground_labels == cluster_id].std(axis=0)
            if np.any(foreground_labels == cluster_id)
            else np.zeros(3, dtype=np.float32)
            for cluster_id in range(best_k)
        ]
    ).astype(np.float32)

    return (
        quantized_lab,
        label_image,
        centers_lab,
        cluster_stds,
        proportions,
        best_k,
        foreground_mask,
    )


def _normalized_hist(
    values: np.ndarray,
    bins: int,
    value_range: tuple[float, float],
) -> np.ndarray:
    counts, _ = np.histogram(values, bins=bins, range=value_range)
    counts = counts.astype(np.float32)
    return counts / max(counts.sum(), 1.0)


def _spatial_cluster_occupancy(
    label_image: np.ndarray,
    center_order: np.ndarray,
    chosen_k: int,
    maximum_k: int,
) -> np.ndarray:
    old_to_sorted = np.empty(chosen_k, dtype=np.int32)
    old_to_sorted[center_order] = np.arange(chosen_k)

    sorted_labels = np.full_like(label_image, -1)
    foreground = label_image >= 0
    sorted_labels[foreground] = old_to_sorted[label_image[foreground]]

    features = []
    for grid_size in (2, 4):
        row_tiles = np.array_split(sorted_labels, grid_size, axis=0)
        for row_tile in row_tiles:
            for tile in np.array_split(row_tile, grid_size, axis=1):
                foreground_labels = tile[tile >= 0]
                histogram = np.bincount(
                    foreground_labels,
                    minlength=maximum_k,
                ).astype(np.float32)
                histogram /= max(histogram.sum(), 1.0)
                features.append(histogram)
    return np.concatenate(features)


def extract_processed_features(
    resized_rgb: np.ndarray,
    config: FeatureConfig,
):
    """Return the SVM vector and images used by the Streamlit preview."""
    foreground_mask = grabcut_foreground_mask(resized_rgb, config)

    masked_rgb = np.zeros_like(resized_rgb)
    masked_rgb[foreground_mask] = resized_rgb[foreground_mask]
    masked_rgb_float = masked_rgb.astype(np.float32) / 255.0
    lab = cv2.cvtColor(masked_rgb_float, cv2.COLOR_RGB2LAB)

    (
        quantized_lab,
        label_image,
        centers,
        cluster_stds,
        proportions,
        chosen_k,
        foreground_mask,
    ) = _adaptive_lab_kmeans(lab, foreground_mask, config)

    lab_hist = np.concatenate(
        [
            _normalized_hist(
                quantized_lab[..., 0][foreground_mask],
                config.hist_bins,
                (0, 100),
            ),
            _normalized_hist(
                quantized_lab[..., 1][foreground_mask],
                config.hist_bins,
                (-128, 128),
            ),
            _normalized_hist(
                quantized_lab[..., 2][foreground_mask],
                config.hist_bins,
                (-128, 128),
            ),
        ]
    )

    order = np.argsort(centers[:, 0])
    lab_scale = np.array([100.0, 128.0, 128.0], dtype=np.float32)
    maximum_k = max(config.k_candidates)
    cluster_summary = np.zeros((maximum_k, 7), dtype=np.float32)
    cluster_summary[:chosen_k, :3] = centers[order] / lab_scale
    cluster_summary[:chosen_k, 3:6] = cluster_stds[order] / lab_scale
    cluster_summary[:chosen_k, 6] = proportions[order]

    spatial_occupancy = _spatial_cluster_occupancy(
        label_image,
        order,
        chosen_k,
        maximum_k,
    )

    feature_vector = np.concatenate(
        [
            lab_hist,
            cluster_summary.ravel(),
            spatial_occupancy,
            np.array([chosen_k / maximum_k], dtype=np.float32),
        ]
    ).astype(np.float32)

    segmented_rgb = cv2.cvtColor(
        quantized_lab.astype(np.float32),
        cv2.COLOR_LAB2RGB,
    )
    segmented_rgb = np.clip(
        segmented_rgb * 255.0,
        0,
        255,
    ).astype(np.uint8)

    foreground_rgb = resized_rgb.copy()
    foreground_rgb[~foreground_mask] = 255

    return feature_vector, {
        "foreground": foreground_rgb,
        "segmented": segmented_rgb,
        "foreground_fraction": float(foreground_mask.mean()),
        "chosen_k": int(chosen_k),
    }
