import torch
import numpy as np
from PIL import Image
import cv2
from skimage import transform, filters, color, measure, feature
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from scipy.stats import skew, kurtosis
import warnings

# Suppress specific scikit-image warnings that might occur with certain image types
warnings.filterwarnings('ignore', category=UserWarning, module='skimage')

def _extract_color_features(image_rgb: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """
    Extracts color features (mean, std, skewness, kurtosis) from RGB, HSV, and L*a*b* channels
    of the masked region of an image.

    Parameters
    ----------
    image_rgb : np.ndarray
        The original RGB image (H, W, 3).
    mask : np.ndarray
        A binary mask (H, W) indicating the region of interest (e.g., the apple).

    Returns
    -------
    np.ndarray
        A 1D array of concatenated color features.
    """
    features = []
    masked_image_rgb = image_rgb[mask]

    if masked_image_rgb.size == 0:
        # Handle cases where the mask is empty
        return np.zeros(36) # 9 channels * 4 stats = 36 features

    # Convert to other color spaces
    image_hsv = color.rgb2hsv(image_rgb)
    image_lab = color.rgb2lab(image_rgb)

    masked_image_hsv = image_hsv[mask]
    masked_image_lab = image_lab[mask]

    # Define channels to process
    channels_rgb = [masked_image_rgb[:, i] for i in range(3)] # R, G, B
    channels_hsv = [masked_image_hsv[:, i] for i in range(3)] # H, S, V
    channels_lab = [masked_image_lab[:, i] for i in range(3)] # L, a, b

    all_channels = channels_rgb + channels_hsv + channels_lab

    for channel_data in all_channels:
        # Ensure channel_data is not empty to avoid NaN/inf
        if channel_data.size == 0:
            features.extend([0.0, 0.0, 0.0, 0.0])
            continue

        # Calculate mean, standard deviation, skewness, and kurtosis
        features.append(np.mean(channel_data))
        features.append(np.std(channel_data))
        features.append(skew(channel_data))
        features.append(kurtosis(channel_data))

    return np.array(features)

def _extract_texture_features(image_rgb: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """
    Extracts texture features (GLCM properties) from the grayscale version
    of the masked region of an image.

    Parameters
    ----------
    image_rgb : np.ndarray
        The original RGB image (H, W, 3).
    mask : np.ndarray
        A binary mask (H, W) indicating the region of interest (e.g., the apple).

    Returns
    -------
    np.ndarray
        A 1D array of concatenated texture features (6 features).
    """
    # Convert to grayscale
    image_gray = color.rgb2gray(image_rgb)
    masked_gray = image_gray[mask]

    if masked_gray.size == 0:
        return np.zeros(6) # 6 GLCM properties

    # Create a sub-image for GLCM calculation
    min_row, min_col, max_row, max_col = _get_bounding_box(mask)
    if min_row is None: # Mask is empty
        return np.zeros(6)

    sub_image_gray = image_gray[min_row:max_row+1, min_col:max_col+1]
    sub_mask = mask[min_row:max_row+1, min_col:max_col+1]

    # Apply mask to the sub-image, setting background to a constant value (e.g., 0)
    # This is important for GLCM to not calculate properties across apple-background boundaries
    # if the mask is not perfectly tight.
    # Alternatively, one could fill the background with the mean of the apple region.
    # For simplicity and common practice, we'll just use the masked region for GLCM.
    # However, skimage's GLCM expects a rectangular image.
    # A common approach is to compute GLCM on the bounding box and then potentially
    # filter results or ensure the mask is applied effectively.
    # For robust GLCM, it's often better to ensure the region is truly isolated.
    # Let's use the bounding box and fill non-masked areas within it with a neutral value.

    # Ensure the sub_image_gray is within 0-255 range and integer type for GLCM
    sub_image_gray_int = (sub_image_gray * 255).astype(np.uint8)
    sub_image_gray_int[~sub_mask] = 0 # Set non-apple pixels in bounding box to 0

    # GLCM parameters
    distances = [1]
    angles = [0, np.pi/4, np.pi/2, 3*np.pi/4]
    levels = 256 # Number of gray levels

    # Calculate GLCM
    # Ensure the image for GLCM is not entirely uniform or too small
    if sub_image_gray_int.shape[0] < 2 or sub_image_gray_int.shape[1] < 2 or np.all(sub_image_gray_int == sub_image_gray_int[0,0]):
        return np.zeros(6)

    try:
        glcm = feature.graycomatrix(sub_image_gray_int, distances=distances, angles=angles, levels=levels, symmetric=True, normed=True)
    except ValueError: # Handle cases where GLCM cannot be computed (e.g., image too small after masking)
        return np.zeros(6)

    # Extract properties and average over angles
    contrast = feature.graycoprops(glcm, 'contrast').mean()
    dissimilarity = feature.graycoprops(glcm, 'dissimilarity').mean()
    homogeneity = feature.graycoprops(glcm, 'homogeneity').mean()
    energy = feature.graycoprops(glcm, 'energy').mean()
    correlation = feature.graycoprops(glcm, 'correlation').mean()
    asm = feature.graycoprops(glcm, 'ASM').mean()

    return np.array([contrast, dissimilarity, homogeneity, energy, correlation, asm])

def _extract_shape_features(mask: np.ndarray) -> np.ndarray:
    """
    Extracts shape features from a binary mask.

    Parameters
    ----------
    mask : np.ndarray
        A binary mask (H, W) indicating the region of interest (e.g., the apple).

    Returns
    -------
    np.ndarray
        A 1D array of concatenated shape features (8 features).
    """
    features = []

    # Label connected regions in the mask
    labeled_mask = measure.label(mask)
    props = measure.regionprops(labeled_mask)

    if not props:
        # If no regions found, return zeros
        return np.zeros(8)

    # Assuming the largest region is the apple
    main_prop = max(props, key=lambda p: p.area)

    # Extract shape properties
    features.append(main_prop.area)
    features.append(main_prop.perimeter)
    features.append(main_prop.eccentricity)
    features.append(main_prop.major_axis_length)
    features.append(main_prop.minor_axis_length)
    features.append(main_prop.equivalent_diameter)
    features.append(main_prop.solidity)
    features.append(main_prop.extent)

    return np.array(features)

def _get_bounding_box(mask: np.ndarray):
    """
    Calculates the bounding box coordinates for a given mask.

    Parameters
    ----------
    mask : np.ndarray
        A binary mask (H, W).

    Returns
    -------
    tuple
        (min_row, min_col, max_row, max_col) or (None, None, None, None) if mask is empty.
    """
    rows = np.any(mask, axis=1)
    cols = np.any(mask, axis=0)
    min_row, max_row = np.where(rows)[0][[0, -1]] if np.any(rows) else (None, None)
    min_col, max_col = np.where(cols)[0][[0, -1]] if np.any(cols) else (None, None)
    return min_row, min_col, max_row, max_col

def _segment_apple(image_rgb: np.ndarray, n_clusters: int = 3) -> np.ndarray:
    """
    Segments the apple from the background using K-means clustering.

    Parameters
    ----------
    image_rgb : np.ndarray
        The input RGB image (H, W, 3).
    n_clusters : int, optional
        Number of clusters for K-means. Typically 2 (apple, background) or 3 (apple, background, defect).
        The default is 3.

    Returns
    -------
    np.ndarray
        A binary mask (H, W) where True indicates the apple region.
    """
    # Reshape the image to a 2D array of pixels and 3 color channels
    pixels = image_rgb.reshape(-1, 3)

    # Perform K-means clustering
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    kmeans.fit(pixels)
    labels = kmeans.labels_
    centers = kmeans.cluster_centers_

    # Identify the cluster corresponding to the apple
    # Heuristic: The apple cluster is often the largest non-dark cluster.
    # Or, the cluster whose centroid is furthest from black (0,0,0) and has a reasonable size.
    # Let's find the cluster with the largest area that is not predominantly dark.
    cluster_areas = [np.sum(labels == i) for i in range(n_clusters)]
    cluster_brightness = [np.mean(centers[i]) for i in range(n_clusters)] # Average brightness of cluster center

    apple_cluster_idx = -1
    max_apple_area = -1
    min_dark_threshold = 50 # A threshold to consider a cluster "dark" (out of 255)

    for i in range(n_clusters):
        if cluster_brightness[i] > min_dark_threshold and cluster_areas[i] > max_apple_area:
            max_apple_area = cluster_areas[i]
            apple_cluster_idx = i

    if apple_cluster_idx == -1:
        # Fallback: if all clusters are dark or no clear apple, pick the largest cluster
        apple_cluster_idx = np.argmax(cluster_areas)

    # Create the binary mask
    mask = (labels == apple_cluster_idx).reshape(image_rgb.shape[:2])

    # Post-processing: morphological operations to clean up the mask
    kernel = np.ones((5,5), np.uint8)
    mask = cv2.morphologyEx(mask.astype(np.uint8), cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    return mask.astype(bool)


def identify_apple_disease_and_sort(
    image_input: [str, Image.Image, np.ndarray],
    svm_model: SVC,
    scaler: StandardScaler,
    image_size: tuple = (256, 256),
    kmeans_n_clusters: int = 3,
    disease_labels: dict = None,
    sorting_categories: dict = None
) -> dict:
    """
    Targets: smartthings-iot-hub (function)

    Identifies diseases in apple fruit and suggests a sorting category based on
    image processing, feature extraction, and a pre-trained Support Vector Machine (SVM) model.

    Source Paper:
    "Using Machine Learning to Identify Diseases and Perform Sorting in Apple Fruit"
    URL: https://www.semanticscholar.org/paper/0d2c38b39f73003ec23ee3e2382e319e20bd5d18

    Mathematical Idea:
    The core idea involves a multi-stage pipeline:
    1.  **Image Preprocessing**: The input image is resized and denoised using a median filter.
        Then, K-means clustering is applied to segment the apple fruit from its background,
        creating a binary mask of the apple region.
    2.  **Feature Extraction**: From the segmented apple region, a comprehensive set of features
        is extracted:
        *   **Color Features**: Mean, standard deviation, skewness, and kurtosis are calculated
            for each channel in the RGB, HSV, and L*a*b* color spaces. This yields 9 channels * 4 statistics = 36 features.
        *   **Texture Features**: Gray-Level Co-occurrence Matrix (GLCM) properties, specifically
            Contrast, Dissimilarity, Homogeneity, Energy, Correlation, and Angular Second Moment (ASM),
            are extracted from the grayscale version of the segmented apple. These are typically
            averaged over multiple angles, resulting in 6 features.
        *   **Shape Features**: Geometric properties of the apple's silhouette (derived from the mask)
            such as Area, Perimeter, Eccentricity, Major Axis Length, Minor Axis Length, Equivalent Diameter,
            Solidity, and Extent are calculated. This yields 8 features.
        These 50 features are concatenated into a single feature vector.
    3.  **Feature Scaling**: The raw feature vector is scaled using a pre-trained StandardScaler
        to normalize the feature ranges, which is crucial for SVM performance.
    4.  **Classification**: The scaled feature vector is fed into a pre-trained Support Vector Machine (SVM)
        classifier, which predicts the disease class of the apple. The predicted class is then mapped
        to a human-readable disease name and a corresponding sorting category.

    Key Hyperparameters and Default Values:
    -   `image_size`: Target dimensions for image resizing. Default: `(256, 256)`.
    -   `kmeans_n_clusters`: Number of clusters for K-means segmentation. Default: `3`.
    -   `disease_labels`: A dictionary mapping numerical SVM output to disease names.
        Default: `{0: "Healthy", 1: "Apple Scab", 2: "Black Rot", 3: "Cedar Apple Rust"}`.
    -   `sorting_categories`: A dictionary mapping disease names to sorting actions.
        Default: `{"Healthy": "Grade A", "Apple Scab": "Grade B", "Black Rot": "Discard", "Cedar Apple Rust": "Grade B"}`.

    Parameters
    ----------
    image_input : str or PIL.Image.Image or np.ndarray
        The input apple image. Can be a file path (str), a PIL Image object,
        or a NumPy array (H, W, 3) in RGB format.
    svm_model : sklearn.svm.SVC
        A pre-trained Support Vector Machine classifier model.
    scaler : sklearn.preprocessing.StandardScaler
        A pre-trained StandardScaler used to normalize feature vectors.
    image_size : tuple, optional
        The target size (height, width) for resizing the input image. Default is (256, 256).
    kmeans_n_clusters : int, optional
        The number of clusters to use for K-means segmentation. Default is 3.
    disease_labels : dict, optional
        A dictionary mapping numerical class labels (from SVM) to human-readable
        disease names. If None, a default mapping is used.
    sorting_categories : dict, optional
        A dictionary mapping disease names to sorting categories. If None, a
        default mapping is used.

    Returns
    -------
    dict
        A dictionary containing the predicted disease and the recommended sorting category.
        Example: `{"disease": "Healthy", "sorting_category": "Grade A"}`
    """
    # Default mappings for disease labels and sorting categories
    if disease_labels is None:
        disease_labels = {
            0: "Healthy",
            1: "Apple Scab",
            2: "Black Rot",
            3: "Cedar Apple Rust"
        }
    if sorting_categories is None:
        sorting_categories = {
            "Healthy": "Grade A",
            "Apple Scab": "Grade B",
            "Black Rot": "Discard",
            "Cedar Apple Rust": "Grade B"
        }

    # 1. Image Preprocessing
    # Load image
    if isinstance(image_input, str):
        image_pil = Image.open(image_input).convert("RGB")
    elif isinstance(image_input, Image.Image):
        image_pil = image_input.convert("RGB")
    elif isinstance(image_input, np.ndarray):
        if image_input.ndim == 2: # Grayscale
            image_pil = Image.fromarray(image_input).convert("RGB")
        else: # Assume RGB
            image_pil = Image.fromarray(image_input)
    else:
        raise ValueError("image_input must be a file path (str), PIL Image, or NumPy array.")

    # Convert to NumPy array for processing
    image_rgb = np.array(image_pil)

    # Resize image
    # skimage.transform.resize expects float input [0,1] or uint8/uint16
    image_resized = transform.resize(image_rgb, image_size, anti_aliasing=True)
    image_resized_uint8 = (image_resized * 255).astype(np.uint8)

    # Apply median filter for noise reduction
    # skimage.filters.median expects uint8/uint16
    image_denoised = filters.median(image_resized_uint8, behavior='ndimage')

    # Segment apple using K-means
    apple_mask = _segment_apple(image_denoised, n_clusters=kmeans_n_clusters)

    # Check if the mask is empty
    if not np.any(apple_mask):
        # If no apple is detected, classify as 'Unknown' or 'No Apple'
        # and assign a default sorting category.
        predicted_disease = "No Apple Detected"
        predicted_sorting = "Unsortable"
        return {"disease": predicted_disease, "sorting_category": predicted_sorting}

    # 2. Feature Extraction
    # Extract color features
    color_features = _extract_color_features(image_denoised, apple_mask)

    # Extract texture features
    texture_features = _extract_texture_features(image_denoised, apple_mask)

    # Extract shape features
    shape_features = _extract_shape_features(apple_mask)

    # Concatenate all features into a single vector
    feature_vector = np.concatenate([color_features, texture_features, shape_features