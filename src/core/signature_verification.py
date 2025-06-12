import cv2
import numpy as np
from skimage import morphology
import matplotlib.pyplot as plt
import torch
from transformers import AutoModelForImageSegmentation
from torchvision import transforms
from PIL import Image

from scipy.stats import pearsonr
from scipy.spatial.distance import euclidean

from skimage.feature import graycomatrix, graycoprops

from .birefnet_processor import birefnet_processor

import cv2
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from core.birefnet_processor import birefnet_processor


def filter_boundary_points(boundary_coords, min_cluster_size=3):
    """Filter boundary points to remove isolated noise"""
    if len(boundary_coords) < min_cluster_size:
        return boundary_coords
    
    points = np.array(boundary_coords)
    
    filtered_points = []
    processed = set()
    
    for i, point in enumerate(points):
        if i in processed:
            continue
            
        distances = np.sqrt(np.sum((points - point) ** 2, axis=1))
        nearby_indices = np.where(distances <= 3)[0]  
        
        if len(nearby_indices) >= 3:
            for idx in nearby_indices:
                if idx not in processed:
                    filtered_points.append(tuple(points[idx]))
                    processed.add(idx)
    
    return filtered_points

def reconstruct_from_boundary_points(boundary_coords, bbox, canvas_shape):
    """Reconstruct image from filtered boundary points"""
    min_x, min_y, max_x, max_y = bbox
    h, w = canvas_shape

    reconstructed = np.zeros((h, w), dtype=np.uint8)
    if len(boundary_coords) == 0:
        return reconstructed

    for x, y in boundary_coords:
        local_x = x - min_x
        local_y = y - min_y
        if 0 <= local_x < w and 0 <= local_y < h:
            reconstructed[local_y, local_x] = 255

    return reconstructed


def center_resize(img, target_size=(512, 512)):
    """Resize and center the image on a fixed canvas"""
    h, w = img.shape
    scale = min(target_size[1] / h, target_size[0] / w)
    new_w, new_h = int(w * scale), int(h * scale)
    resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)

    canvas = np.zeros(target_size, dtype=np.uint8)
    x_offset = (target_size[0] - new_w) // 2
    y_offset = (target_size[1] - new_h) // 2
    canvas[y_offset:y_offset+new_h, x_offset:x_offset+new_w] = resized

    return canvas


def extract_signature_boundary_points(image_path, alpha_thresh=127, min_area=500, padding=10, debug=False):
    """Extract signature from image using BiRefNet and boundary reconstruction"""
    # Load image
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Could not load image from {image_path}")

    # Convert to PIL for BiRefNet
    pil_image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    processed_image, _ = birefnet_processor.remove_background(pil_image)

    # Get alpha channel from RGBA
    rgba_np = np.array(processed_image)
    if rgba_np.shape[2] != 4:
        raise ValueError("Expected RGBA image from BiRefNet")

    alpha_mask = rgba_np[:, :, 3]
    mask_binary = (alpha_mask > alpha_thresh).astype(np.uint8) * 255

    # Find valid contours
    contours, _ = cv2.findContours(mask_binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        print("Warning: No contours found in alpha mask.")
        return mask_binary, mask_binary, []

    # Filter or merge contours
    contours = [cnt for cnt in contours if cv2.contourArea(cnt) > min_area]
    if not contours:
        print("Warning: No significant contours found.")
        return mask_binary, mask_binary, []

    all_pts = np.vstack(contours)
    x, y, w, h = cv2.boundingRect(all_pts)

    # Apply padding and crop
    x = max(0, x - padding)
    y = max(0, y - padding)
    w = min(mask_binary.shape[1] - x, w + 2 * padding)
    h = min(mask_binary.shape[0] - y, h + 2 * padding)

    cropped_mask = mask_binary[y:y+h, x:x+w]
    cropped_image = cv2.cvtColor(rgba_np[y:y+h, x:x+w, :3], cv2.COLOR_RGB2GRAY)

    # Debug preview
    if debug:
        plt.figure(figsize=(12, 4))
        plt.subplot(1, 3, 1)
        plt.imshow(mask_binary, cmap='gray')
        plt.title("Alpha Mask")
        plt.axis("off")

        plt.subplot(1, 3, 2)
        plt.imshow(cropped_image, cmap='gray')
        plt.title("Cropped Image")
        plt.axis("off")

        plt.subplot(1, 3, 3)
        plt.imshow(cropped_mask, cmap='gray')
        plt.title("Cropped Mask")
        plt.axis("off")
        plt.tight_layout()
        plt.show()

    # Extract and reconstruct from boundary points
    boundary_points = np.column_stack(np.where(cropped_mask == 255))
    boundary_coords = list(zip(boundary_points[:, 1], boundary_points[:, 0]))

    boundary_coords = filter_boundary_points(boundary_coords, min_cluster_size=3)

    if boundary_coords:
        reconstructed = reconstruct_from_boundary_points(
            boundary_coords,
            bbox=(0, 0, cropped_mask.shape[1], cropped_mask.shape[0]),
            canvas_shape=cropped_mask.shape
        )
    else:
        reconstructed = cropped_mask

    # Optionally normalize output
    reconstructed_resized = center_resize(reconstructed, target_size=(512, 512))

    return cropped_image, reconstructed_resized, boundary_coords


def extract_enhanced_signature_features(signature, boundary_coords):
    """Extract features from both reconstructed signature and boundary points"""
    
    features = {}
    
    h, w = signature.shape
    features['aspect_ratio'] = w / h if h > 0 else 0
    
    total_pixels = h * w
    signature_pixels = np.sum(signature > 0)
    features['density'] = signature_pixels / total_pixels if total_pixels > 0 else 0
    
    features['num_boundary_points'] = len(boundary_coords)
    features['boundary_density'] = len(boundary_coords) / total_pixels if total_pixels > 0 else 0
    
    if signature_pixels > 0:
        y_coords, x_coords = np.where(signature > 0)
        features['centroid_x'] = np.mean(x_coords) / w if w > 0 else 0
        features['centroid_y'] = np.mean(y_coords) / h if h > 0 else 0
        features['std_x'] = np.std(x_coords) / w if w > 0 else 0
        features['std_y'] = np.std(y_coords) / h if h > 0 else 0
    else:
        features['centroid_x'] = features['centroid_y'] = 0.5
        features['std_x'] = features['std_y'] = 0
    
    if len(boundary_coords) > 0:
        boundary_array = np.array(boundary_coords)
        features['boundary_centroid_x'] = np.mean(boundary_array[:, 0]) / w if w > 0 else 0
        features['boundary_centroid_y'] = np.mean(boundary_array[:, 1]) / h if h > 0 else 0
        features['boundary_std_x'] = np.std(boundary_array[:, 0]) / w if w > 0 else 0
        features['boundary_std_y'] = np.std(boundary_array[:, 1]) / h if h > 0 else 0
        
        min_x, max_x = np.min(boundary_array[:, 0]), np.max(boundary_array[:, 0])
        min_y, max_y = np.min(boundary_array[:, 1]), np.max(boundary_array[:, 1])
        features['boundary_width'] = (max_x - min_x) / w if w > 0 else 0
        features['boundary_height'] = (max_y - min_y) / h if h > 0 else 0
    else:
        features['boundary_centroid_x'] = features['boundary_centroid_y'] = 0.5
        features['boundary_std_x'] = features['boundary_std_y'] = 0
        features['boundary_width'] = features['boundary_height'] = 0
    
    h_projection = np.sum(signature, axis=0)
    v_projection = np.sum(signature, axis=1)
    
    h_projection = h_projection / np.max(h_projection) if np.max(h_projection) > 0 else h_projection
    v_projection = v_projection / np.max(v_projection) if np.max(v_projection) > 0 else v_projection
    
    features['h_projection'] = h_projection
    features['v_projection'] = v_projection
    
    num_labels, labels = cv2.connectedComponents(signature)
    features['num_components'] = num_labels - 1
    
    contours, _ = cv2.findContours(signature, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        largest_contour = max(contours, key=cv2.contourArea)
        contour_area = cv2.contourArea(largest_contour)
        features['contour_area_ratio'] = contour_area / (h * w) if (h * w) > 0 else 0
        
        perimeter = cv2.arcLength(largest_contour, True)
        features['perimeter'] = perimeter / max(h, w) if max(h, w) > 0 else 0
        
        hull = cv2.convexHull(largest_contour)
        hull_area = cv2.contourArea(hull)
        features['convexity'] = contour_area / hull_area if hull_area > 0 else 0
        
        x, y, w_box, h_box = cv2.boundingRect(largest_contour)
        bbox_area = w_box * h_box
        features['bbox_ratio'] = contour_area / bbox_area if bbox_area > 0 else 0
    else:
        features['contour_area_ratio'] = 0
        features['perimeter'] = 0
        features['convexity'] = 0
        features['bbox_ratio'] = 0
    
    return features

def compare_boundary_signatures(img1_path, img2_path, debug=False):
    """Compare signatures using boundary point reconstruction method"""
    
    print("🔄 Extracting boundary-based signatures...")
    
    # Extract signatures using boundary point method
    raw1, recon1, boundary1 = extract_signature_boundary_points(img1_path, debug)
    raw2, recon2, boundary2 = extract_signature_boundary_points(img2_path, debug)
    
    print(f"📊 Signature 1: {len(boundary1)} boundary points")
    print(f"📊 Signature 2: {len(boundary2)} boundary points")
    
    # Extract enhanced features
    features1 = extract_enhanced_signature_features(recon1, boundary1)
    features2 = extract_enhanced_signature_features(recon2, boundary2)
    
    # Compare features
    similarities = compare_enhanced_features(features1, features2)
    
    # Calculate boundary point similarity
    boundary_similarity = compare_boundary_points(boundary1, boundary2)
    
    # Calculate final score with boundary point consideration
    weights = {
        'scalar_features': 0.30,
        'projections': 0.40,
        'boundary_points': 0.15,
        'aspect_ratio': 0.15
    }
    final_score = (
        weights['scalar_features'] * similarities['scalar_avg'] +
        weights['projections'] * (similarities['h_projection_corr'] + 
                                 similarities['v_projection_corr']) / 2 +
        weights['boundary_points'] * boundary_similarity +
        weights['aspect_ratio'] * similarities['aspect_ratio_similarity']
    )
    
    # Enhanced red flag detection
    red_flags = []
    
    if similarities['aspect_ratio_similarity'] < 0.7:
        red_flags.append("Very different aspect ratios")
    
    if similarities['density_similarity'] < 0.6:
        red_flags.append("Very different signature densities")
    
    if abs(len(boundary1) - len(boundary2)) / max(len(boundary1), len(boundary2), 1) > 0.5:
        red_flags.append("Very different number of boundary points")
    
    if boundary_similarity < 0.4:
        red_flags.append("Boundary point distributions are very different")
    
    if similarities['scalar_avg'] < 0.5:
        red_flags.append("Most features are dissimilar")
    
    # Adjust score based on red flags
    penalty = len(red_flags) * 0.08
    adjusted_score = max(0, final_score - penalty)
    
    results = {
        'raw1': raw1, 'raw2': raw2,
        'recon1': recon1, 'recon2': recon2,
        'boundary1': boundary1, 'boundary2': boundary2,
        'features1': features1, 'features2': features2,
        'similarities': similarities,
        'boundary_similarity': boundary_similarity,
        'final_score': final_score,
        'adjusted_score': adjusted_score,
        'red_flags': red_flags
    }
    
    return results

def compare_enhanced_features(features1, features2):
    """Enhanced feature comparison including boundary point features"""
    
    similarities = {}
    
    # Compare scalar features including boundary features
    scalar_features = ['aspect_ratio', 'density', 'centroid_x', 'centroid_y', 
                      'std_x', 'std_y', 'num_components', 'contour_area_ratio', 
                      'perimeter', 'convexity', 'bbox_ratio', 'boundary_density',
                      'boundary_centroid_x', 'boundary_centroid_y', 'boundary_std_x', 
                      'boundary_std_y', 'boundary_width', 'boundary_height']
    
    scalar_diffs = []
    for feature in scalar_features:
        val1 = features1.get(feature, 0)
        val2 = features2.get(feature, 0)
        
        max_val = max(abs(val1), abs(val2), 1e-6)
        diff = abs(val1 - val2) / max_val
        scalar_diffs.append(diff)
        similarities[f'{feature}_similarity'] = 1 - diff
    
    similarities['scalar_avg'] = 1 - np.mean(scalar_diffs)
    
    # Compare projections
    h_proj1 = features1['h_projection']
    h_proj2 = features2['h_projection']
    v_proj1 = features1['v_projection']
    v_proj2 = features2['v_projection']
    
    # Resize projections for comparison
    min_len_h = min(len(h_proj1), len(h_proj2))
    min_len_v = min(len(v_proj1), len(v_proj2))
    
    if min_len_h > 5:
        h_proj1_resized = cv2.resize(h_proj1.reshape(-1, 1), (1, min_len_h)).flatten()
        h_proj2_resized = cv2.resize(h_proj2.reshape(-1, 1), (1, min_len_h)).flatten()
        h_corr, _ = pearsonr(h_proj1_resized, h_proj2_resized)
        similarities['h_projection_corr'] = max(0, h_corr)
    else:
        similarities['h_projection_corr'] = 0
    
    if min_len_v > 5:
        v_proj1_resized = cv2.resize(v_proj1.reshape(-1, 1), (1, min_len_v)).flatten()
        v_proj2_resized = cv2.resize(v_proj2.reshape(-1, 1), (1, min_len_v)).flatten()
        v_corr, _ = pearsonr(v_proj1_resized, v_proj2_resized)
        similarities['v_projection_corr'] = max(0, v_corr)
    else:
        similarities['v_projection_corr'] = 0
    
    return similarities

import numpy as np

def compare_boundary_points(boundary1, boundary2):
    """Compare boundary point distributions with tolerance for minor variations"""
    # Better input validation
    if not boundary1 or not boundary2 or len(boundary1) == 0 or len(boundary2) == 0:
        return 0.0
    
    try:
        # Convert to numpy arrays with explicit dtype
        points1 = np.array(boundary1, dtype=np.float64)
        points2 = np.array(boundary2, dtype=np.float64)
        
        # Validate 2D points
        if points1.ndim != 2 or points2.ndim != 2 or points1.shape[1] != 2 or points2.shape[1] != 2:
            return 0.0
            
    except (ValueError, TypeError):
        return 0.0
    
    # Improved normalization function
    def normalize_points(points):
        if len(points) == 0:
            return points
        min_vals = np.min(points, axis=0)
        max_vals = np.max(points, axis=0)
        range_vals = max_vals - min_vals
        # Better handling of zero range
        range_vals = np.where(range_vals < 1e-10, 1.0, range_vals)
        normalized = (points - min_vals) / range_vals
        # Ensure values are in [0, 1]
        return np.clip(normalized, 0, 1)
    
    norm_points1 = normalize_points(points1)
    norm_points2 = normalize_points(points2)
    
    # Use coarser bins to be more forgiving of small variations
    bins = 20  # Reduced from 50 to be less sensitive to minor shifts
    
    # Create 2D histograms with Gaussian smoothing effect
    try:
        hist1, _, _ = np.histogram2d(
            norm_points1[:, 0], norm_points1[:, 1], 
            bins=bins, range=[[0, 1], [0, 1]], density=True
        )
        hist2, _, _ = np.histogram2d(
            norm_points2[:, 0], norm_points2[:, 1], 
            bins=bins, range=[[0, 1], [0, 1]], density=True
        )
        
        # Apply Gaussian blur to make comparison more forgiving
        from scipy.ndimage import gaussian_filter
        hist1 = gaussian_filter(hist1, sigma=0.8)
        hist2 = gaussian_filter(hist2, sigma=0.8)
        
    except ImportError:
        # Fallback without Gaussian blur
        hist1, _, _ = np.histogram2d(
            norm_points1[:, 0], norm_points1[:, 1], 
            bins=bins, range=[[0, 1], [0, 1]], density=True
        )
        hist2, _, _ = np.histogram2d(
            norm_points2[:, 0], norm_points2[:, 1], 
            bins=bins, range=[[0, 1], [0, 1]], density=True
        )
    except Exception:
        return 0.0
    
    # Normalize histograms properly
    hist1_sum = np.sum(hist1)
    hist2_sum = np.sum(hist2)
    
    if hist1_sum > 0:
        hist1 = hist1 / hist1_sum
    if hist2_sum > 0:
        hist2 = hist2 / hist2_sum
    
    # Calculate multiple similarity metrics that are more forgiving
    hist1_flat = hist1.flatten()
    hist2_flat = hist2.flatten()
    
    try:
        # 1. Histogram intersection (most forgiving)
        intersection = np.sum(np.minimum(hist1_flat, hist2_flat))
        
        # 2. Cosine similarity (angle between vectors)
        dot_product = np.dot(hist1_flat, hist2_flat)
        norm1 = np.linalg.norm(hist1_flat)
        norm2 = np.linalg.norm(hist2_flat)
        cosine_sim = dot_product / (norm1 * norm2 + 1e-10)
        cosine_sim = max(0, cosine_sim)
        
        # 3. Correlation (but with less weight)
        correlation_matrix = np.corrcoef(hist1_flat, hist2_flat)
        correlation = correlation_matrix[0, 1] if not np.isnan(correlation_matrix[0, 1]) else 0
        correlation = max(0, correlation)
        
        # Weight intersection and cosine similarity more heavily
        # These are more forgiving of small shifts
        combined_similarity = (intersection * 0.5 + cosine_sim * 0.3 + correlation * 0.2)
        
        # Apply a boost for high-quality matches to be less harsh
        if combined_similarity > 0.6:
            combined_similarity = combined_similarity * 0.8 + 0.2 * (combined_similarity ** 0.5)
        
        return max(0, min(1, combined_similarity))
        
    except Exception:
        return 0.0
    
def rotate_image(image, angle):
    """Rotate image around its center without cropping."""
    h, w = image.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated = cv2.warpAffine(image, M, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
    return rotated

def template_matching_with_rotation(img1, img2, angle_range=range(-15, 16, 3)):
    """
    Template matching with rotation search.
    :param img1: Template image
    :param img2: Search image
    :param angle_range: Rotation angles to try (degrees)
    """
    # Resize to standard size
    target_size = (200, 200)
    img1_norm = cv2.resize(img1, target_size)
    img2_norm = cv2.resize(img2, target_size)

    methods = [cv2.TM_CCOEFF_NORMED, cv2.TM_CCORR_NORMED]
    best_score = -1  # Initialize with lowest possible match score

    for angle in angle_range:
        rotated_template = rotate_image(img1_norm, angle)

        scores = []
        for method in methods:
            result1 = cv2.matchTemplate(img2_norm, rotated_template, method)
            result2 = cv2.matchTemplate(rotated_template, img2_norm, method)

            _, max_val1, _, _ = cv2.minMaxLoc(result1)
            _, max_val2, _, _ = cv2.minMaxLoc(result2)

            scores.extend([max_val1, max_val2])

        avg_score = np.mean(scores)
        best_score = max(best_score, avg_score)

    return best_score

def clean_signature_advanced(image_path, output_path=None):
    """Clean signature using BiRefNet for background removal and basic cropping"""
    # Read image
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Could not load image from {image_path}")
    
    pil_image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    
    processed_image, mask = birefnet_processor.remove_background(pil_image)    
    processed_np = np.array(processed_image)
    mask_np = np.array(processed_image.split()[-1])  # Get alpha channel
    
    # Convert to grayscale
    gray = cv2.cvtColor(processed_np, cv2.COLOR_RGBA2GRAY)
    
    # Apply mask to get clean signature
    binary = np.zeros_like(gray)
    binary[mask_np > 127] = 255
    
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        # Get the largest contour
        largest_contour = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(largest_contour)
        
        padding = 10
        x = max(0, x - padding)
        y = max(0, y - padding)
        w = min(binary.shape[1] - x, w + 2 * padding)
        h = min(binary.shape[0] - y, h + 2 * padding)
        
        # Crop the image
        cropped = binary[y:y+h, x:x+w]
    else:
        cropped = binary
    
    if output_path:
        cv2.imwrite(output_path, cropped)
    
    return cropped
