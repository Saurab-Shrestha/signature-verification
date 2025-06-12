import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.distance import euclidean
from scipy.stats import pearsonr
from skimage import morphology
import torch
import torch.nn as nn
import functools
import torch.utils.data
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
from torchvision import transforms
from PIL import Image
import os
import random
import pandas as pd
import itertools
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, roc_curve, auc
import seaborn as sns
from skimage.feature import graycomatrix, graycoprops

# --- Start of Signature Verification Core Code (Moved from src/core/signature_verification.py) ---

def extract_signature_boundary_points(image_path, debug=False):
    """Extract signature using boundary point detection and reconstruction"""
    
    # Load and preprocess image
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Could not load image from {image_path}")
    
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Enhanced preprocessing for better edge detection
    # Apply Gaussian blur to reduce noise
    blurred = cv2.GaussianBlur(gray, (3, 3), 0)
    
    # Apply bilateral filter to preserve edges while reducing noise
    filtered = cv2.bilateralFilter(blurred, 9, 75, 75)
    
    # Apply median filter to remove salt and pepper noise
    filtered = cv2.medianBlur(filtered, 3)
    
    # Sobel edge detection with higher precision
    sobelx = cv2.Sobel(filtered, cv2.CV_64F, 1, 0, ksize=3)
    sobely = cv2.Sobel(filtered, cv2.CV_64F, 0, 1, ksize=3)
    
    # Calculate gradient magnitude and direction
    gradient_magnitude = cv2.magnitude(sobelx, sobely)
    gradient_direction = cv2.phase(sobelx, sobely, angleInDegrees=True)
    
    # Normalize gradient magnitude
    gradient_magnitude = cv2.convertScaleAbs(gradient_magnitude)
    
    if debug:
        plt.figure(figsize=(15, 5))
        plt.subplot(1, 4, 1)
        plt.imshow(gray, cmap='gray')
        plt.title('Original')
        plt.axis('off')
        
        plt.subplot(1, 4, 2)
        plt.imshow(gradient_magnitude, cmap='gray')
        plt.title('Gradient Magnitude')
        plt.axis('off')
        
        plt.subplot(1, 4, 3)
        plt.imshow(gradient_direction, cmap='hsv')
        plt.title('Gradient Direction')
        plt.axis('off')
    
    # Dynamic thresholding for edge detection
    # Use Otsu's method to find optimal threshold
    threshold_value, _ = cv2.threshold(gradient_magnitude, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    # Apply threshold with some adjustment
    threshold_value = max(20, threshold_value * 0.5)  # Ensure minimum threshold
    _, binary_edges = cv2.threshold(gradient_magnitude, threshold_value, 255, cv2.THRESH_BINARY)
    
    # Morphological operations to clean up edges
    kernel = np.ones((2, 2), np.uint8)
    
    # Close small gaps in edges
    closed_edges = cv2.morphologyEx(binary_edges, cv2.MORPH_CLOSE, kernel, iterations=1)
    
    # Remove small noise
    opened_edges = cv2.morphologyEx(closed_edges, cv2.MORPH_OPEN, kernel, iterations=1)
    
    # Find boundary points
    boundary_points = np.where(opened_edges == 255)
    boundary_coords = list(zip(boundary_points[1], boundary_points[0]))  # (x, y) format
    
    if len(boundary_coords) == 0:
        print("Warning: No boundary points found")
        return gray, gray, []
    
    # Filter boundary points to remove isolated noise
    boundary_coords = filter_boundary_points(boundary_coords, min_cluster_size=5)
    
    # Find bounding box of boundary points
    if len(boundary_coords) > 0:
        x_coords = [p[0] for p in boundary_coords]
        y_coords = [p[1] for p in boundary_coords]
        
        min_x, max_x = min(x_coords), max(x_coords)
        min_y, max_y = min(y_coords), max(y_coords)
        
        # Add padding
        padding = 5  # Reduced padding
        min_x = max(0, min_x - padding)
        min_y = max(0, min_y - padding)
        max_x = min(gray.shape[1], max_x + padding)
        max_y = min(gray.shape[0], max_y + padding)
        
        # Crop original image
        cropped_signature = gray[min_y:max_y, min_x:max_x]
        
        # Reconstruct signature using only boundary points
        reconstructed_signature = reconstruct_from_boundary_points(
            boundary_coords, (min_x, min_y, max_x, max_y), cropped_signature.shape
        )
    else:
        cropped_signature = gray
        reconstructed_signature = gray
    
    if debug:
        plt.subplot(1, 4, 4)
        plt.imshow(reconstructed_signature, cmap='gray')
        plt.title('Reconstructed from Boundaries')
        plt.axis('off')
        plt.tight_layout()
        plt.show()
    
    return cropped_signature, reconstructed_signature, boundary_coords

def filter_boundary_points(boundary_coords, min_cluster_size=5):
    """Filter boundary points to remove isolated noise"""
    if len(boundary_coords) < min_cluster_size:
        return boundary_coords
    
    # Convert to numpy array for easier processing
    points = np.array(boundary_coords)
    
    # Use DBSCAN-like approach to remove isolated points
    filtered_points = []
    processed = set()
    
    for i, point in enumerate(points):
        if i in processed:
            continue
            
        # Find nearby points with smaller radius
        distances = np.sqrt(np.sum((points - point) ** 2, axis=1))
        nearby_indices = np.where(distances <= 2)[0]  # Reduced from 3 to 2 pixels
        
        if len(nearby_indices) >= min_cluster_size:
            # This is a valid cluster
            for idx in nearby_indices:
                if idx not in processed:
                    filtered_points.append(tuple(points[idx]))
                    processed.add(idx)
    
    return filtered_points

def reconstruct_from_boundary_points(boundary_coords, bbox, target_shape):
    """Reconstruct signature image from boundary points"""
    min_x, min_y, max_x, max_y = bbox
    h, w = target_shape
    
    # Create empty image
    reconstructed = np.zeros((h, w), dtype=np.uint8)
    
    if len(boundary_coords) == 0:
        return reconstructed
    
    # Plot boundary points on the reconstructed image
    for x, y in boundary_coords:
        # Convert to local coordinates
        local_x = x - min_x
        local_y = y - min_y
        
        # Ensure coordinates are within bounds
        if 0 <= local_x < w and 0 <= local_y < h:
            reconstructed[local_y, local_x] = 255
    
    # Apply morphological operations to create continuous strokes
    kernel = np.ones((2, 2), np.uint8)
    
    # Dilate to thicken the lines slightly
    reconstructed = cv2.dilate(reconstructed, kernel, iterations=1)
    
    # Close small gaps
    kernel_close = np.ones((2, 2), np.uint8)  # Reduced from 3x3 to 2x2
    reconstructed = cv2.morphologyEx(reconstructed, cv2.MORPH_CLOSE, kernel_close, iterations=1)
    
    return reconstructed

def extract_enhanced_signature_features(signature, boundary_coords):
    """Extract features from both reconstructed signature and boundary points"""
    
    features = {}
    
    # Basic shape features
    h, w = signature.shape
    features['aspect_ratio'] = w / h if h > 0 else 0
    
    # Density features
    total_pixels = h * w
    signature_pixels = np.sum(signature > 0)
    features['density'] = signature_pixels / total_pixels if total_pixels > 0 else 0
    
    # Boundary point features
    features['num_boundary_points'] = len(boundary_coords)
    features['boundary_density'] = len(boundary_coords) / total_pixels if total_pixels > 0 else 0
    
    # Centroid and spread features
    if signature_pixels > 0:
        y_coords, x_coords = np.where(signature > 0)
        features['centroid_x'] = np.mean(x_coords) / w if w > 0 else 0
        features['centroid_y'] = np.mean(y_coords) / h if h > 0 else 0
        features['std_x'] = np.std(x_coords) / w if w > 0 else 0
        features['std_y'] = np.std(y_coords) / h if h > 0 else 0
    else:
        features['centroid_x'] = features['centroid_y'] = 0.5
        features['std_x'] = features['std_y'] = 0
    
    # Boundary point distribution features
    if len(boundary_coords) > 0:
        boundary_array = np.array(boundary_coords)
        features['boundary_centroid_x'] = np.mean(boundary_array[:, 0]) / w if w > 0 else 0
        features['boundary_centroid_y'] = np.mean(boundary_array[:, 1]) / h if h > 0 else 0
        features['boundary_std_x'] = np.std(boundary_array[:, 0]) / w if w > 0 else 0
        features['boundary_std_y'] = np.std(boundary_array[:, 1]) / h if h > 0 else 0
        
        # Calculate boundary point spread
        min_x, max_x = np.min(boundary_array[:, 0]), np.max(boundary_array[:, 0])
        min_y, max_y = np.min(boundary_array[:, 1]), np.max(boundary_array[:, 1])
        features['boundary_width'] = (max_x - min_x) / w if w > 0 else 0
        features['boundary_height'] = (max_y - min_y) / h if h > 0 else 0
    else:
        features['boundary_centroid_x'] = features['boundary_centroid_y'] = 0.5
        features['boundary_std_x'] = features['boundary_std_y'] = 0
        features['boundary_width'] = features['boundary_height'] = 0
    
    # Projection features
    h_projection = np.sum(signature, axis=0)
    v_projection = np.sum(signature, axis=1)
    
    # Normalize projections
    h_projection = h_projection / np.max(h_projection) if np.max(h_projection) > 0 else h_projection
    v_projection = v_projection / np.max(v_projection) if np.max(v_projection) > 0 else v_projection
    
    features['h_projection'] = h_projection
    features['v_projection'] = v_projection
    
    # Connected components
    num_labels, labels = cv2.connectedComponents(signature)
    features['num_components'] = num_labels - 1
    
    # Contour features (from reconstructed signature)
    contours, _ = cv2.findContours(signature, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        largest_contour = max(contours, key=cv2.contourArea)
        contour_area = cv2.contourArea(largest_contour)
        features['contour_area_ratio'] = contour_area / (h * w) if (h * w) > 0 else 0
        
        perimeter = cv2.arcLength(largest_contour, True)
        features['perimeter'] = perimeter / max(h, w) if max(h, w) > 0 else 0
        
        # Convex hull features
        hull = cv2.convexHull(largest_contour)
        hull_area = cv2.contourArea(hull)
        features['convexity'] = contour_area / hull_area if hull_area > 0 else 0
        
        # Bounding box features
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
    
    # Template matching on reconstructed signatures
    template_score = template_matching_with_validation(recon1, recon2)
    
    # Calculate boundary point similarity
    boundary_similarity = compare_boundary_points(boundary1, boundary2)
    
    # Calculate final score with boundary point consideration
    weights = {
        'template': 0.10,
        'scalar_features': 0.30,
        'projections': 0.35,
        'boundary_points': 0.10,
        'aspect_ratio': 0.15
    }
    final_score = (
        weights['template'] * template_score +
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
        'template_score': template_score,
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

def compare_boundary_points(boundary1, boundary2):
    """Compare boundary point distributions"""
    if len(boundary1) == 0 or len(boundary2) == 0:
        return 0.0
    
    # Convert to numpy arrays
    points1 = np.array(boundary1)
    points2 = np.array(boundary2)
    
    # Normalize coordinates to [0, 1] range
    def normalize_points(points):
        if len(points) == 0:
            return points
        min_vals = np.min(points, axis=0)
        max_vals = np.max(points, axis=0)
        range_vals = max_vals - min_vals
        range_vals[range_vals == 0] = 1  # Avoid division by zero
        return (points - min_vals) / range_vals
    
    norm_points1 = normalize_points(points1)
    norm_points2 = normalize_points(points2)
    
    # Calculate distribution similarity using histogram comparison
    # Create 2D histograms
    hist1, _, _ = np.histogram2d(norm_points1[:, 0], norm_points1[:, 1], bins=20, range=[[0, 1], [0, 1]])
    hist2, _, _ = np.histogram2d(norm_points2[:, 0], norm_points2[:, 1], bins=20, range=[[0, 1], [0, 1]])
    
    # Normalize histograms
    hist1 = hist1 / np.sum(hist1) if np.sum(hist1) > 0 else hist1
    hist2 = hist2 / np.sum(hist2) if np.sum(hist2) > 0 else hist2
    
    # Calculate correlation between histograms
    correlation = np.corrcoef(hist1.flatten(), hist2.flatten())[0, 1]
    correlation = max(0, correlation) if not np.isnan(correlation) else 0
    
    return correlation

def template_matching_with_validation(img1, img2):
    """Enhanced template matching for reconstructed signatures"""
    
    # Normalize images to same size
    target_size = (200, 100)
    img1_norm = cv2.resize(img1, target_size)
    img2_norm = cv2.resize(img2, target_size)
    
    # Multiple template matching methods
    methods = [cv2.TM_CCOEFF_NORMED, cv2.TM_CCORR_NORMED]
    scores = []
    
    for method in methods:
        result1 = cv2.matchTemplate(img2_norm, img1_norm, method)
        result2 = cv2.matchTemplate(img1_norm, img2_norm, method)
        
        _, max_val1, _, _ = cv2.minMaxLoc(result1)
        _, max_val2, _, _ = cv2.minMaxLoc(result2)
        
        scores.extend([max_val1, max_val2])
    
    template_score = np.mean(scores)
    
    return template_score

def visualize_boundary_results(results):
    """Visualize boundary-based signature matching results"""
    
    fig, axes = plt.subplots(3, 4, figsize=(16, 12))
    fig.suptitle('Boundary-Based Signature Analysis', fontsize=16)
    
    # Row 1: Original and reconstructed signatures
    axes[0,0].imshow(results['raw1'], cmap='gray')
    axes[0,0].set_title('Original Signature 1')
    axes[0,0].axis('off')
    
    axes[0,1].imshow(results['recon1'], cmap='gray')
    axes[0,1].set_title(f'Reconstructed 1\n({len(results["boundary1"])} boundary points)')
    axes[0,1].axis('off')
    
    axes[0,2].imshow(results['raw2'], cmap='gray')
    axes[0,2].set_title('Original Signature 2')
    axes[0,2].axis('off')
    
    axes[0,3].imshow(results['recon2'], cmap='gray')
    axes[0,3].set_title(f'Reconstructed 2\n({len(results["boundary2"])} boundary points)')
    axes[0,3].axis('off')
    
    # Row 2: Projections and boundary analysis
    features1 = results['features1']
    features2 = results['features2']
    similarities = results['similarities']
    
    axes[1,0].plot(features1['h_projection'], 'b-', label='Sig 1', alpha=0.7)
    axes[1,0].plot(features2['h_projection'], 'r-', label='Sig 2', alpha=0.7)
    axes[1,0].set_title(f'H-Proj (Corr: {similarities["h_projection_corr"]:.2f})')
    axes[1,0].legend()
    
    axes[1,1].plot(features1['v_projection'], 'b-', label='Sig 1', alpha=0.7)
    axes[1,1].plot(features2['v_projection'], 'r-', label='Sig 2', alpha=0.7)
    axes[1,1].set_title(f'V-Proj (Corr: {similarities["v_projection_corr"]:.2f})')
    axes[1,1].legend()
    
    # Boundary point comparison
    if len(results['boundary1']) > 0 and len(results['boundary2']) > 0:
        boundary1 = np.array(results['boundary1'])
        boundary2 = np.array(results['boundary2'])
        
        axes[1,2].scatter(boundary1[:, 0], boundary1[:, 1], alpha=0.6, s=0.1, label='Sig 1')
        axes[1,2].scatter(boundary2[:, 0], boundary2[:, 1], alpha=0.6, s=0.1, label='Sig 2')
        axes[1,2].set_title('Boundary Points Overlay')
        axes[1,2].legend()
        axes[1,2].invert_yaxis()
    
    # Feature comparison
    feature_names = ['aspect_ratio', 'density', 'boundary_density', 'num_components']
    feature_values1 = [features1.get(f, 0) for f in feature_names]
    feature_values2 = [features2.get(f, 0) for f in feature_names]
    
    x = np.arange(len(feature_names))
    width = 0.35
    
    axes[1,3].bar(x - width/2, feature_values1, width, label='Signature 1', alpha=0.8)
    axes[1,3].bar(x + width/2, feature_values2, width, label='Signature 2', alpha=0.8)
    axes[1,3].set_xticks(x)
    axes[1,3].set_xticklabels([f.replace('_', '\n') for f in feature_names], fontsize=8)
    axes[1,3].set_title('Key Features')
    axes[1,3].legend()
    
    # Row 3: Results and analysis
    # Similarity scores
    sim_names = ['Template', 'Features', 'Boundary', 'Projections']
    sim_scores = [
        results['template_score'],
        similarities['scalar_avg'],
        results['boundary_similarity'],
        (similarities['h_projection_corr'] + similarities['v_projection_corr']) / 2
    ]
    
    colors = ['green' if s > 0.6 else 'orange' if s > 0.4 else 'red' for s in sim_scores]
    axes[2,0].bar(sim_names, sim_scores, color=colors, alpha=0.7)
    axes[2,0].set_title('Similarity Scores')
    axes[2,0].set_ylim(0, 1)
    axes[2,0].tick_params(axis='x', rotation=45)
    
    # Results summary
    axes[2,1].axis('off')
    summary_text = f"""BOUNDARY-BASED RESULTS:

Template Score: {results['template_score']:.3f}
Feature Score: {similarities['scalar_avg']:.3f}
Boundary Score: {results['boundary_similarity']:.3f}

Final Score: {results['final_score']:.3f}
Adjusted Score: {results['adjusted_score']:.3f}

Red Flags: {len(results['red_flags'])}
"""
    axes[2,1].text(0.1, 0.9, summary_text, transform=axes[2,1].transAxes,
                   fontsize=9, verticalalignment='top', fontfamily='monospace')
    
    # Verdict
    axes[2,2].axis('off')
    
    if results['adjusted_score'] > 0.80 and len(results['red_flags']) == 0:
        verdict = "✅ STRONG MATCH"
        color = 'green'
    elif results['adjusted_score'] > 0.65 and len(results['red_flags']) <= 1:
        verdict = "⚠️ POSSIBLE MATCH"
        color = 'orange'
    else:
        verdict = "❌ NO MATCH"
        color = 'red'
    
    axes[2,2].text(0.5, 0.7, verdict, ha='center', va='center', 
                   transform=axes[2,2].transAxes, fontsize=14, 
                   bbox=dict(boxstyle="round,pad=0.3", facecolor=color, alpha=0.3))
    
    if results['red_flags']:# CycleGAN related code
        flag_text = "Red Flags:\n" + "\n".join(f"• {flag}" for flag in results['red_flags'][:3])
        axes[2,2].text(0.5, 0.3, flag_text, ha='center', va='center',
                       transform=axes[2,2].transAxes, fontsize=8, color='red')
    
    # Boundary statistics
    axes[2,3].axis('off')
    boundary_stats = f"""BOUNDARY STATISTICS:

Signature 1:
• Boundary Points: {len(results['boundary1'])}
• Density: {features1.get('boundary_density', 0):.4f}

Signature 2:
• Boundary Points: {len(results['boundary2'])}
• Density: {features2.get('boundary_density', 0):.4f}

Boundary Similarity: {results['boundary_similarity']:.3f}
"""
    axes[2,3].text(0.1, 0.9, boundary_stats, transform=axes[2,3].transAxes,
                   fontsize=9, verticalalignment='top', fontfamily='monospace')
    
    plt.tight_layout()
    return fig

class ResnetBlock(nn.Module):
    def __init__(self, dim, padding_type, norm_layer, use_dropout, use_bias):
        super().__init__()
        conv_block = []
        p = 0
        if padding_type == 'reflect':
            conv_block += [nn.ReflectionPad2d(1)]
        elif padding_type == 'replicate':
            conv_block += [nn.ReplicationPad2d(1)]
        elif padding_type == 'zero':
            p = 1
        else:
            raise NotImplementedError(f'padding [{padding_type}] is not implemented')

        conv_block += [nn.Conv2d(dim, dim, kernel_size=3, padding=p, bias=use_bias),
                       norm_layer(dim),
                       nn.ReLU(True)]

        if use_dropout:
            conv_block += [nn.Dropout(0.5)]

        # Second conv layer
        p = 0
        if padding_type == 'reflect':
            conv_block += [nn.ReflectionPad2d(1)]
        elif padding_type == 'replicate':
            conv_block += [nn.ReplicationPad2d(1)]
        elif padding_type == 'zero':
            p = 1

        conv_block += [nn.Conv2d(dim, dim, kernel_size=3, padding=p, bias=use_bias),
                       norm_layer(dim)]

        self.conv_block = nn.Sequential(*conv_block)

    def forward(self, x):
        return x + self.conv_block(x)

class ResnetGenerator(nn.Module):
    def __init__(self, input_nc, output_nc, ngf=64, norm_layer=nn.InstanceNorm2d,
                 use_dropout=False, n_blocks=9, padding_type='reflect'):
        assert n_blocks >= 0
        super().__init__()
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        model = [nn.ReflectionPad2d(3),
                 nn.Conv2d(input_nc, ngf, kernel_size=7, padding=0, bias=use_bias),
                 norm_layer(ngf),
                 nn.ReLU(True)]

        # Downsampling
        n_downsampling = 2
        for i in range(n_downsampling):
            mult = 2 ** i
            model += [nn.Conv2d(ngf * mult, ngf * mult * 2, kernel_size=3,
                                stride=2, padding=1, bias=use_bias),
                      norm_layer(ngf * mult * 2),
                      nn.ReLU(True)]

        # ResNet blocks
        mult = 2 ** n_downsampling
        for i in range(n_blocks):
            model += [ResnetBlock(ngf * mult, padding_type, norm_layer, use_dropout, use_bias)]

        # Upsampling
        for i in range(n_downsampling):
            mult = 2 ** (n_downsampling - i)
            model += [nn.ConvTranspose2d(ngf * mult, int(ngf * mult / 2),
                                         kernel_size=3, stride=2,
                                         padding=1, output_padding=1,
                                         bias=use_bias),
                      norm_layer(int(ngf * mult / 2)),
                      nn.ReLU(True)]

        model += [nn.ReflectionPad2d(3)]
        model += [nn.Conv2d(ngf, output_nc, kernel_size=7, padding=0)]
        model += [nn.Tanh()]

        self.model = nn.Sequential(*model)

    def forward(self, input):
        return self.model(input)

def transform_image(image_path):
    transform = transforms.Compose([
        transforms.Resize((200, 300)),
        transforms.ToTensor(),
        transforms.Normalize((0.5,) * 3, (0.5,) * 3)
    ])
    image = Image.open(image_path).convert('RGB')
    return transform(image).unsqueeze(0)  # Add batch dimension

def tensor_to_pil(tensor):
    image = tensor.squeeze(0).cpu().detach()
    image = image * 0.5 + 0.5  # De-normalize
    return transforms.ToPILImage()(image)

def clean_image_with_cyclegan(image_path, model_path, output_path, show_plot=True, forge=True):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    netG = ResnetGenerator(input_nc=3, output_nc=3, n_blocks=9).to(device)

    # Load model
    state_dict = torch.load(model_path, map_location=device)
    netG.load_state_dict(state_dict)
    netG.eval()

    # Inference
    input_tensor = transform_image(image_path).to(device)
    with torch.no_grad():
        output_tensor = netG(input_tensor)

    # Convert to PIL
    input_image = tensor_to_pil(input_tensor)
    output_image = tensor_to_pil(output_tensor)

    # Save output
    output_image.save(output_path)
    print(f"✅ Cleaned image saved to: {output_path}")

    if show_plot:
        plt.figure(figsize=(15, 5))
        plt.subplot(1, 2, 1)
        plt.imshow(input_image)
        plt.title("Original")
        plt.axis("off")

        plt.subplot(1, 2, 2)
        plt.imshow(output_image)
        plt.title("CycleGAN Output")
        plt.axis("off")

        plt.tight_layout()
        plt.show()

def clean_signature_advanced(image_path, output_path=None):
    """
    Advanced signature cleaning using adaptive thresholding and morphological operations
    Args:
        image_path: Path to input signature image
        output_path: Optional path to save cleaned image
    Returns:
        Cleaned signature image
    """
    # Read image
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Could not load image from {image_path}")
    
    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Apply bilateral filter to preserve edges while removing noise
    filtered = cv2.bilateralFilter(gray, 9, 75, 75)
    
    # Apply adaptive thresholding
    binary = cv2.adaptiveThreshold(
        filtered,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        11,  # Block size
        2    # C constant
    )
    
    # Remove small noise
    kernel = np.ones((2, 2), np.uint8)
    cleaned = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=1)
    
    # Close small gaps in signature
    kernel_close = np.ones((3, 3), np.uint8)
    cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_CLOSE, kernel_close, iterations=1)
    
    # Enhance contrast
    cleaned = cv2.convertScaleAbs(cleaned, alpha=1.2, beta=0)
    
    # Remove isolated pixels
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(cleaned, connectivity=8)
    min_size = 10  # Minimum size of connected components to keep
    cleaned = np.zeros_like(cleaned)
    for i in range(1, num_labels):
        if stats[i, cv2.CC_STAT_AREA] >= min_size:
            cleaned[labels == i] = 255
    
    # Smooth edges
    cleaned = cv2.GaussianBlur(cleaned, (3, 3), 0)
    _, cleaned = cv2.threshold(cleaned, 127, 255, cv2.THRESH_BINARY)
    
    if output_path:
        cv2.imwrite(output_path, cleaned)
    
    return cleaned

def clean_signature_with_denoising(image_path, output_path=None):
    """
    Clean signature using non-local means denoising and advanced preprocessing
    Args:
        image_path: Path to input signature image
        output_path: Optional path to save cleaned image
    Returns:
        Cleaned signature image
    """
    # Read image
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Could not load image from {image_path}")
    
    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Apply non-local means denoising
    denoised = cv2.fastNlMeansDenoising(gray, None, 10, 7, 21)
    
    # Enhance contrast using CLAHE
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    enhanced = clahe.apply(denoised)
    
    # Apply Otsu's thresholding
    _, binary = cv2.threshold(enhanced, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    
    # Remove small noise
    kernel = np.ones((2, 2), np.uint8)
    cleaned = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=1)
    
    # Close small gaps
    kernel_close = np.ones((3, 3), np.uint8)
    cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_CLOSE, kernel_close, iterations=1)
    
    # Remove isolated components
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(cleaned, connectivity=8)
    min_size = 15  # Minimum size of connected components to keep
    cleaned = np.zeros_like(cleaned)
    for i in range(1, num_labels):
        if stats[i, cv2.CC_STAT_AREA] >= min_size:
            cleaned[labels == i] = 255
    
    if output_path:
        cv2.imwrite(output_path, cleaned)
    
    return cleaned

def clean_signature_with_watershed(image_path, output_path=None):
    """
    Clean signature using watershed algorithm for better separation of touching components
    Args:
        image_path: Path to input signature image
        output_path: Optional path to save cleaned image
    Returns:
        Cleaned signature image
    """
    # Read image
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Could not load image from {image_path}")
    
    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Apply bilateral filter
    filtered = cv2.bilateralFilter(gray, 9, 75, 75)
    
    # Apply Otsu's thresholding
    _, binary = cv2.threshold(filtered, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    
    # Noise removal
    kernel = np.ones((3, 3), np.uint8)
    opening = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=2)
    
    # Sure background area
    sure_bg = cv2.dilate(opening, kernel, iterations=3)
    
    # Finding sure foreground area
    dist_transform = cv2.distanceTransform(opening, cv2.DIST_L2, 5)
    _, sure_fg = cv2.threshold(dist_transform, 0.7 * dist_transform.max(), 255, 0)
    sure_fg = np.uint8(sure_fg)
    
    # Finding unknown region
    unknown = cv2.subtract(sure_bg, sure_fg)
    
    # Marker labelling
    _, markers = cv2.connectedComponents(sure_fg)
    
    # Add one to all labels so that sure background is not 0, but 1
    markers = markers + 1
    
    # Mark the region of unknown with zero
    markers[unknown == 255] = 0
    
    # Apply watershed
    markers = cv2.watershed(image, markers)
    
    # Create output image
    cleaned = np.zeros_like(gray)
    cleaned[markers > 1] = 255
    
    # Remove small components
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(cleaned, connectivity=8)
    min_size = 20  # Minimum size of connected components to keep
    cleaned = np.zeros_like(cleaned)
    for i in range(1, num_labels):
        if stats[i, cv2.CC_STAT_AREA] >= min_size:
            cleaned[labels == i] = 255
    
    if output_path:
        cv2.imwrite(output_path, cleaned)
    
    return cleaned

def extract_texture_features(image_path):
    """
    Extract texture features from signature image using GLCM and gradient analysis
    Args:
        image_path: Path to signature image
    Returns:
        Dictionary of texture and gradient features
    """
    # Read image
    image = cv2.imread(image_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Calculate GLCM features
    distances = [1, 2, 3]
    angles = [0, np.pi/4, np.pi/2, 3*np.pi/4]
    glcm = graycomatrix(gray, distances, angles, 256, symmetric=True, normed=True)
    
    # Extract GLCM properties
    contrast = graycoprops(glcm, 'contrast').flatten()
    dissimilarity = graycoprops(glcm, 'dissimilarity').flatten()
    homogeneity = graycoprops(glcm, 'homogeneity').flatten()
    energy = graycoprops(glcm, 'energy').flatten()
    correlation = graycoprops(glcm, 'correlation').flatten()
    
    # Calculate gradient features
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    gradient_magnitude = np.sqrt(sobelx**2 + sobely**2)
    gradient_direction = np.arctan2(sobely, sobelx)
    
    return {
        'texture_features': {
            'contrast': contrast,
            'dissimilarity': dissimilarity,
            'homogeneity': homogeneity,
            'energy': energy,
            'correlation': correlation
        },
        'gradient_features': {
            'magnitude_mean': np.mean(gradient_magnitude),
            'magnitude_std': np.std(gradient_magnitude),
            'direction_mean': np.mean(gradient_direction),
            'direction_std': np.std(gradient_direction)
        }
    }

def compare_signatures_texture(img1_path, img2_path):
    """
    Compare signatures using texture and gradient features
    Args:
        img1_path: Path to first signature image
        img2_path: Path to second signature image
    Returns:
        Dictionary of comparison results
    """
    # Extract features
    features1 = extract_texture_features(img1_path)
    features2 = extract_texture_features(img2_path)
    
    # Calculate texture similarity
    texture_similarities = {}
    for feature in ['contrast', 'dissimilarity', 'homogeneity', 'energy', 'correlation']:
        texture_similarities[feature] = np.mean(
            np.corrcoef(
                features1['texture_features'][feature],
                features2['texture_features'][feature]
            )[0, 1]
        )
    
    # Calculate gradient similarity
    gradient_similarity = np.mean([
        np.abs(features1['gradient_features']['magnitude_mean'] - 
               features2['gradient_features']['magnitude_mean']),
        np.abs(features1['gradient_features']['direction_mean'] - 
               features2['gradient_features']['direction_mean'])
    ])
    
    # Combine similarities with equal weights
    weights = {
        'texture': 0.5,
        'gradient': 0.5
    }
    
    final_score = (
        weights['texture'] * np.mean(list(texture_similarities.values())) +
        weights['gradient'] * (1 - gradient_similarity)
    )
    
    return {
        'texture_similarities': texture_similarities,
        'gradient_similarity': 1 - gradient_similarity,
        'final_score': final_score
    }

def compare_signatures_with_matching_models(img1_path, img2_path):
    """
    Compare signatures using advanced image matching models
    Args:
        img1_path: Path to first signature image
        img2_path: Path to second signature image
    Returns:
        Dictionary of matching results
    """
    try:
        import kornia
        import torch
        from kornia.feature import LoFTR, DeDoDe, LightGlue
        from kornia.feature import SIFT, SuperPoint
    except ImportError:
        print("Please install required packages: pip install kornia torch")
        return None

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load images
    img1 = cv2.imread(img1_path)
    img2 = cv2.imread(img2_path)
    
    if img1 is None or img2 is None:
        raise ValueError("Could not load one or both images")
    
    # Convert to grayscale
    img1_gray = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    img2_gray = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
    
    # Convert to torch tensors
    img1_tensor = torch.from_numpy(img1_gray).float()[None, None] / 255.0
    img2_tensor = torch.from_numpy(img2_gray).float()[None, None] / 255.0
    
    # Initialize matchers
    matchers = {
        'LoFTR': LoFTR(pretrained='outdoor').to(device),
        'DeDoDe': DeDoDe(pretrained='indoor').to(device),
        'LightGlue': LightGlue(pretrained='superpoint').to(device)
    }
    
    results = {}
    
    # Run matching with each model
    for name, matcher in matchers.items():
        try:
            # Prepare input
            input_dict = {
                'image0': img1_tensor.to(device),
                'image1': img2_tensor.to(device)
            }
            
            # Get matches
            with torch.no_grad():
                matches = matcher(input_dict)
            
            # Calculate matching score
            if 'confidence' in matches:
                score = matches['confidence'].mean().item()
            else:
                # Fallback to number of matches
                score = len(matches['keypoints0']) / max(img1_gray.shape[0] * img1_gray.shape[1], 1)
            
            results[name] = {
                'score': score,
                'num_matches': len(matches['keypoints0']) if 'keypoints0' in matches else 0
            }
            
        except Exception as e:
            print(f"Error with {name} matcher: {str(e)}")
            results[name] = {'score': 0, 'num_matches': 0}
    
    # Calculate final score using weighted average
    weights = {
        'LoFTR': 0.4,
        'DeDoDe': 0.3,
        'LightGlue': 0.3
    }
    
    final_score = sum(results[name]['score'] * weights[name] for name in weights.keys())
    
    return {
        'matcher_results': results,
        'final_score': final_score
    }

def compare_signatures_combined(img1_path, img2_path):
    """
    Compare signatures using all available methods
    Args:
        img1_path: Path to first signature image
        img2_path: Path to second signature image
    Returns:
        Dictionary of combined comparison results
    """
    # Get boundary-based comparison
    boundary_results = compare_boundary_signatures(img1_path, img2_path)
    
    # Get texture-based comparison
    texture_results = compare_signatures_texture(img1_path, img2_path)
    
    # Get matching model comparison
    matching_results = compare_signatures_with_matching_models(img1_path, img2_path)
    
    # Updated weights incorporating matching models
    weights = {
        'boundary': 0.35,      # Reduced from 0.40
        'texture': 0.25,       # Reduced from 0.30
        'matching': 0.25,      # New weight for matching models
        'template': 0.15       # Reduced from 0.20
    }
    
    # Calculate final score
    final_score = (
        weights['boundary'] * boundary_results['adjusted_score'] +
        weights['texture'] * texture_results['final_score'] +
        weights['matching'] * (matching_results['final_score'] if matching_results else 0) +
        weights['template'] * boundary_results['template_score']
    )
    
    return {
        'boundary_results': boundary_results,
        'texture_results': texture_results,
        'matching_results': matching_results,
        'final_score': final_score,
        'weights': weights
    }

class SignatureVerifier:
    """Enhanced signature verification system using multiple feature analysis"""
    
    def __init__(self, threshold=0.75):
        self.threshold = threshold
        self.weights = {
            'boundary': 0.20,      # Boundary-based features (adjusted)
            'matching': 0.40,      # Matching models (adjusted)
            'template': 0.10,      # Template matching (adjusted)
            'projections': 0.30    # Projections (newly added)
        }
        
        # Initialize matching models if available
        self.matching_available = False # Initialize to False by default
        try:
            import kornia
            import torch
            from kornia.feature import LoFTR, DeDoDe, LightGlue
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            self.matchers = {
                'LoFTR': LoFTR(pretrained='outdoor').to(self.device),
                'DeDoDe': DeDoDe(pretrained='indoor').to(self.device),
                'LightGlue': LightGlue(pretrained='superpoint').to(self.device)
            }
            self.matching_available = True
        except ImportError:
            print("Matching models not available. Install with: pip install kornia torch")
            # self.matching_available remains False
            self.matchers = None
    
    def verify_signatures(self, img1_path, img2_path):
        """Main verification function with enhanced features"""
        try:
            # Extract basic features
            raw1, recon1, boundary1 = self.extract_signature_features(img1_path)
            raw2, recon2, boundary2 = self.extract_signature_features(img2_path)
            
            # Clean signatures using different methods
            cleaned1_adv = self.clean_signature_advanced(raw1)
            cleaned2_adv = self.clean_signature_advanced(raw2)
            cleaned1_denoise = self.clean_signature_with_denoising(raw1)
            cleaned2_denoise = self.clean_signature_with_denoising(raw2)
            
            # Extract enhanced features
            features1 = self.extract_enhanced_signature_features(recon1, boundary1)
            features2 = self.extract_enhanced_signature_features(recon2, boundary2)
            
            # Compare features
            similarities = self.compare_enhanced_features(features1, features2)
            template_score = self.template_matching_with_validation(recon1, recon2)
            
            # Get matching model comparison if available
            matching_score = 0
            if self.matching_available:
                matching_results = self.compare_signatures_with_matching_models(img1_path, img2_path)
                if matching_results:
                    matching_score = matching_results['final_score']
            
            # Calculate boundary similarity
            boundary_similarity = self.compare_boundary_points(boundary1, boundary2)
            
            # Calculate projection similarity
            projection_similarity = (similarities['h_projection_corr'] + similarities['v_projection_corr']) / 2
            
            # Calculate final score with all components
            final_score = (
                self.weights['boundary'] * boundary_similarity +
                self.weights['matching'] * matching_score +
                self.weights['template'] * template_score +
                self.weights['projections'] * projection_similarity
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
            
            if projection_similarity < 0.4:
                red_flags.append("Projection profiles are very different")
            
            if self.matching_available and matching_score < 0.3:
                red_flags.append("Feature matching shows significant differences")
            
            # Adjust score based on red flags
            penalty = len(red_flags) * 0.08
            adjusted_score = max(0, final_score - penalty)
            
            # Prepare detailed results dictionary
            results = {
                'adjusted_score': adjusted_score,
                'final_score': final_score,
                'feature_scores': {
                    'Template Matching': template_score,
                    'Scalar Features': similarities['scalar_avg'],
                    'Projections': projection_similarity,
                    'Boundary Points': boundary_similarity,
                    'Matching Models': matching_score
                },
                'red_flags': red_flags,
                'raw1': raw1, 'raw2': raw2,
                'recon1': recon1, 'recon2': recon2,
                'boundary1': boundary1, 'boundary2': boundary2,
                'features1': features1, 'features2': features2,
                'similarities': similarities
            }
            
            return results
            
        except Exception as e:
            print(f"Error processing {img1_path}, {img2_path}: {str(e)}")
            return {'adjusted_score': 0.0, 'final_score': 0.0, 'feature_scores': {},
                    'red_flags': [f"Error: {str(e)}"]}
    
    def compare_signatures_with_matching_models(self, img1_path, img2_path):
        """Compare signatures using advanced image matching models"""
        if not self.matching_available:
            return None
            
        # Load images
        img1 = cv2.imread(img1_path)
        img2 = cv2.imread(img2_path)
        
        if img1 is None or img2 is None:
            raise ValueError("Could not load one or both images")
        
        # Convert to grayscale
        img1_gray = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
        img2_gray = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
        
        # Convert to torch tensors
        img1_tensor = torch.from_numpy(img1_gray).float()[None, None] / 255.0
        img2_tensor = torch.from_numpy(img2_gray).float()[None, None] / 255.0
        
        results = {}
        
        # Run matching with each model
        for name, matcher in self.matchers.items():
            try:
                # Prepare input
                input_dict = {
                    'image0': img1_tensor.to(self.device),
                    'image1': img2_tensor.to(self.device)
                }
                
                # Get matches
                with torch.no_grad():
                    matches = matcher(input_dict)
                
                # Calculate matching score
                if 'confidence' in matches:
                    score = matches['confidence'].mean().item()
                else:
                    # Fallback to number of matches
                    score = len(matches['keypoints0']) / max(img1_gray.shape[0] * img1_gray.shape[1], 1)
                
                results[name] = {
                    'score': score,
                    'num_matches': len(matches['keypoints0']) if 'keypoints0' in matches else 0
                }
                
            except Exception as e:
                print(f"Error with {name} matcher: {str(e)}")
                results[name] = {'score': 0, 'num_matches': 0}
        
        # Calculate final score using weighted average
        matcher_weights = {
            'LoFTR': 0.4,
            'DeDoDe': 0.3,
            'LightGlue': 0.3
        }
        
        final_score = sum(results[name]['score'] * matcher_weights[name] for name in matcher_weights.keys())
        
        return {
            'matcher_results': results,
            'final_score': final_score
        }
# --- End of Signature Verification Core Code ---

def random_images(dataset_folder):
    random_images = []
    for person_folder in os.listdir(dataset_folder):
        person_folder_path = os.path.join(dataset_folder, person_folder)
        filenames = os.listdir(person_folder_path)
        if len(filenames) > 7:
            filenames = random.sample(filenames, 7)  # Select 7 random filenames
        for filename in filenames:
            random_images.append(os.path.join(person_folder_path, filename))
    return random_images

def duplet_dataset_preparation(dataset_folder):
    image_paths = random_images(dataset_folder)
    all_data = []

    for person_folder in os.listdir(dataset_folder):
        person_folder_path = os.path.join(dataset_folder, person_folder)
        genuine_images = []
        forged_images = []

        for filename in os.listdir(person_folder_path):
            filepath = os.path.join(person_folder_path, filename)
            if 'original' in filename or '-G-' in filename:
                genuine_images.append(filepath)
            if 'forgeries' in filename or '-F-' in filename or 'forge' in filename:
                forged_images.append(filepath)

        additional_images = random.sample(image_paths, 10)
        forged_images.extend(additional_images)

        num_genuine_images = len(genuine_images)
        num_forged_images = len(forged_images)
        num_combinations = min(num_genuine_images * (num_genuine_images - 1) // 2, 
                               num_genuine_images * num_forged_images)

        if num_combinations == 0:
            continue

        genuine_combinations = random.sample(list(itertools.combinations(genuine_images, 2)), num_combinations)
        forged_combinations = random.sample(list(itertools.product(genuine_images, forged_images)), num_combinations)

        for (image_1, image_2), (genuine_image, forged_image) in zip(genuine_combinations, forged_combinations):
            all_data.append([image_1, image_2, 0])  # genuine pair
            all_data.append([genuine_image, forged_image, 1])  # forgery pair

    df = pd.DataFrame(all_data, columns=['image1', 'image2', 'label'])
    return df

class DupletDataset(Dataset):
    """Dataset class for signature pairs"""
    
    def __init__(self, dataframe, transform=None):
        self.dataframe = dataframe
        self.transform = transform
    
    def __len__(self):
        return len(self.dataframe)
    
    def __getitem__(self, idx):
        row = self.dataframe.iloc[idx]
        return {
            'image1': row['image1'],
            'image2': row['image2'],
            'label': row['label']
        }

def evaluate_model(verifier, test_loader):
    """Evaluate the signature verification model"""
    predictions = []
    true_labels = []
    scores = []
    feature_scores_list = []
    image_pairs = []
    
    print("Evaluating model...")
    for i, batch in enumerate(test_loader):
        for j in range(len(batch['image1'])):
            img1_path = batch['image1'][j]
            img2_path = batch['image2'][j]
            true_label = batch['label'][j].item()
            
            # Get all results from the verifier
            verification_results = verifier.verify_signatures(img1_path, img2_path)
            
            # Extract data from the results dictionary
            score = verification_results['adjusted_score']
            prediction = verifier.predict(img1_path, img2_path)
            feature_scores = verification_results['feature_scores']
            
            predictions.append(prediction)
            true_labels.append(true_label)
            scores.append(score)
            feature_scores_list.append(feature_scores)
            image_pairs.append((img1_path, img2_path))
            
            if (i * len(batch['image1']) + j + 1) % 100 == 0:
                print(f"Processed {i * len(batch['image1']) + j + 1} samples...")
    
    return np.array(predictions), np.array(true_labels), np.array(scores), feature_scores_list, image_pairs

def calculate_metrics(predictions, true_labels, scores):
    """Calculate various performance metrics"""
    # Basic metrics
    accuracy = accuracy_score(true_labels, predictions)
    
    # Confusion matrix
    cm = confusion_matrix(true_labels, predictions)
    
    # Handle edge cases where confusion matrix might not be 2x2
    if cm.size == 1:
        # If all predictions are the same, create a 2x2 matrix
        if predictions[0] == 0:
            cm = np.array([[len(predictions), 0], [0, 0]])
        else:
            cm = np.array([[0, 0], [0, len(predictions)]])
    
    # Calculate FRR and FAR
    tn, fp, fn, tp = cm.ravel()
    
    # False Rejection Rate (FRR) - rejecting genuine signatures
    frr = fn / (fn + tp) if (fn + tp) > 0 else 0
    
    # False Acceptance Rate (FAR) - accepting forged signatures
    far = fp / (fp + tn) if (fp + tn) > 0 else 0
    
    # ROC curve
    fpr, tpr, thresholds = roc_curve(true_labels, scores)
    roc_auc = auc(fpr, tpr)
    
    return {
        'accuracy': accuracy,
        'confusion_matrix': cm,
        'frr': frr,
        'far': far,
        'roc_auc': roc_auc,
        'fpr': fpr,
        'tpr': tpr
    }

def plot_results(metrics, img1_path=None, img2_path=None, feature_scores=None, verifier=None):
    """Plot evaluation results and signature comparison"""
    if img1_path and img2_path and feature_scores:
        # Create a figure with 1 row and 3 columns
        fig, axes = plt.subplots(1, 3, figsize=(20, 6))
        
        # Load and display images
        img1 = cv2.imread(img1_path)
        img2 = cv2.imread(img2_path)
        
        if img1 is not None and img2 is not None:
            img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)
            img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)
            
            axes[0].imshow(img1)
            axes[0].set_title('Signature 1')
            axes[0].axis('off')
            
            axes[1].imshow(img2)
            axes[1].set_title('Signature 2')
            axes[1].axis('off')
        
        # Display scores
        axes[2].axis('off')
        score_text = "Feature Scores:\n\n"
        for feature, score in feature_scores.items():
            score_text += f"{feature}: {score:.3f}\n"
        
        # Calculate and add overall score using verifier's weights
        if verifier:
            weights = verifier.weights
        else:
            # Fallback weights if verifier is not provided (e.g., for overall metrics plot)
            weights = {
                'Boundary Points': 0.30,
                'Matching Models': 0.40,
                'Template Matching': 0.10,
                'Projections': 0.20
            }

        # Map feature_scores keys to weights keys if necessary
        # Note: 'Boundary Points' from feature_scores maps to 'boundary' in verifier.weights
        # 'Matching Models' from feature_scores maps to 'matching' in verifier.weights
        # 'Template Matching' from feature_scores maps to 'template' in verifier.weights
        # 'Projections' from feature_scores maps to 'projections' in verifier.weights
        
        mapped_feature_scores = {
            'boundary': feature_scores.get('Boundary Points', 0),
            'matching': feature_scores.get('Matching Models', 0),
            'template': feature_scores.get('Template Matching', 0),
            'projections': feature_scores.get('Projections', 0)
        }

        overall_score = sum(mapped_feature_scores[key] * weights[key] for key in weights.keys())

        score_text += f"\nOverall Score: {overall_score:.3f}"
        if overall_score >= 0.75:
            score_text += f"\n Matched"
        else:
            score_text += f"\n Unmatched"
            
        axes[2].text(0.1, 0.5, score_text, fontsize=12, fontfamily='monospace',
                    verticalalignment='center')
        
    else:
        # Original plotting code for when no images are provided
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # Confusion Matrix
        sns.heatmap(metrics['confusion_matrix'], annot=True, fmt='d', cmap='Blues', ax=axes[0,0])
        axes[0,0].set_title('Confusion Matrix')
        axes[0,0].set_xlabel('Predicted')
        axes[0,0].set_ylabel('Actual')
        axes[0,0].set_xticklabels(['Forged', 'Genuine'])
        axes[0,0].set_yticklabels(['Forged', 'Genuine'])
        
        # ROC Curve
        axes[0,1].plot(metrics['fpr'], metrics['tpr'], color='darkorange', lw=2,
                      label=f'ROC curve (AUC = {metrics["roc_auc"]*100:.2f}%)')
        axes[0,1].plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        axes[0,1].set_xlim([0.0, 1.0])
        axes[0,1].set_ylim([0.0, 1.05])
        axes[0,1].set_xlabel('False Positive Rate')
        axes[0,1].set_ylabel('True Positive Rate')
        axes[0,1].set_title('ROC Curve')
        axes[0,1].legend(loc="lower right")
        
        # Metrics Bar Plot
        metric_names = ['Accuracy', 'FRR', 'FAR', 'AUC']
        metric_values = [metrics['accuracy'], metrics['frr'], metrics['far'], metrics['roc_auc']]
        colors = ['green', 'orange', 'red', 'blue']
        
        bars = axes[1,0].bar(metric_names, metric_values, color=colors, alpha=0.7)
        axes[1,0].set_title('Performance Metrics')
        axes[1,0].set_ylim(0, 1)
        
        # Add value labels on bars
        for bar, value in zip(bars, metric_values):
            axes[1,0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                          f'{value:.3f}', ha='center', va='bottom')
        
        # Summary Text
        axes[1,1].axis('off')
        summary_text = f"""
        SIGNATURE VERIFICATION RESULTS
        
        Accuracy: {metrics['accuracy']:.3f}
        
        False Rejection Rate (FRR): {metrics['frr']:.3f}
        (Genuine signatures incorrectly rejected)
        
        False Acceptance Rate (FAR): {metrics['far']:.3f}
        (Forged signatures incorrectly accepted)
        
        AUC Score: {metrics['roc_auc']:.3f}
        
        Confusion Matrix:
        True Negatives (TN): {metrics['confusion_matrix'][0,0]}
        False Positives (FP): {metrics['confusion_matrix'][0,1]}
        False Negatives (FN): {metrics['confusion_matrix'][1,0]}
        True Positives (TP): {metrics['confusion_matrix'][1,1]}
        """
        
        axes[1,1].text(0.1, 0.9, summary_text, transform=axes[1,1].transAxes,
                      fontsize=11, verticalalignment='top', fontfamily='monospace')
    
    plt.tight_layout()
    plt.show()

def main(cedar_dataset):
    """Main function to run the evaluation"""
    # Prepare dataset
    print("Preparing dataset...")
    cedar_duplet = duplet_dataset_preparation(cedar_dataset)
    
    # Sample data for testing (remove this line for full dataset)
    cedar_duplet = cedar_duplet.sample(min(10, len(cedar_duplet)))
    
    print(f"Dataset prepared with {len(cedar_duplet)} pairs")
    print(f"Genuine pairs: {len(cedar_duplet[cedar_duplet['label']==1])}")
    print(f"Forged pairs: {len(cedar_duplet[cedar_duplet['label']==0])}")
    
    # Create dataset
    dataset = DupletDataset(cedar_duplet)
    
    # Split dataset
    indices = list(range(len(dataset)))
    split = int(np.floor(0.80 * len(dataset)))
    validation = int(np.floor(0.70 * split))
    np.random.shuffle(indices)
    
    train_indices, validation_indices, test_indices = (
        indices[:validation],
        indices[validation:split],
        indices[split:],
    )
    
    # Create data loaders
    batch_size = 32
    test_sampler = SubsetRandomSampler(test_indices)
    test_loader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, sampler=test_sampler
    )
    
    print(f"Test set size: {len(test_indices)}")
    
    # Initialize verifier
    verifier = SignatureVerifier(threshold=0.75)
    
    # Evaluate model
    predictions, true_labels, scores, feature_scores_list, image_pairs = evaluate_model(verifier, test_loader)
    
    # Calculate metrics
    metrics = calculate_metrics(predictions, true_labels, scores)
    
    # Print results
    print("\n" + "="*50)
    print("SIGNATURE VERIFICATION RESULTS")
    print("="*50)
    print(f"Accuracy: {metrics['accuracy']:.4f}")
    print(f"False Rejection Rate (FRR): {metrics['frr']:.4f}")
    print(f"False Acceptance Rate (FAR): {metrics['far']:.4f}")
    print(f"AUC Score: {metrics['roc_auc']:.4f}")
    print("\nConfusion Matrix:")
    print(metrics['confusion_matrix'])
    
    # Plot results for a few sample pairs
    print("\nPlotting sample signature comparisons...")
    num_samples = min(5, len(image_pairs))
    sample_indices = np.random.choice(len(image_pairs), num_samples, replace=False)
    
    for idx in sample_indices:
        img1_path, img2_path = image_pairs[idx]
        feature_scores = feature_scores_list[idx]
        plot_results(metrics, img1_path, img2_path, feature_scores, verifier)
    
    # Plot overall metrics
    print("\nPlotting overall performance metrics...")
    plot_results(metrics)
    
    return verifier, metrics