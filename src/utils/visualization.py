import matplotlib.pyplot as plt
import numpy as np
import cv2
import pywt

def apply_dwt(image, wavelet='db1', level=2):
    """Apply Discrete Wavelet Transform to the image"""
    # Ensure image is float32
    image = image.astype(np.float32)
    
    # Apply DWT
    coeffs = pywt.wavedec2(image, wavelet, level=level)
    
    # Extract coefficients
    cA = coeffs[0]  # Approximation coefficients
    cH = coeffs[1][0]  # Horizontal detail coefficients
    cV = coeffs[1][1]  # Vertical detail coefficients
    cD = coeffs[1][2]  # Diagonal detail coefficients
    
    return cA, cH, cV, cD

def visualize_boundary_results(results):
    """Visualize boundary-based signature matching results"""
    fig = plt.figure(figsize=(20, 16))
    fig.suptitle('Signature Analysis with DWT and SWIFT Features', fontsize=16)
    
    # Original and reconstructed signatures
    plt.subplot(4, 4, 1)
    plt.imshow(results['raw1'], cmap='gray')
    plt.title('Original Signature 1')
    plt.axis('off')
    
    plt.subplot(4, 4, 2)
    plt.imshow(results['recon1'], cmap='gray')
    plt.title('Reconstructed Signature 1')
    plt.axis('off')
    
    plt.subplot(4, 4, 3)
    plt.imshow(results['raw2'], cmap='gray')
    plt.title('Original Signature 2')
    plt.axis('off')
    
    plt.subplot(4, 4, 4)
    plt.imshow(results['recon2'], cmap='gray')
    plt.title('Reconstructed Signature 2')
    plt.axis('off')
    
    # Projections
    plt.subplot(4, 4, 5)
    plt.plot(results['features1']['h_projection'], label='Sig 1')
    plt.plot(results['features2']['h_projection'], label='Sig 2')
    plt.title('Horizontal Projections')
    plt.legend()
    
    plt.subplot(4, 4, 6)
    plt.plot(results['features1']['v_projection'], label='Sig 1')
    plt.plot(results['features2']['v_projection'], label='Sig 2')
    plt.title('Vertical Projections')
    plt.legend()
    
    # Boundary analysis - Combined plot
    plt.subplot(4, 4, 7) # Using the first slot for the combined plot
    boundary1 = np.array(results['boundary1'])
    boundary2 = np.array(results['boundary2'])
    
    if len(boundary1) > 0:
        plt.scatter(boundary1[:, 0], boundary1[:, 1], s=1, alpha=0.5, color='blue', label='Signature 1')
    if len(boundary2) > 0:
        plt.scatter(boundary2[:, 0], boundary2[:, 1], s=1, alpha=0.5, color='orange', label='Signature 2')
    
    plt.title('Boundary Points Overlay')
    plt.axis('equal')
    plt.gca().invert_yaxis() # Invert y-axis to match image coordinates
    plt.legend()
    
    # DWT Coefficients (Shifted to accommodate removed subplot)
    cA1, cH1, cV1, cD1 = apply_dwt(results['recon1'])
    cA2, cH2, cV2, cD2 = apply_dwt(results['recon2'])
    
    plt.subplot(4, 4, 8)
    plt.imshow(cA1, cmap='gray')
    plt.title('DWT cA1')
    plt.axis('off')
    
    plt.subplot(4, 4, 9)
    plt.imshow(cH1, cmap='gray')
    plt.title('DWT cH1')
    plt.axis('off')
    
    plt.subplot(4, 4, 10)
    plt.imshow(cA2, cmap='gray')
    plt.title('DWT cA2')
    plt.axis('off')
    
    plt.subplot(4, 4, 11)
    plt.imshow(cH2, cmap='gray')
    plt.title('DWT cH2')
    plt.axis('off')
    
    # SWIFT Features (Shifted to accommodate removed subplot)
    plt.subplot(4, 4, 12)
    swift_energy1 = results['features1']['swift_energy']
    swift_energy2 = results['features2']['swift_energy']
    plt.bar(['cA', 'cH', 'cV', 'cD'], [swift_energy1['cA'], swift_energy1['cH'], 
                                      swift_energy1['cV'], swift_energy1['cD']], 
            alpha=0.5, label='Sig 1')
    plt.bar(['cA', 'cH', 'cV', 'cD'], [swift_energy2['cA'], swift_energy2['cH'], 
                                      swift_energy2['cV'], swift_energy2['cD']], 
            alpha=0.5, label='Sig 2')
    plt.title('SWIFT Energy Features')
    plt.legend()
    
    plt.subplot(4, 4, 13)
    swift_entropy1 = results['features1']['swift_entropy']
    swift_entropy2 = results['features2']['swift_entropy']
    plt.bar(['cA', 'cH', 'cV', 'cD'], [swift_entropy1['cA'], swift_entropy1['cH'], 
                                      swift_entropy1['cV'], swift_entropy1['cD']], 
            alpha=0.5, label='Sig 1')
    plt.bar(['cA', 'cH', 'cV', 'cD'], [swift_entropy2['cA'], swift_entropy2['cH'], 
                                      swift_entropy2['cV'], swift_entropy2['cD']], 
            alpha=0.5, label='Sig 2')
    plt.title('SWIFT Entropy Features')
    plt.legend()
    
    # Similarity scores (Shifted to accommodate removed subplot)
    plt.subplot(4, 4, 14)
    scores = {
        'Scalar Features': results['similarities']['scalar_avg'],
        'Projections': (results['similarities']['h_projection_corr'] + 
                       results['similarities']['v_projection_corr']) / 2,
        'Boundary': results['boundary_similarity'],
        'SWIFT': results['similarities']['swift_avg']
    }
    plt.bar(scores.keys(), scores.values())
    plt.title('Similarity Scores')
    plt.xticks(rotation=45)
    
    # Results summary (Shifted to accommodate removed subplot)
    plt.subplot(4, 4, 15)
    plt.axis('off')
    summary_text = (
        f"Verification Results:\n\n"
        f"Final Score: {results['final_score']:.2f}\n"
        f"Adjusted Score: {results['adjusted_score']:.2f}\n\n"
        f"Individual Scores:\n"
        f"  Scalar Features: {results['similarities']['scalar_avg']:.2f}\n"
        f"  Projections: {((results['similarities']['h_projection_corr'] + results['similarities']['v_projection_corr']) / 2):.2f}\n"
        f"  Boundary: {results['boundary_similarity']:.2f}\n"
        f"  SWIFT: {results['similarities']['swift_avg']:.2f}\n\n"
        f"Red Flags:\n"
    )
    for flag in results['red_flags']:
        summary_text += f"• {flag}\n"
    plt.text(0.1, 0.5, summary_text, fontsize=10, va='center')
    
    plt.tight_layout()
    return fig 