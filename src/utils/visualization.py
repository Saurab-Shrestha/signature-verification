import matplotlib.pyplot as plt
import numpy as np

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
        
        axes[1,2].scatter(boundary1[:, 0], boundary1[:, 1], alpha=0.6, s=1, label='Sig 1')
        axes[1,2].scatter(boundary2[:, 0], boundary2[:, 1], alpha=0.6, s=1, label='Sig 2')
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
    sim_names = ['Features', 'Boundary', 'Projections']
    sim_scores = [
        similarities['scalar_avg'],
        results['boundary_similarity'],
        (similarities['h_projection_corr'] + similarities['v_projection_corr']) / 2
    ]
    
    colors = ['green' if s > 0.80 else 'orange' if s > 0.65 else 'red' for s in sim_scores]
    axes[2,0].bar(sim_names, sim_scores, color=colors, alpha=0.7)
    axes[2,0].set_title('Similarity Scores')
    axes[2,0].set_ylim(0, 1)
    axes[2,0].tick_params(axis='x', rotation=45)
    
    # Results summary
    axes[2,1].axis('off')
    summary_text = f"""BOUNDARY-BASED RESULTS:

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
    
    if results['adjusted_score'] > 0.75 and len(results['red_flags']) == 0:
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
    
    if results['red_flags']:
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